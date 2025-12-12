# ---------------------------------------------
# Sparse Temporal Self-Attention for BEVFormer with DiffFP
# ---------------------------------------------
# Only updates BEV cells that have significant changes
# Maintains full context from previous frame while sparsely updating current
# ---------------------------------------------

import warnings
import torch
import torch.nn as nn
import math
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32


@ATTENTION.register_module()
class SparseTemporalSelfAttention(BaseModule):
    """Sparse Temporal Self-Attention with DiffFP support.
    
    This module extends the original TemporalSelfAttention to support
    sparse updates based on DiffFP masks. Only BEV cells marked as 
    "changed" will be updated through attention, while unchanged cells
    directly inherit from the previous BEV.
    
    Two modes of operation:
    1. Full mode (sparse_mask=None): Behaves like original TemporalSelfAttention
    2. Sparse mode (sparse_mask provided): Only updates masked positions
    
    Args:
        embed_dims (int): Embedding dimension. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        num_levels (int): Number of feature levels. Default: 4.
        num_points (int): Number of sampling points per query. Default: 4.
        num_bev_queue (int): Number of BEV frames (current + history). Default: 2.
        im2col_step (int): im2col step for CUDA kernel. Default: 64.
        dropout (float): Dropout rate. Default: 0.1.
        batch_first (bool): Whether batch is first dim. Default: True.
        sparse_mode (str): How to handle sparse updates.
            - 'query_sparse': Only query changed positions (Option 2 in design)
            - 'kv_sparse': Full query, but K/V only from changed positions
            - 'hybrid': Query sparse + K/V from full prev_bev
            Default: 'query_sparse'.
        use_residual_gate (bool): Use learnable gate for residual connection.
            Default: False.
    """
    
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 sparse_mode='query_sparse',
                 use_residual_gate=False):
        super().__init__(init_cfg)
        
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False
        self.sparse_mode = sparse_mode
        self.use_residual_gate = use_residual_gate
        
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(f'invalid input for _is_power_of_2: {n}')
            return (n & (n - 1) == 0) and n != 0
        
        if not _is_power_of_2(dim_per_head):
            warnings.warn("dim_per_head should be power of 2 for CUDA efficiency")
            
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        
        # Same architecture as original TemporalSelfAttention
        self.sampling_offsets = nn.Linear(
            embed_dims * self.num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims * self.num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        # Optional: learnable gate for residual connection
        if use_residual_gate:
            self.residual_gate = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, 1),
                nn.Sigmoid()
            )
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32
        ) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2
        ).repeat(1, self.num_levels * self.num_bev_queue, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
            
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True
        
    def forward_full(self,
                     query,
                     key=None,
                     value=None,
                     identity=None,
                     query_pos=None,
                     reference_points=None,
                     spatial_shapes=None,
                     level_start_index=None,
                     **kwargs):
        """Full attention mode (original behavior).
        
        This is essentially the same as original TemporalSelfAttention.
        """
        if value is None:
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)
            
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
            
        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2
        
        # Concat prev_bev features with current query for offset/weight prediction
        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)
        
        value = value.reshape(bs * self.num_bev_queue, num_value, self.num_heads, -1)
        
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_bev_queue,
            self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue,
            self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_bev_queue,
            self.num_levels, self.num_points
        )
        
        # Reshape for deformable attention
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5).reshape(
            bs * self.num_bev_queue, num_query, self.num_heads,
            self.num_levels, self.num_points
        ).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            bs * self.num_bev_queue, num_query, self.num_heads,
            self.num_levels, self.num_points, 2
        )
        
        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f'reference_points last dim must be 2 or 4')
            
        # Apply deformable attention
        if torch.cuda.is_available() and value.is_cuda:
            MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights, self.im2col_step
            )
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
            
        # Fuse temporal outputs by averaging
        output = output.permute(1, 2, 0)  # [num_query, embed_dims, bs*num_bev_queue]
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)  # Average over temporal dimension
        output = output.permute(2, 0, 1)  # [bs, num_query, embed_dims]
        
        output = self.output_proj(output)
        
        return self.dropout(output) + identity
    
    def forward_sparse(self,
                       query,
                       prev_bev,
                       sparse_mask,
                       identity=None,
                       query_pos=None,
                       reference_points=None,
                       spatial_shapes=None,
                       level_start_index=None,
                       bev_h=None,
                       bev_w=None,
                       **kwargs):
        """Sparse attention mode - only update masked positions.
        
        Args:
            query: Current BEV query [B, H*W, C]
            prev_bev: Previous BEV features [B, H*W, C] 
            sparse_mask: Boolean mask [B, H*W] where True = needs update
            identity: Residual connection tensor
            query_pos: Positional encoding for query
            reference_points: Reference points for deformable attention
            spatial_shapes: Spatial shapes tensor
            level_start_index: Level start indices
            bev_h, bev_w: BEV grid dimensions
            
        Returns:
            Updated BEV features [B, H*W, C]
        """
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
            
        bs, num_query, embed_dims = query.shape
        device = query.device
        
        # Initialize output with identity (unchanged positions keep original)
        output = identity.clone()
        
        if sparse_mask is None or sparse_mask.all():
            # Fallback to full attention if no mask or all positions need update
            value = torch.stack([prev_bev, query], 1).reshape(bs * 2, num_query, -1)
            return self.forward_full(
                query, value=value, identity=identity,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs
            )
        
        # Process each batch item separately due to variable sparse counts
        for b in range(bs):
            mask_b = sparse_mask[b]  # [H*W]
            sparse_indices = mask_b.nonzero(as_tuple=True)[0]  # [K]
            num_sparse = sparse_indices.shape[0]
            
            if num_sparse == 0:
                # No updates needed for this batch - keep prev_bev
                output[b] = prev_bev[b]
                continue
                
            # Extract sparse queries
            query_sparse = query[b:b+1, sparse_indices, :]  # [1, K, C]
            prev_bev_sparse = prev_bev[b:b+1, sparse_indices, :]  # [1, K, C]
            identity_sparse = identity[b:b+1, sparse_indices, :]  # [1, K, C]
            
            # Get reference points for sparse positions
            ref_points_sparse = reference_points[b:b+1, sparse_indices, :, :]  # [1, K, L, 2]
            
            # Build value: [prev_bev_sparse, query_sparse] for temporal fusion
            if self.sparse_mode == 'hybrid':
                # hybrid mode requires custom deformable attention implementation
                # that can sample from full prev_bev while using sparse queries.
                # For now, fall back to query_sparse mode with a warning.
                # TODO: Implement proper hybrid mode with full K/V sampling
                raise NotImplementedError(
                    "hybrid mode is not yet implemented. Use 'query_sparse' instead. "
                    "hybrid mode would require sampling from full prev_bev grid, "
                    "which needs custom deformable attention logic."
                )
            
            # query_sparse mode: both Q and K/V from sparse positions
            # This is efficient and maintains consistency
            value = torch.stack([prev_bev_sparse, query_sparse], 1)
            value = value.reshape(2, num_sparse, -1)
            value = self.value_proj(value)
                
            # Concat for offset/weight prediction
            query_concat = torch.cat([prev_bev_sparse, query_sparse], -1)  # [1, K, 2C]
            
            value = value.reshape(2, num_sparse, self.num_heads, -1)
            
            # Compute sampling offsets and attention weights
            sampling_offsets = self.sampling_offsets(query_concat)
            sampling_offsets = sampling_offsets.view(
                1, num_sparse, self.num_heads, self.num_bev_queue,
                self.num_levels, self.num_points, 2
            )
            attention_weights = self.attention_weights(query_concat).view(
                1, num_sparse, self.num_heads, self.num_bev_queue,
                self.num_levels * self.num_points
            )
            attention_weights = attention_weights.softmax(-1).view(
                1, num_sparse, self.num_heads, self.num_bev_queue,
                self.num_levels, self.num_points
            )
            
            # Reshape for deformable attention
            attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5).reshape(
                2, num_sparse, self.num_heads, self.num_levels, self.num_points
            ).contiguous()
            sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
                2, num_sparse, self.num_heads, self.num_levels, self.num_points, 2
            )
            
            # Compute sampling locations (using sparse reference points)
            # Expand ref_points to match [2, K, 1, L, 1, 2]
            ref_pts_expanded = ref_points_sparse.expand(2, -1, -1, -1)
            
            if ref_pts_expanded.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
                )
                sampling_locations = ref_pts_expanded[:, :, None, :, None, :] \
                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            else:
                sampling_locations = ref_pts_expanded[:, :, None, :, None, :2] \
                    + sampling_offsets / self.num_points \
                    * ref_pts_expanded[:, :, None, :, None, 2:] * 0.5
            
            # Apply deformable attention
            if torch.cuda.is_available() and value.is_cuda:
                output_sparse = MultiScaleDeformableAttnFunction_fp32.apply(
                    value, spatial_shapes, level_start_index,
                    sampling_locations, attention_weights, self.im2col_step
                )
            else:
                output_sparse = multi_scale_deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights
                )
            
            # Fuse temporal outputs
            output_sparse = output_sparse.permute(1, 2, 0)  # [K, C, 2]
            output_sparse = output_sparse.mean(-1)  # [K, C]
            output_sparse = self.output_proj(output_sparse)
            
            # Apply residual connection and dropout
            if self.use_residual_gate:
                gate = self.residual_gate(
                    torch.cat([output_sparse, identity_sparse.squeeze(0)], -1)
                )
                output_sparse = gate * output_sparse + (1 - gate) * identity_sparse.squeeze(0)
            else:
                output_sparse = self.dropout(output_sparse) + identity_sparse.squeeze(0)
            
            # Place sparse outputs back into full output
            output[b, sparse_indices] = output_sparse
            
            # For unchanged positions, use previous BEV
            unchanged_indices = (~mask_b).nonzero(as_tuple=True)[0]
            if len(unchanged_indices) > 0:
                output[b, unchanged_indices] = prev_bev[b, unchanged_indices]
        
        return output
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                sparse_mask=None,
                prev_bev_aligned=None,
                flag='decoder',
                **kwargs):
        """Forward function.
        
        Args:
            query: Current BEV query [B, H*W, C]
            key: Key tensor (unused in temporal self-attention)
            value: Value tensor - if None, uses query stacked with itself
            identity: Tensor for residual connection
            query_pos: Positional encoding
            sparse_mask: Optional boolean mask [B, H*W] for sparse updates
            prev_bev_aligned: Previous BEV aligned to current coordinates
            Other args: Standard attention arguments
            
        Returns:
            Updated BEV features [B, H*W, C]
        """
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            if value is not None:
                value = value.permute(1, 0, 2)
                
        # Decide whether to use sparse or full mode
        if sparse_mask is not None and prev_bev_aligned is not None:
            # Sparse mode
            output = self.forward_sparse(
                query=query,
                prev_bev=prev_bev_aligned,
                sparse_mask=sparse_mask,
                identity=identity,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs
            )
        else:
            # Full mode (original behavior)
            if value is None and self.batch_first:
                bs, len_bev, c = query.shape
                value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)
            
            output = self.forward_full(
                query=query,
                key=key,
                value=value,
                identity=identity,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs
            )
            
        if not self.batch_first:
            output = output.permute(1, 0, 2)
            
        return output
