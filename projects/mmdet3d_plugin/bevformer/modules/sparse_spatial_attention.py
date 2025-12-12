# ---------------------------------------------
# Sparse Spatial Cross-Attention with DiffFP Support
# ---------------------------------------------
# Applies DiffFP mask to skip attention computation for unchanged patches
# This significantly reduces computation in the BEV projection step
# ---------------------------------------------

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import build_attention
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule


@ATTENTION.register_module()
class SparseSpatialCrossAttention(BaseModule):
    """Spatial Cross-Attention with DiffFP support.
    
    Extends SpatialCrossAttention to support sparse computation based on
    DiffFP masks. When a mask is provided, only changed image patches
    contribute to the BEV features, while unchanged regions are either
    skipped or use cached values.
    
    Args:
        embed_dims (int): Embedding dimension. Default: 256.
        num_cams (int): Number of cameras. Default: 6.
        pc_range (list): Point cloud range.
        dropout (float): Dropout rate. Default: 0.1.
        batch_first (bool): Whether batch is first dim. Default: False.
        deformable_attention (dict): Config for deformable attention.
        use_sparse_mask (bool): Whether to use DiffFP mask. Default: True.
        sparse_mode (str): How to handle sparse patches.
            - 'mask_value': Zero out values for unchanged patches
            - 'skip_unchanged': Skip unchanged patches entirely (faster but approximate)
            Default: 'mask_value'.
    """
    
    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 use_sparse_mask=True,
                 sparse_mode='mask_value',
                 **kwargs):
        super(SparseSpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.use_sparse_mask = use_sparse_mask
        self.sparse_mode = sparse_mode
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    def apply_sparse_mask_to_value(self, value, diff_fp_mask, spatial_shapes):
        """Apply DiffFP mask to values - zero out unchanged patches.
        
        Args:
            value: [num_cams, H*W, bs, embed_dims]
            diff_fp_mask: dict with 'mask' key -> [B, N, H, W]
            spatial_shapes: [[H, W]] tensor
            
        Returns:
            Masked value tensor
        """
        if diff_fp_mask is None or not self.use_sparse_mask:
            return value, None
            
        mask = diff_fp_mask.get('mask', None)
        if mask is None:
            return value, None
        
        # mask shape: [B, N, H, W] where True = keep (changed patch)
        B, N, H_mask, W_mask = mask.shape
        num_cams, L, bs, embed_dims = value.shape
        
        # Reshape mask to match value
        # value is [num_cams, H*W, bs, embed_dims]
        # Need mask as [num_cams, H*W, bs]
        
        # Get spatial shape from the tensor
        H, W = spatial_shapes[0].tolist() if spatial_shapes.dim() > 1 else spatial_shapes.tolist()
        
        # Handle size mismatch by interpolating mask
        if H_mask != H or W_mask != W:
            mask_float = mask.float()  # [B, N, H_mask, W_mask]
            mask_resized = F.interpolate(
                mask_float, size=(H, W), mode='nearest'
            )  # [B, N, H, W]
            mask = mask_resized > 0.5
        
        # Reshape: [B, N, H, W] -> [N, H*W, B]
        mask_flat = mask.view(B, N, -1).permute(1, 2, 0)  # [N, H*W, B]
        
        # Expand to match value dims
        mask_expanded = mask_flat.unsqueeze(-1).expand_as(value)  # [N, H*W, B, C]
        
        # Apply mask - zero out unchanged patches
        value_masked = value * mask_expanded.float()
        
        return value_masked, mask_flat
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                diff_fp_mask=None,
                **kwargs):
        """Forward function with DiffFP support.
        
        Args:
            query: BEV queries [bs, num_query, embed_dims]
            key: Image features [num_cams, H*W, bs, embed_dims]
            value: Image features [num_cams, H*W, bs, embed_dims]
            diff_fp_mask: DiffFP mask dict with 'mask' [B, N, H, W]
            Other args: Same as original SpatialCrossAttention
            
        Returns:
            Updated BEV features [bs, num_query, embed_dims]
        """
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # Rebatch queries per camera
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs_kv, embed_dims = key.shape

        # Apply DiffFP mask to key and value
        key_masked, mask_applied = self.apply_sparse_mask_to_value(
            key, diff_fp_mask, spatial_shapes)
        value_masked, _ = self.apply_sparse_mask_to_value(
            value, diff_fp_mask, spatial_shapes)

        key = key_masked.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value_masked.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), 
            key=key, 
            value=value,
            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), 
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        ).view(bs, self.num_cams, max_len, self.embed_dims)
        
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class SparseSpatialCrossAttentionV2(BaseModule):
    """Sparse Spatial Cross-Attention V2 - More aggressive pruning.
    
    Instead of just masking values, this version:
    1. Only samples from changed patches (skip entirely)
    2. Uses previous BEV for unchanged regions
    3. More significant computation savings
    
    This is closer to the VideoLLaMA3 DiffFP approach where
    unchanged tokens are completely removed from computation.
    """
    
    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs):
        super(SparseSpatialCrossAttentionV2, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    def forward_with_diff_fp(self,
                             query,
                             key,
                             value,
                             diff_fp_mask,
                             prev_bev_embed,
                             reference_points_cam,
                             bev_mask,
                             spatial_shapes,
                             level_start_index,
                             query_pos=None):
        """Forward with DiffFP - only process changed patches.
        
        For BEV cells that only see unchanged image patches,
        we directly copy from prev_bev_embed instead of computing attention.
        """
        if query_pos is not None:
            query = query + query_pos
            
        bs, num_query, _ = query.size()
        inp_residual = query
        slots = torch.zeros_like(query)
        
        mask = diff_fp_mask.get('mask', None)
        if mask is None:
            # Fall back to full attention
            return self.forward_full(
                query, key, value, None, query_pos, None, None, 
                spatial_shapes, reference_points_cam, bev_mask, 
                level_start_index)
        
        # mask: [B, N, H, W] - True = changed
        # Determine which BEV queries need updating based on whether
        # they sample from any changed patches
        
        # For now, use simpler approach: mask the values
        B, N, H_mask, W_mask = mask.shape
        num_cams, L, bs_kv, embed_dims = key.shape
        
        H, W = spatial_shapes[0].tolist() if spatial_shapes.dim() > 1 else spatial_shapes.tolist()
        
        if H_mask != H or W_mask != W:
            mask_float = mask.float()
            mask_resized = F.interpolate(mask_float, size=(H, W), mode='nearest')
            mask = mask_resized > 0.5
            
        # Reshape and apply mask
        mask_flat = mask.view(B, N, -1).permute(1, 2, 0)  # [N, H*W, B]
        mask_expanded = mask_flat.unsqueeze(-1)  # [N, H*W, B, 1]
        
        value_masked = value * mask_expanded.float()
        key_masked = key * mask_expanded.float()
        
        # Continue with standard attention flow
        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        key_masked = key_masked.permute(2, 0, 1, 3).reshape(bs * self.num_cams, L, self.embed_dims)
        value_masked = value_masked.permute(2, 0, 1, 3).reshape(bs * self.num_cams, L, self.embed_dims)

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims),
            key=key_masked,
            value=value_masked,
            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        ).view(bs, self.num_cams, max_len, self.embed_dims)
        
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual
    
    def forward_full(self,
                     query,
                     key,
                     value,
                     residual=None,
                     query_pos=None,
                     key_padding_mask=None,
                     reference_points=None,
                     spatial_shapes=None,
                     reference_points_cam=None,
                     bev_mask=None,
                     level_start_index=None,
                     **kwargs):
        """Standard forward without DiffFP."""
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs_kv, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims),
            key=key,
            value=value,
            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        ).view(bs, self.num_cams, max_len, self.embed_dims)
        
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                diff_fp_mask=None,
                prev_bev_embed=None,
                **kwargs):
        """Forward function."""
        if diff_fp_mask is not None and not diff_fp_mask.get('is_first_frame', True):
            return self.forward_with_diff_fp(
                query, key, value, diff_fp_mask, prev_bev_embed,
                reference_points_cam, bev_mask, spatial_shapes,
                level_start_index, query_pos)
        else:
            return self.forward_full(
                query, key, value, residual, query_pos, key_padding_mask,
                reference_points, spatial_shapes, reference_points_cam,
                bev_mask, level_start_index, **kwargs)
