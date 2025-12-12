# ---------------------------------------------
# DiffFP-enabled BEVFormer Encoder
# ---------------------------------------------
# Integrates DiffFP for sparse temporal updates
# Maintains full prev_bev context while only updating changed cells
# ---------------------------------------------

import numpy as np
import torch
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
from mmcv.cnn.bricks.transformer import build_transformer_layer

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
from .diff_fp import DiffFP, DiffFPBEV

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DiffFPBEVFormerEncoder(TransformerLayerSequence):
    """BEVFormer Encoder with DiffFP for sparse temporal updates.
    
    This encoder extends the original BEVFormerEncoder with:
    1. DiffFP module to identify changed BEV cells
    2. Sparse temporal attention that only updates changed cells
    3. Efficient carry-over of unchanged cells from prev_bev
    
    Args:
        transformerlayers: Config for transformer layers
        num_layers: Number of encoder layers
        pc_range: Point cloud range
        num_points_in_pillar: Number of points per pillar for 3D reference
        return_intermediate: Whether to return intermediate outputs
        use_diff_fp: Whether to enable DiffFP (can be disabled for comparison)
        diff_fp_cfg: Configuration for DiffFP module
    """
    
    def __init__(self, 
                 *args, 
                 pc_range=None, 
                 num_points_in_pillar=4, 
                 return_intermediate=False,
                 dataset_type='nuscenes',
                 use_diff_fp=True,
                 diff_fp_cfg=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        
        # DiffFP configuration
        self.use_diff_fp = use_diff_fp
        if use_diff_fp:
            if diff_fp_cfg is None:
                diff_fp_cfg = dict(
                    embed_dims=256,
                    prune_ratio=0.5,
                    threshold=0.0,
                    min_keep_ratio=0.1,
                    max_keep_ratio=0.9,
                )
            # Get embed_dims from transformerlayers config
            if 'embed_dims' not in diff_fp_cfg:
                diff_fp_cfg['embed_dims'] = kwargs.get('transformerlayers', {}).get(
                    'attn_cfgs', [{}])[0].get('embed_dims', 256)
            self.diff_fp = DiffFP(**diff_fp_cfg)
        else:
            self.diff_fp = None
            
    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', 
                             bs=1, device='cuda', dtype=torch.float):
        """Get reference points for SCA and TSA.
        
        Same as original BEVFormerEncoder.
        """
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(
                                    num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(
                                    num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(
                                    num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
    
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range, img_metas):
        """Sample points from 3D to 2D camera coordinates.
        
        Same as original BEVFormerEncoder.
        """
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32),
            reference_points.to(torch.float32)
        ).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], 
            torch.ones_like(reference_points_cam[..., 2:3]) * eps
        )

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, bev_mask
    
    def compute_diff_fp_mask(self, bev_query, prev_bev):
        """Compute DiffFP mask for sparse updates.
        
        Args:
            bev_query: Current BEV query [B, H*W, C]
            prev_bev: Previous BEV features [B, H*W, C]
            
        Returns:
            dict with mask and statistics
        """
        if not self.use_diff_fp or self.diff_fp is None:
            return None
            
        if prev_bev is None:
            # First frame - update all positions
            return None
            
        # Compute DiffFP mask
        diff_result = self.diff_fp(bev_query, prev_bev)
        return diff_result

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function with DiffFP support.
        
        Args:
            bev_query: BEV query [num_query, bs, embed_dims]
            key, value: Multi-camera features
            bev_h, bev_w: BEV grid dimensions
            bev_pos: BEV positional encoding
            spatial_shapes: Feature spatial shapes
            level_start_index: Level start indices
            prev_bev: Previous frame BEV features
            shift: Ego motion shift
            
        Returns:
            Updated BEV features
        """
        output = bev_query
        intermediate = []

        # Get reference points
        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),
            device=bev_query.device, dtype=bev_query.dtype
        )
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1),
            device=bev_query.device, dtype=bev_query.dtype
        )

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas']
        )

        # Handle shift default value - convert scalar to proper tensor
        bs = bev_query.size(1)
        if isinstance(shift, (int, float)):
            shift = ref_2d.new_tensor([[shift, shift]]).expand(bs, -1)
        elif shift.dim() == 0:
            shift = shift.unsqueeze(0).unsqueeze(0).expand(bs, 2)
        
        # Shifted reference points for temporal alignment
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # Convert to batch-first format
        bev_query = bev_query.permute(1, 0, 2)  # [bs, num_query, embed_dims]
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape

        # Compute DiffFP mask if enabled
        sparse_mask = None
        diff_fp_result = None
        
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)  # [bs, num_query, embed_dims]
            
            # Compute sparse mask using DiffFP
            if self.use_diff_fp:
                diff_fp_result = self.compute_diff_fp_mask(bev_query, prev_bev)
                if diff_fp_result is not None:
                    sparse_mask = diff_fp_result.get('mask_flat', None)
                    
            # Prepare hybrid reference points for temporal attention
            # For full temporal attention (non-sparse path)
            prev_bev_stacked = torch.stack([prev_bev, bev_query], 1).reshape(
                bs * 2, len_bev, -1
            )
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )
        else:
            prev_bev_stacked = None
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )

        # Process through transformer layers
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev_stacked,
                prev_bev_aligned=prev_bev,  # For sparse attention
                sparse_mask=sparse_mask,
                **kwargs
            )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class DiffFPBEVFormerLayer(MyCustomBaseTransformerLayer):
    """BEVFormer Layer with DiffFP-aware temporal attention.
    
    This layer extends BEVFormerLayer to support sparse temporal updates.
    When a sparse_mask is provided, only masked positions are updated
    through temporal attention, while others inherit from prev_bev.
    """
    
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super().__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs
        )
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn']
        )
        
    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                prev_bev_aligned=None,
                sparse_mask=None,
                **kwargs):
        """Forward with optional sparse temporal attention.
        
        Args:
            query: Current BEV query
            prev_bev: Stacked [prev_bev, query] for full temporal attention
            prev_bev_aligned: Aligned prev_bev for sparse attention
            sparse_mask: Boolean mask for sparse updates
            Other args: Standard transformer layer arguments
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f'Use same attn_mask in all attentions')
        else:
            assert len(attn_masks) == self.num_attn

        for layer in self.operation_order:
            if layer == 'self_attn':
                # Temporal self-attention (potentially sparse)
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device
                    ),
                    level_start_index=torch.tensor([0], device=query.device),
                    # Additional args for sparse attention
                    sparse_mask=sparse_mask,
                    prev_bev_aligned=prev_bev_aligned,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    **kwargs
                )
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                # Spatial cross-attention (same as original)
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs
                )
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None
                )
                ffn_index += 1

        return query
