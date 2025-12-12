# ---------------------------------------------
# DiffFP: Differential Feature Pruning for Temporal BEV
# ---------------------------------------------
# Prunes redundant image features based on temporal differences
# Only keeps tokens that have changed significantly between frames
# ---------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class DiffFP(BaseModule):
    """Differential Feature Pruning module.
    
    Computes the difference between current and previous frame features,
    and generates a mask indicating which spatial locations have significant changes.
    
    Args:
        embed_dims (int): Feature dimension. Default: 256.
        prune_ratio (float): Ratio of tokens to keep (0-1). Default: 0.5.
        threshold (float): If > 0, use threshold-based pruning instead of top-k.
            Default: 0.0 (use top-k).
        min_keep_ratio (float): Minimum ratio of tokens to keep. Default: 0.1.
        max_keep_ratio (float): Maximum ratio of tokens to keep. Default: 0.9.
        learnable (bool): Whether to use learnable difference computation. Default: False.
        reduction (str): How to reduce channel dimension for difference.
            Options: 'mean', 'max', 'l2'. Default: 'mean'.
    """
    
    def __init__(self,
                 embed_dims=256,
                 prune_ratio=0.5,
                 threshold=0.0,
                 min_keep_ratio=0.1,
                 max_keep_ratio=0.9,
                 learnable=False,
                 reduction='mean',
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.embed_dims = embed_dims
        self.prune_ratio = prune_ratio
        self.threshold = threshold
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio
        self.learnable = learnable
        self.reduction = reduction
        
        if learnable:
            # Learnable projection for computing difference importance
            self.diff_proj = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, 1),
                nn.Sigmoid()
            )
        
    def compute_difference(self, feat_curr, feat_prev):
        """Compute per-location difference score.
        
        Args:
            feat_curr: Current frame features [B, N, C, H, W] or [B, N*H*W, C]
            feat_prev: Previous frame features (same shape)
            
        Returns:
            diff_score: Difference score per location [B, N*H*W] or [B, N, H, W]
        """
        if feat_curr.dim() == 5:
            # [B, N, C, H, W] format
            B, N, C, H, W = feat_curr.shape
            feat_curr_flat = feat_curr.permute(0, 1, 3, 4, 2).reshape(B, N*H*W, C)
            feat_prev_flat = feat_prev.permute(0, 1, 3, 4, 2).reshape(B, N*H*W, C)
            spatial_shape = (N, H, W)
        else:
            # [B, L, C] format
            B, L, C = feat_curr.shape
            feat_curr_flat = feat_curr
            feat_prev_flat = feat_prev
            spatial_shape = None
            
        if self.learnable:
            # Learnable difference computation
            concat_feat = torch.cat([feat_curr_flat, feat_prev_flat], dim=-1)
            diff_score = self.diff_proj(concat_feat).squeeze(-1)  # [B, L]
        else:
            # Simple difference computation
            diff = feat_curr_flat - feat_prev_flat  # [B, L, C]
            
            if self.reduction == 'mean':
                diff_score = diff.abs().mean(dim=-1)  # [B, L]
            elif self.reduction == 'max':
                diff_score = diff.abs().max(dim=-1)[0]  # [B, L]
            elif self.reduction == 'l2':
                diff_score = torch.norm(diff, p=2, dim=-1)  # [B, L]
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")
                
        return diff_score, spatial_shape
    
    def generate_mask(self, diff_score, num_tokens):
        """Generate binary mask based on difference scores.
        
        Args:
            diff_score: [B, L] difference scores
            num_tokens: Total number of tokens L
            
        Returns:
            mask: [B, L] boolean mask (True = keep)
            keep_indices: [B, K] indices of kept tokens
        """
        B, L = diff_score.shape
        
        if self.threshold > 0:
            # Threshold-based pruning
            mask = diff_score > self.threshold
            
            # Ensure min/max keep ratio
            keep_counts = mask.sum(dim=1)  # [B]
            min_keep = int(L * self.min_keep_ratio)
            max_keep = int(L * self.max_keep_ratio)
            
            for b in range(B):
                if keep_counts[b] < min_keep:
                    # Keep top-k if too few
                    _, topk_idx = diff_score[b].topk(min_keep)
                    mask[b] = False
                    mask[b, topk_idx] = True
                elif keep_counts[b] > max_keep:
                    # Keep top-k if too many
                    _, topk_idx = diff_score[b].topk(max_keep)
                    mask[b] = False
                    mask[b, topk_idx] = True
        else:
            # Top-k based pruning
            k = max(int(L * self.prune_ratio), int(L * self.min_keep_ratio))
            k = min(k, int(L * self.max_keep_ratio))
            
            _, topk_indices = diff_score.topk(k, dim=1)  # [B, k]
            mask = torch.zeros(B, L, dtype=torch.bool, device=diff_score.device)
            mask.scatter_(1, topk_indices, True)
            
        # Get indices of kept tokens
        # Note: Number of kept tokens may vary per batch item
        keep_indices = [mask[b].nonzero(as_tuple=True)[0] for b in range(B)]
        
        return mask, keep_indices
    
    def forward(self, feat_curr, feat_prev=None, return_full_mask=True):
        """Forward pass of DiffFP.
        
        Args:
            feat_curr: Current frame features 
                [B, N, C, H, W] for multi-cam or [B, H*W, C] for BEV
            feat_prev: Previous frame features (same shape as feat_curr)
                If None, return all tokens as valid (first frame case)
            return_full_mask: If True, return mask in original spatial shape
            
        Returns:
            dict containing:
                - mask: Boolean mask [B, L] or [B, N, H, W]
                - diff_score: Difference scores [B, L]
                - keep_ratio: Actual keep ratio per batch
                - pruned_feat: (optional) Pruned features if easy to compute
        """
        if feat_prev is None:
            # First frame - keep all tokens
            if feat_curr.dim() == 5:
                B, N, C, H, W = feat_curr.shape
                L = N * H * W
                mask = torch.ones(B, N, H, W, dtype=torch.bool, device=feat_curr.device)
                diff_score = torch.ones(B, L, device=feat_curr.device)
            else:
                B, L, C = feat_curr.shape
                mask = torch.ones(B, L, dtype=torch.bool, device=feat_curr.device)
                diff_score = torch.ones(B, L, device=feat_curr.device)
                
            return {
                'mask': mask,
                'diff_score': diff_score,
                'keep_ratio': torch.ones(B, device=feat_curr.device),
                'keep_indices': None,
                'is_first_frame': True
            }
        
        # Compute difference scores
        diff_score, spatial_shape = self.compute_difference(feat_curr, feat_prev)
        
        # Generate mask
        B, L = diff_score.shape
        mask, keep_indices = self.generate_mask(diff_score, L)
        
        # Compute actual keep ratio
        keep_ratio = mask.float().mean(dim=1)  # [B]
        
        # Reshape mask to spatial shape if requested
        if return_full_mask and spatial_shape is not None:
            N, H, W = spatial_shape
            mask_spatial = mask.view(B, N, H, W)
        else:
            mask_spatial = mask
            
        return {
            'mask': mask_spatial,
            'mask_flat': mask,
            'diff_score': diff_score,
            'keep_ratio': keep_ratio,
            'keep_indices': keep_indices,
            'is_first_frame': False
        }


@PLUGIN_LAYERS.register_module()
class ImageDiffFP(BaseModule):
    """DiffFP for image feature level (multi-camera).
    
    Computes differences between current and previous frame image features
    to identify which image patches have changed. This enables pruning of
    redundant patches before BEV projection.
    
    Works with multi-camera features: [B, N_cams, C, H, W]
    
    Args:
        embed_dims (int): Feature channel dimension. Default: 256.
        num_cams (int): Number of cameras. Default: 6.
        prune_ratio (float): Ratio of patches to keep per camera. Default: 0.5.
        threshold (float): If > 0, use threshold-based pruning. Default: 0.0.
        min_keep_ratio (float): Minimum patches to keep. Default: 0.1.
        max_keep_ratio (float): Maximum patches to keep. Default: 0.9.
        per_camera (bool): Apply pruning per camera independently. Default: True.
        reduction (str): How to reduce channels ('mean', 'max', 'l2'). Default: 'l2'.
    """
    
    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 prune_ratio=0.5,
                 threshold=0.0,
                 min_keep_ratio=0.1,
                 max_keep_ratio=0.9,
                 per_camera=True,
                 reduction='l2',
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.prune_ratio = prune_ratio
        self.threshold = threshold
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio
        self.per_camera = per_camera
        self.reduction = reduction
        
    def compute_difference_per_camera(self, feat_curr, feat_prev):
        """Compute difference score per spatial location for each camera.
        
        Args:
            feat_curr: [B, N, C, H, W] current features
            feat_prev: [B, N, C, H, W] previous features
            
        Returns:
            diff_score: [B, N, H, W] difference scores
        """
        # Compute absolute difference
        diff = feat_curr - feat_prev  # [B, N, C, H, W]
        
        if self.reduction == 'mean':
            diff_score = diff.abs().mean(dim=2)  # [B, N, H, W]
        elif self.reduction == 'max':
            diff_score = diff.abs().max(dim=2)[0]  # [B, N, H, W]
        elif self.reduction == 'l2':
            diff_score = torch.norm(diff, p=2, dim=2)  # [B, N, H, W]
        elif self.reduction == 'l1':
            diff_score = torch.norm(diff, p=1, dim=2)  # [B, N, H, W]
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
            
        return diff_score
    
    def generate_mask_per_camera(self, diff_score):
        """Generate mask per camera.
        
        Args:
            diff_score: [B, N, H, W]
            
        Returns:
            mask: [B, N, H, W] boolean mask
        """
        B, N, H, W = diff_score.shape
        
        if self.per_camera:
            # Process each camera independently
            mask = torch.zeros_like(diff_score, dtype=torch.bool)
            
            for cam_idx in range(N):
                cam_score = diff_score[:, cam_idx]  # [B, H, W]
                cam_score_flat = cam_score.view(B, -1)  # [B, H*W]
                L = H * W
                
                k = max(int(L * self.prune_ratio), int(L * self.min_keep_ratio))
                k = min(k, int(L * self.max_keep_ratio))
                
                if self.threshold > 0:
                    cam_mask = cam_score_flat > self.threshold
                    # Ensure min/max constraints
                    keep_counts = cam_mask.sum(dim=1)
                    for b in range(B):
                        if keep_counts[b] < int(L * self.min_keep_ratio):
                            _, topk_idx = cam_score_flat[b].topk(int(L * self.min_keep_ratio))
                            cam_mask[b] = False
                            cam_mask[b, topk_idx] = True
                        elif keep_counts[b] > int(L * self.max_keep_ratio):
                            _, topk_idx = cam_score_flat[b].topk(int(L * self.max_keep_ratio))
                            cam_mask[b] = False
                            cam_mask[b, topk_idx] = True
                else:
                    _, topk_indices = cam_score_flat.topk(k, dim=1)
                    cam_mask = torch.zeros(B, L, dtype=torch.bool, device=diff_score.device)
                    cam_mask.scatter_(1, topk_indices, True)
                
                mask[:, cam_idx] = cam_mask.view(B, H, W)
        else:
            # Global pruning across all cameras
            score_flat = diff_score.view(B, -1)  # [B, N*H*W]
            L = N * H * W
            
            k = max(int(L * self.prune_ratio), int(L * self.min_keep_ratio))
            k = min(k, int(L * self.max_keep_ratio))
            
            _, topk_indices = score_flat.topk(k, dim=1)
            mask_flat = torch.zeros(B, L, dtype=torch.bool, device=diff_score.device)
            mask_flat.scatter_(1, topk_indices, True)
            mask = mask_flat.view(B, N, H, W)
            
        return mask
    
    def forward(self, feat_curr, feat_prev=None):
        """Forward pass.
        
        Args:
            feat_curr: Current frame features [B, N, C, H, W]
            feat_prev: Previous frame features [B, N, C, H, W]
            
        Returns:
            dict containing:
                - mask: [B, N, H, W] boolean mask (True = keep/changed)
                - mask_flat: [B, N*H*W] flattened mask
                - diff_score: [B, N, H, W] difference scores
                - keep_ratio: [B] actual keep ratio
                - keep_ratio_per_cam: [B, N] keep ratio per camera
        """
        B, N, C, H, W = feat_curr.shape
        
        if feat_prev is None:
            # First frame - keep all
            mask = torch.ones(B, N, H, W, dtype=torch.bool, device=feat_curr.device)
            return {
                'mask': mask,
                'mask_flat': mask.view(B, -1),
                'diff_score': torch.ones(B, N, H, W, device=feat_curr.device),
                'keep_ratio': torch.ones(B, device=feat_curr.device),
                'keep_ratio_per_cam': torch.ones(B, N, device=feat_curr.device),
                'is_first_frame': True
            }
        
        # Compute difference
        diff_score = self.compute_difference_per_camera(feat_curr, feat_prev)
        
        # Generate mask
        mask = self.generate_mask_per_camera(diff_score)
        
        # Compute keep ratios
        keep_ratio = mask.float().mean(dim=(1, 2, 3))  # [B]
        keep_ratio_per_cam = mask.float().mean(dim=(2, 3))  # [B, N]
        
        return {
            'mask': mask,
            'mask_flat': mask.view(B, -1),
            'diff_score': diff_score,
            'keep_ratio': keep_ratio,
            'keep_ratio_per_cam': keep_ratio_per_cam,
            'is_first_frame': False
        }
@PLUGIN_LAYERS.register_module()
class DiffFPBEV(BaseModule):
    """DiffFP specifically designed for BEV space.
    
    Works on BEV grid features to identify which BEV cells need updating.
    
    Args:
        embed_dims (int): BEV feature dimension. Default: 256.
        bev_h (int): BEV height. Default: 200.
        bev_w (int): BEV width. Default: 200.
        prune_ratio (float): Ratio of BEV cells to update. Default: 0.5.
        use_ego_motion (bool): Whether to consider ego motion for warping. Default: True.
    """
    
    def __init__(self,
                 embed_dims=256,
                 bev_h=200,
                 bev_w=200,
                 prune_ratio=0.5,
                 threshold=0.0,
                 min_keep_ratio=0.1,
                 max_keep_ratio=0.9,
                 use_ego_motion=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.prune_ratio = prune_ratio
        self.threshold = threshold
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio
        self.use_ego_motion = use_ego_motion
        
        # Core DiffFP module
        self.diff_fp = DiffFP(
            embed_dims=embed_dims,
            prune_ratio=prune_ratio,
            threshold=threshold,
            min_keep_ratio=min_keep_ratio,
            max_keep_ratio=max_keep_ratio,
            learnable=False,
            reduction='l2'
        )
        
    def warp_bev(self, bev_prev, rotation_angle, translation):
        """Warp previous BEV to current coordinate frame.
        
        Args:
            bev_prev: Previous BEV features [B, H*W, C] or [B, C, H, W]
            rotation_angle: Ego rotation angle in radians [B]
            translation: Ego translation [B, 2] (x, y in BEV grid units)
            
        Returns:
            Warped BEV features
        """
        # This is a simplified version - in practice, use the existing
        # BEVFormer rotate/shift logic or grid_sample
        B = bev_prev.shape[0]
        
        if bev_prev.dim() == 3:
            # [B, H*W, C] -> [B, C, H, W]
            # Correct reshape: first view to spatial, then permute channels
            # [B, H*W, C] -> [B, H, W, C] -> [B, C, H, W]
            bev_prev = bev_prev.view(B, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2)
            was_flat = True
        else:
            was_flat = False
            
        # Create rotation matrix
        cos_a = torch.cos(rotation_angle)
        sin_a = torch.sin(rotation_angle)
        
        # Affine transformation matrix
        theta = torch.zeros(B, 2, 3, device=bev_prev.device, dtype=bev_prev.dtype)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        
        # Add translation
        if translation is not None:
            theta[:, 0, 2] = translation[:, 0] / (self.bev_w / 2)
            theta[:, 1, 2] = translation[:, 1] / (self.bev_h / 2)
        
        grid = F.affine_grid(theta, bev_prev.shape, align_corners=False)
        bev_warped = F.grid_sample(bev_prev, grid, mode='bilinear', 
                                   padding_mode='zeros', align_corners=False)
        
        if was_flat:
            # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
            bev_warped = bev_warped.permute(0, 2, 3, 1).reshape(B, self.bev_h * self.bev_w, -1)
            
        return bev_warped
    
    def forward(self, bev_curr, bev_prev=None, img_metas=None):
        """Forward pass.
        
        Args:
            bev_curr: Current BEV query/features [B, H*W, C]
            bev_prev: Previous BEV features [B, H*W, C]
            img_metas: Image metas containing ego motion info
            
        Returns:
            dict with mask and related info
        """
        if bev_prev is None:
            # First frame
            return self.diff_fp(bev_curr, None)
        
        # Optionally warp previous BEV to current coordinate frame
        if self.use_ego_motion and img_metas is not None:
            # Extract ego motion from img_metas
            rotation_angles = torch.tensor(
                [meta['can_bus'][-1] for meta in img_metas],
                device=bev_prev.device, dtype=bev_prev.dtype
            )
            translations = torch.tensor(
                [[meta['can_bus'][0], meta['can_bus'][1]] for meta in img_metas],
                device=bev_prev.device, dtype=bev_prev.dtype
            )
            bev_prev_warped = self.warp_bev(bev_prev, rotation_angles, translations)
        else:
            bev_prev_warped = bev_prev
            
        # Compute difference on aligned BEV
        result = self.diff_fp(bev_curr, bev_prev_warped)
        
        # Reshape mask to BEV grid
        B = bev_curr.shape[0]
        result['mask_bev'] = result['mask_flat'].view(B, self.bev_h, self.bev_w)
        
        return result


@PLUGIN_LAYERS.register_module()
class ImageLevelDiffFP(BaseModule):
    """Image-Level DiffFP for sparse SpatialCrossAttention.
    
    Computes differences between current and previous frame image features
    to identify which image patches have changed. Unchanged patches are skipped
    in SpatialCrossAttention, using previous BEV values instead.
    
    This is the core module for image-level temporal pruning in BEVFormer.
    
    Args:
        embed_dims (int): Feature channel dimension. Default: 256.
        num_cams (int): Number of cameras. Default: 6.
        keep_ratio (float): Ratio of patches to keep (changed patches). Default: 0.7.
        threshold (float): If not None, use threshold-based pruning. Default: None.
        score_type (str): How to compute difference ('l1', 'l2', 'cosine'). Default: 'l1'.
        learnable (bool): Whether to learn scoring weights. Default: True.
        temperature (float): Softmax temperature for learnable scoring. Default: 1.0.
    """
    
    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 keep_ratio=0.7,
                 threshold=None,
                 score_type='l1',
                 learnable=True,
                 temperature=1.0,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.keep_ratio = keep_ratio
        self.threshold = threshold
        self.score_type = score_type
        self.learnable = learnable
        self.temperature = temperature
        
        if learnable:
            # Learnable channel-wise importance weights
            self.channel_weights = nn.Parameter(torch.ones(embed_dims))
            # Optional: learnable per-camera weights
            self.camera_weights = nn.Parameter(torch.ones(num_cams))
    
    def compute_difference(self, feat_curr, feat_prev):
        """Compute difference scores between current and previous features.
        
        Args:
            feat_curr: [B, N, C, H, W] or list of [B, N, C, H, W]
            feat_prev: [B, N, C, H, W] or list of [B, N, C, H, W]
            
        Returns:
            diff_scores: list of [B, N, H, W] per level, or [B, N, H, W] if single level
        """
        # Handle multi-level features (FPN output)
        if isinstance(feat_curr, (list, tuple)):
            return [self._compute_diff_single(fc, fp) 
                    for fc, fp in zip(feat_curr, feat_prev)]
        return self._compute_diff_single(feat_curr, feat_prev)
    
    def _compute_diff_single(self, feat_curr, feat_prev):
        """Compute difference for single level features.
        
        Args:
            feat_curr: [B, N, C, H, W]
            feat_prev: [B, N, C, H, W]
            
        Returns:
            diff_score: [B, N, H, W]
        """
        if self.learnable:
            # Apply channel importance weights
            weights = F.softmax(self.channel_weights / self.temperature, dim=0)
            weights = weights.view(1, 1, -1, 1, 1)  # [1, 1, C, 1, 1]
            feat_curr = feat_curr * weights
            feat_prev = feat_prev * weights
        
        # Compute difference
        diff = feat_curr - feat_prev  # [B, N, C, H, W]
        
        if self.score_type == 'l1':
            diff_score = diff.abs().mean(dim=2)  # [B, N, H, W]
        elif self.score_type == 'l2':
            diff_score = torch.norm(diff, p=2, dim=2)  # [B, N, H, W]
        elif self.score_type == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            cos_sim = F.cosine_similarity(feat_curr, feat_prev, dim=2)  # [B, N, H, W]
            diff_score = 1 - cos_sim
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")
        
        if self.learnable:
            # Apply per-camera weights
            cam_weights = F.softmax(self.camera_weights / self.temperature, dim=0)
            cam_weights = cam_weights.view(1, -1, 1, 1)  # [1, N, 1, 1]
            diff_score = diff_score * cam_weights * self.num_cams  # Normalize
            
        return diff_score
    
    def generate_mask(self, diff_scores):
        """Generate masks indicating which patches to keep (changed patches).
        
        Args:
            diff_scores: [B, N, H, W] or list of [B, N, H, W]
            
        Returns:
            masks: Same shape as input, True = changed (keep), False = unchanged (skip)
        """
        if isinstance(diff_scores, (list, tuple)):
            return [self._generate_mask_single(ds) for ds in diff_scores]
        return self._generate_mask_single(diff_scores)
    
    def _generate_mask_single(self, diff_score):
        """Generate mask for single level.
        
        Args:
            diff_score: [B, N, H, W]
            
        Returns:
            mask: [B, N, H, W] boolean
        """
        B, N, H, W = diff_score.shape
        
        # Process each camera independently
        masks = []
        for cam_idx in range(N):
            cam_score = diff_score[:, cam_idx]  # [B, H, W]
            cam_score_flat = cam_score.view(B, -1)  # [B, H*W]
            L = H * W
            
            if self.threshold is not None:
                # Threshold-based: keep patches with diff > threshold
                cam_mask = cam_score_flat > self.threshold
            else:
                # Top-k based: keep top keep_ratio patches
                k = max(1, int(L * self.keep_ratio))
                _, topk_indices = cam_score_flat.topk(k, dim=1)
                cam_mask = torch.zeros(B, L, dtype=torch.bool, device=diff_score.device)
                cam_mask.scatter_(1, topk_indices, True)
            
            masks.append(cam_mask.view(B, H, W))
        
        # Stack: [B, N, H, W]
        mask = torch.stack(masks, dim=1)
        return mask
    
    def forward(self, feat_curr, feat_prev=None):
        """Forward pass.
        
        Args:
            feat_curr: Current features [B, N, C, H, W] or list
            feat_prev: Previous features [B, N, C, H, W] or list (None for first frame)
            
        Returns:
            dict containing:
                - mask: [B, N, H, W] or list, True = keep (changed patch)
                - diff_score: [B, N, H, W] or list
                - keep_ratio: actual keep ratio
                - is_first_frame: bool
        """
        is_list = isinstance(feat_curr, (list, tuple))
        
        if feat_prev is None:
            # First frame - keep all patches
            if is_list:
                masks = [torch.ones(f.shape[0], f.shape[1], f.shape[3], f.shape[4],
                                   dtype=torch.bool, device=f.device) for f in feat_curr]
                diff_scores = [torch.ones_like(m, dtype=torch.float) for m in masks]
            else:
                B, N, C, H, W = feat_curr.shape
                masks = torch.ones(B, N, H, W, dtype=torch.bool, device=feat_curr.device)
                diff_scores = torch.ones(B, N, H, W, device=feat_curr.device)
            
            return {
                'mask': masks,
                'diff_score': diff_scores,
                'keep_ratio': 1.0,
                'is_first_frame': True
            }
        
        # Compute differences
        diff_scores = self.compute_difference(feat_curr, feat_prev)
        
        # Generate masks
        masks = self.generate_mask(diff_scores)
        
        # Compute actual keep ratio
        if is_list:
            total_keep = sum(m.float().sum().item() for m in masks)
            total_elem = sum(m.numel() for m in masks)
            keep_ratio = total_keep / total_elem
        else:
            keep_ratio = masks.float().mean().item()
        
        return {
            'mask': masks,
            'diff_score': diff_scores,
            'keep_ratio': keep_ratio,
            'is_first_frame': False
        }
