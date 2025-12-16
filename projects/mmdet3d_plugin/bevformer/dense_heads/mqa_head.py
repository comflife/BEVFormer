# Copyright (c) OpenMMLab. All rights reserved.
"""
MQA Head for Multi-task Question Answering.

Downstream heads for different QA task types:
1. CountHead: Object counting (0-20 classification)
2. ClassHead: Object class prediction
3. DistanceHead: Distance regression
4. LocationHead: (x, y) coordinate regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS, build_loss
from mmdet.models.losses import FocalLoss
from typing import Dict, List, Optional, Tuple


@HEADS.register_module()
class MQAHead(BaseModule):
    """Multi-task QA Head for Language-guided BEVFormer.
    
    This head performs prior-weighted pooling on BEV features and
    predicts answers for different question types:
    - Count: How many objects of a class
    - Class: What type of object
    - Distance: How far is the nearest object
    - Location: Where is the nearest object (x, y)
    
    Args:
        in_channels: Input feature channels (from BEV)
        text_channels: Text embedding channels
        max_count: Maximum count for classification
        num_classes: Number of object classes
        hidden_dim: Hidden layer dimension
        num_layers: Number of MLP layers for regression heads
        pc_range: Point cloud range for location normalization
        loss_count: Loss config for count prediction
        loss_class: Loss config for class prediction
        loss_distance: Loss config for distance regression
        loss_location: Loss config for location regression
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        text_channels: int = 256,
        max_count: int = 20,
        num_classes: int = 13,
        hidden_dim: int = 256,
        num_layers: int = 2,
        pc_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        loss_count: dict = dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        ),
        loss_class: dict = dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        ),
        loss_distance: dict = dict(
            type='SmoothL1Loss',
            beta=1.0,
            loss_weight=1.0
        ),
        loss_location: dict = dict(
            type='SmoothL1Loss',
            beta=1.0,
            loss_weight=1.0
        ),
        init_cfg: dict = None,
    ):
        super().__init__(init_cfg)
        
        self.in_channels = in_channels
        self.text_channels = text_channels
        self.max_count = max_count
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pc_range = pc_range
        
        # Fusion layer: combine BEV pooled features with text embedding
        fusion_input_dim = in_channels + text_channels
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
        )
        
        # Count head: predicts count from 0 to max_count
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, max_count + 1),  # 0 to max_count
        )
        
        # Class head: predicts object class
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # Distance head: predicts distance in meters
        dist_layers = []
        in_dim = hidden_dim
        for _ in range(num_layers - 1):
            dist_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=False),
            ])
            in_dim = hidden_dim
        dist_layers.append(nn.Linear(hidden_dim, 1))
        self.distance_head = nn.Sequential(*dist_layers)
        
        # Location head: predicts (x, y) coordinates
        loc_layers = []
        in_dim = hidden_dim
        for _ in range(num_layers - 1):
            loc_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=False),
            ])
            in_dim = hidden_dim
        loc_layers.append(nn.Linear(hidden_dim, 2))  # (x, y)
        self.location_head = nn.Sequential(*loc_layers)
        
        # Build losses
        self.loss_count = build_loss(loss_count)
        self.loss_class = build_loss(loss_class)
        self.loss_distance = build_loss(loss_distance)
        self.loss_location = build_loss(loss_location)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def prior_weighted_pooling(
        self,
        bev_feat: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Pool BEV features weighted by spatial prior.

        Now receives modulated features from PriorGuidedModulation,
        so the features already contain language-guided information.

        Args:
            bev_feat: [B, C, H, W] BEV feature map (modulated)
            prior: [B, H, W] spatial prior (probabilities)

        Returns:
            pooled: [B, C] pooled feature vector
        """
        B, C, H, W = bev_feat.shape

        # Normalize prior to sum to 1
        prior_flat = prior.reshape(B, H * W)  # [B, H*W]
        prior_norm = prior_flat / (prior_flat.sum(dim=1, keepdim=True) + 1e-6)
        prior_norm = prior_norm.reshape(B, 1, H, W)  # [B, 1, H, W]

        # Weighted sum: combine modulated features with spatial attention
        pooled = (bev_feat * prior_norm).sum(dim=[2, 3])  # [B, C]

        return pooled
    
    def forward(
        self,
        bev_feat: torch.Tensor,
        prior: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            bev_feat: [B, C, H, W] BEV feature map
            prior: [B, H, W] spatial prior
            text_embedding: [B, D] text embedding
            
        Returns:
            Dict of predictions:
                - count_logits: [B, max_count+1]
                - class_logits: [B, num_classes]
                - distance_pred: [B, 1]
                - location_pred: [B, 2]
        """
        # Pool BEV features using prior
        bev_pooled = self.prior_weighted_pooling(bev_feat, prior)  # [B, C]
        
        # Fuse with text embedding
        fused = torch.cat([bev_pooled, text_embedding], dim=1)  # [B, C+D]
        fused = self.fusion(fused)  # [B, hidden_dim]
        
        # Predictions
        count_logits = self.count_head(fused)  # [B, max_count+1]
        class_logits = self.class_head(fused)  # [B, num_classes]
        distance_pred = self.distance_head(fused)  # [B, 1]
        location_pred = self.location_head(fused)  # [B, 2]
        
        # Denormalize location to actual coordinates
        # location_pred is in [-1, 1], convert to actual range
        x_pred = location_pred[:, 0:1] * (self.pc_range[3] - self.pc_range[0]) / 2
        y_pred = location_pred[:, 1:2] * (self.pc_range[4] - self.pc_range[1]) / 2
        location_pred = torch.cat([x_pred, y_pred], dim=1)
        
        # Distance should be positive
        distance_pred = F.softplus(distance_pred)
        
        return {
            'count_logits': count_logits,
            'class_logits': class_logits,
            'distance_pred': distance_pred,
            'location_pred': location_pred,
            'fused_features': fused,
        }
    
    @force_fp32(apply_to=('preds',))
    def loss(
        self,
        preds: Dict[str, torch.Tensor],
        gt_counts: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_distances: torch.Tensor,
        gt_locations: torch.Tensor,
        has_distance: torch.Tensor,
        has_location: torch.Tensor,
        question_type_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute losses.
        
        Args:
            preds: Prediction dict from forward()
            gt_counts: [B] ground truth counts
            gt_classes: [B] ground truth primary object class
            gt_distances: [B, 1] ground truth distances
            gt_locations: [B, 2] ground truth locations
            has_distance: [B] whether sample has distance label
            has_location: [B] whether sample has location label
            question_type_ids: [B] question type (for task-specific loss weighting)
            
        Returns:
            Dict of losses
        """
        losses = {}

        def _connected_zero(tensor_like: torch.Tensor) -> torch.Tensor:
            # Return a zero scalar that is still connected to `tensor_like`'s autograd graph.
            # This avoids per-iteration unused parameters under DDP when a task has no
            # valid labels in the current batch.
            return tensor_like.sum() * 0.0
        
        # Count loss (always computed)
        count_logits = preds['count_logits']
        loss_count = self.loss_count(count_logits, gt_counts)
        losses['loss_count'] = loss_count
        
        # Class loss (for samples with valid class)
        valid_class_mask = gt_classes >= 0
        if valid_class_mask.any():
            class_logits = preds['class_logits'][valid_class_mask]
            gt_cls = gt_classes[valid_class_mask]
            loss_class = self.loss_class(class_logits, gt_cls)
            losses['loss_class'] = loss_class
        else:
            losses['loss_class'] = _connected_zero(preds['class_logits'])
        
        # Distance loss (for samples with distance labels)
        has_distance_mask = has_distance.bool()
        if has_distance_mask.any():
            dist_pred = preds['distance_pred'][has_distance_mask]
            gt_dist = gt_distances[has_distance_mask]
            loss_dist = self.loss_distance(dist_pred, gt_dist)
            losses['loss_distance'] = loss_dist
        else:
            losses['loss_distance'] = _connected_zero(preds['distance_pred'])
        
        # Location loss (for samples with location labels)
        has_location_mask = has_location.bool()
        if has_location_mask.any():
            loc_pred = preds['location_pred'][has_location_mask]
            gt_loc = gt_locations[has_location_mask]
            loss_loc = self.loss_location(loc_pred, gt_loc)
            losses['loss_location'] = loss_loc
        else:
            losses['loss_location'] = _connected_zero(preds['location_pred'])
        
        return losses
    
    def get_predictions(
        self,
        preds: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Convert logits to predictions.
        
        Args:
            preds: Prediction dict from forward()
            
        Returns:
            Dict with:
                - count_pred: [B] predicted counts
                - class_pred: [B] predicted classes
                - distance_pred: [B] predicted distances
                - location_pred: [B, 2] predicted locations
        """
        return {
            'count_pred': preds['count_logits'].argmax(dim=1),
            'class_pred': preds['class_logits'].argmax(dim=1),
            'distance_pred': preds['distance_pred'].squeeze(1),
            'location_pred': preds['location_pred'],
        }


@HEADS.register_module()
class PriorSupervisionHead(BaseModule):
    """Auxiliary head for prior supervision.
    
    Supervises the prior map to match ground truth spatial attention.
    This helps prevent the prior from collapsing to trivial solutions.
    
    Args:
        bev_h, bev_w: BEV grid dimensions
        loss_prior: Loss config for prior supervision
    """
    
    def __init__(
        self,
        bev_h: int = 150,
        bev_w: int = 150,
        loss_prior: dict = dict(
            type='BCELoss',
            loss_weight=0.1,
        ),
        init_cfg: dict = None,
    ):
        super().__init__(init_cfg)
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        # BCE loss for prior supervision
        self.loss_prior = nn.BCELoss(reduction='mean')
        self.loss_weight = loss_prior.get('loss_weight', 0.1)
    
    @force_fp32(apply_to=('prior', 'gt_prior'))
    def loss(
        self,
        prior: torch.Tensor,
        gt_prior: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute prior supervision loss.
        
        Args:
            prior: [B, H, W] predicted prior
            gt_prior: [B, H, W] ground truth prior (camera sector mask)
            
        Returns:
            Dict with loss_prior
        """
        # Ensure same shape
        if prior.shape != gt_prior.shape:
            gt_prior = F.interpolate(
                gt_prior.unsqueeze(1),
                size=(self.bev_h, self.bev_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # BCE loss
        loss = self.loss_prior(prior, gt_prior) * self.loss_weight
        
        return {'loss_prior': loss}


@HEADS.register_module() 
class EntropyRegularizer(BaseModule):
    """Regularizer to prevent prior collapse.
    
    Encourages the prior to be neither too peaked nor too uniform.
    Uses negative entropy as regularization (minimize to increase entropy).
    
    Args:
        target_entropy_ratio: Target entropy as ratio of maximum (0-1)
        loss_weight: Weight for entropy regularization loss
    """
    
    def __init__(
        self,
        target_entropy_ratio: float = 0.5,
        loss_weight: float = 0.01,
        init_cfg: dict = None,
    ):
        super().__init__(init_cfg)
        
        self.target_entropy_ratio = target_entropy_ratio
        self.loss_weight = loss_weight
    
    @force_fp32(apply_to=('prior',))
    def loss(
        self,
        prior: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute entropy regularization loss.
        
        Args:
            prior: [B, H, W] predicted prior
            
        Returns:
            Dict with loss_entropy
        """
        B, H, W = prior.shape
        
        # Flatten and normalize to get probability distribution
        prior_flat = prior.view(B, -1)  # [B, H*W]
        prior_norm = prior_flat / (prior_flat.sum(dim=1, keepdim=True) + 1e-6)
        
        # Compute entropy: -sum(p * log(p))
        log_prior = torch.log(prior_norm + 1e-6)
        entropy = -(prior_norm * log_prior).sum(dim=1)  # [B]
        
        # Maximum possible entropy for uniform distribution
        max_entropy = torch.log(torch.tensor(H * W, dtype=prior.dtype, device=prior.device))
        
        # Target entropy
        target_entropy = self.target_entropy_ratio * max_entropy
        
        # Loss: penalize deviation from target entropy
        # If entropy is too low (peaked), push it up
        # If entropy is too high (uniform), push it down
        loss = ((entropy - target_entropy) ** 2).mean() * self.loss_weight
        
        return {'loss_entropy': loss}
