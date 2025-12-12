# Copyright (c) OpenMMLab. All rights reserved.
"""
BEVFormer with Language-guided MQA (Markup Question-Answering).

This model extends BEVFormer to incorporate language understanding
for question-answering tasks on autonomous driving scenes.

Architecture:
    1. BEVFormer backbone (frozen or low LR)
    2. Text Encoder (BERT, frozen)
    3. Prior Head (generates spatial attention prior from text)
    4. MQA Head (multi-task prediction: count, class, distance, location)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS, build_head
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.bevformer.modules.language_prior import (
    TextEncoder, PriorHead, LanguagePriorModule
)
import copy
import numpy as np


@DETECTORS.register_module()
class BEVFormerMQA(MVXTwoStageDetector):
    """BEVFormer with Language-guided MQA.
    
    This model:
    1. Extracts BEV features using BEVFormer
    2. Encodes question text using BERT
    3. Generates spatial prior from language
    4. Predicts answers (count, class, distance, location)
    
    Args:
        use_grid_mask: Whether to use grid mask augmentation
        video_test_mode: Whether to use temporal information during inference
        freeze_backbone: Whether to freeze BEVFormer backbone
        freeze_text_encoder: Whether to freeze text encoder
        text_encoder_cfg: Config for text encoder
        prior_head_cfg: Config for prior head
        mqa_head_cfg: Config for MQA prediction head
        prior_supervision_cfg: Config for prior supervision head
        use_prior_injection: Whether to inject prior into attention
    """
    
    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        # MQA specific args
        freeze_backbone=True,
        freeze_neck=True,
        freeze_bev_encoder=True,  # Freeze BEVFormer encoder
        freeze_text_encoder=True,
        text_encoder_cfg=None,
        prior_head_cfg=None,
        mqa_head_cfg=None,
        prior_supervision_cfg=None,
        use_prior_injection=True,
        # Loss weights
        loss_weight_count=1.0,
        loss_weight_class=1.0,
        loss_weight_distance=1.0,
        loss_weight_location=1.0,
        loss_weight_prior=0.1,
    ):
        super(BEVFormerMQA, self).__init__(
            pts_voxel_layer, pts_voxel_encoder,
            pts_middle_encoder, pts_fusion_layer,
            img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained
        )
        
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.video_test_mode = video_test_mode
        
        self.freeze_backbone = freeze_backbone
        self.freeze_neck = freeze_neck
        self.freeze_bev_encoder = freeze_bev_encoder
        self.freeze_text_encoder = freeze_text_encoder
        self.use_prior_injection = use_prior_injection
        
        # Loss weights
        self.loss_weight_count = loss_weight_count
        self.loss_weight_class = loss_weight_class
        self.loss_weight_distance = loss_weight_distance
        self.loss_weight_location = loss_weight_location
        self.loss_weight_prior = loss_weight_prior
        
        # Build language modules
        if text_encoder_cfg is None:
            text_encoder_cfg = dict(
                pretrained_model='bert-base-uncased',
                freeze=freeze_text_encoder,
                output_dim=256,
                pooling='cls',
            )
        self.text_encoder = TextEncoder(**text_encoder_cfg)
        
        if prior_head_cfg is None:
            prior_head_cfg = dict(
                text_dim=256,
                bev_h=150,
                bev_w=150,
                hidden_dim=512,
                num_layers=3,
                use_rule_prior=True,
                rule_prior_weight=0.5,
            )
        self.prior_head = PriorHead(**prior_head_cfg)
        
        if mqa_head_cfg is None:
            mqa_head_cfg = dict(
                type='MQAHead',
                in_channels=256,
                text_channels=256,
                max_count=20,
                num_classes=13,
                hidden_dim=256,
                num_layers=2,
            )
        self.mqa_head = build_head(mqa_head_cfg)
        
        # Prior supervision head (optional)
        if prior_supervision_cfg is not None:
            self.prior_supervision_head = build_head(prior_supervision_cfg)
        else:
            self.prior_supervision_head = None
        
        # Freeze backbone if specified
        self._freeze_modules()
        
        # Temporal info for video mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
    
    def _freeze_modules(self):
        """Freeze specified modules."""
        if self.freeze_backbone and self.img_backbone is not None:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
            self.img_backbone.eval()
        
        if self.freeze_neck and self.img_neck is not None:
            for param in self.img_neck.parameters():
                param.requires_grad = False
            self.img_neck.eval()
        
        if self.freeze_bev_encoder and self.pts_bbox_head is not None:
            for param in self.pts_bbox_head.parameters():
                param.requires_grad = False
            self.pts_bbox_head.eval()
        
        # Note: Text encoder freezing is handled in TextEncoder class
    
    def train(self, mode=True):
        """Override train mode to keep frozen modules in eval."""
        super().train(mode)
        
        if self.freeze_backbone and self.img_backbone is not None:
            self.img_backbone.eval()
        
        if self.freeze_neck and self.img_neck is not None:
            self.img_neck.eval()
        
        if self.freeze_bev_encoder and self.pts_bbox_head is not None:
            self.pts_bbox_head.eval()
    
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        return img_feats_reshaped
    
    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images."""
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats
    
    def get_bev_features(self, img_feats, img_metas, prev_bev=None):
        """Get BEV features from image features.
        
        Returns:
            bev_embed: [B, C, H, W] BEV feature map
        """
        # Use pts_bbox_head to get BEV features
        bev_embed = self.pts_bbox_head(
            img_feats, img_metas, prev_bev, only_bev=True)
        
        # Reshape from [B, H*W, C] to [B, C, H, W]
        B = bev_embed.shape[0]
        bev_h = self.pts_bbox_head.bev_h
        bev_w = self.pts_bbox_head.bev_w
        C = bev_embed.shape[-1]
        
        bev_embed = bev_embed.permute(0, 2, 1).view(B, C, bev_h, bev_w)
        
        return bev_embed
    
    def forward(self, return_loss=True, **kwargs):
        """Forward function."""
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    @auto_fp16(apply_to=('img',))
    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        # MQA specific inputs
        question=None,
        question_type_id=None,
        target_counts=None,
        total_count=None,
        primary_object_class=None,
        primary_object_count=None,
        target_location=None,
        has_location=None,
        target_distance=None,
        has_distance=None,
        camera_prior_mask=None,
        camera_dir_id=None,
        **kwargs
    ):
        """Forward training function for MQA.
        
        Args:
            img: [B, N, C, H, W] input images (N is num cameras)
            img_metas: List of image meta info
            question: List of question strings
            question_type_id: [B] question type indices
            total_count: [B] total object count
            primary_object_class: [B] primary object class
            primary_object_count: [B] primary object count
            target_location: [B, 2] target (x, y) location
            has_location: [B] whether sample has location label
            target_distance: [B, 1] target distance
            has_distance: [B] whether sample has distance label
            camera_prior_mask: [B, H, W] rule-based camera prior
        """
        losses = dict()
        
        # Extract image features
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        # Get BEV features (no temporal for MQA single frame)
        bev_feat = self.get_bev_features(img_feats, img_metas, prev_bev=None)
        
        # Encode question text
        # question is a list of strings
        if isinstance(question, (list, tuple)):
            questions = question
        else:
            questions = [question]
        text_embedding = self.text_encoder(questions)  # [B, D]
        
        # Generate spatial prior
        prior = self.prior_head(text_embedding, camera_prior_mask)  # [B, H, W]
        
        # MQA predictions
        preds = self.mqa_head(bev_feat, prior, text_embedding)
        
        # Compute losses
        mqa_losses = self.mqa_head.loss(
            preds=preds,
            gt_counts=primary_object_count,
            gt_classes=primary_object_class,
            gt_distances=target_distance,
            gt_locations=target_location,
            has_distance=has_distance,
            has_location=has_location,
            question_type_ids=question_type_id,
        )
        
        # Apply loss weights
        losses['loss_count'] = mqa_losses['loss_count'] * self.loss_weight_count
        losses['loss_class'] = mqa_losses['loss_class'] * self.loss_weight_class
        losses['loss_distance'] = mqa_losses['loss_distance'] * self.loss_weight_distance
        losses['loss_location'] = mqa_losses['loss_location'] * self.loss_weight_location
        
        # Prior supervision loss (if ground truth prior is available)
        if self.prior_supervision_head is not None and camera_prior_mask is not None:
            prior_losses = self.prior_supervision_head.loss(prior, camera_prior_mask)
            losses['loss_prior'] = prior_losses['loss_prior'] * self.loss_weight_prior
        
        return losses
    
    def forward_test(self, img_metas, img=None, **kwargs):
        """Forward test function."""
        # Extract question from kwargs
        question = kwargs.get('question', None)
        camera_prior_mask = kwargs.get('camera_prior_mask', None)
        
        if question is None:
            # No question provided, return empty results
            return [{}]
        
        # Handle batch dimension
        if not isinstance(img_metas[0], list):
            img_metas = [img_metas]
        
        img_feats = self.extract_feat(img=img, img_metas=img_metas[0])
        bev_feat = self.get_bev_features(img_feats, img_metas[0], prev_bev=None)
        
        # Encode text
        if isinstance(question, (list, tuple)):
            questions = question
        else:
            questions = [question]
        text_embedding = self.text_encoder(questions)
        
        # Generate prior
        prior = self.prior_head(text_embedding, camera_prior_mask)
        
        # Predictions
        preds = self.mqa_head(bev_feat, prior, text_embedding)
        predictions = self.mqa_head.get_predictions(preds)
        
        # Convert to list of dicts
        B = bev_feat.shape[0]
        results = []
        for i in range(B):
            results.append({
                'count_pred': predictions['count_pred'][i].item(),
                'class_pred': predictions['class_pred'][i].item(),
                'distance_pred': predictions['distance_pred'][i].item(),
                'location_pred': predictions['location_pred'][i].cpu().numpy(),
                'prior': prior[i].cpu().numpy(),
            })
        
        return results
    
    def simple_test(self, img_metas, img=None, **kwargs):
        """Test function without augmentation."""
        return self.forward_test(img_metas, img=img, **kwargs)


@DETECTORS.register_module()
class BEVFormerMQA_FP16(BEVFormerMQA):
    """FP16 version of BEVFormerMQA for mixed precision training."""
    
    @auto_fp16(apply_to=('img',))
    def forward_train(self, *args, **kwargs):
        return super().forward_train(*args, **kwargs)
