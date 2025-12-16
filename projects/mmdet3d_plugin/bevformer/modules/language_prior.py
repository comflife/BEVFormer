# Copyright (c) OpenMMLab. All rights reserved.
"""
Language Prior Module for Language-guided BEVFormer.

This module contains:
1. TextEncoder: BERT-based text encoding
2. PriorHead: Generates spatial prior from text embedding
3. PriorInjection: Injects prior into BEVFormer attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models import HEADS
import math
from collections import OrderedDict


class TextEncoder(BaseModule):
    """BERT-based Text Encoder for question embedding.
    
    Encodes question text into a fixed-size embedding using BERT.
    The encoder is frozen during training by default.
    
    Args:
        pretrained_model: HuggingFace model name (default: 'bert-base-uncased')
        freeze: Whether to freeze the encoder weights
        output_dim: Output embedding dimension
        pooling: Pooling strategy ('cls', 'mean', 'max')
    """
    
    def __init__(
        self,
        pretrained_model: str = 'bert-base-uncased',
        pretrained_weights: str = None,  # Path to custom pretrained weights
        freeze: bool = True,
        output_dim: int = 256,
        pooling: str = 'cls',
        max_length: int = 128,
        enable_cache: bool = True,
        cache_size: int = 10000,
    ):
        super().__init__()
        
        self.pretrained_model = pretrained_model
        self.freeze = freeze
        self.output_dim = output_dim
        self.pooling = pooling
        self.max_length = max_length
        self.enable_cache = enable_cache
        self.cache_size = cache_size

        # Cache for repeated questions.
        # We cache the pooled BERT output (pre-projection) on CPU when BERT is frozen,
        # so projection stays trainable and gradients still flow through projection.
        self._pooled_cache = OrderedDict()  # str -> torch.Tensor([bert_hidden]) on CPU
        
        # Load BERT model and tokenizer
        from transformers import BertModel, BertTokenizer
        
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.bert = BertModel.from_pretrained(
            pretrained_model, 
            use_safetensors=False  # For compatibility with older PyTorch
        )
        
        # Load custom pretrained weights if provided
        if pretrained_weights is not None:
            import os
            import torch
            
            print(f"Loading custom BERT weights from: {pretrained_weights}")
            
            # Check if it's a .pth file or directory
            if os.path.isfile(pretrained_weights) and pretrained_weights.endswith('.pth'):
                # Load from full checkpoint
                checkpoint = torch.load(pretrained_weights, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    # Extract BERT weights from full model checkpoint
                    bert_state_dict = {}
                    for key, value in checkpoint['model_state_dict'].items():
                        if key.startswith('bert.'):
                            # Remove 'bert.' prefix
                            new_key = key[5:]
                            bert_state_dict[new_key] = value
                    
                    if bert_state_dict:
                        self.bert.load_state_dict(bert_state_dict, strict=False)
                        print("✓ Custom BERT weights loaded successfully from checkpoint")
                    else:
                        print("Warning: No BERT weights found in checkpoint, using base BERT")
                else:
                    print("Warning: Invalid checkpoint format, using base BERT weights")
            else:
                print(f"Warning: {pretrained_weights} not found or invalid, using base BERT weights")
        
        # BERT hidden size is typically 768
        bert_hidden_size = self.bert.config.hidden_size
        
        # Projection layer to match BEVFormer embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=False),
        )
        
        # Freeze BERT if specified
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, questions: list) -> torch.Tensor:
        """Encode questions to embeddings.
        
        Args:
            questions: List of question strings
            
        Returns:
            Text embeddings of shape [B, output_dim]
        """
        device = next(self.projection.parameters()).device
        
        # If BERT is frozen, we can safely cache the pooled (pre-projection) output.
        # If BERT is trainable (freeze=False), NEVER detach pooled outputs, otherwise
        # BERT parameters will not receive gradients and DDP will error on unused params.
        use_cache = bool(self.enable_cache and self.freeze and self.cache_size and self.cache_size > 0)

        if not use_cache:
            inputs = self.tokenizer(
                questions,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.set_grad_enabled(not self.freeze):
                outputs = self.bert(**inputs)

            if self.pooling == 'cls':
                pooled = outputs.pooler_output
            elif self.pooling == 'mean':
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                pooled = (outputs.last_hidden_state * attention_mask).sum(1)
                pooled = pooled / attention_mask.sum(1)
            elif self.pooling == 'max':
                pooled = outputs.last_hidden_state.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling}")

            proj_dtype = self.projection[0].weight.dtype
            pooled = pooled.to(dtype=proj_dtype)
            return self.projection(pooled)

        pooled_cpu_list = [None] * len(questions)
        missing_questions = []
        missing_indices = []

        for idx, q in enumerate(questions):
            cached = self._pooled_cache.get(q)
            if cached is not None:
                # LRU: refresh
                self._pooled_cache.move_to_end(q)
                pooled_cpu_list[idx] = cached
            else:
                missing_questions.append(q)
                missing_indices.append(idx)

        if len(missing_questions) > 0:
            # Tokenize only missing questions
            inputs = self.tokenizer(
                missing_questions,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get BERT outputs (frozen)
            with torch.set_grad_enabled(not self.freeze):
                outputs = self.bert(**inputs)

            # Pool the outputs
            if self.pooling == 'cls':
                pooled = outputs.pooler_output
            elif self.pooling == 'mean':
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                pooled = (outputs.last_hidden_state * attention_mask).sum(1)
                pooled = pooled / attention_mask.sum(1)
            elif self.pooling == 'max':
                pooled = outputs.last_hidden_state.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling}")

            # Move pooled to CPU for caching, one item per question.
            pooled_cpu = pooled.detach().to('cpu')

            for j, idx in enumerate(missing_indices):
                pooled_cpu_list[idx] = pooled_cpu[j]
                q = questions[idx]
                self._pooled_cache[q] = pooled_cpu[j]
                self._pooled_cache.move_to_end(q)
                # Evict LRU
                while len(self._pooled_cache) > self.cache_size:
                    self._pooled_cache.popitem(last=False)

        # Stack pooled outputs in original order and move to device for projection
        pooled = torch.stack(pooled_cpu_list, dim=0)
        proj_dtype = self.projection[0].weight.dtype
        pooled = pooled.to(device=device, dtype=proj_dtype)
        return self.projection(pooled)


@HEADS.register_module()
class PriorHead(BaseModule):
    """Prior Head for generating spatial prior from text embedding.
    
    Takes text embedding and generates a probability map over the BEV grid,
    indicating which regions the language refers to.
    
    Args:
        text_dim: Input text embedding dimension
        bev_h, bev_w: BEV grid dimensions
        hidden_dim: Hidden layer dimension
        num_layers: Number of MLP layers
        use_rule_prior: Whether to combine with rule-based camera prior
        rule_prior_weight: Initial weight for rule prior (α in P = α*P_rule + (1-α)*P_text)
    """
    
    def __init__(
        self,
        text_dim: int = 256,
        bev_h: int = 150,
        bev_w: int = 150,
        hidden_dim: int = 512,
        num_layers: int = 3,
        use_rule_prior: bool = True,
        rule_prior_weight: float = 0.5,
        init_cfg: dict = None,
    ):
        super().__init__(init_cfg)
        
        self.text_dim = text_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.hidden_dim = hidden_dim
        self.use_rule_prior = use_rule_prior
        
        # Learnable mixing weight (initialized to give more weight to rule prior)
        if use_rule_prior:
            self.rule_prior_weight = nn.Parameter(
                torch.tensor(rule_prior_weight))
        
        # MLP to generate low-resolution prior
        low_res_h = bev_h // 4  # 37 for 150
        low_res_w = bev_w // 4
        
        layers = []
        in_dim = text_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=False),  # Changed to avoid gradient issues
            ])
            in_dim = hidden_dim
        
        # Final layer outputs low-res prior
        layers.append(nn.Linear(hidden_dim, low_res_h * low_res_w))
        
        self.mlp = nn.Sequential(*layers)
        
        self.low_res_h = low_res_h
        self.low_res_w = low_res_w
        
        # Upsampling to full resolution
        self.upsample = nn.Upsample(
            size=(bev_h, bev_w),
            mode='bilinear',
            align_corners=False
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize final layer to output near-uniform distribution
        final_layer = self.mlp[-1]
        nn.init.zeros_(final_layer.weight)
        if final_layer.bias is not None:
            nn.init.zeros_(final_layer.bias)
    
    def forward(
        self,
        text_embedding: torch.Tensor,
        rule_prior=None,
    ) -> torch.Tensor:
        """Generate spatial prior from text embedding.
        
        Args:
            text_embedding: [B, text_dim] text embedding
            rule_prior: [B, H, W] optional rule-based prior mask
            
        Returns:
            prior: [B, H, W] spatial prior probability map (after sigmoid)
        """
        B = text_embedding.shape[0]

        # Generate text-based prior
        x = self.mlp(text_embedding)  # [B, low_res_h * low_res_w]
        x = x.reshape(B, 1, self.low_res_h, self.low_res_w)
        x = self.upsample(x)  # [B, 1, bev_h, bev_w]

        # Apply sigmoid, squeeze, and clone to create independent tensor
        # This avoids inplace modification errors during backward pass
        p_text = torch.sigmoid(x).squeeze(1).clone()  # [B, bev_h, bev_w]
        
        # Combine with rule prior if available.
        # IMPORTANT (DDP): `rule_prior_weight` is a learnable Parameter created when
        # `use_rule_prior=True`. Some batches/ranks may not provide `rule_prior`.
        # If we return `p_text` without touching `rule_prior_weight`, DDP can throw
        # "Expected to have finished reduction..." due to unused parameters.
        # We keep `rule_prior_weight` in the autograd graph even when `rule_prior`
        # is missing by adding a connected zero term.
        if self.use_rule_prior:
            alpha = torch.sigmoid(self.rule_prior_weight)

        if self.use_rule_prior and rule_prior is not None:
            # Dataloader may provide rule_prior as a list (when not stacked).
            # Normalize to a tensor of shape [B, H, W].
            if isinstance(rule_prior, (list, tuple)):
                if len(rule_prior) == 0:
                    rule_prior = None
                elif torch.is_tensor(rule_prior[0]):
                    rule_prior = torch.stack(rule_prior, dim=0)
                else:
                    rule_prior = torch.as_tensor(rule_prior)

            if rule_prior is None:
                # Keep `rule_prior_weight` connected for DDP.
                return p_text + (alpha * 0.0)

            # Ensure rule_prior is on same device
            rule_prior = rule_prior.to(p_text.device)
            
            # Mix priors: P = α * P_rule + (1-α) * P_text
            # Clone the result to ensure it's an independent tensor
            prior = (alpha * rule_prior + (1 - alpha) * p_text).clone()
        else:
            # If rule prior isn't provided, still keep `rule_prior_weight` connected.
            if self.use_rule_prior:
                prior = p_text + (alpha * 0.0)
            else:
                prior = p_text
        
        return prior


class PriorInjection(BaseModule):
    """Module to inject spatial prior into attention mechanism.
    
    This module modifies attention weights based on the spatial prior,
    encouraging the model to attend more to regions indicated by language.
    
    Two injection strategies:
    1. Additive bias: logit' = logit + λ * log(prior + ε)
    2. Multiplicative weight: weight' = weight * (prior + ε), then normalize
    
    Args:
        num_layers: Number of encoder layers to inject into
        injection_type: 'additive' or 'multiplicative'
        init_lambda: Initial value for λ (learnable per layer)
        epsilon: Small value for numerical stability
    """
    
    def __init__(
        self,
        num_layers: int = 3,
        injection_type: str = 'additive',
        init_lambda: float = 1.0,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.injection_type = injection_type
        self.epsilon = epsilon
        
        # Learnable λ for each layer
        self.lambdas = nn.ParameterList([
            nn.Parameter(torch.tensor(init_lambda))
            for _ in range(num_layers)
        ])
    
    def get_attention_bias(
        self,
        prior: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute attention bias from prior.
        
        Args:
            prior: [B, H, W] spatial prior
            layer_idx: Index of the encoder layer
            
        Returns:
            bias: [B, H*W] attention bias
        """
        B, H, W = prior.shape
        
        # Flatten prior to [B, H*W]
        prior_flat = prior.view(B, H * W)
        
        if self.injection_type == 'additive':
            # log(prior + ε) as additive bias
            # High prior -> positive bias -> higher attention
            # Low prior -> negative bias -> lower attention
            lam = self.lambdas[layer_idx]
            bias = lam * torch.log(prior_flat + self.epsilon)
        else:
            # For multiplicative, return the prior directly
            bias = prior_flat
        
        return bias
    
    def apply_to_attention_weights(
        self,
        attention_weights: torch.Tensor,
        prior: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Apply prior to attention weights (multiplicative method).
        
        Args:
            attention_weights: [B, num_heads, num_queries, H*W]
            prior: [B, H, W]
            layer_idx: Index of the encoder layer
            
        Returns:
            modified_weights: Same shape as attention_weights
        """
        B, num_heads, num_queries, HW = attention_weights.shape
        H = W = int(math.sqrt(HW))
        
        # Get prior as multiplicative factor
        prior_flat = prior.view(B, 1, 1, HW)  # [B, 1, 1, HW]
        lam = self.lambdas[layer_idx]
        
        # Multiply and renormalize
        weights = attention_weights * (prior_flat + self.epsilon) ** lam
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.epsilon)
        
        return weights


@HEADS.register_module()
class PriorGuidedModulation(BaseModule):
    """Prior-Guided BEV Feature Modulation Module.

    This module uses language prior to modulate BEV features through:
    1. Channel-wise modulation (FiLM) based on text embedding
    2. Spatial modulation using prior map as additional channel
    3. Residual connection to preserve original BEV information

    Args:
        text_dim: Text embedding dimension
        bev_channels: BEV feature channels
        bev_h, bev_w: BEV grid dimensions
        hidden_dim: Hidden dimension for prior head
        num_prior_layers: Number of layers in prior head
        use_rule_prior: Whether to use rule-based camera prior
        rule_prior_weight: Initial weight for rule prior mixing
    """

    def __init__(
        self,
        text_dim: int = 256,
        bev_channels: int = 256,
        bev_h: int = 150,
        bev_w: int = 150,
        hidden_dim: int = 512,
        num_prior_layers: int = 3,
        use_rule_prior: bool = True,
        rule_prior_weight: float = 0.5,
        init_cfg: dict = None,
    ):
        super().__init__(init_cfg)

        self.text_dim = text_dim
        self.bev_channels = bev_channels
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Prior head for generating spatial prior
        self.prior_head = PriorHead(
            text_dim=text_dim,
            bev_h=bev_h,
            bev_w=bev_w,
            hidden_dim=hidden_dim,
            num_layers=num_prior_layers,
            use_rule_prior=use_rule_prior,
            rule_prior_weight=rule_prior_weight,
        )

        # Channel-wise modulation (FiLM)
        self.gamma_fc = nn.Linear(text_dim, bev_channels)
        self.beta_fc = nn.Linear(text_dim, bev_channels)

        # Spatial refinement: concat prior as extra channel
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(bev_channels + 1, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=False),  # Changed from inplace=True to avoid gradient issues
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training.

        Strategy: Start from identity (language has no effect initially)
        - gamma = 1.0 (preserve original features)
        - beta = 0.0 (no shift)
        - spatial refinement ≈ 0 (via residual connection)
        """
        # Initialize FiLM parameters to output constant values
        # gamma_fc should output 1.0 for any input
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.ones_(self.gamma_fc.bias)   # gamma = 0*text + 1 = 1.0

        # beta_fc should output 0.0 for any input
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)   # beta = 0*text + 0 = 0.0

        # Initialize spatial refinement to near-zero output
        # This makes the residual connection start from identity
        for m in self.spatial_refine.modules():
            if isinstance(m, nn.Conv2d):
                # Small random initialization (not kaiming)
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        bev_feat: torch.Tensor,
        text_embedding: torch.Tensor,
        camera_prior: torch.Tensor = None,
    ) -> tuple:
        """Forward pass for BEV feature modulation.

        Args:
            bev_feat: [B, C, H, W] raw BEV features from BEVFormer
            text_embedding: [B, D] text embedding from BERT
            camera_prior: [B, H, W] optional rule-based camera prior

        Returns:
            tuple of:
                - modulated_feat: [B, C, H, W] modulated BEV features
                - prior: [B, H, W] spatial prior map
        """
        B, C, H, W = bev_feat.shape

        # 1. Generate spatial prior from text
        prior = self.prior_head(text_embedding, camera_prior)  # [B, H, W]

        # 2. Channel-wise modulation (FiLM)
        gamma = self.gamma_fc(text_embedding).reshape(B, C, 1, 1)  # [B, C, 1, 1]
        beta = self.beta_fc(text_embedding).reshape(B, C, 1, 1)    # [B, C, 1, 1]

        # Apply FiLM: modulate each channel
        # Use clone() to create a completely independent tensor
        feat_modulated = (gamma * bev_feat + beta).clone()  # [B, C, H, W]

        # 3. Spatial modulation: concatenate prior as additional channel
        # Clone prior to create a completely new tensor that doesn't share storage
        # This avoids the inplace modification error during backward (UnsqueezeBackward0)
        prior_for_concat = prior.clone().unsqueeze(1)  # [B, 1, H, W]
        feat_with_prior = torch.cat([feat_modulated, prior_for_concat], dim=1)  # [B, C+1, H, W]

        # Apply spatial refinement
        feat_refined = self.spatial_refine(feat_with_prior)  # [B, C, H, W]

        # 4. Residual connection to preserve original information
        modulated_feat = feat_refined + bev_feat  # [B, C, H, W]

        return modulated_feat, prior


class LanguagePriorModule(BaseModule):
    """Complete Language Prior Module combining all components.

    This module:
    1. Encodes question text to embedding
    2. Generates spatial prior from text
    3. Provides interface for attention injection

    Args:
        text_encoder_cfg: Config for TextEncoder
        prior_head_cfg: Config for PriorHead
        prior_injection_cfg: Config for PriorInjection
    """

    def __init__(
        self,
        text_encoder_cfg: dict = None,
        prior_head_cfg: dict = None,
        prior_injection_cfg: dict = None,
    ):
        super().__init__()

        # Default configs
        if text_encoder_cfg is None:
            text_encoder_cfg = {}
        if prior_head_cfg is None:
            prior_head_cfg = {}
        if prior_injection_cfg is None:
            prior_injection_cfg = {}

        # Build modules
        self.text_encoder = TextEncoder(**text_encoder_cfg)
        self.prior_head = PriorHead(**prior_head_cfg)
        self.prior_injection = PriorInjection(**prior_injection_cfg)

    def forward(
        self,
        questions: list,
        rule_prior: torch.Tensor = None,
    ) -> dict:
        """Forward pass.

        Args:
            questions: List of question strings
            rule_prior: Optional rule-based prior [B, H, W]

        Returns:
            dict containing:
                - text_embedding: [B, D] text embedding
                - prior: [B, H, W] spatial prior
        """
        # Encode text
        text_embedding = self.text_encoder(questions)

        # Generate prior
        prior = self.prior_head(text_embedding, rule_prior)

        return {
            'text_embedding': text_embedding,
            'prior': prior,
        }

    def get_attention_bias(
        self,
        prior: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Get attention bias for a specific layer."""
        return self.prior_injection.get_attention_bias(prior, layer_idx)
