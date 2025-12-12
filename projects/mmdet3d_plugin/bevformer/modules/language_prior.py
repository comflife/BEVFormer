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
        freeze: bool = True,
        output_dim: int = 256,
        pooling: str = 'cls',
        max_length: int = 128,
    ):
        super().__init__()
        
        self.pretrained_model = pretrained_model
        self.freeze = freeze
        self.output_dim = output_dim
        self.pooling = pooling
        self.max_length = max_length
        
        # Load BERT model and tokenizer
        from transformers import BertModel, BertTokenizer
        
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.bert = BertModel.from_pretrained(
            pretrained_model, 
            use_safetensors=False  # For compatibility with older PyTorch
        )
        
        # BERT hidden size is typically 768
        bert_hidden_size = self.bert.config.hidden_size
        
        # Projection layer to match BEVFormer embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
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
        
        # Tokenize questions
        inputs = self.tokenizer(
            questions,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get BERT outputs
        with torch.set_grad_enabled(not self.freeze):
            outputs = self.bert(**inputs)
        
        # Pool the outputs
        if self.pooling == 'cls':
            # Use [CLS] token embedding
            pooled = outputs.pooler_output
        elif self.pooling == 'mean':
            # Mean pooling over all tokens
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            pooled = (outputs.last_hidden_state * attention_mask).sum(1)
            pooled = pooled / attention_mask.sum(1)
        elif self.pooling == 'max':
            # Max pooling over all tokens
            pooled = outputs.last_hidden_state.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # Project to output dimension
        text_embedding = self.projection(pooled)
        
        return text_embedding


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
                nn.ReLU(inplace=True),
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
        x = x.view(B, 1, self.low_res_h, self.low_res_w)
        x = self.upsample(x)  # [B, 1, bev_h, bev_w]
        x = x.squeeze(1)  # [B, bev_h, bev_w]
        
        # Apply sigmoid to get probabilities
        p_text = torch.sigmoid(x)
        
        # Combine with rule prior if available
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
                return p_text

            # Ensure rule_prior is on same device
            rule_prior = rule_prior.to(p_text.device)
            
            # Clamp weight to [0, 1]
            alpha = torch.sigmoid(self.rule_prior_weight)
            
            # Mix priors: P = α * P_rule + (1-α) * P_text
            prior = alpha * rule_prior + (1 - alpha) * p_text
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
