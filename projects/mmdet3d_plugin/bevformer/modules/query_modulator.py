"""
Query-Level Language Modulation for BEVFormer

Instead of modulating the entire BEV feature map (expensive),
we modulate only the object queries (300 queries vs 2500 BEV features).

This is 8.3x more efficient while maintaining the language-guidance capability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.runner import BaseModule


class QueryModulator(BaseModule):
    """Modulate object queries with language features.

    This module applies FiLM-style modulation to object queries based on
    text embeddings, providing language guidance at the query level.

    Args:
        query_dim (int): Dimension of object queries
        text_dim (int): Dimension of text embeddings
        num_layers (int): Number of modulation layers (1 or 2)
        use_residual (bool): Whether to use residual connection
    """

    def __init__(
        self,
        query_dim=256,
        text_dim=256,
        num_layers=1,
        use_residual=True,
        init_cfg=None,
    ):
        super(QueryModulator, self).__init__(init_cfg=init_cfg)

        self.query_dim = query_dim
        self.text_dim = text_dim
        self.num_layers = num_layers
        self.use_residual = use_residual

        # FiLM-style modulation parameters
        # gamma and beta for adaptive feature modulation
        self.gamma_fc = nn.Linear(text_dim, query_dim)
        self.beta_fc = nn.Linear(text_dim, query_dim)

        if num_layers == 2:
            # Optional: deeper modulation
            self.pre_norm = nn.LayerNorm(query_dim)
            self.post_fc = nn.Sequential(
                nn.Linear(query_dim, query_dim),
                nn.ReLU(inplace=True),
                nn.Linear(query_dim, query_dim),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights to ensure identity mapping at start."""
        # Initialize gamma to 1 (multiplicative identity)
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.ones_(self.gamma_fc.bias)

        # Initialize beta to 0 (additive identity)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

        # Initialize post layers if present
        if self.num_layers == 2:
            for m in self.post_fc.modules():
                if isinstance(m, nn.Linear):
                    xavier_init(m, distribution='uniform')

    def forward(self, queries, text_emb):
        """Forward function.

        Args:
            queries (Tensor): Object queries [B, N_q, D] or [N_q, D]
            text_emb (Tensor): Text embeddings [B, D]

        Returns:
            Tensor: Modulated queries [B, N_q, D] or [N_q, D]
        """
        # Handle both [B, N_q, D] and [N_q, D] formats
        is_batched = queries.dim() == 3
        if not is_batched:
            queries = queries.unsqueeze(0)  # [N_q, D] -> [1, N_q, D]
            text_emb = text_emb.unsqueeze(0)  # [D] -> [1, D]

        B, N_q, D = queries.shape

        # Compute FiLM parameters from text
        gamma = self.gamma_fc(text_emb).unsqueeze(1)  # [B, 1, D]
        beta = self.beta_fc(text_emb).unsqueeze(1)    # [B, 1, D]

        # Apply FiLM modulation
        modulated = gamma * queries + beta  # [B, N_q, D]

        # Optional: deeper transformation
        if self.num_layers == 2:
            residual = modulated
            modulated = self.pre_norm(modulated)
            modulated = self.post_fc(modulated)
            if self.use_residual:
                modulated = modulated + residual

        # Remove batch dim if input wasn't batched
        if not is_batched:
            modulated = modulated.squeeze(0)  # [1, N_q, D] -> [N_q, D]

        return modulated


class SparseQueryModulator(BaseModule):
    """Query modulator with sparse BEV attention using camera prior.

    This is a more advanced version that selectively attends to relevant
    BEV regions based on camera prior mask.

    Args:
        query_dim (int): Dimension of object queries
        text_dim (int): Dimension of text embeddings
        bev_dim (int): Dimension of BEV features
        num_heads (int): Number of attention heads
        top_k (int): Number of top BEV locations to attend to
    """

    def __init__(
        self,
        query_dim=256,
        text_dim=256,
        bev_dim=256,
        num_heads=8,
        top_k=100,
        init_cfg=None,
    ):
        super(SparseQueryModulator, self).__init__(init_cfg=init_cfg)

        self.query_dim = query_dim
        self.text_dim = text_dim
        self.bev_dim = bev_dim
        self.num_heads = num_heads
        self.top_k = top_k

        # Text-guided query transformation
        self.query_proj = nn.Linear(query_dim + text_dim, query_dim)

        # Sparse cross-attention to BEV
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Layer norm
        self.norm = nn.LayerNorm(query_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        xavier_init(self.query_proj, distribution='uniform')

    def forward(self, queries, text_emb, bev_feat=None, camera_prior=None):
        """Forward function with sparse BEV attention.

        Args:
            queries (Tensor): Object queries [B, N_q, D]
            text_emb (Tensor): Text embeddings [B, D]
            bev_feat (Tensor, optional): BEV features [B, C, H, W]
            camera_prior (Tensor, optional): Camera prior mask [B, H, W]

        Returns:
            Tensor: Modulated queries [B, N_q, D]
        """
        B, N_q, D = queries.shape

        # Text-guided query features
        text_expanded = text_emb.unsqueeze(1).expand(-1, N_q, -1)  # [B, N_q, text_dim]
        query_text = torch.cat([queries, text_expanded], dim=-1)  # [B, N_q, D+text_dim]
        query_features = self.query_proj(query_text)  # [B, N_q, D]

        # If BEV and camera prior provided, do sparse attention
        if bev_feat is not None and camera_prior is not None:
            _, C, H, W = bev_feat.shape

            # Select top-k BEV locations based on camera prior
            prior_flat = camera_prior.view(B, -1)  # [B, H*W]
            topk_values, topk_indices = torch.topk(prior_flat, k=min(self.top_k, H*W), dim=1)

            # Extract relevant BEV features
            bev_flat = bev_feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

            # Gather top-k features
            relevant_bev = torch.gather(
                bev_flat,
                1,
                topk_indices.unsqueeze(-1).expand(-1, -1, C)
            )  # [B, top_k, C]

            # Project BEV features to query dimension if needed
            if C != D:
                relevant_bev = F.linear(relevant_bev, self.cross_attn.in_proj_weight[:D*3:3, :C])

            # Cross-attention: queries attend to relevant BEV regions
            modulated, _ = self.cross_attn(
                query_features,  # query
                relevant_bev,    # key
                relevant_bev,    # value
            )

            # Residual + norm
            modulated = self.norm(modulated + query_features)
        else:
            # No BEV attention, just return text-guided queries
            modulated = query_features

        return modulated
