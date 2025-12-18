"""
CLIP Contrastive Learning Model for Multi-Camera Images and QA Text

Uses a small CLIP model to align:
- Visual: Multiple camera views from NuScenes (6 cameras)
- Text: MQA question-answer pairs

Model: openai/clip-vit-base-patch32 (smaller and faster than ViT-B/16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPConfig


class CLIPMultiViewContrastive(nn.Module):
    """CLIP model for multi-view images and QA text alignment.

    Visual encoder: Process 6 camera views and aggregate features
    Text encoder: Process question-answer pairs
    Contrastive loss: Align visual and text representations
    """

    def __init__(
        self,
        model_name: str = 'openai/clip-vit-base-patch32',
        embedding_dim: int = 256,
        temperature: float = 0.07,
        visual_aggregation: str = 'mean',  # 'mean', 'max', 'attention'
        freeze_vision: bool = False,
        freeze_text: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.visual_aggregation = visual_aggregation

        # Load CLIP model
        self.clip = CLIPModel.from_pretrained(model_name)

        # Store original vocab size for potential token expansion
        self.original_vocab_size = self.clip.text_model.config.vocab_size

        # Freeze components if needed
        if freeze_vision:
            for param in self.clip.vision_model.parameters():
                param.requires_grad = False

        if freeze_text:
            for param in self.clip.text_model.parameters():
                param.requires_grad = False

        # Get CLIP hidden dimensions
        clip_vision_dim = self.clip.config.vision_config.hidden_size  # 768 for base
        clip_text_dim = self.clip.config.text_config.hidden_size      # 512 for base

        # Camera positional embeddings (6 cameras: front, front_left, front_right, back, back_left, back_right)
        self.camera_positional = nn.Parameter(torch.zeros(6, clip_vision_dim))
        nn.init.normal_(self.camera_positional, mean=0.0, std=0.02)

        # Attention-based aggregation for multi-view
        if visual_aggregation == 'attention':
            self.view_attention = nn.Sequential(
                nn.Linear(clip_vision_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )

        # Projection heads to common embedding space
        self.visual_projection = nn.Sequential(
            nn.Linear(clip_vision_dim, clip_vision_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(clip_vision_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(clip_text_dim, clip_text_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(clip_text_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def resize_token_embeddings(self, new_num_tokens: int):
        """Manually resize token embeddings for CLIP text model.

        Note: CLIPTextTransformer doesn't have resize_token_embeddings() method,
        so we need to manually create a new embedding layer.

        Args:
            new_num_tokens: New vocabulary size
        """
        old_embeddings = self.clip.text_model.embeddings.token_embedding

        if new_num_tokens == old_embeddings.num_embeddings:
            return  # Already correct size

        # Create new embedding layer with same dtype and device
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
        new_embeddings = new_embeddings.to(device=old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # Copy old weights
        new_embeddings.weight.data[:old_embeddings.num_embeddings] = old_embeddings.weight.data

        # Initialize new token embeddings with small random values
        if new_num_tokens > old_embeddings.num_embeddings:
            new_embeddings.weight.data[old_embeddings.num_embeddings:].normal_(mean=0.0, std=0.02)

        # Replace the embedding layer
        self.clip.text_model.embeddings.token_embedding = new_embeddings

        # Update config to match (important for save/load)
        self.clip.text_model.config.vocab_size = new_num_tokens
        self.clip.config.text_config.vocab_size = new_num_tokens

        print(f"Resized token embeddings: {old_embeddings.num_embeddings} → {new_num_tokens}")

    def encode_multiview_images(self, pixel_values):
        """Encode multi-view images.

        Args:
            pixel_values: [B, N_views, C, H, W] where N_views=6 for NuScenes

        Returns:
            visual_embeddings: [B, embedding_dim] aggregated visual features
        """
        B, N, C, H, W = pixel_values.shape

        # Reshape to process all views together (use reshape for non-contiguous tensors)
        pixel_values = pixel_values.reshape(B * N, C, H, W)

        # CLIP vision encoding
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output  # [B*N, vision_dim]

        # Reshape back to [B, N, vision_dim]
        pooled_output = pooled_output.view(B, N, -1)

        # Add camera positional embeddings
        # Camera order: [front, front_left, front_right, back, back_left, back_right]
        pooled_output = pooled_output + self.camera_positional.unsqueeze(0)  # [B, N, vision_dim]

        # Aggregate multi-view features
        if self.visual_aggregation == 'mean':
            aggregated = pooled_output.mean(dim=1)  # [B, vision_dim]
        elif self.visual_aggregation == 'max':
            aggregated = pooled_output.max(dim=1)[0]  # [B, vision_dim]
        elif self.visual_aggregation == 'attention':
            # Attention weights for each view
            attn_scores = self.view_attention(pooled_output)  # [B, N, 1]
            attn_weights = F.softmax(attn_scores, dim=1)  # [B, N, 1]
            aggregated = (pooled_output * attn_weights).sum(dim=1)  # [B, vision_dim]
        else:
            raise ValueError(f"Unknown aggregation: {self.visual_aggregation}")

        # Project to embedding space
        visual_embeddings = self.visual_projection(aggregated)  # [B, embedding_dim]

        # L2 normalize
        visual_embeddings = F.normalize(visual_embeddings, p=2, dim=1)

        return visual_embeddings

    def encode_text(self, input_ids, attention_mask):
        """Encode text (question-answer pairs).

        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask

        Returns:
            text_embeddings: [B, embedding_dim] text features
        """
        # CLIP text encoding
        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = text_outputs.pooler_output  # [B, text_dim]

        # Project to embedding space
        text_embeddings = self.text_projection(pooled_output)  # [B, embedding_dim]

        # L2 normalize
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        return text_embeddings

    def forward(self, pixel_values, input_ids, attention_mask):
        """Forward pass.

        Args:
            pixel_values: [B, N_views, C, H, W] multi-view images
            input_ids: [B, L] text token IDs
            attention_mask: [B, L] text attention mask

        Returns:
            visual_embeddings: [B, embedding_dim]
            text_embeddings: [B, embedding_dim]
        """
        visual_embeddings = self.encode_multiview_images(pixel_values)
        text_embeddings = self.encode_text(input_ids, attention_mask)

        return visual_embeddings, text_embeddings


class CLIPInfoNCELoss(nn.Module):
    """InfoNCE Loss for CLIP-style contrastive learning with hard negative mining.

    Hard negatives: QA pairs with similar structure but different counts/objects/cameras
    This forces the model to pay attention to numerical and categorical details.

    Symmetric loss: image->text and text->image
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_weight: float = 1.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def _extract_count(self, text: str) -> str:
        """Extract count from <cnt>N</cnt> tag."""
        import re
        match = re.search(r'<cnt>(\d+)</cnt>', text)
        return match.group(1) if match else None

    def _extract_object(self, text: str) -> str:
        """Extract primary object from <obj>X</obj> tag."""
        import re
        match = re.search(r'<obj>([^<]+)</obj>', text)
        return match.group(1) if match else None

    def _extract_camera(self, text: str) -> str:
        """Extract camera from <cam>X</cam> tag."""
        import re
        match = re.search(r'<cam>([^<]+)</cam>', text)
        return match.group(1) if match else None

    def _get_hard_negative_mask(self, texts: list, sample_tokens: list, device=None) -> torch.Tensor:
        """Create mask for hard negatives in the batch.

        Hard negative criteria (only applied to different images):
        - Same camera but different count
        - Same camera but different object
        - Same object but different count

        Args:
            texts: List of text strings
            sample_tokens: List of sample tokens (scene IDs) to identify same/different images
            device: Optional device to create tensor on

        Returns:
            mask: [B, B] binary mask (1 = hard negative, 0 = easy negative/positive)
        """
        batch_size = len(texts)
        mask = torch.zeros(batch_size, batch_size, device=device, dtype=torch.float32)

        # Extract features for all texts
        features = []
        for text in texts:
            features.append({
                'count': self._extract_count(text),
                'object': self._extract_object(text),
                'camera': self._extract_camera(text),
            })

        # Find hard negatives
        for i in range(batch_size):
            feat_i = features[i]
            token_i = sample_tokens[i]

            for j in range(batch_size):
                if i == j:
                    continue  # Skip self

                token_j = sample_tokens[j]

                # CRITICAL: Only consider pairs from DIFFERENT images
                if token_i == token_j:
                    continue  # Same image -> skip (will be handled by positive mask)

                feat_j = features[j]

                # Hard negative criteria (only for different images)
                is_hard = False

                # 1. Same camera but different count
                if (feat_i['camera'] == feat_j['camera'] and
                    feat_i['count'] != feat_j['count'] and
                    feat_i['count'] is not None and feat_j['count'] is not None):
                    is_hard = True

                # 2. Same camera but different object
                if (feat_i['camera'] == feat_j['camera'] and
                    feat_i['object'] != feat_j['object'] and
                    feat_i['object'] is not None and feat_j['object'] is not None):
                    is_hard = True

                # 3. Same object but different count
                if (feat_i['object'] == feat_j['object'] and
                    feat_i['count'] != feat_j['count'] and
                    feat_i['count'] is not None and feat_j['count'] is not None):
                    is_hard = True

                if is_hard:
                    mask[i, j] = 1.0

        return mask

    def _get_positive_mask(self, sample_tokens: list, device=None) -> torch.Tensor:
        """Create mask for positive pairs (same image).

        Args:
            sample_tokens: List of sample tokens (scene IDs)
            device: Optional device to create tensor on

        Returns:
            mask: [B, B] binary mask (1 = positive pair, 0 = negative pair)
        """
        batch_size = len(sample_tokens)
        mask = torch.zeros(batch_size, batch_size, device=device, dtype=torch.float32)

        for i in range(batch_size):
            for j in range(batch_size):
                if sample_tokens[i] == sample_tokens[j]:
                    mask[i, j] = 1.0

        # Ensure diagonal is always 1 (self must be positive)
        mask.fill_diagonal_(1.0)

        return mask

    def forward(
        self,
        visual_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        texts: list = None,
        sample_tokens: list = None,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss with multi-positive support and hard negative weighting.

        Properly implements weighted InfoNCE by applying weights to denominator exponentials,
        not by scaling logits.

        Args:
            visual_embeddings: [B, D] normalized visual embeddings
            text_embeddings: [B, D] normalized text embeddings
            texts: Optional list of text strings for hard negative mining
            sample_tokens: Optional list of sample tokens for identifying same images

        Returns:
            loss: scalar loss
        """
        batch_size = visual_embeddings.shape[0]
        device = visual_embeddings.device

        # Compute similarity matrix [B, B]
        logits = torch.matmul(visual_embeddings, text_embeddings.T) / self.temperature

        # Get positive mask (same image = positive)
        if sample_tokens is not None:
            pos_mask = self._get_positive_mask(sample_tokens, device=device)
        else:
            # Default: diagonal only
            pos_mask = torch.eye(batch_size, device=device)

        # Get hard negative mask and weights
        if texts is not None and sample_tokens is not None and self.hard_negative_weight > 1.0:
            hard_neg_mask = self._get_hard_negative_mask(texts, sample_tokens, device=device)
        else:
            hard_neg_mask = torch.zeros(batch_size, batch_size, device=device, dtype=torch.float32)

        # Compute weighted InfoNCE loss (image-to-text)
        loss_i2t = self._compute_weighted_infonce(logits, pos_mask, hard_neg_mask)

        # Compute weighted InfoNCE loss (text-to-image)
        loss_t2i = self._compute_weighted_infonce(logits.T, pos_mask.T, hard_neg_mask.T)

        loss = (loss_i2t + loss_t2i) / 2

        return loss

    def _compute_weighted_infonce(
        self,
        logits: torch.Tensor,
        pos_mask: torch.Tensor,
        hard_neg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted InfoNCE loss with numerically stable logsumexp.

        Uses logsumexp to prevent exp overflow when logits are large.

        Args:
            logits: [B, B] similarity matrix / temperature
            pos_mask: [B, B] binary mask for positive pairs
            hard_neg_mask: [B, B] binary mask for hard negatives

        Returns:
            loss: scalar loss
        """
        batch_size = logits.shape[0]
        device = logits.device

        # Create weight matrix for negatives
        # Positive pairs: exclude from negative denominator (will be in numerator)
        # Hard negatives: weight = hard_negative_weight
        # Easy negatives: weight = 1.0
        neg_mask = 1.0 - pos_mask  # [B, B] negatives

        # Compute log weights for stable logsumexp
        # log(weight) where weight = 1.0 + (hard_negative_weight - 1.0) * hard_neg_mask
        log_weights = torch.log(
            neg_mask * (1.0 + (self.hard_negative_weight - 1.0) * hard_neg_mask) + 1e-10
        )  # [B, B]

        # Mask out non-negatives (positives) with -inf so they don't contribute to logsumexp
        log_weights = torch.where(
            neg_mask.bool(),
            log_weights,
            torch.full_like(log_weights, float('-inf'))
        )

        # Weighted negative logsumexp: logsumexp(logits + log_weights)
        # This is numerically stable version of: log(sum(weight * exp(logit)))
        weighted_neg_logsumexp = torch.logsumexp(
            logits + log_weights, dim=1, keepdim=True
        )  # [B, 1]

        # Positive logsumexp: logsumexp over positive pairs only
        pos_logits = torch.where(
            pos_mask.bool(),
            logits,
            torch.full_like(logits, float('-inf'))
        )
        pos_logsumexp = torch.logsumexp(pos_logits, dim=1, keepdim=True)  # [B, 1]

        # Denominator: log(sum(weighted_neg_exp) + sum(pos_exp))
        # = logsumexp([weighted_neg_logsumexp, pos_logsumexp])
        denominator_logsumexp = torch.logsumexp(
            torch.cat([weighted_neg_logsumexp, pos_logsumexp], dim=1),
            dim=1,
            keepdim=True
        )  # [B, 1]

        # Loss: -log(pos_sum / denominator) = -(pos_logsumexp - denominator_logsumexp)
        log_prob = pos_logsumexp - denominator_logsumexp
        loss = -log_prob.mean()

        return loss


if __name__ == '__main__':
    # Test model
    model = CLIPMultiViewContrastive(
        model_name='openai/clip-vit-base-patch32',
        embedding_dim=256,
        temperature=0.07,
        visual_aggregation='attention',
    )

    # Dummy inputs
    batch_size = 2
    n_views = 6
    pixel_values = torch.randn(batch_size, n_views, 3, 224, 224)
    input_ids = torch.randint(0, 49408, (batch_size, 77))  # CLIP vocab size
    attention_mask = torch.ones_like(input_ids)

    # Forward
    visual_emb, text_emb = model(pixel_values, input_ids, attention_mask)
    print(f"Visual embedding shape: {visual_emb.shape}")
    print(f"Text embedding shape: {text_emb.shape}")
    print(f"Visual normalized: {torch.norm(visual_emb, p=2, dim=1)}")
    print(f"Text normalized: {torch.norm(text_emb, p=2, dim=1)}")

    # Test loss
    criterion = CLIPInfoNCELoss(temperature=0.07)
    loss = criterion(visual_emb, text_emb)
    print(f"Loss: {loss.item():.4f}")
