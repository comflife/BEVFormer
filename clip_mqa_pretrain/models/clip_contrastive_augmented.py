"""
Enhanced CLIP Contrastive Learning with Hard Negative Mining for MQA

Inspired by NLP tag embedding papers, this version adds:
1. Hard negative mining - samples with same structure but different counts/objects
2. Auxiliary count/object classification loss for better numerical understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
import re


class CLIPInfoNCEWithHardNegatives(nn.Module):
    """Enhanced InfoNCE Loss with hard negative mining.

    Hard negatives are samples that share similar structure but differ in
    semantically critical parts (counts, object types, camera positions).

    Example:
        Anchor:   "How many <obj>cars</obj> in <cam>front</cam>? <cnt>2</cnt>"
        Positive: Same image, same QA
        Hard Neg: "How many <obj>trucks</obj> in <cam>front</cam>? <cnt>2</cnt>" (different object)
        Hard Neg: "How many <obj>cars</obj> in <cam>front</cam>? <cnt>3</cnt>" (different count)
        Easy Neg: Completely different QA pairs
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_weight: float = 2.0,  # Weight hard negatives more
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def _extract_count(self, text: str) -> str:
        """Extract count from text for hard negative detection."""
        match = re.search(r'<cnt>(\d+)</cnt>', text)
        return match.group(1) if match else None

    def _extract_object(self, text: str) -> str:
        """Extract primary object type from text."""
        match = re.search(r'<obj>([^<]+)</obj>', text)
        return match.group(1) if match else None

    def _extract_camera(self, text: str) -> str:
        """Extract camera direction from text."""
        match = re.search(r'<cam>([^<]+)</cam>', text)
        return match.group(1) if match else None

    def _find_hard_negatives(
        self,
        texts: list,
        anchor_idx: int,
    ) -> list:
        """Find hard negative indices for a given anchor.

        Hard negatives share same camera/question type but differ in count/object.
        """
        anchor_text = texts[anchor_idx]
        anchor_count = self._extract_count(anchor_text)
        anchor_obj = self._extract_object(anchor_text)
        anchor_cam = self._extract_camera(anchor_text)

        hard_negatives = []
        for idx, text in enumerate(texts):
            if idx == anchor_idx:
                continue

            # Hard negative criteria:
            # 1. Same camera position but different count
            # 2. Same camera position but different object type
            cam = self._extract_camera(text)
            count = self._extract_count(text)
            obj = self._extract_object(text)

            if cam == anchor_cam:  # Same camera
                if count != anchor_count or obj != anchor_obj:
                    hard_negatives.append(idx)

        return hard_negatives

    def forward(
        self,
        visual_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        texts: list = None,  # Optional for hard negative mining
    ) -> torch.Tensor:
        """Compute InfoNCE loss with optional hard negative weighting.

        Args:
            visual_embeddings: [B, D] normalized visual embeddings
            text_embeddings: [B, D] normalized text embeddings
            texts: Optional list of text strings for hard negative mining

        Returns:
            loss: scalar loss
        """
        batch_size = visual_embeddings.shape[0]
        device = visual_embeddings.device

        # Compute similarity matrix [B, B]
        logits = torch.matmul(visual_embeddings, text_embeddings.T) / self.temperature

        # Standard InfoNCE: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=device)

        # Apply hard negative weighting if texts provided
        if texts is not None and self.hard_negative_weight > 1.0:
            # Create weight matrix (default 1.0 for all negatives)
            weight_matrix = torch.ones_like(logits)

            # Identify and upweight hard negatives
            for i in range(batch_size):
                hard_neg_indices = self._find_hard_negatives(texts, i)
                if hard_neg_indices:
                    weight_matrix[i, hard_neg_indices] = self.hard_negative_weight

            # Apply weights by scaling logits
            # Note: Only scale negative pairs (not diagonal)
            mask = torch.eye(batch_size, device=device).bool()
            logits = torch.where(mask, logits, logits * weight_matrix)

        # Symmetric loss
        loss_i2t = F.cross_entropy(logits, labels)  # image to text
        loss_t2i = F.cross_entropy(logits.T, labels)  # text to image

        loss = (loss_i2t + loss_t2i) / 2

        return loss


class CLIPMultiViewContrastiveAugmented(nn.Module):
    """CLIP model with auxiliary count/object classification heads.

    These auxiliary heads help the model learn better representations
    for numerical and categorical information in MQA.
    """

    def __init__(
        self,
        model_name: str = 'openai/clip-vit-base-patch32',
        embedding_dim: int = 256,
        visual_aggregation: str = 'attention',
        freeze_vision: bool = False,
        freeze_text: bool = False,
        # Auxiliary classification
        use_count_head: bool = True,
        use_object_head: bool = True,
        max_count: int = 20,
        num_object_classes: int = 13,
    ):
        super().__init__()

        from clip_mqa_pretrain.models.clip_contrastive import CLIPMultiViewContrastive

        # Use existing CLIP model as base
        self.clip_model = CLIPMultiViewContrastive(
            model_name=model_name,
            embedding_dim=embedding_dim,
            temperature=0.07,
            visual_aggregation=visual_aggregation,
            freeze_vision=freeze_vision,
            freeze_text=freeze_text,
        )

        self.use_count_head = use_count_head
        self.use_object_head = use_object_head

        # Auxiliary classification heads
        if use_count_head:
            self.count_classifier = nn.Linear(embedding_dim, max_count + 1)

        if use_object_head:
            self.object_classifier = nn.Linear(embedding_dim, num_object_classes)

    def forward(self, pixel_values, input_ids, attention_mask):
        """Forward pass with auxiliary outputs."""
        visual_emb, text_emb = self.clip_model(pixel_values, input_ids, attention_mask)

        aux_outputs = {}

        if self.use_count_head:
            # Predict count from text embedding
            count_logits = self.count_classifier(text_emb)
            aux_outputs['count_logits'] = count_logits

        if self.use_object_head:
            # Predict object class from text embedding
            object_logits = self.object_classifier(text_emb)
            aux_outputs['object_logits'] = object_logits

        return visual_emb, text_emb, aux_outputs

    @property
    def original_vocab_size(self):
        """Expose vocab size for token embedding resizing."""
        return self.clip_model.original_vocab_size

    @property
    def clip(self):
        """Expose CLIP model for weight loading."""
        return self.clip_model.clip


if __name__ == '__main__':
    # Test enhanced loss
    texts = [
        "How many <obj>car</obj> in <cam>front</cam>? <cnt>2</cnt>",
        "How many <obj>car</obj> in <cam>front</cam>? <cnt>3</cnt>",  # Hard negative (different count)
        "How many <obj>truck</obj> in <cam>front</cam>? <cnt>2</cnt>",  # Hard negative (different object)
        "How many <obj>car</obj> in <cam>back</cam>? <cnt>2</cnt>",  # Easy negative (different camera)
    ]

    # Dummy embeddings
    visual_emb = torch.randn(4, 256)
    text_emb = torch.randn(4, 256)
    visual_emb = F.normalize(visual_emb, p=2, dim=1)
    text_emb = F.normalize(text_emb, p=2, dim=1)

    # Test standard loss
    criterion_standard = CLIPInfoNCEWithHardNegatives(
        temperature=0.07,
        hard_negative_weight=1.0,  # No hard negative weighting
    )
    loss_standard = criterion_standard(visual_emb, text_emb, texts)
    print(f"Standard InfoNCE loss: {loss_standard.item():.4f}")

    # Test with hard negative weighting
    criterion_hard = CLIPInfoNCEWithHardNegatives(
        temperature=0.07,
        hard_negative_weight=2.0,  # 2x weight for hard negatives
    )
    loss_hard = criterion_hard(visual_emb, text_emb, texts)
    print(f"Hard negative InfoNCE loss: {loss_hard.item():.4f}")

    # Test auxiliary heads
    model = CLIPMultiViewContrastiveAugmented(
        use_count_head=True,
        use_object_head=True,
        max_count=20,
        num_object_classes=13,
    )

    pixel_values = torch.randn(2, 6, 3, 224, 224)
    input_ids = torch.randint(0, 49408, (2, 77))
    attention_mask = torch.ones_like(input_ids)

    visual_emb, text_emb, aux = model(pixel_values, input_ids, attention_mask)
    print(f"\nAuxiliary outputs:")
    print(f"  Count logits shape: {aux['count_logits'].shape}")
    print(f"  Object logits shape: {aux['object_logits'].shape}")
