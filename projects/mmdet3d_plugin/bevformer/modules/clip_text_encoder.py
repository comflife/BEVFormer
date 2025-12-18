"""
CLIP-based Text Encoder for Language-guided BEVFormer.

This module provides a drop-in replacement for BERT TextEncoder using CLIP.
Maintains the same interface for compatibility with existing code.
"""

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from collections import OrderedDict


class CLIPTextEncoder(BaseModule):
    """CLIP-based Text Encoder for question embedding.

    Uses CLIP's text encoder instead of BERT, with projection to match
    BEVFormer's embedding dimension.

    Args:
        pretrained_model: CLIP model name (default: 'openai/clip-vit-base-patch32')
        pretrained_weights: Path to pretrained CLIP weights from contrastive learning
        freeze: Whether to freeze the encoder weights
        output_dim: Output embedding dimension
        max_length: Maximum sequence length (CLIP uses 77)
        enable_cache: Whether to enable question caching
        cache_size: Maximum cache size
    """

    def __init__(
        self,
        pretrained_model: str = 'openai/clip-vit-base-patch32',
        pretrained_weights: str = None,
        freeze: bool = True,
        output_dim: int = 256,
        max_length: int = 77,
        enable_cache: bool = True,
        cache_size: int = 10000,
    ):
        super().__init__()

        self.pretrained_model = pretrained_model
        self.freeze = freeze
        self.output_dim = output_dim
        self.max_length = max_length
        self.enable_cache = enable_cache
        self.cache_size = cache_size

        # Cache for repeated questions
        self._pooled_cache = OrderedDict()

        # Load CLIP model and processor
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(pretrained_model)
        self.clip = CLIPModel.from_pretrained(pretrained_model)

        # Store original vocab size for potential token expansion
        self.original_vocab_size = self.clip.text_model.config.vocab_size

        # Add special tokens for MQA markup (same as pretraining)
        # Note: Camera positions (front, back, etc.) are regular text inside <cam></cam> tags
        special_tokens = [
            '<cam>', '</cam>', '<obj>', '</obj>', '<cnt>', '</cnt>', '<target>', '</target>'
        ]
        num_added = self.processor.tokenizer.add_tokens(special_tokens)
        if num_added > 0:
            print(f"Added {num_added} special tokens to CLIP tokenizer")
            # Resize token embeddings to match
            # Note: CLIPTextTransformer doesn't have resize_token_embeddings(), need manual resize
            new_vocab_size = len(self.processor.tokenizer)
            old_embeddings = self.clip.text_model.embeddings.token_embedding

            # Create new embedding with same dtype and device
            new_embeddings = nn.Embedding(new_vocab_size, old_embeddings.embedding_dim)
            new_embeddings = new_embeddings.to(device=old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

            # Copy old weights
            new_embeddings.weight.data[:old_embeddings.num_embeddings] = old_embeddings.weight.data

            # Initialize new token embeddings
            if new_vocab_size > old_embeddings.num_embeddings:
                new_embeddings.weight.data[old_embeddings.num_embeddings:].normal_(mean=0.0, std=0.02)

            # Replace embedding layer
            self.clip.text_model.embeddings.token_embedding = new_embeddings

            # Update config to match (important for save/load)
            self.clip.text_model.config.vocab_size = new_vocab_size
            self.clip.config.text_config.vocab_size = new_vocab_size

            print(f"Resized token embeddings: {self.original_vocab_size} → {new_vocab_size}")

        # Load custom pretrained weights if provided
        if pretrained_weights is not None:
            import os

            print(f"Loading custom CLIP weights from: {pretrained_weights}")

            if os.path.isfile(pretrained_weights) and pretrained_weights.endswith('.pth'):
                checkpoint = torch.load(pretrained_weights, map_location='cpu')

                if 'model_state_dict' in checkpoint:
                    # Extract CLIP weights from checkpoint
                    clip_state_dict = {}

                    # Look for CLIP text model weights
                    for key, value in checkpoint['model_state_dict'].items():
                        if key.startswith('clip.text_model.'):
                            # Remove 'clip.' prefix
                            new_key = key[5:]
                            clip_state_dict[new_key] = value
                        elif key.startswith('text_projection.'):
                            # Save text projection separately
                            pass

                    if clip_state_dict:
                        self.clip.text_model.load_state_dict(clip_state_dict, strict=False)
                        print("✓ Custom CLIP text model weights loaded successfully")
                    else:
                        print("Warning: No CLIP weights found in checkpoint, using base CLIP")
                else:
                    print("Warning: Invalid checkpoint format, using base CLIP weights")
            else:
                print(f"Warning: {pretrained_weights} not found or invalid, using base CLIP weights")

        # CLIP text hidden size is typically 512
        clip_text_dim = self.clip.config.text_config.hidden_size

        # Projection layer to match BEVFormer embedding dimension
        # If we have pretrained weights with projection, try to load it
        self.projection = nn.Sequential(
            nn.Linear(clip_text_dim, clip_text_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(clip_text_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Try to load projection weights from checkpoint
        if pretrained_weights is not None and os.path.isfile(pretrained_weights):
            checkpoint = torch.load(pretrained_weights, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                proj_state_dict = {}
                for key, value in checkpoint['model_state_dict'].items():
                    if key.startswith('text_projection.'):
                        # Remove 'text_projection.' prefix and map to our projection layers
                        new_key = key.replace('text_projection.', '')
                        proj_state_dict[new_key] = value

                if proj_state_dict:
                    self.projection.load_state_dict(proj_state_dict, strict=False)
                    print("✓ Custom CLIP projection weights loaded successfully")

        # Freeze CLIP if specified
        if freeze:
            for param in self.clip.text_model.parameters():
                param.requires_grad = False

    def forward(self, questions: list) -> torch.Tensor:
        """Encode questions to embeddings.

        Args:
            questions: List of question strings

        Returns:
            Text embeddings of shape [B, output_dim]
        """
        device = next(self.projection.parameters()).device

        # Check if we can use cache
        use_cache = bool(self.enable_cache and self.freeze and self.cache_size and self.cache_size > 0)

        if not use_cache:
            # Tokenize with CLIP processor
            text_inputs = self.processor(
                text=questions,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            # Get CLIP text features
            with torch.set_grad_enabled(not self.freeze):
                text_outputs = self.clip.text_model(**text_inputs)
                pooled = text_outputs.pooler_output  # [B, text_dim]

            # Project to output dimension
            proj_dtype = self.projection[0].weight.dtype
            pooled = pooled.to(dtype=proj_dtype)
            return self.projection(pooled)

        # Use cache
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
            # Tokenize missing questions
            text_inputs = self.processor(
                text=missing_questions,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            # Get CLIP text features (frozen)
            with torch.set_grad_enabled(not self.freeze):
                text_outputs = self.clip.text_model(**text_inputs)
                pooled = text_outputs.pooler_output

            # Move to CPU for caching
            pooled_cpu = pooled.detach().to('cpu')

            for j, idx in enumerate(missing_indices):
                pooled_cpu_list[idx] = pooled_cpu[j]
                q = questions[idx]
                self._pooled_cache[q] = pooled_cpu[j]
                self._pooled_cache.move_to_end(q)
                # Evict LRU
                while len(self._pooled_cache) > self.cache_size:
                    self._pooled_cache.popitem(last=False)

        # Stack pooled outputs and project
        pooled = torch.stack(pooled_cpu_list, dim=0)
        proj_dtype = self.projection[0].weight.dtype
        pooled = pooled.to(device=device, dtype=proj_dtype)
        return self.projection(pooled)


if __name__ == '__main__':
    # Test CLIP text encoder
    encoder = CLIPTextEncoder(
        pretrained_model='openai/clip-vit-base-patch32',
        freeze=False,
        output_dim=256,
    )

    questions = [
        "How many cars are in front?",
        "Where is the nearest pedestrian?",
    ]

    embeddings = encoder(questions)
    print(f"Output shape: {embeddings.shape}")
    print(f"Output dtype: {embeddings.dtype}")
