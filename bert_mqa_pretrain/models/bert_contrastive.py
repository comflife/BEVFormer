"""
BERT Contrastive Learning Model

Implements Multiple Negatives Ranking Loss (InfoNCE) for BERT pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class BERTContrastive(nn.Module):
    """BERT with contrastive learning head.
    
    Uses [CLS] token embedding for contrastive loss.
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        embedding_dim: int = 256,
        temperature: float = 0.07,
        from_scratch: bool = False,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # BERT encoder
        if from_scratch:
            # Train from scratch (like SDTagNet paper)
            config = BertConfig.from_pretrained(model_name)
            config.output_attentions = False
            config.output_hidden_states = False
            self.bert = BertModel(config)
        else:
            # Start from pretrained (faster convergence)
            self.bert = BertModel.from_pretrained(model_name)
            # Disable outputs that cause inplace operations
            self.bert.config.output_attentions = False
            self.bert.config.output_hidden_states = False
        
        bert_hidden_size = self.bert.config.hidden_size
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden_size, bert_hidden_size),
            nn.ReLU(),
            nn.Linear(bert_hidden_size, embedding_dim),
        )
        
        # For final inference, we use pooler output + projection
        self.output_projection = nn.Sequential(
            nn.Linear(bert_hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )
        
        # Disable gradient checkpointing to avoid inplace operation errors
        if hasattr(self.bert, 'gradient_checkpointing'):
            self.bert.gradient_checkpointing = False
    
    def forward(self, input_ids, attention_mask):
        """Forward pass.
        
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask
            
        Returns:
            embeddings: [B, embedding_dim] normalized embeddings
        """
        # BERT encoding - disable return_dict to avoid inplace operations
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,  # Keep as True, but access attributes carefully
        )
        
        # Use [CLS] token (pooler output)
        # Access with .pooler_output to avoid any indexing issues
        cls_output = outputs.pooler_output  # [B, hidden_size]
        
        # Project to embedding space
        embeddings = self.projection(cls_output)  # [B, embedding_dim]
        
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_inference_embedding(self, input_ids, attention_mask):
        """Get embedding for inference (after training)."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        cls_output = outputs.pooler_output
        embeddings = self.output_projection(cls_output)
        
        return embeddings


class InfoNCELoss(nn.Module):
    """InfoNCE Loss (Multiple Negatives Ranking Loss).
    
    Given a batch of (anchor, positive) pairs:
    - Positive: the paired sample
    - Negatives: all other samples in the batch
    
    This is the same as SimCLR loss.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.
        
        Args:
            anchor_embeddings: [B, D] anchor embeddings (normalized)
            positive_embeddings: [B, D] positive embeddings (normalized)
            
        Returns:
            loss: scalar loss
        """
        batch_size = anchor_embeddings.shape[0]
        
        # Compute similarity matrix [B, B]
        # sim[i, j] = cosine similarity between anchor[i] and positive[j]
        similarity_matrix = torch.matmul(
            anchor_embeddings, positive_embeddings.T
        ) / self.temperature  # [B, B]
        
        # Positive pairs are on the diagonal
        # Labels: each anchor should match with its corresponding positive
        labels = torch.arange(batch_size, device=anchor_embeddings.device)
        
        # Cross entropy loss
        # For each anchor, we want highest similarity with its positive
        # and lower similarity with all negatives (other positives in batch)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


if __name__ == '__main__':
    # Test model
    model = BERTContrastive(
        model_name='bert-base-uncased',
        embedding_dim=256,
        temperature=0.07,
        from_scratch=False,
    )
    
    # Dummy input
    batch_size = 4
    seq_length = 32
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    # Forward
    embeddings = model(input_ids, attention_mask)
    print(f"Output shape: {embeddings.shape}")
    print(f"Normalized: {torch.norm(embeddings, p=2, dim=1)}")
    
    # Test loss
    criterion = InfoNCELoss(temperature=0.07)
    
    anchor_emb = torch.randn(batch_size, 256)
    positive_emb = torch.randn(batch_size, 256)
    anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
    positive_emb = F.normalize(positive_emb, p=2, dim=1)
    
    loss = criterion(anchor_emb, positive_emb)
    print(f"Loss: {loss.item():.4f}")
