"""
Debug tokenizer and model vocab size mismatch
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import CLIPProcessor, CLIPModel

# Load CLIP
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

print(f"Original tokenizer vocab size: {processor.tokenizer.vocab_size}")
print(f"Original model embedding size: {model.text_model.embeddings.token_embedding.num_embeddings}")

# Add special tokens
special_tokens = [
    '<cam>', '</cam>', '<obj>', '</obj>', '<cnt>', '</cnt>', '<target>', '</target>'
]
num_added = processor.tokenizer.add_tokens(special_tokens)
print(f"\nAdded {num_added} special tokens")
print(f"New vocab size: {processor.tokenizer.vocab_size}")
print(f"New vocab size (len): {len(processor.tokenizer)}")

# Resize model - need to do it manually for CLIPTextTransformer
print(f"\nBefore resize - Model embedding size: {model.text_model.embeddings.token_embedding.num_embeddings}")

# Manual resize
import torch.nn as nn
old_embeddings = model.text_model.embeddings.token_embedding
new_num_tokens = len(processor.tokenizer)
new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
new_embeddings.weight.data[:old_embeddings.num_embeddings] = old_embeddings.weight.data
model.text_model.embeddings.token_embedding = new_embeddings

print(f"After resize - Model embedding size: {model.text_model.embeddings.token_embedding.num_embeddings}")

# Test tokenization
test_text = "Question: How many <obj>car</obj> in <cam>front</cam>? Answer: <cnt>2</cnt>"
tokens = processor.tokenizer(test_text, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
print(f"\nTokenized IDs: {tokens['input_ids'][0]}")
print(f"Max token ID: {tokens['input_ids'].max().item()}")
print(f"Model can handle up to: {model.text_model.embeddings.token_embedding.num_embeddings - 1}")

if tokens['input_ids'].max().item() >= model.text_model.embeddings.token_embedding.num_embeddings:
    print("\n❌ ERROR: Token ID exceeds model embedding size!")
else:
    print("\n✅ OK: All token IDs within range")
