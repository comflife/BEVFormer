"""
Debug script to verify that data loading is correct.

Checks:
1. Same scene → same sample_token for all QA pairs
2. Different scenes → different sample_tokens
3. Images are actually being paired with texts correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from clip_mqa_pretrain.data.nuscenes_mqa_dataset_optimized import (
    NuScenesMQASceneDataset,
    clip_scene_collate_fn
)
from clip_mqa_pretrain.models.clip_contrastive import (
    CLIPMultiViewContrastive,
    CLIPInfoNCELoss
)

# Load dataset
print("Loading dataset...")
dataset = NuScenesMQASceneDataset(
    data_root='data/nuscenes/',
    mqa_csv='df_train_mqa.csv',
    nuscenes_info_pkl='data/nuscenes/nuscenes_infos_temporal_train_bg.pkl',
    qa_per_scene=8,
    max_samples=10,
)

print(f"Dataset size: {len(dataset)} scenes\n")

# Test single sample
print("="*60)
print("Test 1: Single Sample")
print("="*60)
sample = dataset[0]
print(f"Sample token: {sample['sample_token']}")
print(f"Pixel values shape: {sample['pixel_values'].shape}")  # Should be [8, 6, 3, H, W]
print(f"Number of QA pairs: {len(sample['texts'])}")
print(f"\nAll 8 QA pairs from same scene:")
for i, text in enumerate(sample['texts']):
    print(f"  [{i}] {text[:80]}...")

# Test dataloader
print("\n" + "="*60)
print("Test 2: DataLoader Batch (2 scenes × 8 QA = 16 samples)")
print("="*60)

loader = DataLoader(
    dataset,
    batch_size=2,  # 2 scenes
    shuffle=False,
    num_workers=0,
    collate_fn=clip_scene_collate_fn,
)

batch = next(iter(loader))
print(f"Pixel values shape: {batch['pixel_values'].shape}")  # Should be [16, 6, 3, H, W]
print(f"Input IDs shape: {batch['input_ids'].shape}")  # Should be [16, 77]
print(f"Number of texts: {len(batch['texts'])}")  # Should be 16
print(f"Number of sample_tokens: {len(batch['sample_tokens'])}")  # Should be 16

print(f"\nSample tokens (should show 2 groups of 8):")
for i, token in enumerate(batch['sample_tokens']):
    print(f"  [{i}] {token}")

print(f"\nUnique tokens in batch: {len(set(batch['sample_tokens']))}")  # Should be 2
print(f"Expected multi-positive pairs per image: 8")

# Test 3: Verify positive mask
print("\n" + "="*60)
print("Test 3: Verify Positive Mask Logic")
print("="*60)

criterion = CLIPInfoNCELoss(temperature=0.07)
pos_mask = criterion._get_positive_mask(batch['sample_tokens'], device='cpu')

print(f"Positive mask shape: {pos_mask.shape}")  # Should be [16, 16]
print(f"\nPositive mask (1 = same image, 0 = different image):")
print(pos_mask)

print(f"\nExpected pattern:")
print("  First 8x8 block: all 1s (scene 1)")
print("  Second 8x8 block (diagonal): all 1s (scene 2)")
print("  Off-diagonal blocks: all 0s (different scenes)")

# Count positives per row
positives_per_row = pos_mask.sum(dim=1)
print(f"\nPositives per row (should be 8 for all rows):")
print(positives_per_row)

# Test 4: Verify image-text alignment
print("\n" + "="*60)
print("Test 4: Image-Text Alignment Check")
print("="*60)

print(f"Are the first 8 samples all from the same scene? {len(set(batch['sample_tokens'][:8])) == 1}")
print(f"Scene 1 token: {batch['sample_tokens'][0]}")
print(f"\nFirst 8 texts (should all be from same scene):")
for i in range(8):
    print(f"  [{i}] {batch['texts'][i][:80]}...")

print(f"\nAre they using the SAME images? (they should be)")
pixel_diff = (batch['pixel_values'][0] - batch['pixel_values'][7]).abs().sum()
print(f"Pixel difference between sample 0 and 7: {pixel_diff.item():.2f}")
print(f"(Should be 0.0 - same images)")

# Test 5: Forward pass
print("\n" + "="*60)
print("Test 5: Forward Pass Test")
print("="*60)

model = CLIPMultiViewContrastive(
    model_name='openai/clip-vit-base-patch32',
    embedding_dim=256,
    visual_aggregation='attention',
)

# Resize token embeddings
new_vocab_size = len(dataset.processor.tokenizer)
if new_vocab_size > model.original_vocab_size:
    model.resize_token_embeddings(new_vocab_size)

pixel_values = batch['pixel_values']  # [16, 6, 3, H, W]
input_ids = batch['input_ids']  # [16, 77]
attention_mask = batch['attention_mask']  # [16, 77]

visual_emb, text_emb = model(pixel_values, input_ids, attention_mask)

print(f"Visual embeddings shape: {visual_emb.shape}")  # Should be [16, 256]
print(f"Text embeddings shape: {text_emb.shape}")  # Should be [16, 256]
print(f"Visual normalized: {torch.norm(visual_emb, p=2, dim=1)}")  # Should be all 1.0
print(f"Text normalized: {torch.norm(text_emb, p=2, dim=1)}")  # Should be all 1.0

# Compute similarity matrix
similarity = torch.matmul(visual_emb, text_emb.T)  # [16, 16]
print(f"\nSimilarity matrix shape: {similarity.shape}")
print(f"Similarity matrix (first 8x8 block should have high diagonals):")
print(similarity[:8, :8])

# Compute loss
loss = criterion(visual_emb, text_emb, batch['texts'], batch['sample_tokens'])
print(f"\nLoss: {loss.item():.4f}")

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
print("\nIf everything looks correct, the issue might be:")
print("1. Learning rate too high (causing divergence)")
print("2. Temperature too low (causing numerical issues)")
print("3. Hard negative weight too high (causing instability)")
print("4. Batch size too small (noisy gradients)")
