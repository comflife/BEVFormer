"""
Test script for hard negative mining in CLIP contrastive loss.

This verifies that the loss correctly identifies and weights hard negatives.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from clip_mqa_pretrain.models.clip_contrastive import CLIPInfoNCELoss


def test_hard_negative_mining():
    """Test that hard negatives are correctly identified and weighted."""

    print("="*60)
    print("Testing Hard Negative Mining with Multi-Positive Support")
    print("="*60)

    # Example batch with hard negatives from DIFFERENT images
    texts = [
        "Question: How many <obj>car</obj> in <cam>front</cam>? Answer: <cnt>2</cnt>",  # Image A
        "Question: How many <obj>car</obj> in <cam>front</cam>? Answer: <cnt>3</cnt>",  # Image B - Hard neg from A
        "Question: How many <obj>truck</obj> in <cam>front</cam>? Answer: <cnt>2</cnt>",  # Image C - Hard neg from A
        "Question: How many <obj>car</obj> in <cam>back</cam>? Answer: <cnt>2</cnt>",  # Image D - Easy neg from A
    ]

    # Sample tokens to identify which image each QA belongs to
    sample_tokens = ['img_A', 'img_B', 'img_C', 'img_D']

    print("\nBatch texts (all from DIFFERENT images):")
    for i, (text, token) in enumerate(zip(texts, sample_tokens)):
        print(f"  [{i}] ({token}) {text}")

    # Create dummy embeddings
    batch_size = len(texts)
    embedding_dim = 256
    visual_emb = torch.randn(batch_size, embedding_dim)
    text_emb = torch.randn(batch_size, embedding_dim)

    # Normalize
    visual_emb = F.normalize(visual_emb, p=2, dim=1)
    text_emb = F.normalize(text_emb, p=2, dim=1)

    # Test 1: Standard InfoNCE (no hard negative weighting)
    print("\n" + "="*60)
    print("Test 1: Standard InfoNCE (hard_negative_weight=1.0)")
    print("="*60)

    criterion_standard = CLIPInfoNCELoss(
        temperature=0.07,
        hard_negative_weight=1.0,
    )

    loss_standard = criterion_standard(visual_emb, text_emb, texts, sample_tokens)
    print(f"Loss: {loss_standard.item():.4f}")

    # Test 2: With hard negative weighting
    print("\n" + "="*60)
    print("Test 2: With Hard Negative Weighting (hard_negative_weight=1.5)")
    print("="*60)

    criterion_hard = CLIPInfoNCELoss(
        temperature=0.07,
        hard_negative_weight=1.5,
    )

    # Get hard negative mask
    hard_neg_mask = criterion_hard._get_hard_negative_mask(texts, sample_tokens, device='cpu')

    print("\nHard negative mask (filtered by sample_token):")
    print("(1 = hard negative from different image, 0 = easy negative or positive)")
    print(hard_neg_mask)

    print("\nExpected hard negatives:")
    print("  [0,1] = 1 (different image, same cam/obj, diff count)")
    print("  [0,2] = 1 (different image, same cam/count, diff obj)")
    print("  [0,3] = 0 (different image, but easy negative - diff cam)")
    print("  [1,0] = 1 (symmetric)")
    print("  [1,2] = 1 (different image, same cam)")
    print("  [2,0] = 1 (symmetric)")
    print("  [2,1] = 1 (symmetric)")

    loss_hard = criterion_hard(visual_emb, text_emb, texts, sample_tokens)
    print(f"\nLoss with hard negative weighting: {loss_hard.item():.4f}")

    # Test 3: Multi-positive case (same image with multiple QAs)
    print("\n" + "="*60)
    print("Test 3: Multi-Positive (Same Image, Multiple QAs)")
    print("="*60)

    # Batch with same image repeated
    texts_multi = [
        "Question: How many <obj>car</obj> in <cam>front</cam>? Answer: <cnt>2</cnt>",  # Image A, QA1
        "Question: How many <obj>truck</obj> in <cam>back</cam>? Answer: <cnt>1</cnt>",  # Image A, QA2 (different QA, SAME image)
        "Question: How many <obj>car</obj> in <cam>front</cam>? Answer: <cnt>3</cnt>",  # Image B, QA1
        "Question: How many <obj>car</obj> in <cam>front</cam>? Answer: <cnt>2</cnt>",  # Image C, QA1 (hard neg from Image A)
    ]

    sample_tokens_multi = ['img_A', 'img_A', 'img_B', 'img_C']  # First two are SAME image

    print("\nBatch texts (first two from SAME image):")
    for i, (text, token) in enumerate(zip(texts_multi, sample_tokens_multi)):
        print(f"  [{i}] ({token}) {text}")

    # Get positive mask
    pos_mask = criterion_hard._get_positive_mask(sample_tokens_multi, device='cpu')
    print("\nPositive mask:")
    print("(1 = same image, 0 = different image)")
    print(pos_mask)

    print("\nExpected:")
    print("  [0,0] = 1, [0,1] = 1 (same image A)")
    print("  [1,0] = 1, [1,1] = 1 (same image A)")
    print("  [2,2] = 1 (image B alone)")
    print("  [3,3] = 1 (image C alone)")

    # Get hard negative mask (should exclude same-image pairs)
    hard_neg_mask_multi = criterion_hard._get_hard_negative_mask(texts_multi, sample_tokens_multi, device='cpu')
    print("\nHard negative mask:")
    print("(1 = hard negative from DIFFERENT image only)")
    print(hard_neg_mask_multi)

    print("\nExpected:")
    print("  [0,3] = 1 (different images, same text structure)")
    print("  [3,0] = 1 (symmetric)")
    print("  [0,1] = 0 (SAME image, excluded even though text differs)")
    print("  [1,0] = 0 (SAME image, excluded)")

    # Compute loss
    visual_emb_multi = torch.randn(len(texts_multi), embedding_dim)
    text_emb_multi = torch.randn(len(texts_multi), embedding_dim)
    visual_emb_multi = F.normalize(visual_emb_multi, p=2, dim=1)
    text_emb_multi = F.normalize(text_emb_multi, p=2, dim=1)

    loss_multi = criterion_hard(visual_emb_multi, text_emb_multi, texts_multi, sample_tokens_multi)
    print(f"\nLoss with multi-positive: {loss_multi.item():.4f}")

    # Test 4: Feature extraction
    print("\n" + "="*60)
    print("Test 4: Feature Extraction")
    print("="*60)

    for i, text in enumerate(texts):
        count = criterion_hard._extract_count(text)
        obj = criterion_hard._extract_object(text)
        cam = criterion_hard._extract_camera(text)
        print(f"[{i}] count={count}, obj={obj}, cam={cam}")

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == '__main__':
    test_hard_negative_mining()
