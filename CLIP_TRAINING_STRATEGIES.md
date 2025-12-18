# CLIP Training Strategies to Fix Overfitting

## Problem
- **Train loss**: Decreasing ✓
- **Val loss**: Increasing ✗
- **Diagnosis**: Clear overfitting

## Root Causes
1. Fine-tuning entire CLIP (vision + text) on small dataset
2. Learning rate might be too high for fine-tuning
3. Hard negative weight causing instability on validation set

---

## Solution 1: Conservative Config (RECOMMENDED)

**File**: `clip_mqa_pretrain/configs/pretrain_config_conservative.py`

### Changes:
1. **Freeze vision encoder** (`freeze_vision=True`)
   - Only fine-tune text encoder + projection layers
   - Prevents catastrophic forgetting of pretrained vision features
   - Reduces trainable parameters significantly

2. **Lower learning rate** (`lr=2e-6`, was `1e-5`)
   - More conservative updates
   - Prevents overshooting good solutions

3. **Reduce hard negative weight** (`hard_negative_weight=1.2`, was `1.5`)
   - Less aggressive training
   - More stable on validation set

4. **Early stopping** (`patience=3`)
   - Automatically stops when val loss stops improving
   - Prevents wasting compute on overfitting

### Training command:
```bash
python clip_mqa_pretrain/scripts/train_single_gpu.py \
    --config clip_mqa_pretrain/configs/pretrain_config_conservative.py \
    --gpu 0
```

---

## Solution 2: Aggressive Regularization

If conservative config still overfits, try:

### Additional changes:
```python
# In config
train = dict(
    batch_size=6,  # Larger batch = more stable gradients
    lr=1e-6,  # Even lower LR
    weight_decay=0.05,  # Stronger L2 regularization (was 0.01)
    hard_negative_weight=1.0,  # Disable hard negative weighting
    gradient_accumulation_steps=4,  # Larger effective batch
)
```

---

## Solution 3: Data Augmentation

Add data augmentation to training:

```python
# In dataset
from torchvision import transforms

self.augmentation = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
])
```

---

## Solution 4: Freeze Everything Except Projection

Most conservative approach:

```python
model = dict(
    freeze_vision=True,
    freeze_text=True,  # ← Also freeze text encoder
)

# Only train:
# - Visual projection
# - Text projection
# - Camera positional embeddings
# - View attention
```

---

## Monitoring

Watch these metrics in WandB:

1. **Train vs Val loss gap**
   - Should be small (<0.5)
   - If growing → overfitting

2. **Best val loss epoch**
   - Should be near the end of training
   - If early → stopped too late

3. **Learning rate schedule**
   - Should decay smoothly
   - No sudden jumps

---

## Expected Results

With conservative config:

- **Train loss**: Still decreases (but slower)
- **Val loss**: Should decrease (or at least not increase)
- **Gap**: Train-val gap should be small
- **Convergence**: ~5-10 epochs

---

## Debugging Commands

### Quick test (10 scenes):
```bash
python clip_mqa_pretrain/scripts/train_single_gpu.py \
    --config clip_mqa_pretrain/configs/pretrain_config_conservative.py \
    --gpu 0 \
    --debug
```

### Check model parameters:
```python
# Which parameters are trainable?
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"✓ {name}: {param.numel():,} params")
    else:
        print(f"✗ {name}: frozen")
```

### Monitor loss during training:
```bash
watch -n 5 'tail -n 50 wandb/latest-run/logs/debug-internal.log | grep "loss"'
```
