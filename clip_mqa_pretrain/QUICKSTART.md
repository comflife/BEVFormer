# Quick Start Guide: CLIP Multi-View Pretraining

## Step 1: Train CLIP Model

Train CLIP on NuScenes multi-view images + MQA text pairs:

```bash
# Single GPU (recommended for testing)
python clip_mqa_pretrain/scripts/train_single_gpu.py \
    --config clip_mqa_pretrain/configs/pretrain_config.py \
    --gpu 0

# Multi-GPU (faster)
python clip_mqa_pretrain/scripts/train_ddp.py \
    --config clip_mqa_pretrain/configs/pretrain_config.py \
    --gpus 2

# Debug mode (small dataset for testing)
python clip_mqa_pretrain/scripts/train_single_gpu.py \
    --config clip_mqa_pretrain/configs/pretrain_config.py \
    --gpu 0 \
    --debug
```

**Expected Training Time:**
- Single GPU (V100): ~2-3 hours per epoch
- 2x GPU: ~1-1.5 hours per epoch
- Much faster than BERT pretraining!

**Checkpoints will be saved to:**
- `clip_mqa_pretrain/checkpoints/clip_mqa_best.pth` (best validation loss)
- `clip_mqa_pretrain/checkpoints/clip_mqa_epoch{N}.pth` (every N epochs)

## Step 2: Train BEVFormer with CLIP

Once CLIP pretraining is complete, train BEVFormer:

```bash
# Make sure the checkpoint path in config is correct
# Edit: projects/configs/bevformer/bevformer_mqa_clip_tiny.py
# Line 207: pretrained_weights='clip_mqa_pretrain/checkpoints/clip_mqa_best.pth'

# Train BEVFormer
bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_mqa_clip_tiny.py \
    2 \
    --work-dir work_dirs/bevformer_mqa_clip_tiny
```

## Step 3: Compare with BERT

You can compare training speed and performance:

```bash
# BERT version (slower)
bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_mqa_tiny.py \
    2 \
    --work-dir work_dirs/bevformer_mqa_bert_tiny

# CLIP version (faster)
bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_mqa_clip_tiny.py \
    2 \
    --work-dir work_dirs/bevformer_mqa_clip_tiny
```

## Configuration

### CLIP Pretraining Config

Edit `clip_mqa_pretrain/configs/pretrain_config.py`:

```python
# Model size (smaller = faster)
model = dict(
    model_name='openai/clip-vit-base-patch32',  # Small CLIP
    embedding_dim=256,
    temperature=0.07,
    visual_aggregation='attention',  # How to combine 6 camera views
)

# Training hyperparameters
train = dict(
    batch_size=16,  # Adjust based on GPU memory
    num_epochs=10,
    lr=1e-5,
    gradient_accumulation_steps=2,  # Effective batch = 16*2 = 32
)
```

### BEVFormer Config

The CLIP version config is at:
`projects/configs/bevformer/bevformer_mqa_clip_tiny.py`

Key differences from BERT version:

```python
# BERT version (old)
text_encoder_cfg=dict(
    pretrained_model='bert-base-uncased',
    pretrained_weights='bert_mqa_pretrain/checkpoints/bert_mqa_epoch3.pth',
    freeze=True,
    output_dim=256,
    pooling='cls',
    max_length=128,
)

# CLIP version (new)
text_encoder_cfg=dict(
    type='CLIPTextEncoder',  # Use CLIP instead
    pretrained_model='openai/clip-vit-base-patch32',
    pretrained_weights='clip_mqa_pretrain/checkpoints/clip_mqa_best.pth',
    freeze=True,
    output_dim=256,
    max_length=77,  # CLIP uses 77 tokens
)
```

## Monitoring Training

WandB logs are automatically uploaded to your project:

- CLIP pretraining: `difffp/clip_mqa_pretrain`
- BEVFormer training: `difffp/bevformer_mqa_clip_tiny_training`

## Troubleshooting

### Out of Memory

Reduce batch size in config:

```python
# CLIP pretraining
train = dict(
    batch_size=8,  # Reduce from 16
    gradient_accumulation_steps=4,  # Increase to maintain effective batch size
)

# BEVFormer training
data = dict(
    samples_per_gpu=1,  # Reduce from 2
)
```

### CLIP Weights Not Found

Make sure you completed Step 1 first:

```bash
# Check if checkpoint exists
ls -lh clip_mqa_pretrain/checkpoints/clip_mqa_best.pth

# If not, train CLIP first
python clip_mqa_pretrain/scripts/train_single_gpu.py \
    --config clip_mqa_pretrain/configs/pretrain_config.py \
    --gpu 0
```

### Slow Training

Use multiple GPUs:

```bash
# CLIP pretraining with 2 GPUs
python clip_mqa_pretrain/scripts/train_ddp.py \
    --config clip_mqa_pretrain/configs/pretrain_config.py \
    --gpus 2

# BEVFormer training with 4 GPUs
bash tools/dist_train.sh \
    projects/configs/bevformer/bevformer_mqa_clip_tiny.py \
    4 \
    --work-dir work_dirs/bevformer_mqa_clip_tiny
```

## Expected Results

CLIP should converge faster than BERT:

- **BERT**: ~10-15 hours to converge, large model
- **CLIP**: ~3-5 hours to converge, smaller model
- **Performance**: Should be comparable or better due to vision-language pre-alignment

## Next Steps

After training completes:

1. Evaluate on validation set
2. Compare metrics with BERT baseline
3. Experiment with different CLIP model sizes:
   - `openai/clip-vit-base-patch32` (current, fastest)
   - `openai/clip-vit-base-patch16` (slower, more accurate)
   - `openai/clip-vit-large-patch14` (largest, best performance)
