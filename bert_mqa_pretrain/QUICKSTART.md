# Quick Start Guide

## 1. Setup Environment

```bash
cd /home/byounggun/BEVFormer/bert_mqa_pretrain

# Install dependencies
pip install transformers torch accelerate wandb scikit-learn
```

## 2. Prepare Data

Make sure you have the MQA CSV files:
```
/home/byounggun/BEVFormer/data/nuscenes/df_train_mqa.csv
/home/byounggun/BEVFormer/data/nuscenes/df_val_mqa.csv
```

## 3. Train BERT

### Option A: Use the shell script (recommended)
```bash
cd /home/byounggun/BEVFormer/bert_mqa_pretrain

# Edit NUM_GPUS in scripts/train.sh if needed
bash scripts/train.sh
```

### Option B: Manual command
```bash
# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 scripts/train_ddp.py --config configs/pretrain_config.py

# Single GPU
python scripts/train_ddp.py --config configs/pretrain_config.py
```

## 4. Monitor Training

- **WandB**: Check https://wandb.ai/YOUR_USER/difffp
- **Local logs**: `logs/` directory
- **Checkpoints**: Saved in `checkpoints/`

## 5. Evaluate Model

```bash
python scripts/eval.py \
    --checkpoint checkpoints/bert_mqa_final.pth \
    --csv ../data/nuscenes/df_val_mqa.csv \
    --top_k 5
```

## 6. Use Pretrained BERT in BEVFormer-MQA

After training, load the pretrained BERT:

```python
# In bevformer_mqa config, change:
text_encoder_cfg=dict(
    pretrained_model='bert_mqa_pretrain/checkpoints/bert_model_final',  # Use your trained model
    freeze=False,  # Can fine-tune further
    output_dim=256,
    pooling='cls',
)
```

## Training Tips

### 1. Adjust Batch Size
```python
# In configs/pretrain_config.py
batch_size=128  # Larger = better negatives, but needs more GPU memory
```

### 2. Temperature Tuning
```python
temperature=0.07  # Lower = harder negatives, higher = softer
```

### 3. From Scratch vs Fine-tuning
```python
from_scratch=False  # False: faster convergence
from_scratch=True   # True: like SDTagNet paper, more domain-specific
```

### 4. Monitor Loss
- InfoNCE loss should decrease from ~6-7 to ~2-3
- If loss doesn't decrease: increase learning rate or batch size
- If loss oscillates: decrease learning rate

## Expected Results

- **Training time**: ~2-4 hours on 4x RTX 3090 (10 epochs)
- **Retrieval Accuracy@5**: 60-80% 
- **Improved MQA performance**: 5-10% boost in downstream tasks

## Troubleshooting

### OOM (Out of Memory)
```python
# Reduce batch size
batch_size=64  # or 32

# Enable gradient checkpointing
# In models/bert_contrastive.py, add:
self.bert.gradient_checkpointing_enable()
```

### Slow training
```python
# Increase num_workers
num_workers=8

# Use larger batch size with more GPUs
```

### Loss not decreasing
```python
# Increase learning rate
lr=1e-4

# Increase batch size (more negatives)
batch_size=256

# Check data: make sure positive pairs are actually similar
```
