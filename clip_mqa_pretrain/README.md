# CLIP Multi-View Contrastive Learning for NuScenes MQA

This module trains a CLIP model to align multi-view camera images from NuScenes with question-answer text pairs from MQA dataset.

## Overview

- **Visual Encoder**: CLIP ViT-Base-Patch32 (processes 6 camera views)
- **Text Encoder**: CLIP Text Transformer (processes QA pairs)
- **Loss**: Symmetric InfoNCE (CLIP-style contrastive loss)
- **Goal**: Learn aligned representations of visual scenes and textual descriptions

## Advantages over BERT

1. **Faster Training**: CLIP is pre-trained on image-text pairs, requires less fine-tuning
2. **Multi-Modal**: Native support for both vision and text
3. **Smaller Model**: ViT-Base-Patch32 is more efficient than BERT-large
4. **Better Alignment**: Designed for vision-language tasks

## Setup

```bash
# Install dependencies
pip install transformers torch torchvision tqdm wandb

# Prepare data
# Make sure you have:
# - data/nuscenes/ (NuScenes dataset)
# - df_train_mqa.csv (training annotations)
# - df_val_mqa.csv (validation annotations)
```

## Training

### Single GPU Training

```bash
python clip_mqa_pretrain/scripts/train_single_gpu.py \
    --config clip_mqa_pretrain/configs/pretrain_config.py \
    --gpu 0
```

### Multi-GPU Training (DDP)

```bash
python clip_mqa_pretrain/scripts/train_ddp.py \
    --config clip_mqa_pretrain/configs/pretrain_config.py \
    --gpus 2
```

### Debug Mode

```bash
python clip_mqa_pretrain/scripts/train_single_gpu.py \
    --config clip_mqa_pretrain/configs/pretrain_config.py \
    --gpu 0 \
    --debug
```

## Model Architecture

```
Input: 6 camera images + QA text
    ↓
┌─────────────────────────────────────┐
│ CLIP Vision Encoder                 │
│ - Process each view independently   │
│ - Extract visual features           │
│ - Aggregate with attention          │
└─────────────────────────────────────┘
    ↓ [B, 768]
┌─────────────────────────────────────┐
│ Visual Projection                   │
│ - Project to embedding space (256)  │
│ - L2 normalize                      │
└─────────────────────────────────────┘
    ↓ [B, 256]

Input: "Question: ... Answer: ..."
    ↓
┌─────────────────────────────────────┐
│ CLIP Text Encoder                   │
│ - Tokenize and encode text          │
│ - Extract text features             │
└─────────────────────────────────────┘
    ↓ [B, 512]
┌─────────────────────────────────────┐
│ Text Projection                     │
│ - Project to embedding space (256)  │
│ - L2 normalize                      │
└─────────────────────────────────────┘
    ↓ [B, 256]

┌─────────────────────────────────────┐
│ Symmetric InfoNCE Loss              │
│ - Image-to-text similarity          │
│ - Text-to-image similarity          │
└─────────────────────────────────────┘
```

## Configuration

Edit [configs/pretrain_config.py](configs/pretrain_config.py):

```python
# Model
model = dict(
    model_name='openai/clip-vit-base-patch32',
    embedding_dim=256,
    temperature=0.07,
    visual_aggregation='attention',  # 'mean', 'max', 'attention'
)

# Training
train = dict(
    batch_size=16,
    num_epochs=10,
    lr=1e-5,
    gradient_accumulation_steps=2,
)
```

## Checkpoints

Trained models are saved to `clip_mqa_pretrain/checkpoints/`:

- `clip_mqa_best.pth`: Best validation loss
- `clip_mqa_epoch{N}.pth`: Checkpoints every N epochs
- `clip_mqa_final.pth`: Final model

## Using Trained Model

```python
from clip_mqa_pretrain.models.clip_contrastive import CLIPMultiViewContrastive

# Load model
model = CLIPMultiViewContrastive(
    model_name='openai/clip-vit-base-patch32',
    embedding_dim=256,
)

# Load weights
checkpoint = torch.load('clip_mqa_pretrain/checkpoints/clip_mqa_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
visual_emb = model.encode_multiview_images(images)  # [B, 256]
text_emb = model.encode_text(input_ids, attention_mask)  # [B, 256]
```

## Integration with BEVFormer

After training, update BEVFormer config to use CLIP encoder:

```python
# In bevformer_mqa_tiny.py
text_encoder_cfg=dict(
    type='CLIPTextEncoder',
    pretrained_model='openai/clip-vit-base-patch32',
    pretrained_weights='clip_mqa_pretrain/checkpoints/clip_mqa_best.pth',
    freeze=True,
    output_dim=256,
)
```
