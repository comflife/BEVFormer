# BERT Contrastive Pretraining for MQA

## Overview
This directory contains code for pre-training BERT on MQA (Multi-task Question Answering) questions using contrastive learning, inspired by SDTagNet paper.

## Key Idea
- **Problem**: Pre-trained BERT is trained on general text, not domain-specific QA patterns
- **Solution**: Use contrastive learning to make BERT understand MQA question semantics
- **Method**: Multiple Negatives Ranking Loss (InfoNCE-based)

## Directory Structure
```
bert_mqa_pretrain/
├── data/                   # Dataset and dataloader
│   └── mqa_dataset.py
├── models/                 # Model definitions
│   └── bert_contrastive.py
├── scripts/                # Training scripts
│   ├── train_ddp.py       # DDP training
│   └── eval.py            # Evaluation
├── configs/                # Configuration files
│   └── pretrain_config.py
├── logs/                   # Training logs
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install transformers torch accelerate wandb
```

### 2. Train BERT
```bash
# Single GPU
python scripts/train_ddp.py --config configs/pretrain_config.py

# Multi-GPU DDP
torchrun --nproc_per_node=4 scripts/train_ddp.py --config configs/pretrain_config.py
```

## Contrastive Learning Strategy

### Positive Pairs
Questions with semantically similar patterns:
- "How many cars in front?" ↔ "How many vehicles ahead?"
- "Where is the nearest pedestrian?" ↔ "Where is the closest person?"

### Negative Pairs
Questions with different semantics:
- "How many cars in front?" ↔ "Where is the traffic cone?"
- "Distance to car" ↔ "Count of pedestrians"

### Data Augmentation
- Synonym replacement (car → vehicle)
- Direction paraphrasing (front → ahead)
- Number variation (same semantic but different counts)
