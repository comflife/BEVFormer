"""
Conservative Configuration for CLIP Multi-View Contrastive Learning

Changes from original config to prevent overfitting:
1. Freeze vision encoder (only fine-tune text + projection layers)
2. Lower learning rate
3. Reduce hard negative weight
4. Add dropout for regularization
"""

# Model configuration
model = dict(
    model_name='openai/clip-vit-base-patch32',
    embedding_dim=256,
    temperature=0.07,
    visual_aggregation='attention',
    freeze_vision=True,   # ← FREEZE vision encoder to prevent overfitting
    freeze_text=False,    # Fine-tune text encoder only
)

# Dataset configuration
data_root = 'data/nuscenes/'
train_mqa_csv = 'df_train_mqa.csv'
val_mqa_csv = 'df_val_mqa.csv'
train_nuscenes_pkl = 'data/nuscenes/nuscenes_infos_temporal_train_bg.pkl'
val_nuscenes_pkl = 'data/nuscenes/nuscenes_infos_temporal_val_bg.pkl'

# Training configuration
train = dict(
    batch_size=4,
    qa_per_scene=8,
    num_workers=4,
    num_epochs=20,  # More epochs since we're more conservative
    lr=2e-6,  # ← Lower LR (was 1e-5)
    weight_decay=0.01,
    warmup_steps=500,
    temperature=0.07,
    hard_negative_weight=1.2,  # ← Reduce from 1.5 to 1.2
    save_interval=2,
    log_interval=50,
    gradient_accumulation_steps=2,
)

# Output directory
output_dir = 'clip_mqa_pretrain/checkpoints_conservative'

# WandB logging
wandb = dict(
    project='difffp',
    name='clip_mqa_pretrain_conservative',
    entity=None,
)

# Evaluation
eval_interval = 1

# Early stopping
early_stopping = dict(
    patience=3,  # Stop if val loss doesn't improve for 3 epochs
    min_delta=0.001,  # Minimum improvement to consider as progress
)
