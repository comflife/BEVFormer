"""
Configuration for CLIP Multi-View Contrastive Learning
"""

# Model configuration
model = dict(
    model_name='openai/clip-vit-base-patch32',  # Small CLIP model
    embedding_dim=256,
    temperature=0.07,
    visual_aggregation='attention',  # 'mean', 'max', 'attention'
    freeze_vision=False,  # Fine-tune vision encoder
    freeze_text=False,    # Fine-tune text encoder
)

# Dataset configuration
data_root = 'data/nuscenes/'
train_mqa_csv = 'df_train_mqa.csv'
val_mqa_csv = 'df_val_mqa.csv'
train_nuscenes_pkl = 'data/nuscenes/nuscenes_infos_temporal_train_bg.pkl'
val_nuscenes_pkl = 'data/nuscenes/nuscenes_infos_temporal_val_bg.pkl'

# Training configuration
train = dict(
    batch_size=4,  # Scenes per GPU (reduced due to qa_per_scene)
    qa_per_scene=8,  # Sample 8 QA pairs per scene (effective batch = 4*8 = 32 per GPU)
    num_workers=4,  # Reduced from 8 to avoid I/O bottleneck
    num_epochs=10,
    lr=1e-5,  # Lower learning rate for fine-tuning
    weight_decay=0.01,
    warmup_steps=500,
    temperature=0.07,
    hard_negative_weight=1.5,  # Weight for hard negatives (same cam but different count/obj)
    save_interval=2,
    log_interval=50,
    gradient_accumulation_steps=2,  # Effective batch size = 4*8*2*4 = 256
)

# Output directory
output_dir = 'clip_mqa_pretrain/checkpoints'

# WandB logging
wandb = dict(
    project='difffp',
    name='clip_mqa_pretrain',
    entity=None,  # Set to your wandb entity
)

# Evaluation
eval_interval = 1  # Evaluate every N epochs
