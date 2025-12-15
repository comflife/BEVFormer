"""
Configuration for BERT Contrastive Pretraining
"""

# Data (CSV files are in parent directory)
train_csv = '../df_train_mqa.csv'  # 1,204,771 samples
val_csv = '../df_val_mqa.csv'      # 255,164 samples

# Model
model = dict(
    model_name='bert-base-uncased',
    embedding_dim=256,
    temperature=0.07,
    from_scratch=False,  # True: train from scratch, False: fine-tune pretrained
)

# Training
train = dict(
    batch_size=32,  # Large batch size for contrastive learning (per GPU)
    num_epochs=10,
    num_workers=4,
    
    # Optimizer
    lr=5e-5,  # Learning rate
    weight_decay=0.01,
    warmup_steps=1000,
    
    # Loss
    temperature=0.07,
    
    # Checkpointing
    save_interval=1,  # Save every N epochs
    log_interval=50,  # Log every N steps
)

# Dataset
dataset = dict(
    augmentation=True,
    num_positives=20,  # Positive pairs per semantic group
)

# DDP
ddp = dict(
    backend='nccl',  # 'nccl' for GPU, 'gloo' for CPU
    find_unused_parameters=False,
)

# Output
output_dir = 'bert_mqa_pretrain/checkpoints'
log_dir = 'bert_mqa_pretrain/logs'

# WandB
wandb = dict(
    project='difffp',
    name='bert_mqa_contrastive',
    entity=None,  # Use default logged-in account
)
