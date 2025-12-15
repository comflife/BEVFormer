#!/bin/bash
# Training script for BERT Contrastive Learning on MQA

# Configuration
NUM_GPUS=4  # Number of GPUs to use
CONFIG="configs/pretrain_config.py"

# Conda environment
CONDA_ENV="bert"  # Change this to your environment name

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Set CUDA visible devices (optional)
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Check transformers is installed
python -c "import transformers; print(f'✓ transformers {transformers.__version__} found')" || {
    echo "Error: transformers not found. Please install:"
    echo "  conda activate $CONDA_ENV"
    echo "  pip install transformers accelerate wandb scikit-learn"
    exit 1
}

# DDP training
echo "Starting BERT contrastive pretraining with $NUM_GPUS GPUs..."
echo "Using Python: $(which python)"

# Use python -m torch.distributed.run instead of torchrun
# This ensures the conda environment is properly inherited
python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    scripts/train_ddp.py \
    --config $CONFIG

echo "Training complete!"
