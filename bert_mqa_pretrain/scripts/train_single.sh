#!/bin/bash
# Single GPU training script for BERT Contrastive Learning

CONDA_ENV="bert"
GPU_ID=0  # Change this to use different GPU

source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "Starting BERT contrastive pretraining on single GPU..."
echo "Using Python: $(which python)"
echo "GPU: $GPU_ID"

# Check dependencies
python -c "import transformers; print(f'✓ transformers {transformers.__version__} found')" || {
    echo "Error: transformers not found. Please install:"
    echo "  conda activate $CONDA_ENV"
    echo "  pip install transformers accelerate wandb scikit-learn"
    exit 1
}

# Single GPU training
export CUDA_VISIBLE_DEVICES=$GPU_ID

python scripts/train_single_gpu.py \
    --config configs/pretrain_config.py \
    --gpu 0

echo "Training complete!"
