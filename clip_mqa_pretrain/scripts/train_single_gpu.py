"""
Single GPU Training Script for CLIP Multi-View Contrastive Learning

Uses gradient accumulation for larger effective batch size.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

from clip_mqa_pretrain.data.nuscenes_mqa_dataset_optimized import (
    NuScenesMQASceneDataset,
    clip_scene_collate_fn
)
from clip_mqa_pretrain.models.clip_contrastive import (
    CLIPMultiViewContrastive,
    CLIPInfoNCELoss
)


def load_config(config_path):
    """Load config from Python file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    epoch,
    device,
    accumulation_steps=1,
    log_interval=50,
):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for step, batch in enumerate(pbar):
        # Move to device
        pixel_values = batch['pixel_values'].to(device)  # [B, 6, 3, H, W]
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        texts = batch['texts']  # Keep on CPU for hard negative mining
        sample_tokens = batch['sample_tokens']  # Keep on CPU for positive/negative identification

        # Forward
        visual_emb, text_emb = model(pixel_values, input_ids, attention_mask)

        # Loss with multi-positive support and hard negative mining
        loss = criterion(visual_emb, text_emb, texts, sample_tokens)
        loss = loss / accumulation_steps  # Scale loss for gradient accumulation

        # Backward
        loss.backward()

        # Update every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        total_loss += loss.item() * accumulation_steps

        if step % log_interval == 0:
            avg_loss = total_loss / (step + 1)
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{lr:.2e}'
            })

            # WandB
            wandb.log({
                'train/loss': loss.item() * accumulation_steps,
                'train/avg_loss': avg_loss,
                'train/lr': lr,
                'epoch': epoch,
                'step': epoch * len(dataloader) + step,
            })

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion_val, device):
    """Evaluate on validation set with standard CLIP loss."""
    model.eval()
    total_loss = 0

    pbar = tqdm(dataloader, desc='Validation')

    for batch in pbar:
        # Move to device
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        texts = batch['texts']  # Keep on CPU for hard negative mining
        sample_tokens = batch['sample_tokens']  # Keep on CPU for positive/negative identification

        # Forward
        visual_emb, text_emb = model(pixel_values, input_ids, attention_mask)

        # Standard CLIP loss (no hard negative weighting for validation)
        loss = criterion_val(visual_emb, text_emb, texts, sample_tokens)
        total_loss += loss.item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--debug', action='store_true', help='Debug mode with small dataset')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # WandB
    wandb.init(
        project=config.wandb['project'],
        name=config.wandb['name'] + '_single_gpu',
        entity=config.wandb['entity'],
        config={
            'model': config.model,
            'train': config.train,
            'single_gpu': True,
        }
    )

    # Datasets (optimized: scene-based loading)
    print("Loading datasets...")

    qa_per_scene = config.train.get('qa_per_scene', 8)

    train_dataset = NuScenesMQASceneDataset(
        data_root=config.data_root,
        mqa_csv=config.train_mqa_csv,
        nuscenes_info_pkl=config.train_nuscenes_pkl,
        clip_processor_name=config.model['model_name'],
        qa_per_scene=qa_per_scene,
        max_samples=100 if args.debug else None,
    )

    val_dataset = NuScenesMQASceneDataset(
        data_root=config.data_root,
        mqa_csv=config.val_mqa_csv,
        nuscenes_info_pkl=config.val_nuscenes_pkl,
        clip_processor_name=config.model['model_name'],
        qa_per_scene=qa_per_scene,
        max_samples=50 if args.debug else None,
    )

    # Model
    print("Loading CLIP model...")
    model = CLIPMultiViewContrastive(**config.model)

    # Resize token embeddings if special tokens were added
    # Dataset adds special tokens, so model needs to match
    # IMPORTANT: Resize BEFORE moving to GPU
    # Note: Use len() not vocab_size property, as vocab_size doesn't update after adding tokens
    new_vocab_size = len(train_dataset.processor.tokenizer)
    if new_vocab_size > model.original_vocab_size:
        print(f"Resizing token embeddings: {model.original_vocab_size} → {new_vocab_size}")
        model.resize_token_embeddings(new_vocab_size)

    # Now move to GPU
    model = model.to(device)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # DataLoaders
    batch_size = config.train['batch_size']
    accumulation_steps = config.train['gradient_accumulation_steps']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.train['num_workers'],
        collate_fn=clip_scene_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.train['num_workers'],
        collate_fn=clip_scene_collate_fn,
        pin_memory=True,
    )

    print(f'Train scenes: {len(train_dataset)}')
    print(f'Val scenes: {len(val_dataset)}')
    print(f'QA per scene: {qa_per_scene}')
    print(f'Effective train samples: {len(train_dataset) * qa_per_scene}')
    print(f'Batch size (scenes): {batch_size}')
    print(f'Batch size (QA pairs): {batch_size * qa_per_scene}')
    print(f'Gradient accumulation steps: {accumulation_steps}')
    print(f'Effective batch size: {batch_size * qa_per_scene * accumulation_steps}')

    # Loss with hard negative mining (for training)
    criterion = CLIPInfoNCELoss(
        temperature=config.train['temperature'],
        hard_negative_weight=config.train.get('hard_negative_weight', 1.5),
    )
    print(f'Train hard negative weight: {config.train.get("hard_negative_weight", 1.5)}')

    # Standard CLIP loss for validation (no hard negative weighting)
    criterion_val = CLIPInfoNCELoss(
        temperature=config.train['temperature'],
        hard_negative_weight=1.0,  # Standard loss
    )
    print(f'Val uses standard CLIP loss (hard_negative_weight=1.0)')

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train['lr'],
        weight_decay=config.train['weight_decay']
    )

    # Scheduler
    total_steps = len(train_loader) * config.train['num_epochs'] // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.train['warmup_steps'],
        num_training_steps=total_steps
    )

    # Create output dir
    os.makedirs(config.output_dir, exist_ok=True)

    # Early stopping setup
    early_stopping_config = getattr(config, 'early_stopping', None)
    if early_stopping_config:
        patience = early_stopping_config.get('patience', 5)
        min_delta = early_stopping_config.get('min_delta', 0.001)
        epochs_without_improvement = 0
        print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}")
    else:
        patience = None

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, config.train['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.train['num_epochs']}")
        print(f"{'='*50}")

        # Train
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            device=device,
            accumulation_steps=accumulation_steps,
            log_interval=config.train['log_interval'],
        )

        print(f'Train Loss: {train_loss:.4f}')

        # Evaluate
        if epoch % config.eval_interval == 0:
            val_loss = evaluate(
                model=model,
                dataloader=val_loader,
                criterion_val=criterion_val,
                device=device,
            )

            print(f'Val Loss: {val_loss:.4f}')

            # WandB
            wandb.log({
                'val/loss': val_loss,
                'train/loss_epoch': train_loss,
                'epoch': epoch,
            })

            # Save best model and check early stopping
            if val_loss < best_val_loss - (min_delta if patience else 0):
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                epochs_without_improvement = 0

                best_path = os.path.join(config.output_dir, 'clip_mqa_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                }, best_path)
                print(f'✓ Saved best model: {best_path} (improved by {improvement:.4f})')
            else:
                if patience:
                    epochs_without_improvement += 1
                    print(f'No improvement for {epochs_without_improvement} epoch(s). Best val loss: {best_val_loss:.4f}')

                    if epochs_without_improvement >= patience:
                        print(f'\nEarly stopping triggered! No improvement for {patience} epochs.')
                        print(f'Best val loss: {best_val_loss:.4f} at epoch {epoch - patience}')
                        break

        # Save checkpoint
        if epoch % config.train['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }

            save_path = os.path.join(
                config.output_dir,
                f'clip_mqa_epoch{epoch}.pth'
            )
            torch.save(checkpoint, save_path)
            print(f'Saved checkpoint: {save_path}')

    # Final save
    final_path = os.path.join(config.output_dir, 'clip_mqa_final.pth')
    torch.save({
        'epoch': config.train['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, final_path)
    print(f'Training complete. Final model saved to {final_path}')

    wandb.finish()


if __name__ == '__main__':
    main()
