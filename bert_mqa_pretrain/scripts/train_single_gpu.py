"""
Single GPU Training Script for BERT Contrastive Learning

Avoids DDP issues while maintaining large effective batch size through gradient accumulation.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

from data.mqa_dataset import MQAContrastiveDataset, contrastive_collate_fn
from models.bert_contrastive import BERTContrastive, InfoNCELoss


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
    tokenizer,
    epoch,
    device,
    accumulation_steps=4,
    log_interval=50,
):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for step, batch in enumerate(pbar):
        # Tokenize
        anchors = batch['anchors']
        positives = batch['positives']
        
        anchor_inputs = tokenizer(
            anchors,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        positive_inputs = tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        anchor_input_ids = anchor_inputs['input_ids'].to(device)
        anchor_attention_mask = anchor_inputs['attention_mask'].to(device)
        positive_input_ids = positive_inputs['input_ids'].to(device)
        positive_attention_mask = positive_inputs['attention_mask'].to(device)
        
        # Forward
        anchor_emb = model(anchor_input_ids, anchor_attention_mask)
        positive_emb = model(positive_input_ids, positive_attention_mask)
        
        # Loss
        loss = criterion(anchor_emb, positive_emb)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
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
    
    # Model
    model = BERTContrastive(**config.model).to(device)
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model['model_name'])
    
    # Dataset
    train_dataset = MQAContrastiveDataset(
        csv_file=config.train_csv,
        **config.dataset
    )
    
    # Adjust batch size for single GPU
    # Use smaller batch size but larger accumulation steps
    batch_size_per_gpu = config.train['batch_size'] // 4  # 128 -> 32
    accumulation_steps = 4  # Effective batch = 32 * 4 = 128
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=config.train['num_workers'],
        collate_fn=contrastive_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    print(f'Batch size per GPU: {batch_size_per_gpu}')
    print(f'Gradient accumulation steps: {accumulation_steps}')
    print(f'Effective batch size: {batch_size_per_gpu * accumulation_steps}')
    
    # Loss
    criterion = InfoNCELoss(temperature=config.train['temperature'])
    
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
    
    # Training loop
    for epoch in range(1, config.train['num_epochs'] + 1):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            epoch=epoch,
            device=device,
            accumulation_steps=accumulation_steps,
            log_interval=config.train['log_interval'],
        )
        
        print(f'Epoch {epoch} - Avg Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if epoch % config.train['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                # Don't save config module (not picklable)
            }
            
            save_path = os.path.join(
                config.output_dir,
                f'bert_mqa_epoch{epoch}.pth'
            )
            torch.save(checkpoint, save_path)
            print(f'Saved checkpoint: {save_path}')
            
            # Also save BERT model for easy loading
            model.bert.save_pretrained(
                os.path.join(config.output_dir, f'bert_model_epoch{epoch}')
            )
    
    # Final save
    final_path = os.path.join(config.output_dir, 'bert_mqa_final.pth')
    torch.save(checkpoint, final_path)
    model.bert.save_pretrained(
        os.path.join(config.output_dir, 'bert_model_final')
    )
    print(f'Training complete. Final model saved to {final_path}')
    
    wandb.finish()


if __name__ == '__main__':
    main()
