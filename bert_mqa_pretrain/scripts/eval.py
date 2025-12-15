"""
Evaluation script for BERT contrastive model

Tests the learned embeddings on similarity tasks.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer
from tqdm import tqdm

from models.bert_contrastive import BERTContrastive
from data.mqa_dataset import MQAContrastiveDataset


def load_model(checkpoint_path, device='cuda'):
    """Load trained BERT model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    config = checkpoint['config']
    
    # Create model
    model = BERTContrastive(**config.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def encode_questions(model, tokenizer, questions, device='cuda', batch_size=32):
    """Encode questions to embeddings."""
    embeddings = []
    
    for i in tqdm(range(0, len(questions), batch_size), desc='Encoding'):
        batch = questions[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            batch_emb = model(
                inputs['input_ids'],
                inputs['attention_mask']
            )
        
        embeddings.append(batch_emb.cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def evaluate_retrieval(model, tokenizer, dataset, device='cuda', top_k=5):
    """Evaluate retrieval accuracy.
    
    For each question, find top-k most similar questions.
    Measure what percentage of top-k are from the same semantic group.
    """
    # Get all questions
    questions = [q['question'] for q in dataset.questions]
    
    # Encode all questions
    embeddings = encode_questions(model, tokenizer, questions, device)
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # For each question, check if retrieved questions are semantically similar
    accuracies = []
    
    for i, q in enumerate(questions):
        # Get ground truth semantic group
        semantic_key = dataset._extract_semantic_key(q)
        gt_group = set(dataset.question_groups[semantic_key])
        
        # Get top-k most similar (excluding self)
        similarities = sim_matrix[i]
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        # Check how many are in the same group
        correct = sum(1 for idx in top_indices if idx in gt_group)
        accuracy = correct / top_k
        accuracies.append(accuracy)
    
    return np.mean(accuracies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint')
    parser.add_argument('--csv', type=str, default='data/nuscenes/df_val_mqa.csv',
                        help='CSV file for evaluation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()
    
    # Load model
    print(f'Loading model from {args.checkpoint}...')
    model, config = load_model(args.checkpoint, args.device)
    tokenizer = BertTokenizer.from_pretrained(config.model['model_name'])
    
    # Load dataset
    print(f'Loading dataset from {args.csv}...')
    dataset = MQAContrastiveDataset(
        csv_file=args.csv,
        augmentation=False,
        num_positives=1,
    )
    
    # Evaluate
    print(f'Evaluating retrieval accuracy (top-{args.top_k})...')
    accuracy = evaluate_retrieval(
        model, tokenizer, dataset,
        device=args.device,
        top_k=args.top_k
    )
    
    print(f'\nRetrieval Accuracy@{args.top_k}: {accuracy:.4f}')
    
    # Example: show some similar questions
    print(f'\nExample similar questions:')
    
    questions = [q['question'] for q in dataset.questions[:100]]
    embeddings = encode_questions(model, tokenizer, questions, args.device)
    
    # Show top-3 similar questions for first 3 questions
    sim_matrix = cosine_similarity(embeddings)
    
    for i in range(min(3, len(questions))):
        print(f'\nQuery: {questions[i]}')
        similarities = sim_matrix[i]
        top_indices = np.argsort(similarities)[::-1][1:4]
        
        for rank, idx in enumerate(top_indices, 1):
            print(f'  {rank}. [{similarities[idx]:.3f}] {questions[idx]}')


if __name__ == '__main__':
    main()
