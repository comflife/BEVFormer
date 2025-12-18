"""
Pre-cache CLIP text embeddings for all unique questions in MQA dataset.

This dramatically speeds up training by avoiding redundant CLIP text encoding.
Run this script once before training to generate cached embeddings.
"""

import argparse
import os
import sys
from pathlib import Path
import csv
from tqdm import tqdm
import torch
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from projects.mmdet3d_plugin.bevformer.modules.clip_text_encoder import CLIPTextEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mqa-csv', type=str, required=True, help='Path to MQA CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output pickle file path')
    parser.add_argument('--clip-model', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load CLIP text encoder
    print(f'Loading CLIP model: {args.clip_model}')
    text_encoder = CLIPTextEncoder(
        pretrained_model=args.clip_model,
        freeze=True,
        output_dim=256,
    ).to(device).eval()

    # Extract unique questions
    print(f'Reading questions from: {args.mqa_csv}')
    unique_questions = set()
    with open(args.mqa_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc='Scanning CSV'):
            question = row['question']
            unique_questions.add(question)

    unique_questions = sorted(list(unique_questions))
    print(f'Found {len(unique_questions)} unique questions')

    # Pre-compute embeddings
    print('Pre-computing CLIP embeddings...')
    text_embeddings = {}

    with torch.no_grad():
        # Process in batches for efficiency
        batch_size = 32
        for i in tqdm(range(0, len(unique_questions), batch_size)):
            batch_questions = unique_questions[i:i+batch_size]
            batch_embeddings = text_encoder(batch_questions)  # [N, 256]

            for q, emb in zip(batch_questions, batch_embeddings):
                text_embeddings[q] = emb.cpu()  # Store on CPU to save GPU memory

    # Save to pickle
    print(f'Saving embeddings to: {args.output}')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(text_embeddings, f)

    print(f'✓ Done! Cached {len(text_embeddings)} text embeddings')
    print(f'  File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB')


if __name__ == '__main__':
    main()
