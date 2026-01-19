#!/usr/bin/env python3
"""Encode corpus with NV-Embed-v2 in dedicated conda environment.

This script should be run in the nv-embed-v2 conda environment to avoid
dependency conflicts. It encodes the sentence corpus and saves embeddings
to cache for later use.

Usage:
    conda activate nv-embed-v2
    python scripts/encode_corpus_nv_embed.py --config configs/default.yaml
    conda deactivate
"""

import argparse
import json
import pickle
from pathlib import Path
import yaml
import torch
import numpy as np
from tqdm import tqdm

# Import NV-Embed-v2 specific imports
from transformers import AutoTokenizer, AutoModel


def load_sentence_corpus(corpus_path: Path):
    """Load sentence corpus from JSONL file."""
    sentences = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sentences.append({
                'sent_uid': data['sent_uid'],
                'post_id': data['post_id'],
                'sentence': data['text'],  # Field is 'text' in JSONL
            })
    return sentences


def encode_with_nv_embed(sentences, model_id="nvidia/NV-Embed-v2", batch_size=8, max_length=512):
    """Encode sentences with NV-Embed-v2 using the model's _do_encode() method."""

    print(f"Loading NV-Embed-v2 model: {model_id}")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Encoding {len(sentences)} sentences with batch_size={batch_size}")

    # Extract texts
    texts = [s['sentence'] for s in sentences]

    # Use NV-Embed-v2's custom _do_encode() method
    # For passages (corpus), no instruction prefix is needed
    with torch.no_grad():
        corpus_embeddings = model._do_encode(
            texts,
            batch_size=batch_size,
            instruction="",  # No instruction for passages
            max_length=max_length,
            num_workers=0,
        )

    # Convert to numpy and normalize
    corpus_embeddings = corpus_embeddings.cpu().numpy()
    norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    embeddings = corpus_embeddings / (norms + 1e-8)

    print(f"Encoded {len(embeddings)} sentences, embedding shape: {embeddings.shape}")

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Encode corpus with NV-Embed-v2")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Load sentence corpus
    corpus_path = Path(cfg['paths']['sentence_corpus'])
    print(f"Loading sentence corpus from {corpus_path}")
    sentences = load_sentence_corpus(corpus_path)
    print(f"Loaded {len(sentences)} sentences")

    # Encode with NV-Embed-v2
    model_id = cfg['models'].get('retriever_model_id', 'nvidia/NV-Embed-v2')
    embeddings = encode_with_nv_embed(
        sentences,
        model_id=model_id,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Save embeddings to cache
    cache_dir = Path(cfg['paths']['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings as numpy array
    embeddings_file = cache_dir / "corpus_embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f"Saved embeddings to {embeddings_file}")

    # Save sentence metadata (mapping sent_uid -> index)
    metadata = {
        'sent_uids': [s['sent_uid'] for s in sentences],
        'post_ids': [s['post_id'] for s in sentences],
        'model_id': model_id,
        'embedding_dim': embeddings.shape[1],
        'n_sentences': len(sentences),
    }

    metadata_file = cache_dir / "corpus_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {metadata_file}")

    print("\n" + "="*60)
    print("ENCODING COMPLETE")
    print("="*60)
    print(f"Embeddings: {embeddings_file}")
    print(f"Metadata: {metadata_file}")
    print(f"Shape: {embeddings.shape}")
    print(f"\nYou can now run the evaluation in the main environment:")
    print(f"  conda activate llmhe")
    print(f"  python scripts/eval_with_cached_embeddings.py --config {args.config}")


if __name__ == "__main__":
    main()
