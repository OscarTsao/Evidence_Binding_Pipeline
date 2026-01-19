#!/usr/bin/env python3
"""Encode corpus with a retriever model.

This script pre-computes embeddings for the sentence corpus and caches them.
It should be run in the appropriate conda environment for the chosen retriever.

For NV-Embed-v2, use the nv-embed-v2 environment (transformers<=4.44).

Usage:
    conda activate nv-embed-v2
    python scripts/encode_corpus.py \
        --retriever nv-embed-v2 \
        --corpus data/groundtruth/sentence_corpus.jsonl \
        --output data/cache/nv-embed-v2 \
        --batch_size 8
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm


def load_corpus(corpus_path: Path) -> List[Dict[str, Any]]:
    """Load sentence corpus from JSONL file."""
    sentences = []
    with open(corpus_path) as f:
        for line in f:
            sentences.append(json.loads(line))
    return sentences


def compute_corpus_hash(sentences: List[Dict[str, Any]]) -> str:
    """Compute hash of corpus for cache validation."""
    content = json.dumps([s.get("sent_uid", "") for s in sentences], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def encode_with_nv_embed_v2(
    sentences: List[str],
    batch_size: int = 8,
    device: str = "cuda",
) -> np.ndarray:
    """Encode sentences with NV-Embed-v2.

    Requires transformers<=4.44.x
    """
    import torch
    from transformers import AutoModel

    print("Loading NV-Embed-v2 model...")
    model = AutoModel.from_pretrained(
        "nvidia/NV-Embed-v2",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)
    model.eval()

    print(f"Encoding {len(sentences)} sentences with batch_size={batch_size}...")
    all_embeddings = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
        batch = sentences[i:i + batch_size]
        with torch.no_grad():
            embeddings = model._do_encode(
                batch,
                batch_size=len(batch),
                instruction="",
                max_length=512,
                num_workers=0,
                return_numpy=True,
            )
            all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


def encode_with_bge_m3(
    sentences: List[str],
    batch_size: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """Encode sentences with BGE-M3."""
    from FlagEmbedding import BGEM3FlagModel

    print("Loading BGE-M3 model...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)

    print(f"Encoding {len(sentences)} sentences...")
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        max_length=512,
    )["dense_vecs"]

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Encode corpus with retriever")
    parser.add_argument("--retriever", type=str, required=True,
                        choices=["nv-embed-v2", "bge-m3"],
                        help="Retriever model to use")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Path to sentence_corpus.jsonl")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for cached embeddings")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for encoding")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-encoding even if cache exists")
    args = parser.parse_args()

    # Load corpus
    print(f"Loading corpus from {args.corpus}...")
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} sentences")

    # Compute corpus hash
    corpus_hash = compute_corpus_hash(corpus)
    print(f"Corpus hash: {corpus_hash}")

    # Check for existing cache
    args.output.mkdir(parents=True, exist_ok=True)
    cache_file = args.output / f"embeddings_{corpus_hash}.pkl"
    meta_file = args.output / "meta.json"

    if cache_file.exists() and not args.force:
        print(f"Cache already exists at {cache_file}")
        print("Use --force to re-encode")
        return 0

    # Extract sentence texts
    sentences = [s["sentence"] for s in corpus]
    sent_uids = [s["sent_uid"] for s in corpus]

    # Encode
    if args.retriever == "nv-embed-v2":
        embeddings = encode_with_nv_embed_v2(
            sentences,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.retriever == "bge-m3":
        embeddings = encode_with_bge_m3(
            sentences,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        print(f"Unknown retriever: {args.retriever}")
        return 1

    print(f"Embeddings shape: {embeddings.shape}")

    # Save cache
    print(f"Saving embeddings to {cache_file}...")
    cache_data = {
        "embeddings": embeddings,
        "sent_uids": sent_uids,
        "corpus_hash": corpus_hash,
        "retriever": args.retriever,
    }
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)

    # Save metadata
    meta = {
        "retriever": args.retriever,
        "corpus_hash": corpus_hash,
        "n_sentences": len(sentences),
        "embedding_dim": embeddings.shape[1],
        "cache_file": cache_file.name,
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Cache saved successfully!")
    print(f"  - Embeddings: {cache_file}")
    print(f"  - Metadata: {meta_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
