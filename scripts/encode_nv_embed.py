#!/usr/bin/env python3
"""Encode corpus using NV-Embed-v2 in isolated conda environment.

This script runs in the 'nv-embed-v2' conda environment which has
transformers==4.44.2 pinned for compatibility.

Usage:
    conda activate nv-embed-v2
    python scripts/encode_nv_embed.py --config configs/default.yaml

The embeddings are cached to disk and loaded by the main pipeline
running in the 'llmhe' environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml


def get_logger(name: str):
    """Simple logger setup."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(name)


logger = get_logger(__name__)


def patch_dynamic_cache_compat():
    """Patch DynamicCache for backward compatibility with older model code."""
    try:
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, 'get_usable_length'):
            def get_usable_length(self, new_seq_length: int) -> int:
                return self.get_seq_length()
            DynamicCache.get_usable_length = get_usable_length
            logger.debug("Patched DynamicCache.get_usable_length")

        _original_from_legacy_cache = DynamicCache.from_legacy_cache

        @classmethod
        def _patched_from_legacy_cache(cls, past_key_values):
            if past_key_values is None:
                return cls()
            return _original_from_legacy_cache.__func__(cls, past_key_values)

        if not getattr(DynamicCache, '_patched_from_legacy', False):
            DynamicCache.from_legacy_cache = _patched_from_legacy_cache
            DynamicCache._patched_from_legacy = True
            logger.debug("Patched DynamicCache.from_legacy_cache")

    except ImportError:
        pass


def compute_corpus_fingerprint(sentences: List[dict]) -> str:
    """Compute fingerprint of corpus for cache validation."""
    h = hashlib.sha256()
    for sent in sentences:
        h.update(f"{sent['post_id']}|{sent['sid']}|{sent['text']}".encode())
    return h.hexdigest()[:16]


def load_sentence_corpus(corpus_path: Path) -> List[dict]:
    """Load sentence corpus from JSONL file."""
    sentences = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            sentences.append(json.loads(line))
    return sentences


def encode_corpus(
    sentences: List[dict],
    cache_dir: Path,
    model_id: str = "nvidia/NV-Embed-v2",
    batch_size: int = 8,
    max_length: int = 512,
    rebuild: bool = False,
) -> Path:
    """Encode corpus using NV-Embed-v2 and cache to disk.

    Args:
        sentences: List of sentence dicts with 'text', 'post_id', 'sid' keys
        cache_dir: Directory to cache embeddings
        model_id: HuggingFace model ID
        batch_size: Encoding batch size
        max_length: Maximum sequence length
        rebuild: Force rebuild even if cache exists

    Returns:
        Path to cached embeddings
    """
    cache_dir = cache_dir / "nv-embed-v2"
    cache_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = cache_dir / "embeddings.npy"
    fingerprint_path = cache_dir / "fingerprint.json"
    uid_mapping_path = cache_dir / "uid_to_idx.json"

    corpus_fp = compute_corpus_fingerprint(sentences)

    # Check cache
    if not rebuild and embeddings_path.exists() and fingerprint_path.exists():
        with open(fingerprint_path) as f:
            meta = json.load(f)
        if meta.get("corpus") == corpus_fp:
            logger.info(f"Using cached embeddings from {embeddings_path}")
            return embeddings_path

    # Apply compatibility patch
    patch_dynamic_cache_compat()

    # Load model
    logger.info(f"Loading NV-Embed-v2 model: {model_id}")
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_kwargs = {
        "trust_remote_code": True,
    }

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_kwargs["torch_dtype"] = torch.bfloat16
        logger.info("  Using BF16 precision")
    else:
        model_kwargs["torch_dtype"] = torch.float16
        logger.info("  Using FP16 precision")

    model = AutoModel.from_pretrained(model_id, **model_kwargs)
    model = model.to(device)
    model.eval()
    logger.info("  Model loaded successfully!")

    # Encode corpus
    texts = [s["text"] for s in sentences]
    logger.info(f"Encoding {len(texts)} sentences...")

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

    # Save embeddings
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings to {embeddings_path}")
    logger.info(f"  Shape: {embeddings.shape}")

    # Save UID mapping
    uid_to_idx = {s["sent_uid"]: i for i, s in enumerate(sentences)}
    with open(uid_mapping_path, "w") as f:
        json.dump(uid_to_idx, f)
    logger.info(f"Saved UID mapping to {uid_mapping_path}")

    # Save fingerprint
    with open(fingerprint_path, "w") as f:
        json.dump({
            "corpus": corpus_fp,
            "model_id": model_id,
            "num_sentences": len(sentences),
            "embedding_dim": embeddings.shape[1],
            "max_length": max_length,
        }, f, indent=2)

    return embeddings_path


def main():
    parser = argparse.ArgumentParser(
        description="Encode corpus using NV-Embed-v2 (run in nv-embed-v2 conda env)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to sentence corpus JSONL (overrides config)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Encoding batch size",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild cache",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Get paths from config or args
    corpus_path = args.corpus or Path(config["paths"]["sentence_corpus"])
    cache_dir = args.cache_dir or Path(config["paths"]["cache_dir"])

    logger.info(f"Loading corpus from {corpus_path}")
    sentences = load_sentence_corpus(corpus_path)
    logger.info(f"Loaded {len(sentences)} sentences")

    embeddings_path = encode_corpus(
        sentences=sentences,
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        rebuild=args.rebuild,
    )

    logger.info(f"Done! Embeddings cached at: {embeddings_path}")
    logger.info("You can now run the pipeline in the 'llmhe' environment")


if __name__ == "__main__":
    main()
