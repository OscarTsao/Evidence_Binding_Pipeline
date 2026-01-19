#!/usr/bin/env python3
"""Encode all test queries with NV-Embed-v2 in dedicated conda environment.

This script pre-encodes all queries for the test split to avoid loading
NV-Embed-v2 in the main environment during evaluation.

Usage:
    conda activate nv-embed-v2
    python scripts/encode_queries_nv_embed.py --config configs/default.yaml --split test
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
from transformers import AutoModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from final_sc_review.data.io import load_criteria, load_groundtruth
from final_sc_review.data.splits import split_post_ids


def encode_queries_with_nv_embed(queries, model_id="nvidia/NV-Embed-v2", max_length=512):
    """Encode queries with NV-Embed-v2 using the model's encode() method."""

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
    print(f"Encoding {len(queries)} queries")

    # Instruction for queries
    query_instruction = "Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: "

    # Encode all queries
    query_embeddings = {}

    with torch.no_grad():
        for criterion_id, query_text in tqdm(queries.items(), desc="Encoding queries"):
            query_emb = model.encode(
                [query_text],
                instruction=query_instruction,
                max_length=max_length,
            )
            query_emb = query_emb.cpu().numpy()[0]

            # Normalize
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

            query_embeddings[criterion_id] = query_emb

    print(f"Encoded {len(query_embeddings)} queries, embedding dim: {query_embeddings[list(query_embeddings.keys())[0]].shape[0]}")

    return query_embeddings


def main():
    parser = argparse.ArgumentParser(description="Encode queries with NV-Embed-v2")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "train"])
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Load criteria
    print(f"Loading criteria...")
    criteria_path = Path(cfg['paths']['data_dir']) / "DSM5" / "MDD_Criteira.json"
    criteria = load_criteria(criteria_path)
    criteria_map = {c.criterion_id: c.text for c in criteria}
    print(f"Loaded {len(criteria_map)} criteria")

    # Encode queries
    model_id = cfg['models'].get('retriever_model_id', 'nvidia/NV-Embed-v2')
    query_embeddings = encode_queries_with_nv_embed(
        criteria_map,
        model_id=model_id,
        max_length=args.max_length
    )

    # Save query embeddings to cache
    cache_dir = Path(cfg['paths']['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    query_embeddings_file = cache_dir / f"query_embeddings_{args.split}.pkl"
    with open(query_embeddings_file, 'wb') as f:
        pickle.dump(query_embeddings, f)
    print(f"Saved query embeddings to {query_embeddings_file}")

    print("\n" + "="*60)
    print("QUERY ENCODING COMPLETE")
    print("="*60)
    print(f"Query embeddings: {query_embeddings_file}")
    print(f"Number of queries: {len(query_embeddings)}")
    print(f"Embedding dim: {list(query_embeddings.values())[0].shape[0]}")
    print(f"\nYou can now run the evaluation in the main environment:")
    print(f"  conda activate llmhe")
    print(f"  python scripts/eval_with_precomputed_embeddings.py --config {args.config} --split {args.split}")


if __name__ == "__main__":
    main()
