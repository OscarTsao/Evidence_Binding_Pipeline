#!/bin/bash
# Wrapper script to evaluate NV-Embed-v2 pipeline with automatic environment switching
#
# Usage:
#   bash scripts/eval_nv_embed_wrapper.sh --split test

set -e  # Exit on error

echo "======================================================================"
echo "NV-EMBED-V2 EVALUATION WRAPPER"
echo "======================================================================"
echo ""

# Parse arguments
SPLIT="test"
SAVE_RANKINGS=""
SKIP_RERANKING=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --save-per-query-rankings)
            SAVE_RANKINGS="--save_per_query_rankings"
            shift
            ;;
        --skip-reranking)
            SKIP_RERANKING="--skip_reranking"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if cache already exists
CACHE_DIR="data/cache/nv-embed-v2"
EMBEDDINGS_FILE="$CACHE_DIR/corpus_embeddings.npy"

if [ -f "$EMBEDDINGS_FILE" ]; then
    echo "✓ Found cached NV-Embed-v2 embeddings at $EMBEDDINGS_FILE"
    echo "  Skipping corpus encoding (delete cache to re-encode)"
    echo ""
else
    echo "STAGE 1: Encoding corpus with NV-Embed-v2"
    echo "----------------------------------------------------------------------"
    echo "  Switching to nv-embed-v2 conda environment..."
    echo ""

    # Run encoding in nv-embed-v2 environment
    eval "$(conda shell.bash hook)"
    conda activate nv-embed-v2

    python scripts/encode_corpus_nv_embed.py --config configs/default.yaml

    conda deactivate
    echo ""
fi

echo "STAGE 2: Running retrieval + reranking"
echo "----------------------------------------------------------------------"
echo "  Switching to llmhe conda environment..."
echo ""

# Run evaluation in main environment
eval "$(conda shell.bash hook)"
conda activate llmhe

python scripts/eval_with_cached_embeddings.py \
    --config configs/default.yaml \
    --split "$SPLIT" \
    $SAVE_RANKINGS \
    $SKIP_RERANKING

conda deactivate

echo ""
echo "======================================================================"
echo "EVALUATION COMPLETE"
echo "======================================================================"
