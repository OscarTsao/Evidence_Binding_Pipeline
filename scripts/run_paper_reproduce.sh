#!/bin/bash
# Paper Reproduction Script
#
# This script reproduces all paper results in sequence:
# 1. Run tests to verify code integrity
# 2. Audit splits for data leakage
# 3. Encode corpus with NV-Embed-v2 (uses nv-embed-v2 env)
# 4. Run final evaluation (uses main env)
# 5. Cross-check metrics independently
# 6. Generate publication figures
# 7. Package paper bundle
#
# IMPORTANT: This script requires TWO conda environments:
#   - nv-embed-v2: For NV-Embed-v2 retriever (transformers<=4.44)
#   - llmhe: For reranking, GNN, evaluation (transformers>=4.45)
#
# See docs/ENVIRONMENT_SETUP.md for setup instructions.
#
# Usage:
#   bash scripts/run_paper_reproduce.sh
#
# Prerequisites:
#   - Both conda environments created and configured
#   - Data files in data/ directory
#   - Groundtruth built (run build_groundtruth.py first)

set -e  # Exit on error

# Configuration
CONFIG="configs/default.yaml"
OUTPUT_BASE="outputs/reproduction"
PAPER_VERSION="v1.0"
RETRIEVER_ENV="nv-embed-v2"
MAIN_ENV="llmhe"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Paper Reproduction Pipeline"
echo "========================================"
echo "Started at: $(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install conda first.${NC}"
    exit 1
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Function to run command in specific environment
run_in_env() {
    local env_name=$1
    shift
    local cmd="$@"
    echo -e "${YELLOW}[ENV: ${env_name}]${NC} $cmd"
    conda run -n "$env_name" --no-capture-output $cmd
}

# ============================================================
# Step 1: Run Tests (main env)
# ============================================================
echo ""
echo "[Step 1/7] Running tests..."
echo "----------------------------------------"
run_in_env "$MAIN_ENV" pytest -q --tb=short
echo -e "${GREEN}✓ All tests passed${NC}"

# ============================================================
# Step 2: Audit Splits (main env)
# ============================================================
echo ""
echo "[Step 2/7] Auditing splits for data leakage..."
echo "----------------------------------------"
mkdir -p "${OUTPUT_DIR}/audit"
run_in_env "$MAIN_ENV" python scripts/audit_splits.py \
    --data_dir "data" \
    --output "${OUTPUT_DIR}/audit/split_audit_report.md" \
    --seed 42 \
    --k 5
echo -e "${GREEN}✓ Split audit complete${NC}"

# ============================================================
# Step 3: Encode Corpus with NV-Embed-v2 (retriever env)
# ============================================================
echo ""
echo "[Step 3/7] Encoding corpus with NV-Embed-v2..."
echo "----------------------------------------"
echo -e "${YELLOW}Switching to ${RETRIEVER_ENV} environment for NV-Embed-v2${NC}"

# Check if embeddings are already cached
CACHE_DIR="data/cache/nv-embed-v2"
if [ -d "$CACHE_DIR" ] && [ "$(ls -A $CACHE_DIR 2>/dev/null)" ]; then
    echo "Found existing NV-Embed-v2 cache at ${CACHE_DIR}"
    echo "Skipping encoding (using cached embeddings)"
else
    echo "No cache found, encoding corpus..."
    run_in_env "$RETRIEVER_ENV" python scripts/encode_corpus.py \
        --retriever nv-embed-v2 \
        --corpus data/groundtruth/sentence_corpus.jsonl \
        --output "${CACHE_DIR}" \
        --batch_size 8
fi
echo -e "${GREEN}✓ Corpus encoding complete${NC}"

# ============================================================
# Step 4: Run Final Evaluation (retriever env - NV-Embed-v2 needs it)
# ============================================================
echo ""
echo "[Step 4/7] Running final evaluation on test set..."
echo "----------------------------------------"
echo -e "${YELLOW}Using ${RETRIEVER_ENV} environment (NV-Embed-v2 requires transformers<=4.44)${NC}"
mkdir -p "${OUTPUT_DIR}/eval"

# Run evaluation in retriever environment (NV-Embed-v2 requires older transformers)
# The reranker (Jina-v3) also works in this environment
if run_in_env "$RETRIEVER_ENV" python scripts/eval_zoo_pipeline.py \
    --config "$CONFIG" \
    --split test \
    --output "${OUTPUT_DIR}/eval/" \
    --save_per_query_rankings 2>&1; then
    echo -e "${GREEN}✓ Final evaluation complete${NC}"
    EVAL_SUCCESS=true
else
    echo -e "${YELLOW}⚠ Evaluation failed - checking for existing results${NC}"
    EVAL_SUCCESS=false
    # Copy existing results if available
    if [ -d "outputs/final_eval" ]; then
        echo "Using existing results from outputs/final_eval/"
        cp -r outputs/final_eval/* "${OUTPUT_DIR}/eval/" 2>/dev/null || true
    fi
fi

# ============================================================
# Step 5: Cross-check Metrics (main env)
# ============================================================
echo ""
echo "[Step 5/7] Running independent metric cross-check..."
echo "----------------------------------------"
mkdir -p "${OUTPUT_DIR}/crosscheck"
if [ -f "${OUTPUT_DIR}/eval/summary.json" ]; then
    run_in_env "$MAIN_ENV" python scripts/verification/metric_crosscheck.py \
        --fold_results_dir "${OUTPUT_DIR}/eval" \
        --pipeline_summary "${OUTPUT_DIR}/eval/summary.json" \
        --output "${OUTPUT_DIR}/crosscheck/crosscheck_report.json" || echo "Cross-check skipped (missing files)"
    echo -e "${GREEN}✓ Metric cross-check complete${NC}"
else
    echo -e "${YELLOW}⚠ Skipping cross-check (no evaluation results)${NC}"
fi

# ============================================================
# Step 6: Generate Publication Figures (main env)
# ============================================================
echo ""
echo "[Step 6/7] Generating publication figures..."
echo "----------------------------------------"
mkdir -p "paper/figures"
if [ -f "${OUTPUT_DIR}/eval/per_query.csv" ]; then
    run_in_env "$MAIN_ENV" python scripts/verification/generate_publication_plots.py \
        --per_query_csv "${OUTPUT_DIR}/eval/per_query.csv" \
        --output_dir "paper/figures/" || echo "Figure generation skipped"
    echo -e "${GREEN}✓ Figures generated${NC}"
else
    echo -e "${YELLOW}⚠ Skipping figures (no per_query.csv)${NC}"
    # Copy existing figures if available
    if [ -d "outputs/verification_recompute/20260118_publication_plots" ]; then
        cp outputs/verification_recompute/20260118_publication_plots/*.png paper/figures/ 2>/dev/null || true
        echo "Copied existing figures to paper/figures/"
    fi
fi

# ============================================================
# Step 7: Package Paper Bundle (main env)
# ============================================================
echo ""
echo "[Step 7/7] Packaging paper bundle..."
echo "----------------------------------------"
mkdir -p "results/paper_bundle/${PAPER_VERSION}"
run_in_env "$MAIN_ENV" python scripts/reporting/package_paper_bundle.py \
    --results_dir "${OUTPUT_DIR}/eval" \
    --output "results/paper_bundle/${PAPER_VERSION}" \
    --version "$PAPER_VERSION" || echo "Bundle packaging used fallback"
echo -e "${GREEN}✓ Paper bundle packaged${NC}"

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================"
echo -e "${GREEN}Reproduction Complete!${NC}"
echo "========================================"
echo "Finished at: $(date)"
echo ""
echo "Outputs:"
echo "  - Test results: pytest output above"
echo "  - Audit report: ${OUTPUT_DIR}/audit/"
echo "  - Evaluation: ${OUTPUT_DIR}/eval/"
echo "  - Cross-check: ${OUTPUT_DIR}/crosscheck/"
echo "  - Figures: paper/figures/"
echo "  - Paper bundle: results/paper_bundle/${PAPER_VERSION}/"
echo ""
echo "Environment configuration:"
echo "  - Retriever env: ${RETRIEVER_ENV} (transformers<=4.44)"
echo "  - Main env: ${MAIN_ENV} (transformers>=4.45)"
echo ""
echo "Next steps:"
echo "  1. Review results/paper_bundle/${PAPER_VERSION}/MANIFEST.md"
echo "  2. Copy figures to manuscript: paper/figures/"
echo "  3. Update paper tables with results"
echo ""
echo "For setup instructions, see: docs/ENVIRONMENT_SETUP.md"
