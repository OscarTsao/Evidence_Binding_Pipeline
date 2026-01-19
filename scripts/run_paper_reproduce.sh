#!/bin/bash
# Paper Reproduction Script
#
# This script reproduces all paper results in sequence:
# 1. Run tests to verify code integrity
# 2. Audit splits for data leakage
# 3. Run final evaluation
# 4. Cross-check metrics independently
# 5. Generate publication figures
# 6. Package paper bundle
#
# Usage:
#   bash scripts/run_paper_reproduce.sh
#
# Prerequisites:
#   - Python environment activated
#   - Data files in data/ directory
#   - Groundtruth built (run build_groundtruth.py first)

set -e  # Exit on error

echo "========================================"
echo "Paper Reproduction Pipeline"
echo "========================================"
echo "Started at: $(date)"
echo ""

# Configuration
CONFIG="configs/default.yaml"
OUTPUT_BASE="results"
PAPER_VERSION="v1.0"

# Step 1: Run Tests
echo ""
echo "[Step 1/6] Running tests..."
echo "----------------------------------------"
pytest -q --tb=short
if [ $? -eq 0 ]; then
    echo "✓ All tests passed"
else
    echo "✗ Tests failed - aborting"
    exit 1
fi

# Step 2: Audit Splits
echo ""
echo "[Step 2/6] Auditing splits and leakage..."
echo "----------------------------------------"
python scripts/verification/audit_splits_and_leakage.py \
    --config "$CONFIG" \
    --output "$OUTPUT_BASE/audit/"
echo "✓ Split audit complete"

# Step 3: Run Final Evaluation
echo ""
echo "[Step 3/6] Running final evaluation on test set..."
echo "----------------------------------------"
python scripts/eval_zoo_pipeline.py \
    --config "$CONFIG" \
    --split test \
    --output "$OUTPUT_BASE/eval/"
echo "✓ Final evaluation complete"

# Step 4: Cross-check Metrics
echo ""
echo "[Step 4/6] Running independent metric cross-check..."
echo "----------------------------------------"
if [ -f "$OUTPUT_BASE/eval/predictions.csv" ]; then
    python scripts/verification/metric_crosscheck.py \
        --predictions "$OUTPUT_BASE/eval/predictions.csv" \
        --groundtruth "data/groundtruth/evidence_sentence_groundtruth.csv" \
        --output "$OUTPUT_BASE/crosscheck/"
    echo "✓ Metric cross-check complete"
else
    echo "⚠ Skipping cross-check (no predictions file)"
fi

# Step 5: Generate Publication Figures
echo ""
echo "[Step 5/6] Generating publication figures..."
echo "----------------------------------------"
python scripts/verification/generate_publication_plots.py \
    --results_dir "$OUTPUT_BASE/eval/" \
    --output "paper/figures/"
echo "✓ Figures generated"

# Step 6: Package Paper Bundle
echo ""
echo "[Step 6/6] Packaging paper bundle..."
echo "----------------------------------------"
python scripts/reporting/package_paper_bundle.py \
    --results_dir "$OUTPUT_BASE/eval/" \
    --output "$OUTPUT_BASE/paper_bundle/$PAPER_VERSION" \
    --version "$PAPER_VERSION"
echo "✓ Paper bundle packaged"

# Summary
echo ""
echo "========================================"
echo "Reproduction Complete!"
echo "========================================"
echo "Finished at: $(date)"
echo ""
echo "Outputs:"
echo "  - Test results: pytest output above"
echo "  - Audit report: $OUTPUT_BASE/audit/"
echo "  - Evaluation: $OUTPUT_BASE/eval/"
echo "  - Cross-check: $OUTPUT_BASE/crosscheck/"
echo "  - Figures: paper/figures/"
echo "  - Paper bundle: $OUTPUT_BASE/paper_bundle/$PAPER_VERSION/"
echo ""
echo "Next steps:"
echo "  1. Review $OUTPUT_BASE/paper_bundle/$PAPER_VERSION/MANIFEST.md"
echo "  2. Copy figures to manuscript: paper/figures/"
echo "  3. Update paper tables with results from tables/"
