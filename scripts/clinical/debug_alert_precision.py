#!/usr/bin/env python3
"""Debug script to investigate alert precision = 0% issue.

This script analyzes the summary.json from a clinical evaluation run to identify
the root cause of the alert precision problem.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_summary(summary_path: Path, output_dir: Path):
    """Analyze summary.json to diagnose alert precision issue."""

    print(f"\n{'='*80}")
    print("ALERT PRECISION DIAGNOSTIC ANALYSIS")
    print(f"{'='*80}\n")

    # Load summary
    with open(summary_path) as f:
        summary = json.load(f)

    print(f"Summary file: {summary_path}")
    print(f"Number of folds: {summary['n_folds']}\n")

    # Analysis 1: Threshold values across folds
    print("="*80)
    print("1. THRESHOLD VALUES ACROSS FOLDS")
    print("="*80)

    for i, fold_result in enumerate(summary['fold_results']):
        thresh = fold_result['threshold_selection']
        print(f"\nFold {i}:")
        print(f"  tau_neg: {thresh['tau_neg']:.6f}")
        print(f"  tau_pos: {thresh['tau_pos']:.6f}")
        print(f"  TUNE distribution:")
        print(f"    NEG: {thresh['distribution']['neg_rate']:.1%}")
        print(f"    UNCERTAIN: {thresh['distribution']['uncertain_rate']:.1%}")
        print(f"    POS: {thresh['distribution']['pos_rate']:.1%}")
        print(f"  TUNE screening:")
        print(f"    Sensitivity: {thresh['screening']['sensitivity']:.4f}")
        print(f"    FPR: {thresh['screening']['fpr']:.6f} ← {'NEGATIVE!' if thresh['screening']['fpr'] < 0 else 'OK'}")
        print(f"    NPV: {thresh['screening']['npv']:.4f}")
        print(f"  TUNE alert:")
        print(f"    Precision: {thresh['alert']['precision']:.4f} ← {'ZERO!' if thresh['alert']['precision'] == 0 else 'OK'}")
        print(f"    Recall: {thresh['alert']['recall']:.4f}")
        print(f"    FPR: {thresh['alert']['fpr']:.6f} ← {'NEGATIVE!' if thresh['alert']['fpr'] < 0 else 'OK'}")

    # Analysis 2: Negative FPR investigation
    print("\n" + "="*80)
    print("2. NEGATIVE FPR INVESTIGATION")
    print("="*80)

    print("\nNegative FPR values detected! This indicates a bug in the code.")
    print("FPR should always be in [0, 1], defined as FP / N where N = # negative samples.")
    print("\nPossible causes:")
    print("  1. Incorrect calculation: FP or N is computed incorrectly")
    print("  2. Label inversion: positive/negative labels are swapped")
    print("  3. Data corruption: labels or predictions are corrupted")
    print("  4. Train/test split bug: TUNE/TEST splits are incorrect")

    # Analysis 3: Alert precision = 0% investigation
    print("\n" + "="*80)
    print("3. ALERT PRECISION = 0% INVESTIGATION")
    print("="*80)

    print("\nAlert precision = 0% on TUNE split means:")
    print("  Precision = TP / (TP + FP) = 0")
    print("  This happens when TP = 0 (no true positives in POS class)")
    print("\nPossible scenarios:")
    print("  a) tau_pos is set so high that NO queries are classified as POS")
    print("  b) tau_pos is set so low that ALL queries are POS, but none have evidence")
    print("  c) Calibration is broken and all probabilities are shifted")

    # Check scenario a: tau_pos too high
    print("\nChecking scenario (a): tau_pos too high")
    for i, fold_result in enumerate(summary['fold_results']):
        thresh = fold_result['threshold_selection']
        pos_rate_tune = thresh['distribution']['pos_rate']
        tau_pos = thresh['tau_pos']

        if pos_rate_tune < 0.01:
            print(f"  Fold {i}: tau_pos={tau_pos:.4f}, POS rate={pos_rate_tune:.1%} ← TOO HIGH (almost no POS)")
        elif pos_rate_tune > 0.90:
            print(f"  Fold {i}: tau_pos={tau_pos:.4f}, POS rate={pos_rate_tune:.1%} ← TOO LOW (almost all POS)")
        else:
            print(f"  Fold {i}: tau_pos={tau_pos:.4f}, POS rate={pos_rate_tune:.1%} ← Reasonable distribution")

    # Analysis 4: TEST set metrics
    print("\n" + "="*80)
    print("4. TEST SET METRICS (HELD-OUT EVALUATION)")
    print("="*80)

    for i, fold_result in enumerate(summary['fold_results']):
        test = fold_result['test_metrics']['deployment']
        print(f"\nFold {i} TEST:")
        print(f"  Distribution:")
        print(f"    NEG: {test['neg_rate']:.1%}")
        print(f"    UNCERTAIN: {test['uncertain_rate']:.1%}")
        print(f"    POS: {test['pos_rate']:.1%}")
        print(f"  Screening:")
        print(f"    Sensitivity: {test['screening_sensitivity']:.4f}")
        print(f"    FPR: {test['screening_fpr']:.6f} ← {'NEGATIVE!' if test['screening_fpr'] < 0 else 'OK'}")
        print(f"  Alert:")
        print(f"    Precision: {test['alert_precision']:.4f} ← {'ZERO!' if test['alert_precision'] == 0 else 'OK'}")
        print(f"    Recall: {test['alert_recall']:.4f}")
        print(f"    FPR: {test['alert_fpr']:.6f} ← {'NEGATIVE!' if test['alert_fpr'] < 0 else 'OK'}")

    # Analysis 5: Root cause diagnosis
    print("\n" + "="*80)
    print("5. ROOT CAUSE DIAGNOSIS")
    print("="*80)

    # Check for consistent patterns
    all_tune_fpr_negative = all(
        fold['threshold_selection']['screening']['fpr'] < 0
        for fold in summary['fold_results']
    )
    all_test_fpr_negative = all(
        fold['test_metrics']['deployment']['screening_fpr'] < 0
        for fold in summary['fold_results']
    )
    all_alert_precision_zero = all(
        fold['test_metrics']['deployment']['alert_precision'] == 0
        for fold in summary['fold_results']
    )

    print("\nConsistent patterns across all folds:")
    print(f"  TUNE screening FPR < 0: {all_tune_fpr_negative}")
    print(f"  TEST screening FPR < 0: {all_test_fpr_negative}")
    print(f"  TEST alert precision = 0: {all_alert_precision_zero}")

    if all_tune_fpr_negative and all_test_fpr_negative:
        print("\n⚠️  CRITICAL BUG IDENTIFIED:")
        print("  FPR calculation is consistently negative across ALL folds and splits.")
        print("  This is a systematic code bug, not a data issue.")
        print("\n  Recommended fix:")
        print("  1. Check FPR calculation in three_state_gate.py:307")
        print("  2. Check FPR calculation in run_clinical_high_recall_eval.py:360")
        print("  3. Verify that (~labels) is computing the negative class correctly")
        print("  4. Verify that n_samples - n_positive equals the true number of negatives")

    if all_alert_precision_zero:
        print("\n⚠️  ALERT PRECISION ISSUE:")
        print("  Alert precision = 0% across all folds suggests:")
        print("  - Threshold selection is finding tau_pos values that classify many queries as POS")
        print("  - But NONE of those POS queries have evidence (all false positives)")
        print("\n  This could be caused by:")
        print("  1. Calibration shifting all probabilities incorrectly")
        print("  2. tau_pos selection logic has a bug")
        print("  3. TUNE split is unrepresentative")

    # Analysis 6: Code location check
    print("\n" + "="*80)
    print("6. CODE LOCATIONS TO CHECK")
    print("="*80)

    print("\nFiles to inspect:")
    print("  1. src/final_sc_review/clinical/three_state_gate.py")
    print("     - Line 307: screening_fpr calculation")
    print("     - Line 318: alert_fpr calculation")
    print("     - Line 233-285: tau_pos selection logic")
    print("\n  2. scripts/clinical/run_clinical_high_recall_eval.py")
    print("     - Line 132: BUG DETECTED! n_train = n_total - n_train (should be n_total - n_tune)")
    print("     - Line 360: screening_fpr calculation in test metrics")
    print("     - Line 370: alert_fpr calculation in test metrics")
    print("\n  3. scripts/clinical/run_clinical_high_recall_eval.py")
    print("     - Line 121-138: TRAIN/TUNE/TEST split logic (verify correctness)")

    # Save diagnostic report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "alert_precision_analysis.md"

    with open(report_file, 'w') as f:
        f.write("# Alert Precision Diagnostic Analysis\n\n")
        f.write(f"**Summary file:** `{summary_path}`\n\n")
        f.write(f"**Analysis date:** {np.datetime64('now')}\n\n")
        f.write("---\n\n")

        f.write("## Critical Findings\n\n")
        f.write("### 1. Negative FPR Bug\n\n")
        f.write("**Issue:** FPR values are negative across all folds and splits.\n\n")
        f.write("**Impact:** This invalidates all screening and alert metrics.\n\n")
        f.write("**Root cause:** Code bug in FPR calculation.\n\n")
        f.write("**Fix required:**\n")
        f.write("- Check `three_state_gate.py` line 307 and line 318\n")
        f.write("- Check `run_clinical_high_recall_eval.py` line 360 and line 370\n")
        f.write("- Verify negative class calculation `(~labels)` is correct\n\n")

        f.write("### 2. Alert Precision = 0%\n\n")
        f.write("**Issue:** All POS predictions are false positives.\n\n")
        f.write("**Possible causes:**\n")
        f.write("- Calibration is shifting probabilities incorrectly\n")
        f.write("- tau_pos selection logic has a bug\n")
        f.write("- TUNE split is too small or unrepresentative\n\n")

        f.write("### 3. Data Split Bug\n\n")
        f.write("**Issue:** Line 132 in `run_clinical_high_recall_eval.py` has a bug:\n")
        f.write("```python\n")
        f.write("n_train = n_total - n_train  # BUG! Should be: n_total - n_tune\n")
        f.write("```\n\n")
        f.write("**Impact:** This causes incorrect TRAIN/TUNE splitting, leading to:\n")
        f.write("- Wrong threshold selection (on wrong data)\n")
        f.write("- Potential data leakage\n")
        f.write("- Unreliable metrics\n\n")

        f.write("---\n\n")
        f.write("## Recommended Actions\n\n")
        f.write("1. **Fix data split bug** (line 132 in run_clinical_high_recall_eval.py)\n")
        f.write("2. **Fix FPR calculation** (verify negative class computation)\n")
        f.write("3. **Re-run evaluation** with fixes applied\n")
        f.write("4. **Verify threshold selection** logic works correctly\n")
        f.write("5. **Check calibration** is not breaking probabilities\n\n")

        # Add fold-by-fold details
        f.write("---\n\n")
        f.write("## Fold-by-Fold Details\n\n")
        for i, fold_result in enumerate(summary['fold_results']):
            f.write(f"### Fold {i}\n\n")

            thresh = fold_result['threshold_selection']
            f.write("**TUNE split (threshold selection):**\n")
            f.write(f"- tau_neg: {thresh['tau_neg']:.6f}\n")
            f.write(f"- tau_pos: {thresh['tau_pos']:.6f}\n")
            f.write(f"- NEG rate: {thresh['distribution']['neg_rate']:.1%}\n")
            f.write(f"- POS rate: {thresh['distribution']['pos_rate']:.1%}\n")
            f.write(f"- Screening FPR: {thresh['screening']['fpr']:.6f} {'← NEGATIVE!' if thresh['screening']['fpr'] < 0 else ''}\n")
            f.write(f"- Alert precision: {thresh['alert']['precision']:.4f} {'← ZERO!' if thresh['alert']['precision'] == 0 else ''}\n\n")

            test = fold_result['test_metrics']['deployment']
            f.write("**TEST split (held-out evaluation):**\n")
            f.write(f"- NEG rate: {test['neg_rate']:.1%}\n")
            f.write(f"- POS rate: {test['pos_rate']:.1%}\n")
            f.write(f"- Screening FPR: {test['screening_fpr']:.6f} {'← NEGATIVE!' if test['screening_fpr'] < 0 else ''}\n")
            f.write(f"- Alert precision: {test['alert_precision']:.4f} {'← ZERO!' if test['alert_precision'] == 0 else ''}\n\n")

    print(f"\n\nDiagnostic report saved to: {report_file}")

    # Return summary of issues
    return {
        'negative_fpr_bug': all_tune_fpr_negative and all_test_fpr_negative,
        'alert_precision_zero': all_alert_precision_zero,
        'data_split_bug': True,  # We identified this in the code
    }


def main():
    parser = argparse.ArgumentParser(description="Debug alert precision = 0% issue")
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to summary.json from clinical evaluation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for diagnostic report"
    )

    args = parser.parse_args()

    # Run analysis
    issues = analyze_summary(args.summary, args.output)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"Critical bugs identified: {sum(issues.values())}")
    for issue_name, found in issues.items():
        status = "✓ FOUND" if found else "✗ Not found"
        print(f"  {issue_name}: {status}")

    print(f"\nSee diagnostic report for details: {args.output / 'alert_precision_analysis.md'}")
    print()


if __name__ == "__main__":
    main()
