#!/usr/bin/env bash
# Run this once the expansion Kaggle job's output has been downloaded to
# kaggle_run/expansion_run/output/. Copies the new raw files into results/,
# re-analyzes each condition with all 4 models (old + new), and runs the
# full comparison + multiple-comparisons correction across everything.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
source .venv/bin/activate

echo "=== Copying new raw files ==="
cp kaggle_run/expansion_run/output/cot-failure-modes/results/raw_*.json results/

echo "=== Re-analyzing each condition with all 4 models ==="
# Excludes _x3_ (repeated-sampling consistency-check files) -- those overlap
# with the single-attempt runs and would otherwise silently multiply counts
# (a first pass here produced n=140 instead of n=20 for mistral on
# running-total). _bridge_ files are NOT excluded: for mistral and
# qwen2.5-coder, their difficulty=5 data on the singles condition lives
# *only* in the bridge file (their main singles run predates that problem
# level being merged into arithmetic_singles.json) -- excluding it creates
# a gap, not a fix. llama3.1:8b/qwen3:4b already have difficulty=5 natively
# and have no bridge file, so this exclusion is harmless for them either way.
rm -rf results/steps_v2 results/independent_v2 results/singles_v2
find results -maxdepth 1 -name 'raw_*_arithmetic_steps_*.json' ! -name '*_x3_*' \
  | xargs python3 analyze.py --results-dir results/steps_v2
find results -maxdepth 1 -name 'raw_*_arithmetic_independent_*.json' ! -name '*_x3_*' \
  | xargs python3 analyze.py --results-dir results/independent_v2
find results -maxdepth 1 -name 'raw_*_arithmetic_singles_*.json' ! -name '*_x3_*' \
  | xargs python3 analyze.py --results-dir results/singles_v2

echo "=== compare_models.py per condition ==="
python3 compare_models.py results/steps_v2/analysis_*.csv --results-dir results/steps_v2
python3 compare_models.py results/independent_v2/analysis_*.csv --results-dir results/independent_v2
python3 compare_models.py results/singles_v2/analysis_*.csv --results-dir results/singles_v2

echo "=== full_comparison.py across all 3 conditions, 4 models ==="
rm -rf results/full_comparison_v2
python3 full_comparison.py \
  --condition running_total "results/steps_v2/analysis_*.csv" \
  --condition independent "results/independent_v2/analysis_*.csv" \
  --condition singles "results/singles_v2/analysis_*.csv" \
  --results-dir results/full_comparison_v2

echo "=== Multiple-comparisons correction across the full 4-model test set ==="
python3 correct_multiple_comparisons.py \
  --own-slope "results/full_comparison_v2/own_slope_*.csv" \
  --pairwise "results/full_comparison_v2/pairwise_*.csv" \
  --cross-condition "results/full_comparison_v2/cross_condition_*.csv"

echo "=== Done. Review results/full_comparison_v2/ and the correction output above before updating the report. ==="
