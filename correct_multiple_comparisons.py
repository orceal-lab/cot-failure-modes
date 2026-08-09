"""Apply a Benjamini-Hochberg false-discovery-rate correction across every
significance test this project has run, pooled from full_comparison.py's
saved CSVs plus any additional one-off tests (format compliance, consistency)
supplied via --extra.

Testing many hypotheses and reporting whichever clear a raw p<0.05 is how
spurious findings accumulate. This pools everything and asks: which results
are still solid once you account for how many chances we gave ourselves to
find something?

Usage:
    python correct_multiple_comparisons.py \
        --own-slope results/full_comparison/own_slope_*.csv \
        --pairwise results/full_comparison/pairwise_*.csv \
        --cross-condition results/full_comparison/cross_condition_*.csv \
        --extra "running_total: mistral format-slope" 0.2200 \
        --extra "consistency: mistral vs qwen2.5-coder rate" 0.2518
"""

import argparse
import glob

import pandas as pd
from statsmodels.stats.multitest import multipletests


def load_glob(paths_glob: str | None) -> pd.DataFrame:
    if not paths_glob:
        return pd.DataFrame()
    paths = glob.glob(paths_glob)
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--own-slope", help="Glob for full_comparison.py's own_slope_*.csv output")
    parser.add_argument("--pairwise", help="Glob for full_comparison.py's pairwise_*.csv output")
    parser.add_argument("--cross-condition", help="Glob for full_comparison.py's cross_condition_*.csv output")
    parser.add_argument(
        "--extra", nargs=2, action="append", metavar=("TEST_NAME", "P_VALUE"), default=[],
        help="An additional (name, p-value) pair not covered by the CSVs above -- e.g. format-compliance "
        "or consistency tests computed separately. Repeat for each one.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold (default 0.05)")
    args = parser.parse_args()

    tests = []

    own_slope = load_glob(args.own_slope)
    for _, r in own_slope.iterrows():
        tests.append((f"{r['condition']}: {r['model']} own slope", r["p"]))

    pairwise = load_glob(args.pairwise)
    for _, r in pairwise.iterrows():
        tests.append((f"{r['condition']}: {r['model_a']} vs {r['model_b']} slope diff", r["p"]))

    cross = load_glob(args.cross_condition)
    for _, r in cross.iterrows():
        tests.append((f"{r['model']}: {r['condition_a']} vs {r['condition_b']} slope diff", r["p"]))

    for name, p in args.extra:
        tests.append((name, float(p)))

    if not tests:
        raise SystemExit("No tests found -- supply at least one of --own-slope/--pairwise/--cross-condition/--extra")

    names = [t[0] for t in tests]
    pvals = [t[1] for t in tests]
    reject, p_adj, _, _ = multipletests(pvals, alpha=args.alpha, method="fdr_bh")

    result = pd.DataFrame({"test": names, "p_raw": pvals, "p_adjusted_BH": p_adj, "significant": reject})
    result = result.sort_values("p_raw").reset_index(drop=True)

    pd.set_option("display.max_colwidth", 70)
    pd.set_option("display.width", 150)
    print(result.to_string(index=False))
    print(f"\n{len(tests)} total tests. {int(reject.sum())} survive Benjamini-Hochberg FDR correction at alpha={args.alpha}.")


if __name__ == "__main__":
    main()
