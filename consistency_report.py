"""Measure per-problem consistency from a repeated-sampling run (run_experiment.py
--repeats N). Separates "this model is unreliable" (accuracy varies attempt to
attempt on the same problem) from "this model is reliably wrong on hard problems,
reliably right on easy ones" (accuracy is ~0 or ~1 per problem, consistently).

Usage:
    python consistency_report.py results/analysis_..._x3_....csv
"""

import argparse
import glob
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd


def per_problem_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["model", "difficulty", "problem_id"])["is_correct"]
    rows = []
    for (model, difficulty, problem_id), values in grouped:
        k = len(values)
        correct = int(values.sum())
        rows.append(
            {
                "model": model,
                "difficulty": difficulty,
                "problem_id": problem_id,
                "attempts": k,
                "correct": correct,
                "success_rate": correct / k,
                "consistent": correct == 0 or correct == k,
            }
        )
    return pd.DataFrame(rows)


def plot_consistency(per_problem: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, group in per_problem.groupby("model"):
        by_diff = group.groupby("difficulty")["consistent"].mean()
        ax.plot(by_diff.index, by_diff.values, marker="o", label=model)
    ax.set_xlabel("Number of reasoning steps")
    ax.set_ylabel("Share of problems with all-or-nothing outcomes")
    ax.set_title("Per-problem consistency across repeated attempts")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


def print_readout(per_problem: pd.DataFrame):
    print("\n=== Plain-text summary ===\n")
    for model, group in per_problem.groupby("model"):
        n = len(group)
        consistent = int(group["consistent"].sum())
        mixed = n - consistent
        avg_k = group["attempts"].mean()

        # Null model: if every problem secretly had the SAME success probability
        # p (the model's overall accuracy) and outcomes were pure independent
        # noise, what fraction of problems would still look "consistent" (all-
        # correct or all-wrong) by chance across k attempts? P(all same) =
        # p^k + (1-p)^k. If observed consistency is well above this, that's
        # evidence of real per-problem difficulty variation, not just noise.
        overall_p = group["correct"].sum() / group["attempts"].sum()
        expected_consistent_frac = (group["attempts"].apply(lambda k: overall_p**k + (1 - overall_p) ** k)).mean()

        print(f"{model} (avg {avg_k:.0f} attempts/problem, {n} problems, overall accuracy {overall_p:.0%}):")
        print(f"  {consistent}/{n} problems ({consistent/n:.0%}) were all-correct or all-wrong across attempts (consistent)")
        print(f"  {mixed}/{n} problems ({mixed/n:.0%}) had mixed outcomes across attempts (stochastic)")
        print(f"  Expected consistency under pure noise (same p for every problem): {expected_consistent_frac:.0%}")
        gap = consistent / n - expected_consistent_frac
        if gap > 0.10:
            print(f"  Observed consistency is {gap:+.0%} above the noise baseline — real per-problem difficulty variation, not just unreliability")
        elif gap < -0.10:
            print(f"  Observed consistency is {gap:+.0%} below the noise baseline — unexpected, worth double-checking")
        else:
            print(f"  Observed consistency is close to the noise baseline ({gap:+.0%}) — accuracy looks like it could be explained by uniform per-problem unreliability, not systematic per-problem difficulty")
        if mixed > 0:
            mixed_rates = group.loc[~group["consistent"], "success_rate"]
            print(f"  Mixed problems' per-problem success rate ranged {mixed_rates.min():.0%}-{mixed_rates.max():.0%}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_files", nargs="+", help="Analysis CSV file(s) with an 'attempt' column, repeats>1 (globs ok)")
    parser.add_argument("--results-dir", default="results", help="Directory to write outputs to")
    args = parser.parse_args()

    paths = []
    for pattern in args.analysis_files:
        paths.extend(glob.glob(pattern))
    if not paths:
        raise SystemExit("No analysis CSV files matched.")

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    if "attempt" not in df.columns or df["attempt"].nunique() < 2:
        raise SystemExit("This analysis CSV doesn't have multiple attempts per problem — run with run_experiment.py --repeats N first.")

    per_problem = per_problem_table(df)
    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("=== Per-problem consistency by model x step count ===")
    summary = per_problem.groupby(["model", "difficulty"])["consistent"].mean().unstack("difficulty")
    print(summary.round(2).to_string())

    per_problem.to_csv(os.path.join(args.results_dir, f"consistency_{timestamp}.csv"), index=False)
    plot_consistency(per_problem, os.path.join(args.results_dir, f"consistency_{timestamp}.png"))
    print_readout(per_problem)


if __name__ == "__main__":
    main()
