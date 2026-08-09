"""Combine per-model analysis CSVs (from analyze.py) into a cross-model comparison:
accuracy by model x step-count, format-compliance (explicit vs. fallback answer
extraction) by model x step-count, two comparison plots, and a plain-text readout.

Usage:
    python compare_models.py results/analysis_*.csv
"""

import argparse
import glob
import math
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

# Rough triage threshold for flagging a raw percentage-point drop worth a
# closer look -- NOT a statistical test. A model can clear this threshold
# without its slope being significantly different from a flatter model's
# (see stats_test.py's interaction test, which found exactly that case).
# Use this output to decide what's worth testing properly, not as a verdict.
DEGRADATION_THRESHOLD = 0.4


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion — better behaved
    than the normal approximation at small n or when the proportion is near
    0 or 1, both of which happen constantly with per-step accuracy here."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((center - margin) / denom, (center + margin) / denom)


def load_analysis(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="model", columns="difficulty", values="is_correct", aggfunc="mean")


def accuracy_ci_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, difficulty): n, k, accuracy, and a 95% Wilson CI."""
    rows = []
    grouped = df.groupby(["model", "difficulty"])["is_correct"]
    for (model, difficulty), values in grouped:
        n = len(values)
        k = int(values.sum())
        lo, hi = wilson_interval(k, n)
        rows.append(
            {
                "model": model,
                "difficulty": difficulty,
                "n": n,
                "correct": k,
                "accuracy": k / n,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "difficulty"])


def extraction_table(df: pd.DataFrame) -> pd.DataFrame:
    is_explicit = df["extraction_method"] == "explicit_final_answer"
    return is_explicit.groupby([df["model"], df["difficulty"]]).mean().unstack("difficulty")


def plot_accuracy(acc_ci: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, group in acc_ci.groupby("model"):
        group = group.sort_values("difficulty")
        # Clip to >=0: at accuracy exactly 0% or 100%, the independently-computed
        # Wilson bound can land a hair on the wrong side of it due to floating-point
        # rounding, which matplotlib rejects as a negative yerr.
        yerr_low = (group["accuracy"] - group["ci_low"]).clip(lower=0)
        yerr_high = (group["ci_high"] - group["accuracy"]).clip(lower=0)
        ax.errorbar(
            group["difficulty"], group["accuracy"],
            yerr=[yerr_low, yerr_high],
            marker="o", capsize=3, label=model,
        )
    ax.set_xlabel("Number of reasoning steps")
    ax.set_ylabel("Accuracy (with 95% Wilson CI)")
    ax.set_title("Accuracy vs. reasoning-chain length, by model")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


def plot_extraction(ext: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, row in ext.iterrows():
        ax.plot(row.index, row.values, marker="o", label=model)
    ax.set_xlabel("Number of reasoning steps")
    ax.set_ylabel('Share following "Final Answer:" format')
    ax.set_title("Format compliance vs. reasoning-chain length, by model")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


def print_readout(acc: pd.DataFrame, ext: pd.DataFrame):
    print("\n=== Plain-text summary ===\n")

    degraded_models = []
    stable_models = []
    for model, row in acc.iterrows():
        row = row.dropna().sort_index()
        if len(row) < 2:
            continue
        drop = row.iloc[0] - row.iloc[-1]
        if drop >= DEGRADATION_THRESHOLD:
            degraded_models.append((model, row.iloc[0], row.iloc[-1]))
        else:
            stable_models.append((model, row.min(), row.max()))

    if degraded_models:
        print("Raw drop of >= {:.0%} from lowest to highest step count (triage only, not a significance test — confirm with stats_test.py):".format(DEGRADATION_THRESHOLD))
        for model, first, last in degraded_models:
            print(f"  - {model}: {first:.0%} -> {last:.0%}")
    else:
        print("No model's raw accuracy dropped by >= {:.0%} across step counts.".format(DEGRADATION_THRESHOLD))

    if stable_models:
        print("\nSmaller raw movement (< {:.0%} drop) for:".format(DEGRADATION_THRESHOLD))
        for model, lo, hi in stable_models:
            print(f"  - {model}: {lo:.0%}-{hi:.0%} across step counts — this does NOT mean the slope is zero, just that the raw drop is smaller; run stats_test.py to check whether it's statistically flat")

    print("\nFormat compliance (explicit \"Final Answer:\" rate), lowest vs. highest step count:")
    for model, row in ext.iterrows():
        row = row.dropna().sort_index()
        if len(row) < 2:
            continue
        first, last = row.iloc[0], row.iloc[-1]
        direction = "drifted down" if last < first else ("held steady" if last == first else "improved")
        print(f"  - {model}: {first:.0%} at {row.index[0]} steps -> {last:.0%} at {row.index[-1]} steps ({direction})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_files", nargs="+", help="Analysis CSV file(s) from analyze.py (globs ok)")
    parser.add_argument("--results-dir", default="results", help="Directory to write tables/plots to")
    args = parser.parse_args()

    paths = []
    for pattern in args.analysis_files:
        paths.extend(glob.glob(pattern))
    if not paths:
        raise SystemExit("No analysis CSV files matched.")

    df = load_analysis(paths)
    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    acc = accuracy_table(df)
    acc_ci = accuracy_ci_table(df)
    ext = extraction_table(df)

    print("=== Accuracy by model x step count ===")
    print(acc.round(2).to_string())
    acc.to_csv(os.path.join(args.results_dir, f"compare_accuracy_{timestamp}.csv"))

    print("\n=== Accuracy with 95% Wilson confidence intervals ===")
    ci_display = acc_ci.copy()
    ci_display["accuracy"] = (ci_display["accuracy"] * 100).round(1).astype(str) + "%"
    ci_display["95% CI"] = (
        "["
        + (ci_display["ci_low"] * 100).round(1).astype(str)
        + "%, "
        + (ci_display["ci_high"] * 100).round(1).astype(str)
        + "%]"
    )
    print(ci_display[["model", "difficulty", "n", "correct", "accuracy", "95% CI"]].to_string(index=False))
    acc_ci.to_csv(os.path.join(args.results_dir, f"compare_accuracy_ci_{timestamp}.csv"), index=False)

    print("\n=== Format compliance (explicit rate) by model x step count ===")
    print(ext.round(2).to_string())
    ext.to_csv(os.path.join(args.results_dir, f"compare_extraction_{timestamp}.csv"))

    plot_accuracy(acc_ci, os.path.join(args.results_dir, f"compare_accuracy_{timestamp}.png"))
    plot_extraction(ext, os.path.join(args.results_dir, f"compare_extraction_{timestamp}.png"))

    print_readout(acc, ext)


if __name__ == "__main__":
    main()
