"""Combine per-model analysis CSVs (from analyze.py) into a cross-model comparison:
accuracy by model x step-count, format-compliance (explicit vs. fallback answer
extraction) by model x step-count, two comparison plots, and a plain-text readout.

Usage:
    python compare_models.py results/analysis_*.csv
"""

import argparse
import glob
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

# Threshold for calling a model's accuracy drop "real" rather than sampling
# noise: at n=5 problems per step level, a single flip is 0.2, so we require
# a meaningfully larger swing before flagging degradation.
DEGRADATION_THRESHOLD = 0.4


def load_analysis(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="model", columns="difficulty", values="is_correct", aggfunc="mean")


def extraction_table(df: pd.DataFrame) -> pd.DataFrame:
    is_explicit = df["extraction_method"] == "explicit_final_answer"
    return is_explicit.groupby([df["model"], df["difficulty"]]).mean().unstack("difficulty")


def plot_accuracy(acc: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, row in acc.iterrows():
        ax.plot(row.index, row.values, marker="o", label=model)
    ax.set_xlabel("Number of reasoning steps")
    ax.set_ylabel("Accuracy")
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
        print("Real accuracy degradation (>= {:.0%} drop from lowest to highest step count):".format(DEGRADATION_THRESHOLD))
        for model, first, last in degraded_models:
            print(f"  - {model}: {first:.0%} -> {last:.0%}")
    else:
        print("No model showed a real accuracy drop (>= {:.0%}) across step counts —".format(DEGRADATION_THRESHOLD))
        print("this problem set (up to 8 steps) doesn't reach a reasoning failure for any of them.")

    if stable_models:
        print("\nAccuracy held roughly flat for:")
        for model, lo, hi in stable_models:
            print(f"  - {model}: {lo:.0%}-{hi:.0%} across step counts")

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
    ext = extraction_table(df)

    print("=== Accuracy by model x step count ===")
    print(acc.round(2).to_string())
    acc.to_csv(os.path.join(args.results_dir, f"compare_accuracy_{timestamp}.csv"))

    print("\n=== Format compliance (explicit rate) by model x step count ===")
    print(ext.round(2).to_string())
    ext.to_csv(os.path.join(args.results_dir, f"compare_extraction_{timestamp}.csv"))

    plot_accuracy(acc, os.path.join(args.results_dir, f"compare_accuracy_{timestamp}.png"))
    plot_extraction(ext, os.path.join(args.results_dir, f"compare_extraction_{timestamp}.png"))

    print_readout(acc, ext)


if __name__ == "__main__":
    main()
