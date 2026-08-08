"""Generate basic plots from analysis CSVs: accuracy vs. reasoning-step count,
and a breakdown of failure categories.

Usage:
    python report.py results/analysis_*.csv
"""

import argparse
import glob
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd


def load_analysis(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def plot_accuracy_vs_difficulty(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, group in df.groupby("model"):
        accuracy = group.groupby("difficulty")["is_correct"].mean().sort_index()
        ax.plot(accuracy.index, accuracy.values, marker="o", label=model)

    ax.set_xlabel("Number of reasoning steps")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. reasoning-chain length")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


def plot_failure_categories(df: pd.DataFrame, out_path: str):
    incorrect = df[~df["is_correct"]]
    if incorrect.empty:
        print("No incorrect responses to plot failure categories for.")
        return

    counts = incorrect.groupby(["difficulty", "failure_category"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Number of reasoning steps")
    ax.set_ylabel("Count of incorrect responses")
    ax.set_title("Failure category breakdown by difficulty")
    ax.legend(title="Failure category", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_files", nargs="+", help="Analysis CSV file(s) from analyze.py (globs ok)")
    parser.add_argument("--results-dir", default="results", help="Directory to write plots to")
    args = parser.parse_args()

    paths = []
    for pattern in args.analysis_files:
        paths.extend(glob.glob(pattern))
    if not paths:
        raise SystemExit("No analysis CSV files matched.")

    df = load_analysis(paths)
    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    plot_accuracy_vs_difficulty(df, os.path.join(args.results_dir, f"accuracy_vs_difficulty_{timestamp}.png"))
    plot_failure_categories(df, os.path.join(args.results_dir, f"failure_categories_{timestamp}.png"))


if __name__ == "__main__":
    main()
