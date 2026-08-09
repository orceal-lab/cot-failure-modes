"""Consolidated multi-model, multi-condition comparison: accuracy summary table
plus formal significance tests, replacing the ad-hoc one-off scripts used to
build this report by hand. Takes any number of (condition, analysis CSV) pairs
and reports, for every condition:
  - each model's own difficulty slope (does it decline at all?)
  - every model pair's slope difference within that condition (who declines
    faster than whom, if anyone)
and, for every model, whether its slope differs between conditions (does the
isolation manipulation matter for this model specifically?).

Usage:
    python full_comparison.py \
        --condition running_total results/steps/analysis_*.csv \
        --condition independent results/independent/analysis_*.csv \
        --condition singles results/singles/analysis_*.csv
"""

import argparse
import glob
import itertools

import pandas as pd
import statsmodels.formula.api as smf


def load(paths_glob: str) -> pd.DataFrame:
    paths = glob.glob(paths_glob)
    if not paths:
        raise SystemExit(f"No files matched: {paths_glob}")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["is_correct"] = df["is_correct"].astype(int)
    return df


def short_name(model: str) -> str:
    # "ollama:mistral:7b-instruct-q4_K_M" -> "mistral:7b-instruct-q4_K_M"
    return model.split(":", 1)[1] if ":" in model else model


def own_slope_test(df: pd.DataFrame, model: str) -> tuple[float, float]:
    sub = df[df["model"] == model]
    fit = smf.logit("is_correct ~ difficulty", data=sub).fit(disp=0)
    return fit.params["difficulty"], fit.pvalues["difficulty"]


def pairwise_slope_test(df: pd.DataFrame, model_a: str, model_b: str) -> tuple[float, float]:
    sub = df[df["model"].isin([model_a, model_b])].copy()
    sub["is_model_b"] = pd.Categorical(sub["model"] == model_b, categories=[False, True])
    fit = smf.logit("is_correct ~ difficulty * is_model_b", data=sub).fit(disp=0)
    interaction_term = fit.params.index[-1]
    return fit.params[interaction_term], fit.pvalues[interaction_term]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--condition", nargs=2, action="append", metavar=("LABEL", "CSV_GLOB"), required=True,
        help="A condition label and its analysis CSV glob. Repeat for each condition.",
    )
    args = parser.parse_args()

    conditions = {label: load(csv_glob) for label, csv_glob in args.condition}

    print("=== Accuracy summary: model x condition ===\n")
    for label, df in conditions.items():
        table = df.pivot_table(index="model", columns="difficulty", values="is_correct", aggfunc="mean")
        table.index = [short_name(m) for m in table.index]
        print(f"-- {label} --")
        print(table.round(2).to_string())
        print()

    print("=== Each model's own difficulty slope, per condition ===\n")
    for label, df in conditions.items():
        print(f"-- {label} --")
        for model in sorted(df["model"].unique()):
            coef, p = own_slope_test(df, model)
            verdict = "declines significantly" if (p < 0.05 and coef < 0) else "no significant trend"
            print(f"  {short_name(model):40s} coef={coef:+.4f} p={p:.4f}  ({verdict})")
        print()

    print("=== Pairwise model comparisons within each condition (does one decline faster?) ===\n")
    for label, df in conditions.items():
        print(f"-- {label} --")
        models = sorted(df["model"].unique())
        for model_a, model_b in itertools.combinations(models, 2):
            coef, p = pairwise_slope_test(df, model_a, model_b)
            verdict = "SIGNIFICANT slope difference" if p < 0.05 else "not significant"
            print(f"  {short_name(model_a):30s} vs {short_name(model_b):30s} p={p:.4f}  ({verdict})")
        print()

    print("=== Does each model's slope differ across conditions? ===\n")
    all_models = set()
    for df in conditions.values():
        all_models.update(df["model"].unique())
    condition_pairs = list(itertools.combinations(conditions.keys(), 2))
    for model in sorted(all_models):
        present_in = [label for label, df in conditions.items() if model in df["model"].values]
        if len(present_in) < 2:
            continue
        print(f"-- {short_name(model)} --")
        for label_a, label_b in condition_pairs:
            if label_a not in present_in or label_b not in present_in:
                continue
            df_a = conditions[label_a][conditions[label_a]["model"] == model].copy()
            df_a["condition"] = label_a
            df_b = conditions[label_b][conditions[label_b]["model"] == model].copy()
            df_b["condition"] = label_b
            combined = pd.concat([df_a, df_b], ignore_index=True)
            combined["condition"] = pd.Categorical(combined["condition"], categories=[label_a, label_b])
            fit = smf.logit("is_correct ~ difficulty * condition", data=combined).fit(disp=0)
            interaction_term = fit.params.index[-1]
            p = fit.pvalues[interaction_term]
            verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
            print(f"  {label_a:15s} vs {label_b:15s} p={p:.4f}  ({verdict})")
        print()


if __name__ == "__main__":
    main()
