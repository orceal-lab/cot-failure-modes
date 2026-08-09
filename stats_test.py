"""Formally test whether a model's accuracy-vs-difficulty slope differs between
two conditions (e.g. running-total vs. independent-list framing), instead of
eyeballing whether confidence intervals overlap.

Fits, per model: is_correct ~ difficulty * condition (logistic regression).
The difficulty:condition interaction term answers "does the decline rate
itself differ between conditions?" — a small/insignificant interaction means
the two conditions decline at statistically indistinguishable rates, even if
their absolute accuracy differs.

Usage:
    python stats_test.py --a results/steps/analysis_*.csv --a-label running_total \
                          --b results/independent/analysis_*.csv --b-label independent
"""

import argparse
import glob

import pandas as pd
import statsmodels.formula.api as smf


def load(paths_glob: str) -> pd.DataFrame:
    paths = glob.glob(paths_glob)
    if not paths:
        raise SystemExit(f"No files matched: {paths_glob}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="Analysis CSV(s) for condition A (glob ok)")
    parser.add_argument("--a-label", default="condition_a", help="Label for condition A")
    parser.add_argument("--b", required=True, help="Analysis CSV(s) for condition B (glob ok)")
    parser.add_argument("--b-label", default="condition_b", help="Label for condition B")
    args = parser.parse_args()

    df_a = load(args.a)
    df_a["condition"] = args.a_label
    df_b = load(args.b)
    df_b["condition"] = args.b_label

    df = pd.concat([df_a, df_b], ignore_index=True)
    df["model_short"] = df["model"].str.extract(r":([^:]+(?::[^:]+)?)$").iloc[:, 0].fillna(df["model"])
    df["is_correct"] = df["is_correct"].astype(int)

    print(f"=== Interaction test: does the difficulty slope differ between '{args.a_label}' and '{args.b_label}'? ===\n")
    for model, sub in df.groupby("model_short"):
        sub = sub.copy()
        sub["condition"] = pd.Categorical(sub["condition"], categories=[args.b_label, args.a_label])
        try:
            fit = smf.logit("is_correct ~ difficulty * condition", data=sub).fit(disp=0)
        except Exception as e:
            print(f"{model}: could not fit model ({e})")
            continue
        interaction_term = fit.params.index[-1]
        coef = fit.params[interaction_term]
        p = fit.pvalues[interaction_term]
        verdict = "SIGNIFICANT (p<0.05) — slopes genuinely differ" if p < 0.05 else "not significant — slopes are statistically indistinguishable"
        print(f"{model} (n={len(sub)}): interaction coef={coef:+.4f}, p={p:.4f} — {verdict}")


if __name__ == "__main__":
    main()
