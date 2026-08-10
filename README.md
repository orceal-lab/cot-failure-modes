# cot-failure-modes

A small research project testing where chain-of-thought (CoT) reasoning
breaks down in LLMs, using programmatically-generated problem sets with a
controlled variable (number of chained reasoning steps) and, where possible,
statistical tests rather than eyeballed curves.

Current status and findings: see the latest report (link in project notes)
or reproduce with `compare_models.py` / `full_comparison.py` below. Short
version: across 4 models (mistral:7b-instruct, qwen2.5-coder:7b, llama3.1:8b,
qwen3:4b) and 3 problem-set conditions, own-accuracy declines with difficulty
are the most robust finding — but most cross-model "who declines faster"
claims don't survive correcting for the number of tests run, with two
exceptions (mistral vs. llama3.1:8b and mistral vs. qwen3:4b, both in the
running-total condition). qwen3:4b stands out: its own difficulty slope is
not significant in *any* condition, including the zero-shared-context
condition where every other model tested declines significantly. See
`results/` and the published report for the full statistical breakdown.

## Project structure

```text
problems/
  arithmetic_steps.json            Running-total problem set (generated)
  arithmetic_independent.json      Isolation-control problem set (generated)
  generate_arithmetic_steps.py     Generator for the running-total set
  generate_arithmetic_independent.py  Generator for the independent-list set
models/                            Modular model-calling clients (Anthropic API, Ollama)
run_experiment.py                  Sends problems to one or more models, saves raw responses
analyze.py                         Scores responses, extracts answers, heuristic failure categories
compare_models.py                  Cross-model accuracy/format-compliance tables, plots, Wilson CIs
consistency_report.py              Per-problem consistency from repeated-sampling runs
stats_test.py                      Formal logistic-regression interaction test, two conditions or two models
full_comparison.py                 Same, generalized to N conditions x N models in one pass
report.py                          Single-model accuracy/failure-category plots
kaggle_run/                        Notebooks + kernel-metadata for running experiments on Kaggle
results/                           Raw responses, analysis CSVs, plots (gitignored, not checked in)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY, only needed for the Anthropic client
```

For local models, install [Ollama](https://ollama.com), run `ollama serve`,
and pull a model, e.g. `ollama pull mistral:7b-instruct-q4_K_M`. Everything
in this project has been run against local Ollama models (no API key
required) — see `kaggle_run/` to run them on a free Kaggle GPU instead of
locally.

Never commit `.env` or real API keys — `.gitignore` already excludes it.

## Running on Kaggle (recommended — keeps your own machine free)

`kaggle_run/cot-failure-modes-run.ipynb` + `kernel-metadata.json` install
Ollama, pull models, run both problem sets, and analyze/compare results, all
inside a Kaggle notebook with a free GPU. Push and run with the Kaggle CLI
(requires `~/.kaggle/kaggle.json`):

```bash
cd kaggle_run
kaggle kernels push -p .
kaggle kernels status atharvranjan/cot-failure-modes-run
kaggle kernels output atharvranjan/cot-failure-modes-run -p ./output   # once complete
```

`kaggle_run/consistency_run/` is a smaller, dedicated notebook for repeated-
sampling consistency checks (see below).

## Running an experiment locally

```bash
# Anthropic API
python run_experiment.py --model anthropic:claude-haiku-4-5-20251001

# Local Ollama model(s) — pass multiple --model values to run several in one pass
python run_experiment.py --model ollama:mistral:7b-instruct-q4_K_M ollama:qwen2.5-coder:7b-instruct-q4_K_M
```

Defaults to `problems/arithmetic_steps.json` (override with `--problems`),
sends each problem with a chain-of-thought instruction, and writes raw
responses to `results/raw_<model>_<problems-stem>_<timestamp>.json`.

Add `--repeats N` to ask each problem N times (tagged with an `attempt`
index) — needed to distinguish "this model is unreliable" from "this model
is reliably wrong on hard problems," which a single sample per problem can't
do. Feed the resulting analysis CSV to `consistency_report.py`.

## Analyzing results

```bash
python analyze.py results/raw_*.json
```

Extracts each model's final answer (preferring the *last* `Final Answer:`
line in the response — some models echo it after every step, not just the
end), scores correctness, and (heuristically, by default) categorizes *why*
incorrect responses failed:

- `misread_problem` — none of the expected intermediate values ever appear;
  usually a genuine misunderstanding, but can also be a pure arithmetic slip
  on an otherwise-correct approach if the model never narrates intermediate
  totals. Treat as "early divergence," not a precise diagnosis.
- `lost_track_of_state` — got the early steps right, then the chain breaks —
  can also be a sign/direction error on a single operation, not necessarily
  "forgot the running total." Same caveat as above.
- `arithmetic_slip` — tracked everything correctly, final answer is off by a
  small amount. **In practice this almost never fires** (0/480 graded
  responses across every run so far) — it requires every intermediate value
  except the last to appear verbatim in the text, which real model output
  essentially never does (models write one-shot expressions, skip narrating
  steps, or format numbers differently). Treat it as dead code, not a
  meaningful zero.
- `wrong_final_step` — the whole chain is right except the very last step
- `no_answer_extracted` — couldn't find a parseable final answer

These heuristic labels are directionally useful in aggregate (e.g. "which
category dominates at high difficulty") but have not been validated against
human review at scale — a manual audit of ~13 transcripts found real but
forgivable ambiguity between categories. Don't treat any single label as
ground truth.

Add `--llm-classify` to use a second Anthropic call for classification
instead of the heuristics (slower, costs tokens, needs an API key).

Output: `results/analysis_<models>_<timestamp>.csv`.

## Comparing models

```bash
python compare_models.py results/analysis_*.csv
```

Prints accuracy and format-compliance tables by model x step count, with 95%
Wilson confidence intervals, plus a plain-text readout flagging which models
(if any) show a real accuracy drop vs. sampling noise. Saves comparison CSVs
and plots.

## Formal statistical tests

```bash
python stats_test.py --a results/steps/analysis_*.csv --a-label running_total \
                      --b results/independent/analysis_*.csv --b-label independent
```

Fits `is_correct ~ difficulty * condition` (logistic regression) per model
and reports whether the difficulty slope actually differs between two
conditions — a real test, not eyeballed confidence-interval overlap. Swap
`--a`/`--b` for two model subsets of the same CSV to test model-vs-model
slope differences instead of condition-vs-condition.

For more than two conditions and/or more than two models at once, use
`full_comparison.py` instead — it's what `stats_test.py` calls under the
hood, generalized:

```bash
python full_comparison.py \
    --condition running_total results/steps/analysis_*.csv \
    --condition independent results/independent/analysis_*.csv \
    --condition singles results/singles/analysis_*.csv
```

Prints, for every condition supplied: an accuracy table, each model's own
difficulty-slope significance, every model-pair's slope-difference
significance within that condition, and (for any model present in more than
one condition) whether its slope differs across every pair of conditions.
One command instead of hand-writing a new pandas/statsmodels snippet for
every new comparison.

### Correcting for how many tests you've run

The more significance tests a project runs, the less a single raw p<0.05
means. Once you've run more than a couple, pool every test's p-value and
correct:

```bash
python correct_multiple_comparisons.py \
    --own-slope "results/full_comparison/own_slope_*.csv" \
    --pairwise "results/full_comparison/pairwise_*.csv" \
    --cross-condition "results/full_comparison/cross_condition_*.csv" \
    --extra "some one-off test name" 0.03
```

Applies a Benjamini-Hochberg false-discovery-rate correction across
everything and reports which results actually survive. In this project,
running it across all 22 tests accumulated so far dropped several
previously-"confirmed" results below the line — see the report for the
full story. Use `--extra NAME P_VALUE` (repeatable) for tests not covered
by `full_comparison.py`'s saved CSVs, like the format-compliance or
consistency checks.

## Per-problem consistency

```bash
python run_experiment.py --repeats 3 --model ollama:mistral:7b-instruct-q4_K_M
python analyze.py results/raw_*_x3_*.json --results-dir results/consistency
python consistency_report.py "results/consistency/analysis_*.csv"
```

Separates "unreliable" (accuracy varies across repeated attempts on the same
problem) from "reliably wrong on hard problems" (accuracy is consistently 0
or 1 per problem) — a distinction a single-sample-per-problem run can't make.

## Generating plots (single model)

```bash
python report.py results/analysis_*.csv
```

Saves `accuracy_vs_difficulty_<timestamp>.png` and
`failure_categories_<timestamp>.png` to `results/`.

## Problem sets

All three are generated programmatically (not hand-written) so sample size
can be scaled without linearly scaling authoring effort:

```bash
python problems/generate_arithmetic_steps.py --per-level 4 --seed 42       # 5 families x 4 variants = 20/level
python problems/generate_arithmetic_independent.py --per-level 4 --seed 42
python problems/generate_arithmetic_singles.py --per-level 20 --seed 42
```

- **`arithmetic_steps.json`** — 5 scenario families (money, warehouse
  inventory, truck mileage, bakery, water tank), each instantiated at 2, 4,
  6, 8, 10, 12 chained steps, 20 variants per step level (120 total). Each
  step is a "Then..." narrative sentence that updates a running total —
  answering correctly requires carrying that total across every step.
- **`arithmetic_independent.json`** — same families, same deltas, same RNG
  seed (so the underlying numbers match `arithmetic_steps.json` one-to-one),
  but reframed as an enumerated list of independent day-by-day changes.
  Mathematically identical, but doesn't require tracking a running total
  narratively — used as an isolation control to test whether a model's
  failure is state-tracking-specific or just a function of chain length /
  operation count.
- **`arithmetic_singles.json`** — goes one step further than the independent
  set: N fully independent single-operation facts with *no* shared starting
  quantity or scenario at all (not even a loosely shared narrative), summed
  at the end. Tests the "volume" hypothesis directly — if a model's decline
  is really about the count of correct operations required in one response
  rather than any structural/framing effect, accuracy should still drop with
  N here.

To add a new problem set by hand instead, use the same schema:

```json
{
  "id": "unique_id",
  "category": "your_category",
  "difficulty": 2,
  "prompt": "...",
  "answer": 42,
  "reasoning_chain": [/* optional: expected intermediate values, used by the heuristic classifier */]
}
```

## Adding new models

Add a client in `models/` implementing `ModelClient.generate(prompt) -> str`
(see `models/anthropic_client.py` / `models/ollama_client.py`), then register
it in `models/__init__.py::get_client`.
