# cot-failure-modes

A small research prototype for systematically testing where chain-of-thought
(CoT) reasoning breaks down in LLMs, using hand-built problem sets with
controlled variations (e.g. the same underlying logic at increasing reasoning-chain
lengths).

## Project structure

```
problems/            Problem sets (JSON), each with id/category/difficulty/prompt/answer
models/               Modular model-calling clients (Anthropic API, Ollama)
run_experiment.py     Sends problems to a model, saves raw responses to results/
analyze.py            Scores responses and categorizes *where* reasoning failed
report.py             Generates plots (accuracy vs. difficulty, failure breakdown)
results/              Raw responses, analysis CSVs, plots (gitignored, not checked in)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

For local models, install [Ollama](https://ollama.com), run `ollama serve`,
and pull a model, e.g. `ollama pull llama3`.

Never commit `.env` or real API keys — `.gitignore` already excludes it.

## Running an experiment

```bash
# Anthropic API
python run_experiment.py --model anthropic:claude-haiku-4-5-20251001

# Local Ollama model
python run_experiment.py --model ollama:llama3
```

This loads `problems/arithmetic_steps.json` by default (override with
`--problems`), sends each problem with a chain-of-thought instruction, and
writes raw responses to `results/raw_<model>_<timestamp>.json`.

## Analyzing results

```bash
python analyze.py results/raw_*.json
```

Extracts each model's final answer, scores correctness, and (heuristically,
by default) categorizes *why* incorrect responses failed:

- `misread_problem` — none of the expected intermediate values ever appear;
  the model likely misunderstood the setup
- `lost_track_of_state` — got the early steps right, then the chain breaks
- `arithmetic_slip` — tracked everything correctly, final answer is off by a
  small amount
- `wrong_final_step` — the whole chain is right except the very last step
- `no_answer_extracted` — couldn't find a parseable final answer

Add `--llm-classify` to use a second Anthropic call for classification
instead of the heuristics (slower, costs tokens, but often more accurate).

Output: `results/analysis_<timestamp>.csv`.

## Generating plots

```bash
python report.py results/analysis_*.csv
```

Saves `accuracy_vs_difficulty_<timestamp>.png` and
`failure_categories_<timestamp>.png` to `results/`.

## Adding new problem sets

Add a new JSON file to `problems/` with a list of objects:

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

## Seed problem set: `arithmetic_steps.json`

20 multi-step arithmetic word problems: 5 structurally-identical scenario
families (money, warehouse inventory, truck mileage, bakery, water tank),
each instantiated at 2, 4, 6, and 8 reasoning steps. Each higher difficulty
level extends the same scenario with additional chained operations, so
accuracy can be compared directly across chain length while holding the
"logic shape" fixed.
