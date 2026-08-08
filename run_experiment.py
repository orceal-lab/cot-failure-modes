"""Send problems to one or more LLMs and save the raw responses.

Usage:
    python run_experiment.py --model anthropic:claude-haiku-4-5-20251001
    python run_experiment.py --model ollama:llama3 --problems problems/arithmetic_steps.json
    python run_experiment.py --model ollama:qwen3:4b ollama:mistral:7b-instruct
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

from models import get_client

COT_INSTRUCTION = (
    "Think through this problem step by step, showing your reasoning for "
    "each step. On the final line, state your answer in the exact form:\n"
    "Final Answer: <number>\n\n"
    "Problem: {prompt}"
)


def load_problems(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run(model_spec: str, problems_path: str, results_dir: str) -> str:
    load_dotenv()
    client = get_client(model_spec)
    problems = load_problems(problems_path)

    records = []
    for i, problem in enumerate(problems, start=1):
        print(f"[{i}/{len(problems)}] {problem['id']} (model={model_spec})", file=sys.stderr)
        prompt = COT_INSTRUCTION.format(prompt=problem["prompt"])
        response_text = client.generate(prompt)
        records.append(
            {
                "problem_id": problem["id"],
                "category": problem["category"],
                "difficulty": problem["difficulty"],
                "prompt_sent": prompt,
                "correct_answer": problem["answer"],
                "reasoning_chain": problem.get("reasoning_chain", []),
                "raw_response": response_text,
                "model": model_spec,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    os.makedirs(results_dir, exist_ok=True)
    safe_model = model_spec.replace(":", "-").replace("/", "-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(results_dir, f"raw_{safe_model}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} responses to {out_path}", file=sys.stderr)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Model spec as 'provider:model_name', e.g. anthropic:claude-haiku-4-5-20251001 or ollama:llama3",
    )
    parser.add_argument(
        "--problems",
        default="problems/arithmetic_steps.json",
        help="Path to a problems JSON file (default: problems/arithmetic_steps.json)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to write raw response JSON to (default: results)",
    )
    args = parser.parse_args()
    run(args.model, args.problems, args.results_dir)


if __name__ == "__main__":
    main()
