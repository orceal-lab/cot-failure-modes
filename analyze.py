"""Score raw model responses and categorize *where* failures happened.

Usage:
    python analyze.py results/raw_anthropic-claude-...json
    python analyze.py results/raw_*.json --llm-classify   # use a second LLM call instead of heuristics

Failure categories (heuristic mode):
    correct             - final answer matches ground truth
    no_answer_extracted - couldn't find a "Final Answer: <number>" line
    misread_problem     - none of the expected intermediate values appear anywhere
                           in the response; the model likely misread the setup
    lost_track_of_state - early intermediate values are correct but the chain
                           breaks before the end (model lost the running total)
    arithmetic_slip     - all intermediate values look right but the final
                           number is off by a small amount (a slip, not a
                           conceptual error)
    wrong_final_step    - the full chain up to the second-to-last value is
                           present, but the final step/answer is wrong
"""

import argparse
import glob
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd

FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*\$?(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(response_text: str) -> tuple[float | None, str]:
    """Return (value, method). Prefers the LAST explicit 'Final Answer:' line
    — some models (mistral, notably) echo "Final Answer: <running total>"
    after every step rather than only at the end, so taking the first match
    grabs an early intermediate value instead of the model's actual final
    answer. Falls back to the last number mentioned anywhere in the response,
    since models frequently state the correct total in a closing sentence
    without following the requested format at all."""
    matches = FINAL_ANSWER_RE.findall(response_text)
    if matches:
        try:
            return float(matches[-1].replace(",", "")), "explicit_final_answer"
        except ValueError:
            pass

    all_numbers = NUMBER_RE.findall(response_text)
    if all_numbers:
        try:
            return float(all_numbers[-1].replace(",", "")), "fallback_last_number"
        except ValueError:
            pass

    return None, "none"


def numbers_in_text(text: str) -> set[float]:
    found = set()
    for m in NUMBER_RE.findall(text):
        try:
            found.add(float(m.replace(",", "")))
        except ValueError:
            continue
    return found


def categorize_heuristic(
    response_text: str,
    reasoning_chain: list[float],
    correct_answer: float,
    extracted_answer: float | None,
) -> str:
    if extracted_answer is not None and extracted_answer == correct_answer:
        return "correct"
    if extracted_answer is None:
        return "no_answer_extracted"
    if not reasoning_chain:
        return "wrong_final_step"

    present = numbers_in_text(response_text)
    # How far into the expected chain does the model track correctly before diverging?
    matched_prefix = 0
    for value in reasoning_chain:
        if value in present:
            matched_prefix += 1
        else:
            break

    if matched_prefix == 0:
        return "misread_problem"
    if matched_prefix < len(reasoning_chain) - 1:
        return "lost_track_of_state"

    # Model tracked (almost) the whole chain correctly but the final answer is wrong.
    error_margin = abs(extracted_answer - correct_answer)
    if error_margin <= max(1, 0.05 * abs(correct_answer)):
        return "arithmetic_slip"
    return "wrong_final_step"


LLM_CLASSIFY_PROMPT = """A model was given this word problem and produced the reasoning below. \
The correct final answer is {correct_answer}, but the model answered {extracted_answer}.

Classify the failure into exactly one of these categories, and reply with just the category name:
- misread_problem (misunderstood the setup or a stated quantity)
- lost_track_of_state (started correctly but lost the running total partway through)
- arithmetic_slip (correct approach and tracking, but a computational slip)
- wrong_final_step (correct up to the last step, but the final operation/answer is wrong)
- other (doesn't fit the above)

Problem: {prompt}

Model's response:
{response}

Category:"""


def categorize_llm(client, record: dict, extracted_answer: float | None) -> str:
    prompt = LLM_CLASSIFY_PROMPT.format(
        correct_answer=record["correct_answer"],
        extracted_answer=extracted_answer,
        prompt=record["prompt_sent"],
        response=record["raw_response"],
    )
    reply = client.generate(prompt).strip().lower()
    for category in (
        "misread_problem",
        "lost_track_of_state",
        "arithmetic_slip",
        "wrong_final_step",
        "other",
    ):
        if category in reply:
            return category
    return "other"


def analyze(records: list[dict], use_llm: bool) -> pd.DataFrame:
    llm_client = None
    if use_llm:
        from dotenv import load_dotenv

        from models.anthropic_client import AnthropicClient

        load_dotenv()
        llm_client = AnthropicClient("claude-haiku-4-5-20251001")

    rows = []
    for record in records:
        extracted, extraction_method = extract_answer(record["raw_response"])
        correct = extracted is not None and extracted == record["correct_answer"]

        if correct:
            failure_category = "correct"
        elif use_llm:
            failure_category = categorize_llm(llm_client, record, extracted)
        else:
            failure_category = categorize_heuristic(
                record["raw_response"],
                record.get("reasoning_chain", []),
                record["correct_answer"],
                extracted,
            )

        rows.append(
            {
                "problem_id": record["problem_id"],
                "category": record["category"],
                "difficulty": record["difficulty"],
                "model": record["model"],
                "attempt": record.get("attempt", 1),
                "correct_answer": record["correct_answer"],
                "extracted_answer": extracted,
                "extraction_method": extraction_method,
                "is_correct": correct,
                "failure_category": failure_category,
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_files", nargs="+", help="Raw response JSON file(s) from run_experiment.py (globs ok)")
    parser.add_argument("--llm-classify", action="store_true", help="Use a second Anthropic call to classify failures")
    parser.add_argument("--results-dir", default="results", help="Directory to write the analysis CSV to")
    args = parser.parse_args()

    paths = []
    for pattern in args.raw_files:
        paths.extend(glob.glob(pattern))
    if not paths:
        raise SystemExit("No raw result files matched.")

    all_records = []
    for path in paths:
        with open(path) as f:
            all_records.extend(json.load(f))

    df = analyze(all_records, use_llm=args.llm_classify)

    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    models = sorted(df["model"].unique())
    safe_models = "-".join(m.replace(":", "-").replace("/", "-") for m in models)
    if len(safe_models) > 80:  # avoid unwieldy filenames when analyzing many models at once
        safe_models = f"{len(models)}models"
    out_path = os.path.join(args.results_dir, f"analysis_{safe_models}_{timestamp}.csv")
    df.to_csv(out_path, index=False)

    print(df.groupby("difficulty")["is_correct"].mean())
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    main()
