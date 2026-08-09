"""Generate the arithmetic_independent problem set — the isolation control for
arithmetic_steps.json.

Same scenario families, same step counts, same per-step deltas/vocabulary,
but framed as an enumerated list of independent day-by-day changes rather
than a sequential "Then... then..." narrative. Mathematically identical
(same numbers, same final answer computation), but a model doesn't need to
carry a running total across N sequential steps — it can sum N independent
terms in any order, so a slip on one term doesn't corrupt every term after
it. If mistral's accuracy stays flat here while still declining on the
running-total version, the failure is state-tracking specific, not raw
step-count/length. If both decline the same way, it's chain length itself.

Usage:
    python problems/generate_arithmetic_independent.py [--per-level N] [--seed S]
Writes problems/arithmetic_independent.json.
"""

import argparse
import json
import random

from generate_arithmetic_steps import FAMILIES, STEP_LEVELS

INDEPENDENT_QUESTIONS = {
    "money": "After all of the changes listed above, how much money does Sam have in total?",
    "warehouse": "After all of the changes listed above, how many boxes are in the warehouse in total?",
    "truck": "After all of the changes listed above, how many total miles has he driven?",
    "bakery": "After all of the changes listed above, how many cookies do they have in total?",
    "tank": "After all of the changes listed above, how many liters are in the tank in total?",
}

STRIP_PREFIXES = ["Then ", "He then "]


def clean_lead(sentence: str) -> str:
    for prefix in STRIP_PREFIXES:
        if sentence.startswith(prefix):
            sentence = sentence[len(prefix):]
            break
    return sentence[0].upper() + sentence[1:] if sentence else sentence


def generate_problem(family_name, n_steps, rng):
    family = FAMILIES[family_name]
    start = rng.randint(*family["start_range"])
    value = start
    chain = []
    day_lines = []

    for day in range(1, n_steps + 1):
        for attempt in range(2):
            sign = rng.choice([1, -1])
            if attempt == 1:
                sign = 1
            lo, hi = family["pos_range"] if sign == 1 else family["neg_range"]
            d = rng.randint(lo, hi)
            if value + sign * d >= 0:
                break
        template = rng.choice(family["pos_templates"] if sign == 1 else family["neg_templates"])
        value += sign * d
        chain.append(value)
        day_lines.append(f"Day {day}: {clean_lead(template.format(d=d))}")

    intro = family["intro"].format(start=start)
    body = " ".join(day_lines)
    prompt = f"{intro} {body} {INDEPENDENT_QUESTIONS[family_name]}"
    return prompt, chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-level", type=int, default=4, help="Variants per family per step level (default 4; x5 families = 20/level)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--out", default="problems/arithmetic_independent.json", help="Output path")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    problems = []

    for steps in STEP_LEVELS:
        for family_name in FAMILIES:
            for variant in range(args.per_level):
                prompt, chain = generate_problem(family_name, steps, rng)
                problems.append(
                    {
                        "id": f"arith_ind_{steps}_{family_name}_{variant}",
                        "category": "arithmetic_independent",
                        "difficulty": steps,
                        "prompt": prompt,
                        "answer": chain[-1],
                        "reasoning_chain": chain,
                    }
                )

    with open(args.out, "w") as f:
        json.dump(problems, f, indent=2)

    print(f"Generated {len(problems)} problems ({args.per_level * len(FAMILIES)} per level x {len(STEP_LEVELS)} levels) -> {args.out}")


if __name__ == "__main__":
    main()
