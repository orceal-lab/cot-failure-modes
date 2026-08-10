"""Ceiling-effect check for the volume hypothesis: same structure as
arithmetic_singles.json (N fully independent single-operation facts, summed),
but with numbers scaled up roughly 100x (3-4 digit arithmetic instead of
2-digit) so genuine arithmetic difficulty increases while N (the volume
variable) stays identical.

qwen3:4b showed no significant accuracy decline in any condition of the
original study, including singles -- flat at 95-100% across every difficulty
level. Two explanations: (1) it's near a hard ceiling on arithmetic this
simple regardless of volume, or (2) something about the singles format
specifically suits it. This set tests (1) directly: if the ceiling is really
about problem *simplicity* rather than the volume manipulation, harder raw
arithmetic should push accuracy down even at low N, independent of chain
length.

Usage:
    python problems/generate_arithmetic_singles_hard.py [--per-level N] [--seed S]
Writes problems/arithmetic_singles_hard.json.
"""

import argparse
import json
import random

from generate_arithmetic_steps import FAMILIES, STEP_LEVELS
from generate_arithmetic_singles import FACT_TEMPLATES, POS_VERBS, NEG_VERBS

# Same families/templates as arithmetic_singles.json, ~100x larger magnitudes.
HARD_RANGES = {
    "money": {"start_range": (4000, 9000), "pos_range": (300, 2500), "neg_range": (300, 2500)},
    "warehouse": {"start_range": (4000, 9000), "pos_range": (300, 2500), "neg_range": (300, 2500)},
    "truck": {"start_range": (4000, 9000), "pos_range": (300, 2500), "neg_range": (300, 2500)},
    "bakery": {"start_range": (4000, 9000), "pos_range": (300, 2500), "neg_range": (300, 2500)},
    "tank": {"start_range": (4000, 9000), "pos_range": (300, 2500), "neg_range": (300, 2500)},
}


def generate_fact(family_name, rng):
    ranges = HARD_RANGES[family_name]
    start = rng.randint(*ranges["start_range"])
    for attempt in range(2):
        sign = rng.choice([1, -1])
        if attempt == 1:
            sign = 1
        lo, hi = ranges["pos_range"] if sign == 1 else ranges["neg_range"]
        d = rng.randint(lo, hi)
        if start + sign * d >= 0:
            break
    verb = POS_VERBS[family_name] if sign == 1 else NEG_VERBS[family_name]
    text = FACT_TEMPLATES[family_name].format(who="Sam", start=start, verb=verb, d=d)
    result = start + sign * d
    return text, result


def generate_problem(n_facts, rng):
    families = list(FAMILIES.keys())
    facts = []
    results = []
    for i in range(n_facts):
        family_name = families[i % len(families)]
        text, result = generate_fact(family_name, rng)
        facts.append(f"({i + 1}) {text}")
        results.append(result)

    body = " ".join(facts)
    prompt = (
        f"Here are {n_facts} separate, unrelated situations. Work out the final "
        f"number in each one, then add all {n_facts} of those numbers together. "
        f"{body} What is the sum of all {n_facts} results?"
    )
    running_sum = []
    total = 0
    for r in results:
        total += r
        running_sum.append(total)
    return prompt, running_sum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-level", type=int, default=20, help="Variants per step level (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--out", default="problems/arithmetic_singles_hard.json", help="Output path")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    problems = []

    for n_facts in STEP_LEVELS:
        for variant in range(args.per_level):
            prompt, running_sum = generate_problem(n_facts, rng)
            problems.append(
                {
                    "id": f"arith_single_hard_{n_facts}_{variant}",
                    "category": "arithmetic_singles_hard",
                    "difficulty": n_facts,
                    "prompt": prompt,
                    "answer": running_sum[-1],
                    "reasoning_chain": running_sum,
                }
            )

    with open(args.out, "w") as f:
        json.dump(problems, f, indent=2)

    print(f"Generated {len(problems)} problems ({args.per_level} per level x {len(STEP_LEVELS)} levels) -> {args.out}")


if __name__ == "__main__":
    main()
