"""Generate the arithmetic_singles problem set — the volume-hypothesis test.

arithmetic_independent.json removed the running-total requirement but still
shared one starting scenario/quantity across all N steps. This set goes
further: N fully independent single-operation questions, no shared starting
value, no narrative connection between them at all — just N separate
"start, then one operation" facts, asked together and summed at the end.

If mistral's decline is really about the *volume* of correct operations
required in one response (not narrative structure, not even a loosely shared
scenario), its accuracy should still decline with N here. If accuracy holds
up on this version specifically, that would point back toward *some* kind of
structural/framing effect after all.

Usage:
    python problems/generate_arithmetic_singles.py [--per-level N] [--seed S]
Writes problems/arithmetic_singles.json.
"""

import argparse
import json
import random

from generate_arithmetic_steps import FAMILIES, STEP_LEVELS

# One independent fact per family: a start value, one operation, phrased as
# a fully self-contained single-step question (no "Day n" framing, no shared
# quantity across facts).
FACT_TEMPLATES = {
    "money": "{who} has ${start}. {verb} ${d}.",
    "warehouse": "A warehouse has {start} boxes. {verb} {d} boxes.",
    "truck": "A truck has driven {start} miles. {verb} {d} miles.",
    "bakery": "A bakery has {start} cookies. {verb} {d} cookies.",
    "tank": "A tank has {start} liters. {verb} {d} liters.",
}

POS_VERBS = {
    "money": "Then they earn",
    "warehouse": "Then they receive",
    "truck": "Then it drives",
    "bakery": "Then they bake",
    "tank": "Then it gains",
}
NEG_VERBS = {
    "money": "Then they spend",
    "warehouse": "Then they ship out",
    "truck": "Then it saves via a shortcut",
    "bakery": "Then they sell",
    "tank": "Then it loses",
}


def generate_fact(family_name, rng):
    family = FAMILIES[family_name]
    start = rng.randint(*family["start_range"])
    for attempt in range(2):
        sign = rng.choice([1, -1])
        if attempt == 1:
            sign = 1
        lo, hi = family["pos_range"] if sign == 1 else family["neg_range"]
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
    parser.add_argument("--out", default="problems/arithmetic_singles.json", help="Output path")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    problems = []

    for n_facts in STEP_LEVELS:
        for variant in range(args.per_level):
            prompt, running_sum = generate_problem(n_facts, rng)
            problems.append(
                {
                    "id": f"arith_single_{n_facts}_{variant}",
                    "category": "arithmetic_singles",
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
