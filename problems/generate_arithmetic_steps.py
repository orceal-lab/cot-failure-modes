"""Programmatically generate the arithmetic_steps problem set.

Replaces the original 5 hand-written problems/level with many randomized
variants per (scenario family, step count), so accuracy curves rest on
enough samples to trust their shape, not just 5 problems/level. Each
variant keeps the same narrative template per family (only the numbers
and which template sentence is picked vary) — the controlled variable
stays "number of chained steps", not problem content.

Usage:
    python problems/generate_arithmetic_steps.py [--per-level N] [--seed S]
Writes problems/arithmetic_steps.json.
"""

import argparse
import json
import random

STEP_LEVELS = [2, 4, 6, 8, 10, 12]

FAMILIES = {
    "money": {
        "intro": "Sam has ${start}.",
        "start_range": (40, 90),
        "unit": "$",
        "pos_range": (3, 25),
        "neg_range": (3, 25),
        "pos_templates": [
            "Then she earns ${d} doing a chore.",
            "Then she finds ${d} on the ground.",
            "Then her friend repays her ${d} that she was owed.",
            "Then she gets a ${d} birthday gift from her uncle.",
            "Then she earns ${d} babysitting.",
            "Then she sells an old game for ${d}.",
        ],
        "neg_templates": [
            "She spends ${d} on lunch.",
            "She buys a book for ${d}.",
            "She pays ${d} for parking.",
            "She spends ${d} on a movie ticket.",
            "She donates ${d} to a fundraiser.",
            "She buys a coffee for ${d}.",
        ],
        "question": "How much money does she have now?",
    },
    "warehouse": {
        "intro": "A warehouse has {start} boxes.",
        "start_range": (90, 180),
        "unit": "",
        "pos_range": (15, 65),
        "neg_range": (15, 65),
        "pos_templates": [
            "Then {d} new boxes arrive.",
            "Then {d} boxes arrive from a returned order.",
            "Then {d} more boxes arrive from a new shipment.",
            "Then {d} boxes are returned by customers.",
            "Then {d} new boxes arrive from the supplier.",
            "Then {d} boxes arrive from overseas.",
        ],
        "neg_templates": [
            "{d} boxes are shipped out.",
            "{d} boxes are found damaged and discarded.",
            "{d} boxes are shipped to a store.",
            "{d} boxes are donated.",
            "{d} boxes are shipped to a partner store.",
            "{d} boxes are recalled for inspection.",
        ],
        "question": "How many boxes are in the warehouse now?",
    },
    "truck": {
        "intro": "A truck driver drives {start} miles on the first leg of the trip.",
        "start_range": (60, 110),
        "unit": "",
        "pos_range": (10, 45),
        "neg_range": (5, 20),
        "pos_templates": [
            "Then he drives {d} more miles for a delivery.",
            "He then takes a detour that adds {d} miles.",
            "Then he drives {d} more miles to reach the destination.",
            "He then drives another {d} miles for a second delivery.",
            "Then he drives {d} more miles to pick up cargo.",
        ],
        "neg_templates": [
            "He then takes a shortcut that saves {d} miles.",
            "He backtracks {d} miles to pick up forgotten cargo.",
        ],
        "question": "How many total miles has he driven?",
    },
    "bakery": {
        "intro": "A bakery makes {start} cookies in the morning.",
        "start_range": (100, 180),
        "unit": "",
        "pos_range": (15, 55),
        "neg_range": (15, 55),
        "pos_templates": [
            "Then they bake {d} more cookies.",
            "Then they bake {d} more cookies for the next day.",
            "Then they bake {d} more cookies for a large order.",
            "Then they bake {d} more cookies for the next batch.",
        ],
        "neg_templates": [
            "They sell {d} cookies.",
            "{d} cookies are found burnt and thrown out.",
            "They sell {d} cookies to a catering order.",
            "They donate {d} cookies to a school event.",
            "They sell {d} cookies at a farmers market.",
            "{d} cookies are given away as samples.",
        ],
        "question": "How many cookies do they have now?",
    },
    "tank": {
        "intro": "A water tank contains {start} liters.",
        "start_range": (350, 550),
        "unit": "",
        "pos_range": (30, 150),
        "neg_range": (30, 150),
        "pos_templates": [
            "Then {d} liters of rain water are added.",
            "Then {d} liters are pumped in from a well.",
            "Then {d} liters are added from a backup tank.",
            "Then {d} liters of rainfall are collected.",
            "Then {d} liters are added from a delivery truck.",
            "Then {d} liters are added from a final refill.",
        ],
        "neg_templates": [
            "{d} liters are drained for irrigation.",
            "{d} liters are drained for cleaning.",
            "{d} liters evaporate over a hot week.",
            "{d} liters are drained for a large event.",
            "{d} liters are drained for a farm.",
            "{d} liters evaporate over a dry spell.",
        ],
        "question": "How many liters are in the tank now?",
    },
}


def generate_problem(family_name, n_steps, rng):
    family = FAMILIES[family_name]
    start = rng.randint(*family["start_range"])
    value = start
    chain = []
    sentences = []

    for _ in range(n_steps):
        # Retry with the opposite sign if a negative delta would drive the
        # running value below zero — keeps every intermediate value sane.
        for attempt in range(2):
            sign = rng.choice([1, -1])
            if attempt == 1:
                sign = 1  # force positive on retry
            lo, hi = family["pos_range"] if sign == 1 else family["neg_range"]
            d = rng.randint(lo, hi)
            if value + sign * d >= 0:
                break
        template = rng.choice(family["pos_templates"] if sign == 1 else family["neg_templates"])
        value += sign * d
        chain.append(value)
        sentences.append(template.format(d=d))

    prompt = family["intro"].format(start=start) + " " + " ".join(sentences) + " " + family["question"]
    return prompt, chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-level", type=int, default=4, help="Variants per family per step level (default 4; x5 families = 20/level)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--out", default="problems/arithmetic_steps.json", help="Output path")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    problems = []

    for steps in STEP_LEVELS:
        for family_name in FAMILIES:
            for variant in range(args.per_level):
                prompt, chain = generate_problem(family_name, steps, rng)
                problems.append(
                    {
                        "id": f"arith_{steps}_{family_name}_{variant}",
                        "category": "arithmetic_steps",
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
