#!/usr/bin/python3

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

try:
    from problem_generator.common import get_problem_template
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from problem_generator.common import get_problem_template


TEMPLATE_FILE_PATH = Path(__file__).parent / "template.pddl"


FOOD_POOL = [
    "rice", "pear", "flounder", "okra", "pork", "lamb", "wurst", "shrimp",
    "muffin", "broccoli", "potato", "lettuce", "melon", "tofu", "orange",
    "cherry", "chicken", "apple", "pepper", "bacon", "turkey", "papaya",
]

PLEASURE_POOL = [
    "rest", "satiety", "curiosity", "achievement", "satisfaction",
    "entertainment",
]

PAIN_POOL = [
    "hangover", "depression", "abrasion", "anxiety", "anger", "angina",
    "boils", "grief", "loneliness", "dread", "jealousy", "sciatica",
]


def unique_names(pool: list[str], count: int, offset: int = 0) -> list[str]:
    """Return count unique names from a finite base-name pool.

    If count exceeds the pool size, names wrap around with numeric suffixes.

    :param pool: base-name pool.
    :param count: number of names to produce.
    :param offset: starting offset inside the pool.
    :return: list of unique object names.
    """

    names = []

    for idx in range(count):
        raw_idx = idx + offset
        base = pool[raw_idx % len(pool)]
        suffix = "" if raw_idx < len(pool) else f"-{raw_idx}"
        names.append(f"{base}{suffix}")

    return names


def generate_instance(
    instance_name: str,
    num_foods: int,
    num_pleasures: int,
    num_pains: int,
    max_locale: int,
    rng: Optional[random.Random] = None,
) -> str:
    """Generate a single mystery-prime planning problem instance.

    :param instance_name: the name of the problem instance.
    :param num_foods: number of food objects.
    :param num_pleasures: number of pleasure objects.
    :param num_pains: number of pain objects.
    :param max_locale: maximal locale numeric value.
    :return: the rendered PDDL problem string.
    """

    template = get_problem_template(TEMPLATE_FILE_PATH)
    rng = rng or random

    foods = unique_names(
        pool=FOOD_POOL,
        count=num_foods,
        offset=rng.randint(0, len(FOOD_POOL) - 1),
    )

    pleasures = unique_names(
        pool=PLEASURE_POOL,
        count=num_pleasures,
        offset=rng.randint(0, len(PLEASURE_POOL) - 1),
    )

    pains = unique_names(
        pool=PAIN_POOL,
        count=num_pains,
        offset=rng.randint(0, len(PAIN_POOL) - 1),
    )

    initial_statements = []
    initial_cravings: set[tuple[str, str]] = set()

    # ---------------------------------------------------------
    # Initial numeric fluents
    # ---------------------------------------------------------

    for food in foods:
        initial_statements.append(
            f"(= (locale {food}) {rng.randint(0, max_locale)})"
        )

    for pleasure in pleasures:
        initial_statements.append(
            f"(= (harmony {pleasure}) {rng.randint(1, 3)})"
        )

    # ---------------------------------------------------------
    # Initial predicates
    # ---------------------------------------------------------

    for food in foods:
        targets = rng.sample(
            foods,
            k=rng.randint(1, min(len(foods), 3)),
        )

        for target in targets:
            initial_statements.append(f"(eats {food} {target})")

    feelings = pleasures + pains

    for feeling in feelings:
        targets = rng.sample(
            foods,
            k=rng.randint(1, min(len(foods), 2)),
        )

        for target in targets:
            initial_cravings.add((feeling, target))

    # ---------------------------------------------------------
    # Goal conditions.  Each goal is absent initially but has a guaranteed
    # two-action witness: overcome(pain, pleasure, source) followed by
    # succumb(pain, pleasure, target).  The old generator sampled goals from
    # initial_cravings, making every generated problem a zero-step problem.
    # ---------------------------------------------------------

    num_goal_cravings = rng.randint(1, min(len(pains), len(pleasures), 3))
    goal_conditions = []
    for pain, pleasure in zip(
        rng.sample(pains, k=num_goal_cravings),
        rng.sample(pleasures, k=num_goal_cravings),
    ):
        pain_sources = [food for food in foods if (pain, food) in initial_cravings]
        source = rng.choice(pain_sources)
        target_candidates = [
            food for food in foods if (pain, food) not in initial_cravings
        ]
        target = rng.choice(target_candidates)
        initial_cravings.add((pleasure, source))
        initial_cravings.add((pleasure, target))
        goal_conditions.append(f"(craves {pain} {target})")

    initial_statements.extend(
        f"(craves {feeling} {food})"
        for feeling, food in sorted(initial_cravings)
    )

    # ---------------------------------------------------------
    # Template mapping
    # ---------------------------------------------------------

    template_mapping = {
        "instance_name": instance_name,
        "foods": " ".join(foods),
        "pleasures": " ".join(pleasures),
        "pains": " ".join(pains),
        "initial_statements": "\n".join(initial_statements),
        "goal_conditions": " ".join(goal_conditions),
    }

    return template.substitute(template_mapping)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate mystery-prime planning instances."
    )

    parser.add_argument(
        "--min_foods",
        "--min_locations",
        dest="min_foods",
        type=int,
        required=True,
        help=(
            "Minimal number of food objects. "
            "The alias --min_locations is kept for compatibility."
        ),
    )

    parser.add_argument(
        "--max_foods",
        "--max_locations",
        dest="max_foods",
        type=int,
        required=True,
        help=(
            "Maximal number of food objects. "
            "The alias --max_locations is kept for compatibility."
        ),
    )

    parser.add_argument(
        "--min_pleasures",
        dest="min_pleasures",
        type=int,
        default=1,
        help="Minimal number of pleasure objects. Default: 1.",
    )

    parser.add_argument(
        "--max_pleasures",
        "--max_keys",
        dest="max_pleasures",
        type=int,
        required=True,
        help=(
            "Maximal number of pleasure objects. "
            "The alias --max_keys is kept for compatibility."
        ),
    )

    parser.add_argument(
        "--min_pains",
        dest="min_pains",
        type=int,
        default=2,
        help="Minimal number of pain objects. Default: 2.",
    )

    parser.add_argument(
        "--max_pains",
        dest="max_pains",
        type=int,
        default=None,
        help=(
            "Maximal number of pain objects. "
            "Default: max_pleasures + 4."
        ),
    )

    parser.add_argument(
        "--max_locale",
        "--max_fuel",
        dest="max_locale",
        type=int,
        required=True,
        help=(
            "Maximal locale numeric value. "
            "The alias --max_fuel is kept for compatibility."
        ),
    )

    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to the output folder where the problems will be saved.",
    )

    parser.add_argument(
        "--total_num_problems",
        type=int,
        default=200,
        help="Total number of problems to generate. Default: 200.",
    )

    parser.add_argument(
        "--num_prev_instances",
        type=int,
        default=0,
        help=(
            "Number of previously generated instances. "
            "Used to offset pfile numbering. Default: 0."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic generator seed.",
    )

    return parser.parse_args()


def generate_multiple_problems(
    output_folder,
    total_num_problems: int = 200,
    num_prev_instances: int = 0,
    min_locations: int = 3,
    max_locations: int = 9,
    min_foods: Optional[int] = None,
    max_foods: Optional[int] = None,
    min_keys: int = 1,
    max_keys: int = 4,
    min_pleasures: Optional[int] = None,
    max_pleasures: Optional[int] = None,
    min_pains: int = 2,
    max_pains: Optional[int] = None,
    max_fuel: int = 9,
    max_locale: Optional[int] = None,
    seed: Optional[int] = None,
    **_,
) -> None:
    """Generate multiple mystery-prime problems.

    This keeps compatibility with the shared workflow API.

    Legacy argument mapping:
      - min_locations / max_locations behave as min_foods / max_foods.
      - min_keys / max_keys behave as min_pleasures / max_pleasures.
      - max_fuel behaves as max_locale.

    :param output_folder: folder where generated PDDL files are written.
    :param total_num_problems: number of problems to generate.
    :param num_prev_instances: offset for generated pfile numbering.
    :param min_locations: backwards-compatible alias for min_foods.
    :param max_locations: backwards-compatible alias for max_foods.
    :param min_foods: minimal number of food objects.
    :param max_foods: maximal number of food objects.
    :param min_keys: backwards-compatible alias for min_pleasures.
    :param max_keys: backwards-compatible alias for max_pleasures.
    :param min_pleasures: minimal number of pleasure objects.
    :param max_pleasures: maximal number of pleasure objects.
    :param min_pains: minimal number of pain objects.
    :param max_pains: maximal number of pain objects.
    :param max_fuel: backwards-compatible alias for max_locale.
    :param max_locale: maximal locale numeric value.
    """

    if min_foods is None:
        min_foods = min_locations

    if max_foods is None:
        max_foods = max_locations

    if min_pleasures is None:
        min_pleasures = min_keys

    if max_pleasures is None:
        max_pleasures = max_keys

    if max_pains is None:
        max_pains = max_pleasures + 4

    if max_locale is None:
        max_locale = max_fuel

    if min_foods > max_foods:
        raise ValueError(
            f"min_foods must be <= max_foods, got "
            f"{min_foods} > {max_foods}."
        )

    if min_pleasures > max_pleasures:
        raise ValueError(
            f"min_pleasures must be <= max_pleasures, got "
            f"{min_pleasures} > {max_pleasures}."
        )

    if min_pains > max_pains:
        raise ValueError(
            f"min_pains must be <= max_pains, got "
            f"{min_pains} > {max_pains}."
        )

    if min_foods < 1:
        raise ValueError(f"min_foods must be at least 1, got {min_foods}.")

    if min_pleasures < 0:
        raise ValueError(
            f"min_pleasures must be non-negative, got {min_pleasures}."
        )

    if min_pains < 0:
        raise ValueError(f"min_pains must be non-negative, got {min_pains}.")

    if min_pleasures + min_pains < 1:
        raise ValueError(
            "At least one feeling object is required, but both "
            "min_pleasures and min_pains are 0."
        )

    if max_locale < 0:
        raise ValueError(f"max_locale must be non-negative, got {max_locale}.")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed) if seed is not None else random

    start_index = num_prev_instances or 0

    for idx in range(total_num_problems):
        problem_index = start_index + idx

        num_foods = rng.randint(
            max(4, min_foods),
            max(6, max_foods),
        )

        num_pleasures = rng.randint(
            min_pleasures,
            max_pleasures,
        )

        num_pains = rng.randint(
            min_pains,
            max_pains,
        )

        problem = generate_instance(
            instance_name=f"mprime-x-{problem_index + 1}",
            num_foods=num_foods,
            num_pleasures=num_pleasures,
            num_pains=num_pains,
            max_locale=max_locale,
            rng=rng,
        )

        with open(output_folder / f"pfile{problem_index}.pddl", "wt") as problem_file:
            problem_file.write(problem)


def main() -> None:
    args = parse_arguments()

    generate_multiple_problems(
        output_folder=args.output_path,
        total_num_problems=args.total_num_problems,
        num_prev_instances=args.num_prev_instances,
        min_foods=args.min_foods,
        max_foods=args.max_foods,
        min_pleasures=args.min_pleasures,
        max_pleasures=args.max_pleasures,
        min_pains=args.min_pains,
        max_pains=args.max_pains,
        max_locale=args.max_locale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
