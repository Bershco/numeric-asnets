#!/usr/bin/python3
import random
from pathlib import Path
import sys

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
    "rest", "satiety", "curiosity", "achievement", "satisfaction", "entertainment",
]
PAIN_POOL = [
    "hangover", "depression", "abrasion", "anxiety", "anger", "angina",
    "boils", "grief", "loneliness", "dread", "jealousy", "sciatica",
]


def unique_names(pool: list[str], count: int, offset: int = 0) -> list[str]:
    """Return `count` names from a pool, wrapping with numeric suffixes if needed."""
    names = []
    for idx in range(count):
        base = pool[(idx + offset) % len(pool)]
        suffix = "" if idx + offset < len(pool) else f"-{idx + offset}"
        names.append(f"{base}{suffix}")
    return names


def generate_instance(
        instance_name: str,
        num_foods: int,
        num_pleasures: int,
        num_pains: int,
        max_locale: int,
) -> str:
    """Render a single mystery-prime instance with consistent typed objects."""
    template = get_problem_template(TEMPLATE_FILE_PATH)

    foods = unique_names(FOOD_POOL, num_foods, random.randint(0, len(FOOD_POOL) - 1))
    pleasures = unique_names(PLEASURE_POOL, num_pleasures, random.randint(0, len(PLEASURE_POOL) - 1))
    pains = unique_names(PAIN_POOL, num_pains, random.randint(0, len(PAIN_POOL) - 1))

    initial_statements = []
    craving_facts = []

    for food in foods:
        initial_statements.append(f"(= (locale {food}) {random.randint(0, max_locale)})")

    for pleasure in pleasures:
        initial_statements.append(f"(= (harmony {pleasure}) {random.randint(1, 3)})")

    for food in foods:
        targets = random.sample(foods, k=random.randint(1, min(len(foods), 3)))
        for target in targets:
            initial_statements.append(f"(eats {food} {target})")

    feelings = pleasures + pains
    for feeling in feelings:
        targets = random.sample(foods, k=random.randint(1, min(len(foods), 2)))
        for target in targets:
            fact = f"(craves {feeling} {target})"
            craving_facts.append(fact)
            initial_statements.append(fact)

    goal_conditions = " ".join(
        random.sample(craving_facts, k=max(1, min(len(craving_facts), random.randint(1, 3))))
    )

    template_mapping = {
        "instance_name": instance_name,
        "foods": " ".join(foods),
        "pleasures": " ".join(pleasures),
        "pains": " ".join(pains),
        "initial_statements": "\n".join(initial_statements),
        "goal_conditions": goal_conditions,
    }
    return template.substitute(template_mapping)


def generate_multiple_problems(
        output_folder,
        total_num_problems,
        num_prev_instances=0,
        min_locations=3,
        max_locations=9,
        min_keys=1,
        max_keys=4,
        max_fuel=9,
        **_,
):
    """Generate a batch of mystery-prime instances."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    start_index = num_prev_instances or 0
    for idx in range(total_num_problems):
        num_foods = random.randint(max(4, min_locations), max(6, max_locations))
        num_pleasures = random.randint(1, max_keys)
        num_pains = random.randint(2, max_keys + 4)
        problem = generate_instance(
            f"mprime-x-{start_index + idx + 1}",
            num_foods=num_foods,
            num_pleasures=num_pleasures,
            num_pains=num_pains,
            max_locale=max_fuel,
        )
        with open(output_folder / f"pfile{start_index + idx}.pddl", "wt") as problem_file:
            problem_file.write(problem)
