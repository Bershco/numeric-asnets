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


def generate_supply(num_goods: int, num_markets: int, max_amount: int) -> list[list[int]]:
    """Distribute stock across a bounded number of markets for each good."""
    supply = [[0 for _ in range(num_markets)] for _ in range(num_goods)]
    for good_idx in range(num_goods):
        active_markets = random.sample(
            list(range(num_markets)),
            k=random.randint(1, max(1, min(num_markets, 4))),
        )
        for market_idx in active_markets:
            supply[good_idx][market_idx] = random.randint(1, max_amount)
    return supply


def generate_instance(
        instance_name: str,
        num_markets: int,
        num_goods: int,
        max_cost: int,
        max_capacity: int,
) -> str:
    """Render a travelling purchase problem instance with feasible demand levels."""
    template = get_problem_template(TEMPLATE_FILE_PATH)

    markets = [f"market{i + 1}" for i in range(num_markets)]
    goods = [f"goods{i}" for i in range(num_goods)]
    supply = generate_supply(num_goods, num_markets, max_capacity)
    initial_statements = []

    for good_idx, good in enumerate(goods):
        total_supply = sum(supply[good_idx])
        request = max(1, random.randint(max(1, total_supply // 3), max(1, total_supply)))
        for market_idx, market in enumerate(markets):
            on_sale = supply[good_idx][market_idx]
            price = random.randint(1, max_cost) if on_sale > 0 else 0
            initial_statements.append(f"(= (price {good} {market}) {price})")
            initial_statements.append(f"(= (on-sale {good} {market}) {on_sale})")
        initial_statements.append(f"(= (bought {good}) 0)")
        initial_statements.append(f"(= (request {good}) {request})")

    initial_statements.append("(loc truck0 depot0)")
    locations = ["depot0"] + markets
    for idx, src in enumerate(locations):
        initial_statements.append(f"(= (drive-cost {src} {src}) 0)")
        for jdx in range(idx + 1, len(locations)):
            dst = locations[jdx]
            cost = f"{random.uniform(50.0, float(max_capacity * 60 + 150)):.2f}"
            initial_statements.append(f"(= (drive-cost {src} {dst}) {cost})")
            initial_statements.append(f"(= (drive-cost {dst} {src}) {cost})")
    initial_statements.append("(= (total-cost) 0)")

    goal_conditions = [f"(>= (bought {good}) (request {good}))" for good in goods]
    goal_conditions.append("(loc truck0 depot0)")

    template_mapping = {
        "instance_name": instance_name,
        "markets": " ".join(markets),
        "goods": " ".join(goods),
        "initial_statements": "\n    ".join(initial_statements),
        "goal_conditions": "\n    ".join(goal_conditions),
    }
    return template.substitute(template_mapping)


def generate_multiple_problems(
        output_folder,
        total_num_problems,
        num_prev_instances=0,
        min_markets=4,
        max_markets=10,
        min_products=2,
        max_products=8,
        max_cost=50,
        max_capacity=20,
        **_,
):
    """Generate a batch of TPP instances."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    start_index = num_prev_instances or 0

    for idx in range(total_num_problems):
        num_markets = random.randint(min_markets, max_markets)
        num_goods = random.randint(min_products, max_products)
        problem = generate_instance(
            f"pfile{start_index + idx}",
            num_markets=num_markets,
            num_goods=num_goods,
            max_cost=max_cost,
            max_capacity=max_capacity,
        )
        with open(output_folder / f"pfile{start_index + idx}.pddl", "wt") as problem_file:
            problem_file.write(problem)
