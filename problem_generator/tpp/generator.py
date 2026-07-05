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


def generate_supply(
    num_goods: int,
    num_markets: int,
    max_amount: int,
    min_active_fraction: float = 0.35,
    max_active_fraction: float = 0.75,
) -> list[list[int]]:
    """Distribute stock across markets for each good.

    Uses a market-density range rather than a tiny fixed cap.
    """

    supply = [
        [0 for _ in range(num_markets)]
        for _ in range(num_goods)
    ]

    min_active_markets = max(1, int(round(num_markets * min_active_fraction)))
    max_active_markets = max(min_active_markets, int(round(num_markets * max_active_fraction)))
    max_active_markets = min(max_active_markets, num_markets)

    for good_idx in range(num_goods):
        active_markets = random.sample(
            list(range(num_markets)),
            k=random.randint(min_active_markets, max_active_markets),
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
    min_active_fraction: float = 0.35,
    max_active_fraction: float = 0.75,
) -> str:
    """Generate a single travelling purchase problem instance.

    :param instance_name: the name of the problem instance.
    :param num_markets: number of markets.
    :param num_goods: number of goods/products.
    :param max_cost: maximal item price.
    :param max_capacity: maximal stock amount per active market.
    :return: the rendered PDDL problem string.
    """

    template = get_problem_template(TEMPLATE_FILE_PATH)

    markets = [f"market{i + 1}" for i in range(num_markets)]
    goods = [f"goods{i}" for i in range(num_goods)]

    supply = generate_supply(
        num_goods=num_goods,
        num_markets=num_markets,
        max_amount=max_capacity,
        min_active_fraction=min_active_fraction,
        max_active_fraction=max_active_fraction,
    )
    initial_statements = []

    # ---------------------------------------------------------
    # Goods supply, prices, bought counters, and requests
    # ---------------------------------------------------------

    for good_idx, good in enumerate(goods):
        total_supply = sum(supply[good_idx])

        # Feasible demand: request never exceeds total available supply.
        request = random.randint(
            max(1, total_supply // 3),
            total_supply,
        )

        for market_idx, market in enumerate(markets):
            on_sale = supply[good_idx][market_idx]
            price = random.randint(1, max_cost) if on_sale > 0 else 0

            initial_statements.append(f"(= (price {good} {market}) {price})")
            initial_statements.append(f"(= (on-sale {good} {market}) {on_sale})")

        initial_statements.append(f"(= (bought {good}) 0)")
        initial_statements.append(f"(= (request {good}) {request})")

    # ---------------------------------------------------------
    # Truck location and travel costs
    # ---------------------------------------------------------

    initial_statements.append("(loc truck0 depot0)")

    locations = ["depot0"] + markets

    for idx, src in enumerate(locations):
        initial_statements.append(f"(= (drive-cost {src} {src}) 0)")

        for jdx in range(idx + 1, len(locations)):
            dst = locations[jdx]

            cost = (
                f"{random.uniform(50.0, float(max_capacity * 60 + 150)):.2f}"
            )

            initial_statements.append(f"(= (drive-cost {src} {dst}) {cost})")
            initial_statements.append(f"(= (drive-cost {dst} {src}) {cost})")

    initial_statements.append("(= (total-cost) 0)")

    # ---------------------------------------------------------
    # Goal conditions
    # ---------------------------------------------------------

    goal_conditions = [
        f"(>= (bought {good}) (request {good}))"
        for good in goods
    ]

    goal_conditions.append("(loc truck0 depot0)")

    # ---------------------------------------------------------
    # Template mapping
    # ---------------------------------------------------------

    template_mapping = {
        "instance_name": instance_name,
        "markets": " ".join(markets),
        "goods": " ".join(goods),
        "initial_statements": "\n    ".join(initial_statements),
        "goal_conditions": "\n    ".join(goal_conditions),
    }

    return template.substitute(template_mapping)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate travelling purchase problem instances."
    )

    parser.add_argument(
        "--min_markets",
        type=int,
        required=True,
        help="Minimal number of markets in each generated problem.",
    )

    parser.add_argument(
        "--max_markets",
        type=int,
        required=True,
        help="Maximal number of markets in each generated problem.",
    )

    parser.add_argument(
        "--min_products",
        "--min_goods",
        dest="min_products",
        type=int,
        required=True,
        help=(
            "Minimal number of products/goods in each generated problem. "
            "The alias --min_goods is also supported."
        ),
    )

    parser.add_argument(
        "--max_products",
        "--max_goods",
        dest="max_products",
        type=int,
        required=True,
        help=(
            "Maximal number of products/goods in each generated problem. "
            "The alias --max_goods is also supported."
        ),
    )

    parser.add_argument(
        "--max_cost",
        type=int,
        required=True,
        help="Maximal item price.",
    )

    parser.add_argument(
        "--max_capacity",
        type=int,
        required=True,
        help="Maximal stock amount per active market.",
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

    return parser.parse_args()


def generate_multiple_problems(
    output_folder,
    total_num_problems: int = 200,
    num_prev_instances: int = 0,
    min_markets: int = 4,
    max_markets: int = 40,
    min_products: int = 2,
    max_products: int = 40,
    min_goods: Optional[int] = None,
    max_goods: Optional[int] = None,
    max_cost: int = 50,
    max_capacity: int = 20,
    min_active_fraction: float = 0.35,
    max_active_fraction: float = 0.75,
    **_,
) -> None:
    """Generate multiple travelling purchase problem instances.

    This keeps compatibility with the shared workflow API.

    :param output_folder: folder where generated PDDL files are written.
    :param total_num_problems: number of problems to generate.
    :param num_prev_instances: offset for generated pfile numbering.
    :param min_markets: minimal number of markets.
    :param max_markets: maximal number of markets.
    :param min_products: minimal number of goods/products.
    :param max_products: maximal number of goods/products.
    :param min_goods: alias for min_products.
    :param max_goods: alias for max_products.
    :param max_cost: maximal item price.
    :param max_capacity: maximal stock amount per active market.
    """

    if min_goods is not None:
        min_products = min_goods

    if max_goods is not None:
        max_products = max_goods

    if min_markets > max_markets:
        raise ValueError(
            f"min_markets must be <= max_markets, got "
            f"{min_markets} > {max_markets}."
        )

    if min_products > max_products:
        raise ValueError(
            f"min_products must be <= max_products, got "
            f"{min_products} > {max_products}."
        )

    if min_markets < 1:
        raise ValueError(
            f"min_markets must be at least 1, got {min_markets}."
        )

    if min_products < 1:
        raise ValueError(
            f"min_products must be at least 1, got {min_products}."
        )

    if max_cost < 1:
        raise ValueError(f"max_cost must be at least 1, got {max_cost}.")

    if max_capacity < 1:
        raise ValueError(
            f"max_capacity must be at least 1, got {max_capacity}."
        )

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    start_index = num_prev_instances or 0

    for idx in range(total_num_problems):
        problem_index = start_index + idx

        num_markets = random.randint(min_markets, max_markets)
        num_goods = random.randint(min_products, max_products)

        problem = generate_instance(
            instance_name=f"pfile{problem_index}",
            num_markets=num_markets,
            num_goods=num_goods,
            max_cost=max_cost,
            max_capacity=max_capacity,
            min_active_fraction=min_active_fraction,
            max_active_fraction=max_active_fraction,
        )

        with open(output_folder / f"pfile{problem_index}.pddl", "wt") as problem_file:
            problem_file.write(problem)


def main() -> None:
    args = parse_arguments()

    generate_multiple_problems(
        output_folder=args.output_path,
        total_num_problems=args.total_num_problems,
        num_prev_instances=args.num_prev_instances,
        min_markets=args.min_markets,
        max_markets=args.max_markets,
        min_products=args.min_products,
        max_products=args.max_products,
        max_cost=args.max_cost,
        max_capacity=args.max_capacity,
    )


if __name__ == "__main__":
    main()