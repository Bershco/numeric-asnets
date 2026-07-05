#!/usr/bin/python3

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

try:
    from problem_generator.common import get_problem_template
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from problem_generator.common import get_problem_template


TEMPLATE_FILE_PATH = Path(__file__).parent / "template.pddl"


def generate_distance_matrix(
    num_cities: int,
    max_distance: int,
) -> list[list[int]]:
    """Create a symmetric distance matrix similar to the original instances.

    :param num_cities: number of cities.
    :param max_distance: maximal distance between two different cities.
    :return: symmetric distance matrix.
    """

    matrix = [
        [0 for _ in range(num_cities)]
        for _ in range(num_cities)
    ]

    min_distance = max(1, max_distance // 3)

    for src in range(num_cities):
        for dst in range(src + 1, num_cities):
            distance = random.randint(min_distance, max_distance)

            matrix[src][dst] = distance
            matrix[dst][src] = distance

    return matrix


def generate_instance(
    instance_name: str,
    num_cities: int,
    num_people: int,
    num_aircraft: int,
    max_fuel: int,
    min_fuel: int = 1,
    min_capacity: int = 100,
    max_capacity: int | None = None,
    max_distance: int | None = None,
) -> str:
    """Generate a single zenotravel planning problem instance.

    :param instance_name: the name of the problem instance.
    :param num_cities: number of cities.
    :param num_people: number of passengers.
    :param num_aircraft: number of aircraft.
    :param max_fuel: maximal initial fuel value.
    :return: the rendered PDDL problem string.
    """

    if max_capacity is None:
        max_capacity = max_fuel + 5000

    if max_distance is None:
        max_distance = max(100, max_fuel // 2)
    template = get_problem_template(TEMPLATE_FILE_PATH)

    cities = [f"city{i}" for i in range(num_cities)]
    aircraft = [f"plane{i + 1}" for i in range(num_aircraft)]
    people = [f"person{i + 1}" for i in range(num_people)]

    effective_max_distance = min(max_distance, max(1, max_fuel // 10))
    distances = generate_distance_matrix(
        num_cities=num_cities,
        max_distance=effective_max_distance,
    )
    initial_statements = []
    goal_conditions = []

    # ---------------------------------------------------------
    # Aircraft initial state and optional aircraft goals
    # ---------------------------------------------------------

    for plane in aircraft:
        start_city = random.choice(cities)
        possible_destinations = [
            city
            for city in cities
            if city != start_city
        ]

        initial_statements.append(f"(located {plane} {start_city})")

        slow_burn = random.randint(1, 3)
        fast_burn = random.randint(slow_burn + 1, slow_burn + 4)

        minimum_safe_capacity = max(
            min_capacity,
            max_fuel,
            effective_max_distance * slow_burn,
        )

        if minimum_safe_capacity > max_capacity:
            max_capacity = minimum_safe_capacity

        capacity = random.randint(minimum_safe_capacity, max_capacity)

        # For validation generation, start aircraft full.
        # This avoids false UNSOLVABLE cases caused only by unlucky initial fuel.
        fuel = min(max_fuel, capacity)

        zoom_limit = random.randint(5, 10)

        initial_statements.extend(
            [
                f"(= (capacity {plane}) {capacity})",
                f"(= (fuel {plane}) {fuel})",
                f"(= (slow-burn {plane}) {slow_burn})",
                f"(= (fast-burn {plane}) {fast_burn})",
                f"(= (onboard {plane}) 0)",
                f"(= (zoom-limit {plane}) {zoom_limit})",
            ]
        )

        if random.random() < 0.35:
            destination = random.choice(possible_destinations)
            goal_conditions.append(f"(located {plane} {destination})")

    # ---------------------------------------------------------
    # Passenger initial state and passenger goals
    # ---------------------------------------------------------

    for person in people:
        start_city = random.choice(cities)
        possible_destinations = [
            city
            for city in cities
            if city != start_city
        ]

        destination = random.choice(possible_destinations)

        initial_statements.append(f"(located {person} {start_city})")
        goal_conditions.append(f"(located {person} {destination})")

    # ---------------------------------------------------------
    # Distances and metric fluent
    # ---------------------------------------------------------

    for src, source_city in enumerate(cities):
        for dst, target_city in enumerate(cities):
            initial_statements.append(
                f"(= (distance {source_city} {target_city}) "
                f"{distances[src][dst]})"
            )

    initial_statements.append("(= (total-fuel-used) 0)")

    # ---------------------------------------------------------
    # Template mapping
    # ---------------------------------------------------------

    template_mapping = {
        "instance_name": instance_name,
        "aircraft": " ".join(aircraft),
        "people": " ".join(people),
        "cities": " ".join(cities),
        "initial_statements": "\n    ".join(initial_statements),
        "goal_conditions": "\n    ".join(goal_conditions),
    }

    return template.substitute(template_mapping)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate zenotravel planning instances."
    )

    parser.add_argument(
        "--min_cities",
        type=int,
        required=True,
        help="Minimal number of cities in each generated problem.",
    )

    parser.add_argument(
        "--max_cities",
        type=int,
        required=True,
        help="Maximal number of cities in each generated problem.",
    )

    parser.add_argument(
        "--min_people",
        type=int,
        required=True,
        help="Minimal number of passengers in each generated problem.",
    )

    parser.add_argument(
        "--max_people",
        type=int,
        required=True,
        help="Maximal number of passengers in each generated problem.",
    )

    parser.add_argument(
        "--min_aircraft",
        type=int,
        required=True,
        help="Minimal number of aircraft in each generated problem.",
    )

    parser.add_argument(
        "--max_aircraft",
        type=int,
        required=True,
        help="Maximal number of aircraft in each generated problem.",
    )

    parser.add_argument(
        "--max_fuel",
        type=int,
        required=True,
        help="Maximal initial aircraft fuel value.",
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
    min_cities: int = 3,
    max_cities: int = 8,
    min_people: int = 2,
    max_people: int = 8,
    min_aircraft: int = 1,
    max_aircraft: int = 3,
    max_fuel: int = 5000,
    min_fuel: int = 1,
    min_capacity: int = 100,
    max_capacity: int | None = None,
    max_distance: int | None = None,
    **_,
) -> None:
    """Generate multiple zenotravel problems.

    This keeps compatibility with the shared workflow API where
    output_folder and total_num_problems may be passed externally.

    :param output_folder: folder where generated PDDL files are written.
    :param total_num_problems: number of problems to generate.
    :param num_prev_instances: offset for generated pfile numbering.
    :param min_cities: minimal number of cities.
    :param max_cities: maximal number of cities.
    :param min_people: minimal number of passengers.
    :param max_people: maximal number of passengers.
    :param min_aircraft: minimal number of aircraft.
    :param max_aircraft: maximal number of aircraft.
    :param max_fuel: maximal initial aircraft fuel value.
    """

    if min_cities > max_cities:
        raise ValueError(
            f"min_cities must be <= max_cities, got "
            f"{min_cities} > {max_cities}."
        )

    if min_people > max_people:
        raise ValueError(
            f"min_people must be <= max_people, got "
            f"{min_people} > {max_people}."
        )

    if min_aircraft > max_aircraft:
        raise ValueError(
            f"min_aircraft must be <= max_aircraft, got "
            f"{min_aircraft} > {max_aircraft}."
        )

    if min_cities < 2:
        raise ValueError(
            f"min_cities must be at least 2, got {min_cities}."
        )

    if min_people < 1:
        raise ValueError(
            f"min_people must be at least 1, got {min_people}."
        )

    if min_aircraft < 1:
        raise ValueError(
            f"min_aircraft must be at least 1, got {min_aircraft}."
        )

    if max_fuel < 1:
        raise ValueError(
            f"max_fuel must be at least 1, got {max_fuel}."
        )

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    start_index = num_prev_instances or 0

    for idx in range(total_num_problems):
        problem_index = start_index + idx

        num_cities = random.randint(min_cities, max_cities)
        num_people = random.randint(min_people, max_people)
        num_aircraft = random.randint(min_aircraft, max_aircraft)

        problem = generate_instance(
            instance_name=f"ZTRAVEL-{problem_index + 1}",
            num_cities=num_cities,
            num_people=num_people,
            num_aircraft=num_aircraft,
            max_fuel=max_fuel,
            min_fuel=min_fuel,
            min_capacity=min_capacity,
            max_capacity=max_capacity,
            max_distance=max_distance,
        )

        with open(output_folder / f"pfile{problem_index}.pddl", "wt") as problem_file:
            problem_file.write(problem)


def main() -> None:
    args = parse_arguments()

    generate_multiple_problems(
        output_folder=args.output_path,
        total_num_problems=args.total_num_problems,
        num_prev_instances=args.num_prev_instances,
        min_cities=args.min_cities,
        max_cities=args.max_cities,
        min_people=args.min_people,
        max_people=args.max_people,
        min_aircraft=args.min_aircraft,
        max_aircraft=args.max_aircraft,
        max_fuel=args.max_fuel,
    )


if __name__ == "__main__":
    main()