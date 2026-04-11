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


def generate_distance_matrix(num_cities: int, max_distance: int) -> list[list[int]]:
    """Create a symmetric distance matrix similar to the original instances."""
    matrix = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
    for src in range(num_cities):
        for dst in range(src + 1, num_cities):
            distance = random.randint(max_distance // 3, max_distance)
            matrix[src][dst] = distance
            matrix[dst][src] = distance
    return matrix


def generate_instance(
        instance_name: str,
        num_cities: int,
        num_people: int,
        num_aircraft: int,
        max_fuel: int,
) -> str:
    """Render one zenotravel instance with aircraft, passengers, and fuel data."""
    template = get_problem_template(TEMPLATE_FILE_PATH)

    cities = [f"city{i}" for i in range(num_cities)]
    aircraft = [f"plane{i + 1}" for i in range(num_aircraft)]
    people = [f"person{i + 1}" for i in range(num_people)]
    distances = generate_distance_matrix(num_cities, max(100, max_fuel // 2))

    initial_statements = []
    goal_conditions = []

    for plane in aircraft:
        start_city = random.choice(cities)
        initial_statements.append(f"(located {plane} {start_city})")
        slow_burn = random.randint(2, 6)
        fast_burn = random.randint(slow_burn + 4, slow_burn + 14)
        fuel = random.randint(max_fuel // 2, max_fuel)
        capacity = random.randint(max_fuel + 1000, max_fuel + 5000)
        zoom_limit = random.randint(5, 10)
        initial_statements.extend([
            f"(= (capacity {plane}) {capacity})",
            f"(= (fuel {plane}) {fuel})",
            f"(= (slow-burn {plane}) {slow_burn})",
            f"(= (fast-burn {plane}) {fast_burn})",
            f"(= (onboard {plane}) 0)",
            f"(= (zoom-limit {plane}) {zoom_limit})",
        ])
        if random.random() < 0.35:
            destination = random.choice([city for city in cities if city != start_city] or [start_city])
            goal_conditions.append(f"(located {plane} {destination})")

    for person in people:
        start_city = random.choice(cities)
        destination = random.choice([city for city in cities if city != start_city] or [start_city])
        initial_statements.append(f"(located {person} {start_city})")
        goal_conditions.append(f"(located {person} {destination})")

    for src, source_city in enumerate(cities):
        for dst, target_city in enumerate(cities):
            initial_statements.append(
                f"(= (distance {source_city} {target_city}) {distances[src][dst]})"
            )
    initial_statements.append("(= (total-fuel-used) 0)")

    template_mapping = {
        "instance_name": instance_name,
        "aircraft": " ".join(aircraft),
        "people": " ".join(people),
        "cities": " ".join(cities),
        "initial_statements": "\n    ".join(initial_statements),
        "goal_conditions": "\n    ".join(goal_conditions),
    }
    return template.substitute(template_mapping)


def generate_multiple_problems(
        output_folder,
        total_num_problems,
        num_prev_instances=0,
        min_cities=3,
        max_cities=8,
        min_people=2,
        max_people=8,
        min_aircraft=1,
        max_aircraft=3,
        max_fuel=5000,
        **_,
):
    """Generate a batch of zenotravel instances."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    start_index = num_prev_instances or 0

    for idx in range(total_num_problems):
        num_cities = random.randint(min_cities, max_cities)
        num_people = random.randint(min_people, max_people)
        num_aircraft = random.randint(min_aircraft, max_aircraft)
        problem = generate_instance(
            f"ZTRAVEL-{start_index + idx + 1}",
            num_cities=num_cities,
            num_people=num_people,
            num_aircraft=num_aircraft,
            max_fuel=max_fuel,
        )
        with open(output_folder / f"pfile{start_index + idx}.pddl", "wt") as problem_file:
            problem_file.write(problem)
