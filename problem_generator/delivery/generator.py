#!/usr/bin/python3
import argparse
import random
from collections import deque
from pathlib import Path
import sys

try:
    from problem_generator.common import get_problem_template
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from problem_generator.common import get_problem_template

TEMPLATE_FILE_PATH = Path(__file__).parent / "template.pddl"


def generate_room_names(num_locations: int) -> list[str]:
    """Return deterministic room identifiers matching the observed corpus style."""
    return [f"room{chr(ord('a') + idx)}" for idx in range(num_locations)]


def generate_item_names(num_packages: int) -> list[str]:
    """Return item identifiers in descending order to mirror the source instances."""
    return [f"item{idx}" for idx in range(num_packages, 0, -1)]


def generate_bot_names(num_bots: int) -> list[str]:
    """Return bot identifiers."""
    return [f"bot{idx}" for idx in range(1, num_bots + 1)]


def generate_arm_names(num_bots: int, arms_per_bot: int) -> list[str]:
    """Return arm names grouped by bot using the delivery naming convention."""
    if arms_per_bot == 2:
        arm_labels = ["left", "right"]
    elif arms_per_bot == 3:
        arm_labels = ["left", "mid", "right"]
    else:
        arm_labels = ["left", "mid", "right", "aux"]
    arms = []
    for bot_idx in range(1, num_bots + 1):
        for arm_idx in range(arms_per_bot):
            arms.append(f"{arm_labels[arm_idx]}{bot_idx}")
    return arms


def build_door_graph(num_locations: int) -> list[tuple[str, str]]:
    """Build a small connected room graph shaped like the original delivery cases."""
    rooms = generate_room_names(num_locations)

    if num_locations == 2:
        return [(rooms[0], rooms[1]), (rooms[1], rooms[0])]

    if num_locations == 3:
        return [
            (rooms[0], rooms[1]),
            (rooms[1], rooms[0]),
            (rooms[0], rooms[2]),
            (rooms[2], rooms[0]),
        ]

    if num_locations == 4:
        return [
            (rooms[0], rooms[1]),
            (rooms[1], rooms[0]),
            (rooms[0], rooms[2]),
            (rooms[2], rooms[0]),
            (rooms[3], rooms[1]),
            (rooms[1], rooms[3]),
            (rooms[3], rooms[2]),
            (rooms[2], rooms[3]),
        ]

    if num_locations == 5:
        if random.random() < 0.5:
            return [
                (rooms[0], rooms[1]),
                (rooms[1], rooms[0]),
                (rooms[0], rooms[2]),
                (rooms[2], rooms[0]),
                (rooms[3], rooms[1]),
                (rooms[1], rooms[3]),
                (rooms[3], rooms[2]),
                (rooms[2], rooms[3]),
                (rooms[3], rooms[4]),
                (rooms[4], rooms[3]),
            ]
        return [
            (rooms[0], rooms[1]),
            (rooms[1], rooms[2]),
            (rooms[2], rooms[3]),
            (rooms[3], rooms[4]),
            (rooms[4], rooms[0]),
        ]

    return [
        (rooms[0], rooms[1]),
        (rooms[1], rooms[2]),
        (rooms[2], rooms[3]),
        (rooms[3], rooms[4]),
        (rooms[4], rooms[0]),
        (rooms[0], rooms[5]),
        (rooms[5], rooms[3]),
    ]


def compute_shortest_distances(
        rooms: list[str], doors: list[tuple[str, str]], start_room: str
) -> dict[str, int]:
    """Compute directed shortest-path distances from one room."""
    adjacency = {room: [] for room in rooms}
    for source, target in doors:
        adjacency[source].append(target)

    distances = {start_room: 0}
    queue = deque([start_room])

    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances


def choose_num_bots(num_locations: int, num_packages: int) -> int:
    """Scale the fleet size conservatively with map size and package count."""
    if num_locations <= 2:
        return 1
    if num_locations <= 4:
        return 2
    if num_locations == 5 and num_packages <= 20:
        return 2
    return 3


def choose_arms_per_bot(num_locations: int, num_packages: int) -> int:
    """Use two arms by default and a third only for the largest layouts."""
    if num_locations == 6 and num_packages >= 34:
        return 3
    return 2


def choose_initial_rooms(rooms: list[str], num_packages: int) -> list[str]:
    """Restrict package start rooms so small instances stay clustered and readable."""
    if len(rooms) <= 2 or num_packages <= 12:
        return [rooms[0]]
    if len(rooms) <= 5 or num_packages <= 32:
        return rooms[: min(2, len(rooms))]
    return rooms[: min(3, len(rooms))]


def assign_item_rooms(
        items: list[str],
        rooms: list[str],
        doors: list[tuple[str, str]],
        max_distance: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Assign each item a reachable start room and a reasonably distant goal room."""
    initial_room_pool = choose_initial_rooms(rooms, len(items))
    initial_positions = {}
    goal_positions = {}

    for item in items:
        start_room = random.choice(initial_room_pool)
        distances = compute_shortest_distances(rooms, doors, start_room)

        target_candidates = [
            room
            for room, distance in distances.items()
            if room != start_room and distance <= max_distance
        ]
        if not target_candidates:
            target_candidates = [
                room for room in distances.keys() if room != start_room
            ]

        max_observed_distance = max(distances[room] for room in target_candidates)
        preferred_targets = [
            room
            for room in target_candidates
            if distances[room] >= max(1, max_observed_distance - 1)
        ]

        initial_positions[item] = start_room
        goal_positions[item] = random.choice(preferred_targets)

    return initial_positions, goal_positions


def generate_weights(num_packages: int, max_capacity: int) -> list[int]:
    """Generate light package weights that still force tray and load decisions."""
    max_weight = 1
    if num_packages >= 12:
        max_weight = 2
    if num_packages >= 16:
        max_weight = 3
    if num_packages >= 28:
        max_weight = 4

    max_weight = min(max_weight, max_capacity)
    return [random.randint(1, max_weight) for _ in range(num_packages)]


def choose_load_limit(weights: list[int], max_capacity: int) -> int:
    """Choose a per-bot capacity that keeps every instance feasible."""
    min_capacity = max(4, max(weights))
    capacity = max(min_capacity, int(0.75 * max_capacity))
    return min(capacity, max_capacity)


def generate_instance(
        instance_name: str,
        num_locations: int,
        num_packages: int,
        max_capacity: int,
        max_distance: int,
) -> str:
    """Render a single delivery problem instance from the template."""
    template = get_problem_template(TEMPLATE_FILE_PATH)

    rooms = generate_room_names(num_locations)
    num_bots = choose_num_bots(num_locations, num_packages)
    arms_per_bot = choose_arms_per_bot(num_locations, num_packages)
    items = generate_item_names(num_packages)
    bots = generate_bot_names(num_bots)
    arms = generate_arm_names(num_bots, arms_per_bot)
    doors = build_door_graph(num_locations)
    initial_positions, goal_positions = assign_item_rooms(
        items, rooms, doors, max_distance
    )
    weights = generate_weights(num_packages, max_capacity)
    load_limit = choose_load_limit(weights, max_capacity)

    initial_fluents = []
    for item, weight in zip(items, weights):
        initial_fluents.append(f"(= (weight {item}) {weight})")

    for bot in bots:
        initial_fluents.append(f"(= (current_load {bot}) 0)")
        initial_fluents.append(f"(= (load_limit {bot}) {load_limit})")
    initial_fluents.append("(= (cost) 0)")

    initial_predicates = []
    for item in items:
        initial_predicates.append(f"(at {item} {initial_positions[item]})")
    for bot in bots:
        initial_predicates.append(f"(at-bot {bot} {rooms[0]})")
    for arm in arms:
        initial_predicates.append(f"(free {arm})")
    for arm, bot in zip(arms, [bot for bot in bots for _ in range(arms_per_bot)]):
        initial_predicates.append(f"(mount {arm} {bot})")
    for source, target in doors:
        initial_predicates.append(f"(door {source} {target})")

    goal_conditions = [
        f"(at {item} {goal_positions[item]})" for item in items
    ]

    template_mapping = {
        "instance_name": instance_name,
        "rooms_list": " ".join(rooms),
        "items_list": " ".join(items),
        "bots_list": " ".join(bots),
        "arms_list": " ".join(arms),
        "initial_fluents": "\n    ".join(initial_fluents),
        "initial_predicates": "\n    ".join(initial_predicates),
        "goal_conditions": "\n    ".join(goal_conditions),
    }

    return template.substitute(template_mapping)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate delivery planning instances")
    parser.add_argument("--min_locations", required=True)
    parser.add_argument("--max_locations", required=True)
    parser.add_argument("--min_packages", required=True)
    parser.add_argument("--max_packages", required=True)
    parser.add_argument("--max_capacity", required=True)
    parser.add_argument("--max_distance", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--total_num_problems", default=200)
    parser.add_argument("--num_prev_instances", default=0)
    return parser.parse_args()


def generate_multiple_problems(
        output_folder: Path,
        total_num_problems: int,
        num_prev_instances: int = 0,
        **difficulty_kwargs,
) -> None:
    """Generate a batch of delivery instances using the shared workflow API."""
    min_locations = int(difficulty_kwargs.get("min_locations", 3))
    max_locations = int(difficulty_kwargs.get("max_locations", 6))
    min_packages = int(difficulty_kwargs.get("min_packages", 4))
    max_packages = int(difficulty_kwargs.get("max_packages", 42))
    max_capacity = int(difficulty_kwargs.get("max_capacity", 9))
    max_distance = int(difficulty_kwargs.get("max_distance", max_locations))

    output_folder.mkdir(parents=True, exist_ok=True)

    for offset in range(total_num_problems):
        num_locations = random.randint(min_locations, max_locations)
        num_packages = random.randint(min_packages, max_packages)

        with open(output_folder / f"pfile{num_prev_instances + offset}.pddl", "wt") as problem_file:
            problem_file.write(
                generate_instance(
                    f"delivery-generated-{num_prev_instances + offset}",
                    num_locations=num_locations,
                    num_packages=num_packages,
                    max_capacity=max_capacity,
                    max_distance=max_distance,
                )
            )


def main():
    args = parse_arguments()

    generate_multiple_problems(
        output_folder=Path(args.output_path),
        total_num_problems=int(args.total_num_problems),
        num_prev_instances=int(args.num_prev_instances),
        min_locations=int(args.min_locations),
        max_locations=int(args.max_locations),
        min_packages=int(args.min_packages),
        max_packages=int(args.max_packages),
        max_capacity=int(args.max_capacity),
        max_distance=int(args.max_distance),
    )


if __name__ == "__main__":
    main()
