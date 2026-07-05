#!/usr/bin/python3

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from pathlib import Path

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
    """Return arm names grouped by bot."""

    base_labels = ["left", "mid", "right", "aux"]

    if arms_per_bot <= len(base_labels):
        arm_labels = base_labels[:arms_per_bot]
    else:
        arm_labels = base_labels + [
            f"aux{idx}"
            for idx in range(arms_per_bot - len(base_labels))
        ]

    arms = []

    for bot_idx in range(1, num_bots + 1):
        for arm_label in arm_labels:
            arms.append(f"{arm_label}{bot_idx}")

    return arms

def build_door_graph(num_locations: int) -> list[tuple[str, str]]:
    """Build a connected directed room graph shaped like the original delivery cases."""

    rooms = generate_room_names(num_locations)

    if num_locations == 1:
        return []

    if num_locations == 2:
        return [
            (rooms[0], rooms[1]),
            (rooms[1], rooms[0]),
        ]

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

    if num_locations == 6:
        return [
            (rooms[0], rooms[1]),
            (rooms[1], rooms[2]),
            (rooms[2], rooms[3]),
            (rooms[3], rooms[4]),
            (rooms[4], rooms[0]),
            (rooms[0], rooms[5]),
            (rooms[5], rooms[3]),
        ]

    # Generic fallback for larger maps:
    # directed cycle + a few forward shortcuts.
    doors = []

    for idx in range(num_locations):
        doors.append((rooms[idx], rooms[(idx + 1) % num_locations]))

    for idx in range(num_locations):
        if idx + 2 < num_locations:
            doors.append((rooms[idx], rooms[idx + 2]))

    return doors


def compute_shortest_distances(
    rooms: list[str],
    doors: list[tuple[str, str]],
    start_room: str,
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


def choose_arms_per_bot(
    num_locations: int,
    num_packages: int,
    min_arms_per_bot: int = 2,
    max_arms_per_bot: int = 8,
) -> int:
    """Choose the number of arms per bot.

    Allows octopus-like instances with many arms.
    """

    if min_arms_per_bot > max_arms_per_bot:
        raise ValueError(
            f"min_arms_per_bot must be <= max_arms_per_bot, got "
            f"{min_arms_per_bot} > {max_arms_per_bot}."
        )

    if num_locations <= 2 and num_packages >= 20:
        return max_arms_per_bot

    if num_packages >= 28:
        return min(max_arms_per_bot, max(min_arms_per_bot, 4))

    if num_packages >= 16:
        return min(max_arms_per_bot, max(min_arms_per_bot, 3))

    return min(max_arms_per_bot, max(min_arms_per_bot, 2))

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
                room
                for room in distances.keys()
                if room != start_room
            ]

        if not target_candidates:
            raise ValueError(
                f"No reachable non-start target room exists from {start_room}."
            )

        max_observed_distance = max(
            distances[room]
            for room in target_candidates
        )

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

    return [
        random.randint(1, max_weight)
        for _ in range(num_packages)
    ]


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
    min_arms_per_bot: int = 2,
    max_arms_per_bot: int = 8,
) -> str:
    """Generate a single delivery planning problem instance.

    :param instance_name: the name of the problem instance.
    :param num_locations: number of rooms.
    :param num_packages: number of delivery items.
    :param max_capacity: maximal bot load capacity.
    :param max_distance: maximal preferred source-target room distance.
    :return: the rendered PDDL problem string.
    """

    template = get_problem_template(TEMPLATE_FILE_PATH)

    rooms = generate_room_names(num_locations)

    num_bots = choose_num_bots(
        num_locations=num_locations,
        num_packages=num_packages,
    )

    arms_per_bot = choose_arms_per_bot(
        num_locations=num_locations,
        num_packages=num_packages,
        min_arms_per_bot=min_arms_per_bot,
        max_arms_per_bot=max_arms_per_bot,
    )

    items = generate_item_names(num_packages)
    bots = generate_bot_names(num_bots)
    arms = generate_arm_names(num_bots, arms_per_bot)
    doors = build_door_graph(num_locations)

    initial_positions, goal_positions = assign_item_rooms(
        items=items,
        rooms=rooms,
        doors=doors,
        max_distance=max_distance,
    )

    weights = generate_weights(
        num_packages=num_packages,
        max_capacity=max_capacity,
    )

    load_limit = choose_load_limit(
        weights=weights,
        max_capacity=max_capacity,
    )

    # ---------------------------------------------------------
    # Initial numeric fluents
    # ---------------------------------------------------------

    initial_fluents = []

    for item, weight in zip(items, weights):
        initial_fluents.append(f"(= (weight {item}) {weight})")

    for bot in bots:
        initial_fluents.append(f"(= (current_load {bot}) 0)")
        initial_fluents.append(f"(= (load_limit {bot}) {load_limit})")

    initial_fluents.append("(= (cost) 0)")

    # ---------------------------------------------------------
    # Initial predicates
    # ---------------------------------------------------------

    initial_predicates = []

    for item in items:
        initial_predicates.append(f"(at {item} {initial_positions[item]})")

    for bot in bots:
        initial_predicates.append(f"(at-bot {bot} {rooms[0]})")

    for arm in arms:
        initial_predicates.append(f"(free {arm})")

    mounted_bots = [
        bot
        for bot in bots
        for _ in range(arms_per_bot)
    ]

    for arm, bot in zip(arms, mounted_bots):
        initial_predicates.append(f"(mount {arm} {bot})")

    for source, target in doors:
        initial_predicates.append(f"(door {source} {target})")

    # ---------------------------------------------------------
    # Goal conditions
    # ---------------------------------------------------------

    goal_conditions = [
        f"(at {item} {goal_positions[item]})"
        for item in items
    ]

    # ---------------------------------------------------------
    # Template mapping
    # ---------------------------------------------------------

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
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate delivery planning instances."
    )

    parser.add_argument(
        "--min_locations",
        type=int,
        required=True,
        help="Minimal number of rooms/locations in each generated problem.",
    )

    parser.add_argument(
        "--max_locations",
        type=int,
        required=True,
        help="Maximal number of rooms/locations in each generated problem.",
    )

    parser.add_argument(
        "--min_packages",
        type=int,
        required=True,
        help="Minimal number of packages/items in each generated problem.",
    )

    parser.add_argument(
        "--max_packages",
        type=int,
        required=True,
        help="Maximal number of packages/items in each generated problem.",
    )

    parser.add_argument(
        "--max_capacity",
        type=int,
        required=True,
        help="Maximal bot load capacity.",
    )

    parser.add_argument(
        "--max_distance",
        type=int,
        required=True,
        help="Maximal preferred source-target room distance.",
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
    min_locations: int = 3,
    max_locations: int = 6,
    min_packages: int = 4,
    max_packages: int = 42,
    max_capacity: int = 9,
    max_distance: int | None = None,
    min_arms_per_bot: int = 2,
    max_arms_per_bot: int = 8,
    **_,
) -> None:
    """Generate multiple delivery problems.

    This keeps compatibility with the shared workflow API where
    output_folder and total_num_problems may be passed externally.

    :param output_folder: folder where generated PDDL files are written.
    :param total_num_problems: number of problems to generate.
    :param num_prev_instances: offset for generated pfile numbering.
    :param min_locations: minimal number of rooms.
    :param max_locations: maximal number of rooms.
    :param min_packages: minimal number of packages.
    :param max_packages: maximal number of packages.
    :param max_capacity: maximal bot load capacity.
    :param max_distance: maximal preferred source-target room distance.
    """

    if max_distance is None:
        max_distance = max_locations

    if min_locations > max_locations:
        raise ValueError(
            f"min_locations must be <= max_locations, got "
            f"{min_locations} > {max_locations}."
        )

    if min_packages > max_packages:
        raise ValueError(
            f"min_packages must be <= max_packages, got "
            f"{min_packages} > {max_packages}."
        )

    if min_locations < 2:
        raise ValueError(
            f"min_locations must be at least 2, got {min_locations}."
        )

    if min_packages < 1:
        raise ValueError(
            f"min_packages must be at least 1, got {min_packages}."
        )

    if max_capacity < 1:
        raise ValueError(
            f"max_capacity must be at least 1, got {max_capacity}."
        )

    if max_distance < 1:
        raise ValueError(
            f"max_distance must be at least 1, got {max_distance}."
        )

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    start_index = num_prev_instances or 0

    for offset in range(total_num_problems):
        problem_index = start_index + offset

        num_locations = random.randint(min_locations, max_locations)
        num_packages = random.randint(min_packages, max_packages)

        problem = generate_instance(
            instance_name=f"delivery-generated-{problem_index}",
            num_locations=num_locations,
            num_packages=num_packages,
            max_capacity=max_capacity,
            max_distance=max_distance,
            min_arms_per_bot=min_arms_per_bot,
            max_arms_per_bot=max_arms_per_bot,
        )

        with open(output_folder / f"pfile{problem_index}.pddl", "wt") as problem_file:
            problem_file.write(problem)


def main() -> None:
    args = parse_arguments()

    generate_multiple_problems(
        output_folder=args.output_path,
        total_num_problems=args.total_num_problems,
        num_prev_instances=args.num_prev_instances,
        min_locations=args.min_locations,
        max_locations=args.max_locations,
        min_packages=args.min_packages,
        max_packages=args.max_packages,
        max_capacity=args.max_capacity,
        max_distance=args.max_distance,
    )


if __name__ == "__main__":
    main()