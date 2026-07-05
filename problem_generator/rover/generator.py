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


def build_connected_edges(num_waypoints: int) -> set[tuple[int, int]]:
    """Create a connected visibility graph with a few extra shortcuts.

    :param num_waypoints: number of waypoints.
    :return: directed edge set over waypoint indices.
    """

    edges = set()

    order = list(range(num_waypoints))
    random.shuffle(order)

    # Random connected backbone.
    for idx in range(1, num_waypoints):
        src = order[idx]
        dst = random.choice(order[:idx])

        edges.add((src, dst))
        edges.add((dst, src))

    # Extra shortcuts.
    extra_edges = max(num_waypoints, num_waypoints // 2)

    for _ in range(extra_edges):
        src = random.randrange(num_waypoints)
        dst = random.randrange(num_waypoints)

        if src == dst:
            continue

        edges.add((src, dst))

        if random.random() < 0.8:
            edges.add((dst, src))

    return edges

def build_rover_traversal_edges(
    num_waypoints: int,
    visibility_edges: set[tuple[int, int]],
    keep_probability: float = 0.35,
) -> set[tuple[int, int]]:
    """Build a connected per-rover traversal graph using visible edges.

    The result is a directed edge set. It guarantees that each rover has
    a connected traversal graph, while still allowing different rovers to
    have different can_traverse predicates.
    """

    neighbors = {idx: set() for idx in range(num_waypoints)}

    for src, dst in visibility_edges:
        if src == dst:
            continue

        neighbors[src].add(dst)
        neighbors[dst].add(src)

    start = random.randrange(num_waypoints)
    connected = {start}
    remaining = set(range(num_waypoints)) - connected
    traversal_edges = set()

    # Randomized spanning tree over the visible graph.
    while remaining:
        candidates = []

        for src in connected:
            for dst in neighbors[src]:
                if dst in remaining:
                    candidates.append((src, dst))

        if not candidates:
            raise ValueError(
                "Visibility graph is not connected; cannot build rover traversal graph."
            )

        src, dst = random.choice(candidates)

        traversal_edges.add((src, dst))
        traversal_edges.add((dst, src))

        connected.add(dst)
        remaining.remove(dst)

    # Add extra directed traversal edges.
    for src, dst in visibility_edges:
        if src != dst and random.random() < keep_probability:
            traversal_edges.add((src, dst))

    return traversal_edges

def choose_goal_mode(supported_modes: list[str]) -> str:
    """Prefer the more informative imaging modes when they are available."""

    preferred_order = ["high_res", "colour", "low_res"]

    for mode in preferred_order:
        if mode in supported_modes:
            return mode

    return supported_modes[0]


def generate_instance(
    instance_name: str,
    num_rovers: int,
    num_waypoints: int,
    num_objectives: int,
    num_cameras: int,
    max_energy: int,
    traverse_keep_probability: float = 0.35,
) -> str:
    """Generate a single rover planning problem instance.

    :param instance_name: the name of the problem instance.
    :param num_rovers: number of rovers.
    :param num_waypoints: number of waypoints.
    :param num_objectives: number of objectives.
    :param max_energy: maximal initial rover energy.
    :return: the rendered PDDL problem string.
    """

    template = get_problem_template(TEMPLATE_FILE_PATH)

    rovers = [f"rover{i}" for i in range(num_rovers)]
    forced_soil_rover = random.choice(rovers)
    forced_rock_rover = random.choice(rovers)
    stores = [f"{rover}store" for rover in rovers]
    waypoints = [f"waypoint{i}" for i in range(num_waypoints)]
    cameras = [f"camera{i}" for i in range(num_cameras)]
    objectives = [f"objective{i}" for i in range(num_objectives)]
    modes = ["colour", "high_res", "low_res"]

    camera_to_rover = {
        camera: random.choice(rovers)
        for camera in cameras
    }

    imaging_rovers = set(camera_to_rover.values())

    connectivity = sorted(build_connected_edges(num_waypoints))

    # ---------------------------------------------------------
    # Visibility / traversal graph
    # ---------------------------------------------------------

    visible_edges = [
        f"(visible {waypoints[src]} {waypoints[dst]})"
        for src, dst in connectivity
    ]

    # ---------------------------------------------------------
    # Sample / sun locations
    # ---------------------------------------------------------

    soil_waypoints = random.sample(
        waypoints,
        k=max(1, random.randint(1, num_waypoints)),
    )

    rock_waypoints = random.sample(
        waypoints,
        k=max(1, random.randint(1, num_waypoints)),
    )

    sunny_waypoints = random.sample(
        waypoints,
        k=max(1, random.randint(1, num_waypoints)),
    )

    # ---------------------------------------------------------
    # Initial predicates
    # ---------------------------------------------------------

    initial_predicates = []

    initial_predicates.extend(visible_edges)
    initial_predicates.extend(
        f"(at_soil_sample {waypoint})"
        for waypoint in soil_waypoints
    )
    initial_predicates.extend(
        f"(at_rock_sample {waypoint})"
        for waypoint in rock_waypoints
    )
    initial_predicates.extend(
        f"(in_sun {waypoint})"
        for waypoint in sunny_waypoints
    )
    lander_waypoint = random.choice(waypoints)
    initial_predicates.append(f"(at_lander general {lander_waypoint})")

    initial_predicates.append("(channel_free general)")

    supported_modes_by_camera = {}
    calibration_targets = {}

    for rover, store in zip(rovers, stores):
        rover_location = random.choice(waypoints)

        initial_predicates.append(f"(in {rover} {rover_location})")
        initial_predicates.append(f"(available {rover})")
        initial_predicates.append(f"(store_of {store} {rover})")
        initial_predicates.append(f"(empty {store})")

        if rover == forced_soil_rover or random.random() < 0.7:
            initial_predicates.append(f"(equipped_for_soil_analysis {rover})")

        if rover == forced_rock_rover or random.random() < 0.7:
            initial_predicates.append(f"(equipped_for_rock_analysis {rover})")

        if rover in imaging_rovers:
            initial_predicates.append(f"(equipped_for_imaging {rover})")

    if not any("equipped_for_soil_analysis" in pred for pred in initial_predicates):
        initial_predicates.append(f"(equipped_for_soil_analysis {rovers[0]})")

    if not any("equipped_for_rock_analysis" in pred for pred in initial_predicates):
        initial_predicates.append(f"(equipped_for_rock_analysis {rovers[0]})")

    for rover in rovers:
        rover_edges = build_rover_traversal_edges(
            num_waypoints=num_waypoints,
            visibility_edges=set(connectivity),
            keep_probability=traverse_keep_probability,
        )

        for src, dst in sorted(rover_edges):
            initial_predicates.append(
                f"(can_traverse {rover} {waypoints[src]} {waypoints[dst]})"
            )

    for camera in cameras:
        rover = camera_to_rover[camera]

        supported_modes = [
            mode
            for mode in modes
            if random.random() < 0.6
        ]

        if not supported_modes:
            supported_modes = [random.choice(modes)]

        calibration_target = random.choice(objectives)

        supported_modes_by_camera[camera] = supported_modes
        calibration_targets[camera] = calibration_target

        initial_predicates.append(f"(on_board {camera} {rover})")
        initial_predicates.append(
            f"(calibration_target {camera} {calibration_target})"
        )
        initial_predicates.extend(
            f"(supports {camera} {mode})"
            for mode in supported_modes
        )

    for objective in objectives:
        visible_waypoints = random.sample(
            waypoints,
            k=random.randint(1, min(3, len(waypoints))),
        )

        for waypoint in visible_waypoints:
            initial_predicates.append(f"(visible_from {objective} {waypoint})")

    # ---------------------------------------------------------
    # Initial numeric fluents
    # ---------------------------------------------------------

    initial_fluents = ["(= (recharges) 0)"]

    for rover in rovers:
        initial_fluents.append(
            f"(= (energy {rover}) "
            f"{random.randint(max_energy // 2, max_energy)})"
        )

    # ---------------------------------------------------------
    # Goal conditions
    # ---------------------------------------------------------

    goal_conditions = []

    soil_goal_count = max(1, min(len(soil_waypoints), num_objectives))
    rock_goal_count = max(1, min(len(rock_waypoints), max(1, num_objectives - 1)))

    for waypoint in random.sample(soil_waypoints, k=soil_goal_count):
        goal_conditions.append(f"(communicated_soil_data {waypoint})")

    for waypoint in random.sample(rock_waypoints, k=rock_goal_count):
        goal_conditions.append(f"(communicated_rock_data {waypoint})")

    valid_goal_cameras = [
        camera
        for camera in cameras
        if supported_modes_by_camera.get(camera)
    ]

    if valid_goal_cameras:
        num_image_goals = random.randint(
            1,
            min(len(valid_goal_cameras), max(1, num_objectives)),
        )

        goal_cameras = random.sample(valid_goal_cameras, k=num_image_goals)

        for camera in goal_cameras:
            objective = calibration_targets[camera]
            mode = choose_goal_mode(supported_modes_by_camera[camera])
            goal_conditions.append(f"(communicated_image_data {objective} {mode})")

    # ---------------------------------------------------------
    # Template mapping
    # ---------------------------------------------------------

    template_mapping = {
        "instance_name": instance_name,
        "rovers": " ".join(rovers),
        "stores": " ".join(stores),
        "waypoints": " ".join(waypoints),
        "cameras": " ".join(cameras),
        "objectives": " ".join(objectives),
        "initial_predicates": "\n    ".join(initial_predicates),
        "initial_fluents": "\n    ".join(initial_fluents),
        "goal_conditions": "\n    ".join(goal_conditions),
    }

    return template.substitute(template_mapping)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate rover planning instances."
    )

    parser.add_argument(
        "--min_rovers",
        type=int,
        required=True,
        help="Minimal number of rovers in each generated problem.",
    )

    parser.add_argument(
        "--max_rovers",
        type=int,
        required=True,
        help="Maximal number of rovers in each generated problem.",
    )

    parser.add_argument(
        "--min_waypoints",
        type=int,
        required=True,
        help="Minimal number of waypoints in each generated problem.",
    )

    parser.add_argument(
        "--max_waypoints",
        type=int,
        required=True,
        help="Maximal number of waypoints in each generated problem.",
    )

    parser.add_argument(
        "--min_objectives",
        type=int,
        required=True,
        help="Minimal number of objectives in each generated problem.",
    )

    parser.add_argument(
        "--max_objectives",
        type=int,
        required=True,
        help="Maximal number of objectives in each generated problem.",
    )

    parser.add_argument(
        "--max_energy",
        type=int,
        required=True,
        help="Maximal initial rover energy.",
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
    min_rovers: int = 1,
    max_rovers: int = 3,
    min_waypoints: int = 4,
    max_waypoints: int = 9,
    min_objectives: int = 1,
    max_objectives: int = 4,
    min_cameras: int = 1,
    max_cameras: int = 8,
    max_energy: int = 80,
    traverse_keep_probability: float = 0.35,
    **_,
) -> None:
    """Generate multiple rover problems.

    This keeps compatibility with the shared workflow API where
    output_folder and total_num_problems may be passed externally.

    :param output_folder: folder where generated PDDL files are written.
    :param total_num_problems: number of problems to generate.
    :param num_prev_instances: offset for generated pfile numbering.
    :param min_rovers: minimal number of rovers.
    :param max_rovers: maximal number of rovers.
    :param min_waypoints: minimal number of waypoints.
    :param max_waypoints: maximal number of waypoints.
    :param min_objectives: minimal number of objectives.
    :param max_objectives: maximal number of objectives.
    :param max_energy: maximal initial rover energy.
    """

    if min_rovers > max_rovers:
        raise ValueError(
            f"min_rovers must be <= max_rovers, got "
            f"{min_rovers} > {max_rovers}."
        )

    if min_waypoints > max_waypoints:
        raise ValueError(
            f"min_waypoints must be <= max_waypoints, got "
            f"{min_waypoints} > {max_waypoints}."
        )

    if min_objectives > max_objectives:
        raise ValueError(
            f"min_objectives must be <= max_objectives, got "
            f"{min_objectives} > {max_objectives}."
        )

    if min_rovers < 1:
        raise ValueError(f"min_rovers must be at least 1, got {min_rovers}.")

    if min_waypoints < 2:
        raise ValueError(
            f"min_waypoints must be at least 2, got {min_waypoints}."
        )

    if min_objectives < 1:
        raise ValueError(
            f"min_objectives must be at least 1, got {min_objectives}."
        )

    if max_energy < 1:
        raise ValueError(f"max_energy must be positive, got {max_energy}.")

    if min_cameras > max_cameras:
        raise ValueError(
            f"min_cameras must be <= max_cameras, got "
            f"{min_cameras} > {max_cameras}."
        )

    if min_cameras < 1:
        raise ValueError(
            f"min_cameras must be at least 1, got {min_cameras}."
        )

    max_energy = max(max_energy, 20)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    start_index = num_prev_instances or 0

    for idx in range(total_num_problems):
        problem_index = start_index + idx

        num_rovers = random.randint(min_rovers, max_rovers)

        effective_min_waypoints = max(
            min_waypoints,
            num_rovers + 2,
        )

        if effective_min_waypoints > max_waypoints:
            raise ValueError(
                f"Cannot generate a problem with num_rovers={num_rovers}: "
                f"effective_min_waypoints={effective_min_waypoints} exceeds "
                f"max_waypoints={max_waypoints}."
            )

        num_waypoints = random.randint(
            effective_min_waypoints,
            max_waypoints,
        )

        effective_max_objectives = min(max_objectives, num_waypoints)

        if min_objectives > effective_max_objectives:
            raise ValueError(
                f"Cannot generate a problem with num_waypoints={num_waypoints}: "
                f"min_objectives={min_objectives} exceeds "
                f"effective_max_objectives={effective_max_objectives}."
            )

        num_objectives = random.randint(
            min_objectives,
            effective_max_objectives,
        )

        num_cameras = random.randint(min_cameras, max_cameras)

        problem = generate_instance(
            instance_name=f"roverprob-{problem_index}",
            num_rovers=num_rovers,
            num_waypoints=num_waypoints,
            num_objectives=num_objectives,
            num_cameras=num_cameras,
            max_energy=max_energy,
            traverse_keep_probability=traverse_keep_probability,
        )

        with open(output_folder / f"pfile{problem_index}.pddl", "wt") as problem_file:
            problem_file.write(problem)


def main() -> None:
    args = parse_arguments()

    generate_multiple_problems(
        output_folder=args.output_path,
        total_num_problems=args.total_num_problems,
        num_prev_instances=args.num_prev_instances,
        min_rovers=args.min_rovers,
        max_rovers=args.max_rovers,
        min_waypoints=args.min_waypoints,
        max_waypoints=args.max_waypoints,
        min_objectives=args.min_objectives,
        max_objectives=args.max_objectives,
        max_energy=args.max_energy,
    )


if __name__ == "__main__":
    main()