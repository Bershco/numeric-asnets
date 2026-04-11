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


def build_connected_edges(num_waypoints: int) -> set[tuple[int, int]]:
    """Create a connected visibility graph with a few extra shortcuts."""
    edges = set()
    order = list(range(num_waypoints))
    random.shuffle(order)
    for idx in range(1, num_waypoints):
        src = order[idx]
        dst = random.choice(order[:idx])
        edges.add((src, dst))
        edges.add((dst, src))
    extra_edges = max(num_waypoints, num_waypoints // 2)
    for _ in range(extra_edges):
        src = random.randrange(num_waypoints)
        dst = random.randrange(num_waypoints)
        if src != dst:
            edges.add((src, dst))
            if random.random() < 0.8:
                edges.add((dst, src))
    return edges


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
        max_energy: int,
) -> str:
    """Render a rover instance with reachable samples, images, and communication goals."""
    template = get_problem_template(TEMPLATE_FILE_PATH)

    rovers = [f"rover{i}" for i in range(num_rovers)]
    stores = [f"{rover}store" for rover in rovers]
    waypoints = [f"waypoint{i}" for i in range(num_waypoints)]
    cameras = [f"camera{i}" for i in range(max(1, num_rovers))]
    objectives = [f"objective{i}" for i in range(num_objectives)]
    modes = ["colour", "high_res", "low_res"]

    connectivity = sorted(build_connected_edges(num_waypoints))
    visible_edges = [
        f"(visible {waypoints[src]} {waypoints[dst]})"
        for src, dst in connectivity
    ]
    traverse_edges = []
    for rover in rovers:
        for src, dst in connectivity:
            if random.random() < 0.75:
                traverse_edges.append(f"(can_traverse {rover} {waypoints[src]} {waypoints[dst]})")

    soil_waypoints = random.sample(waypoints, k=max(1, random.randint(1, num_waypoints)))
    rock_waypoints = random.sample(waypoints, k=max(1, random.randint(1, num_waypoints)))
    sunny_waypoints = random.sample(waypoints, k=max(1, random.randint(1, num_waypoints)))

    initial_predicates = []
    initial_predicates.extend(visible_edges)
    initial_predicates.extend(f"(at_soil_sample {waypoint})" for waypoint in soil_waypoints)
    initial_predicates.extend(f"(at_rock_sample {waypoint})" for waypoint in rock_waypoints)
    initial_predicates.extend(f"(in_sun {waypoint})" for waypoint in sunny_waypoints)
    initial_predicates.append(f"(at_lander general {waypoints[0]})")
    initial_predicates.append("(channel_free general)")

    supported_modes_by_camera = {}
    calibration_targets = {}
    visible_from_by_objective = {}
    rover_locations = {}

    for rover, store in zip(rovers, stores):
        rover_locations[rover] = random.choice(waypoints)
        initial_predicates.append(f"(in {rover} {rover_locations[rover]})")
        initial_predicates.append(f"(available {rover})")
        initial_predicates.append(f"(store_of {store} {rover})")
        initial_predicates.append(f"(empty {store})")
        if random.random() < 0.8:
            initial_predicates.append(f"(equipped_for_soil_analysis {rover})")
        if random.random() < 0.8:
            initial_predicates.append(f"(equipped_for_rock_analysis {rover})")
        initial_predicates.append(f"(equipped_for_imaging {rover})")

    if not any("soil_analysis" in predicate for predicate in initial_predicates):
        initial_predicates.append(f"(equipped_for_soil_analysis {rovers[0]})")
    if not any("rock_analysis" in predicate for predicate in initial_predicates):
        initial_predicates.append(f"(equipped_for_rock_analysis {rovers[0]})")

    initial_predicates.extend(traverse_edges)

    for camera in cameras:
        rover = random.choice(rovers)
        supported_modes = [mode for mode in modes if random.random() < 0.6]
        if not supported_modes:
            supported_modes = [random.choice(modes)]
        supported_modes_by_camera[camera] = supported_modes
        calibration_targets[camera] = random.choice(objectives)
        initial_predicates.append(f"(on_board {camera} {rover})")
        initial_predicates.append(f"(calibration_target {camera} {calibration_targets[camera]})")
        initial_predicates.extend(f"(supports {camera} {mode})" for mode in supported_modes)

    for objective in objectives:
        visible_from = random.sample(
            waypoints,
            k=max(1, min(num_waypoints, random.randint(2 if num_waypoints > 1 else 1, num_waypoints))),
        )
        visible_from_by_objective[objective] = visible_from
        initial_predicates.extend(
            f"(visible_from {objective} {waypoint})" for waypoint in visible_from
        )

    initial_fluents = ["(= (recharges) 0)"]
    for rover in rovers:
        initial_fluents.append(f"(= (energy {rover}) {random.randint(max_energy // 2, max_energy)})")

    goal_conditions = []
    for waypoint in random.sample(soil_waypoints, k=max(1, min(len(soil_waypoints), num_objectives))):
        goal_conditions.append(f"(communicated_soil_data {waypoint})")
    for waypoint in random.sample(rock_waypoints, k=max(1, min(len(rock_waypoints), max(1, num_objectives - 1)))):
        goal_conditions.append(f"(communicated_rock_data {waypoint})")
    for camera in cameras:
        objective = calibration_targets[camera]
        mode = choose_goal_mode(supported_modes_by_camera[camera])
        goal_conditions.append(f"(communicated_image_data {objective} {mode})")

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


def generate_multiple_problems(
        output_folder,
        total_num_problems,
        num_prev_instances=0,
        min_rovers=1,
        max_rovers=3,
        min_waypoints=4,
        max_waypoints=9,
        min_objectives=1,
        max_objectives=4,
        max_energy=80,
        **_,
):
    """Generate a batch of rover instances."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    start_index = num_prev_instances or 0
    for idx in range(total_num_problems):
        num_rovers = random.randint(min_rovers, max_rovers)
        num_waypoints = random.randint(max(min_waypoints, num_rovers + 2), max_waypoints)
        num_objectives = random.randint(min_objectives, min(max_objectives, num_waypoints))
        problem = generate_instance(
            f"roverprob-{start_index + idx}",
            num_rovers=num_rovers,
            num_waypoints=num_waypoints,
            num_objectives=num_objectives,
            max_energy=max(max_energy, 20),
        )
        with open(output_folder / f"pfile{start_index + idx}.pddl", "wt") as problem_file:
            problem_file.write(problem)
