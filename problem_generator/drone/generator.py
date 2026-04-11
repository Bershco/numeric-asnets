#!/usr/bin/python3
from pathlib import Path
import random
import sys

try:
    from problem_generator.common import get_problem_template
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from problem_generator.common import get_problem_template

TEMPLATE_FILE_PATH = Path(__file__).parent / "template.pddl"


def generate_instance(instance_name: str, x_size: int, y_size: int, z_size: int) -> str:
    """Render a drone grid instance that requires visiting every coordinate."""
    template = get_problem_template(TEMPLATE_FILE_PATH)

    locations = []
    coordinate_fluents = []
    goal_conditions = []
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                location = f"x{x}y{y}z{z}"
                locations.append(location)
                coordinate_fluents.extend([
                    f"(= (xl {location}) {x})",
                    f"(= (yl {location}) {y})",
                    f"(= (zl {location}) {z})",
                ])
                goal_conditions.append(f"(visited {location})")

    battery_level = max(13, len(locations) * 4)
    template_mapping = {
        "instance_name": instance_name,
        "x_size": x_size,
        "y_size": y_size,
        "z_size": z_size,
        "locations": "\n".join(locations),
        "coordinate_fluents": "\n".join(coordinate_fluents),
        "battery_level": battery_level,
        "goal_conditions": "\n".join(goal_conditions),
    }
    return template.substitute(template_mapping)


def generate_multiple_problems(
        output_folder,
        total_num_problems,
        num_prev_instances=0,
        min_x=1,
        max_x=8,
        min_y=1,
        max_y=8,
        min_z=1,
        max_z=4,
        **_,
):
    """Generate a batch of drone instances with bounded 3D grid sizes."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    start_index = num_prev_instances or 0
    for idx in range(total_num_problems):
        x_size = random.randint(min_x, max_x)
        y_size = random.randint(min_y, max_y)
        z_size = random.randint(min_z, max_z)
        problem = generate_instance(
            f"droneprob_{x_size}_{y_size}_{z_size}",
            x_size=x_size,
            y_size=y_size,
            z_size=z_size,
        )
        with open(output_folder / f"pfile{start_index + idx}.pddl", "wt") as problem_file:
            problem_file.write(problem)
