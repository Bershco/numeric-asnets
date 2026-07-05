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


def generate_instance(
    instance_name: str,
    x_size: int,
    y_size: int,
    z_size: int,
) -> str:
    """Generate a single drone grid planning problem instance.

    The drone must visit every coordinate in a bounded 3D grid.

    :param instance_name: the name of the problem instance.
    :param x_size: grid size along the x dimension.
    :param y_size: grid size along the y dimension.
    :param z_size: grid size along the z dimension.
    :return: the rendered PDDL problem string.
    """

    template = get_problem_template(TEMPLATE_FILE_PATH)

    locations = []
    coordinate_fluents = []
    goal_conditions = []

    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                location = f"x{x}y{y}z{z}"

                locations.append(location)

                coordinate_fluents.extend(
                    [
                        f"(= (xl {location}) {x})",
                        f"(= (yl {location}) {y})",
                        f"(= (zl {location}) {z})",
                    ]
                )

                goal_conditions.append(f"(visited {location})")

    battery_level = 2 * (x_size + y_size + z_size) + 1

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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate drone grid planning instances."
    )

    parser.add_argument(
        "--min_x",
        type=int,
        required=True,
        help="Minimal grid size along the x dimension.",
    )

    parser.add_argument(
        "--max_x",
        type=int,
        required=True,
        help="Maximal grid size along the x dimension.",
    )

    parser.add_argument(
        "--min_y",
        type=int,
        required=True,
        help="Minimal grid size along the y dimension.",
    )

    parser.add_argument(
        "--max_y",
        type=int,
        required=True,
        help="Maximal grid size along the y dimension.",
    )

    parser.add_argument(
        "--min_z",
        type=int,
        required=True,
        help="Minimal grid size along the z dimension.",
    )

    parser.add_argument(
        "--max_z",
        type=int,
        required=True,
        help="Maximal grid size along the z dimension.",
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
    min_x: int = 1,
    max_x: int = 8,
    min_y: int = 1,
    max_y: int = 8,
    min_z: int = 1,
    max_z: int = 4,
    **_,
) -> None:
    """Generate multiple drone grid problems.

    This keeps compatibility with the shared workflow API where
    output_folder and total_num_problems may be passed externally.

    :param output_folder: folder where generated PDDL files are written.
    :param total_num_problems: number of problems to generate.
    :param num_prev_instances: offset for generated pfile numbering.
    :param min_x: minimal x grid size.
    :param max_x: maximal x grid size.
    :param min_y: minimal y grid size.
    :param max_y: maximal y grid size.
    :param min_z: minimal z grid size.
    :param max_z: maximal z grid size.
    """

    if min_x > max_x:
        raise ValueError(f"min_x must be <= max_x, got {min_x} > {max_x}.")

    if min_y > max_y:
        raise ValueError(f"min_y must be <= max_y, got {min_y} > {max_y}.")

    if min_z > max_z:
        raise ValueError(f"min_z must be <= max_z, got {min_z} > {max_z}.")

    if min_x < 1 or min_y < 1 or min_z < 1:
        raise ValueError(
            f"Grid dimensions must be positive, got "
            f"min_x={min_x}, min_y={min_y}, min_z={min_z}."
        )

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    start_index = num_prev_instances or 0

    for idx in range(total_num_problems):
        problem_index = start_index + idx

        x_size = random.randint(min_x, max_x)
        y_size = random.randint(min_y, max_y)
        z_size = random.randint(min_z, max_z)

        problem = generate_instance(
            instance_name=f"droneprob_{x_size}_{y_size}_{z_size}_{problem_index}",
            x_size=x_size,
            y_size=y_size,
            z_size=z_size,
        )

        with open(output_folder / f"pfile{problem_index}.pddl", "wt") as problem_file:
            problem_file.write(problem)


def main() -> None:
    args = parse_arguments()

    generate_multiple_problems(
        output_folder=args.output_path,
        total_num_problems=args.total_num_problems,
        num_prev_instances=args.num_prev_instances,
        min_x=args.min_x,
        max_x=args.max_x,
        min_y=args.min_y,
        max_y=args.max_y,
        min_z=args.min_z,
        max_z=args.max_z,
    )


if __name__ == "__main__":
    main()