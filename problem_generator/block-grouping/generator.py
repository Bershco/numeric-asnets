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


def generate_partition(num_blocks: int, num_groups: int) -> list[list[int]]:
    """Split blocks into non-empty groups.

    The groups define the target co-location classes: blocks in the same
    group should end up with identical x/y coordinates, while blocks in
    different groups should not be co-located.

    :param num_blocks: number of blocks.
    :param num_groups: requested number of groups.
    :return: list of block-index groups.
    """

    num_groups = min(num_groups, num_blocks)

    groups = [[] for _ in range(num_groups)]
    block_indices = list(range(num_blocks))
    random.shuffle(block_indices)

    for idx, block_idx in enumerate(block_indices):
        groups[idx % num_groups].append(block_idx)

    return [group for group in groups if group]


def generate_instance(
    instance_name: str,
    num_blocks: int,
    num_groups: int,
    max_coord: int,
) -> str:
    """Generate a single block-grouping planning problem instance.

    :param instance_name: the name of the problem instance.
    :param num_blocks: the number of blocks in the problem.
    :param num_groups: the number of target block groups.
    :param max_coord: maximal coordinate value.
    :return: the rendered PDDL problem string.
    """

    if num_blocks > max_coord * max_coord:
        raise ValueError(
            f"Cannot place {num_blocks} blocks in unique initial positions "
            f"with max_coord={max_coord}; only {max_coord * max_coord} "
            f"unique coordinates exist."
        )

    template = get_problem_template(TEMPLATE_FILE_PATH)

    blocks = [f"b{i + 1}" for i in range(num_blocks)]

    # ---------------------------------------------------------
    # Initial values
    # Unique random initial coordinates for every block
    # ---------------------------------------------------------

    initial_fluents = []

    for block in blocks:
        x_value = random.randint(1, max_coord)
        y_value = random.randint(1, max_coord)

        initial_fluents.append(f"(= (x {block}) {x_value})")
        initial_fluents.append(f"(= (y {block}) {y_value})")

    # ---------------------------------------------------------
    # Goal generation
    # Same group  -> same x and same y
    # Diff group  -> not both x and y equal
    # ---------------------------------------------------------

    groups = generate_partition(
        num_blocks=num_blocks,
        num_groups=num_groups,
    )

    block_to_group = {}

    for group_idx, group in enumerate(groups):
        for block_idx in group:
            block_to_group[block_idx] = group_idx

    goal_conditions = []

    for left_idx in range(num_blocks):
        for right_idx in range(left_idx + 1, num_blocks):
            left_block = blocks[left_idx]
            right_block = blocks[right_idx]

            same_group = block_to_group[left_idx] == block_to_group[right_idx]

            if same_group:
                goal_conditions.append(
                    f"(= (x {left_block}) (x {right_block}))"
                )
                goal_conditions.append(
                    f"(= (y {left_block}) (y {right_block}))"
                )
            else:
                goal_conditions.append(
                    f"(or "
                    f"(not (= (x {left_block}) (x {right_block}))) "
                    f"(not (= (y {left_block}) (y {right_block})))"
                    f")"
                )

    # ---------------------------------------------------------
    # Template mapping
    # ---------------------------------------------------------

    template_mapping = {
        "instance_name": instance_name,
        "blocks": " ".join(blocks),
        "initial_fluents": "\n    ".join(initial_fluents),
        "max_coord": max_coord,
        "goal_conditions": "\n    ".join(goal_conditions),
    }

    return template.substitute(template_mapping)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate block-grouping planning instances."
    )

    parser.add_argument(
        "--min_blocks",
        type=int,
        required=True,
        help="Minimal number of blocks in each generated problem.",
    )

    parser.add_argument(
        "--max_blocks",
        type=int,
        required=True,
        help="Maximal number of blocks in each generated problem.",
    )

    parser.add_argument(
        "--min_groups",
        type=int,
        required=True,
        help="Minimal number of target block groups.",
    )

    parser.add_argument(
        "--max_groups",
        type=int,
        required=True,
        help="Maximal number of target block groups.",
    )

    parser.add_argument(
        "--max_coord",
        "--max_values",
        dest="max_coord",
        type=int,
        required=True,
        help=(
            "Maximal coordinate value. "
            "The alias --max_values is kept for compatibility."
        ),
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
    min_blocks: int = 5,
    max_blocks: int = 20,
    min_groups: int = 2,
    max_groups: int = 6,
    max_values: int = 40,
    max_coord: Optional[int] = None,
    **_,
) -> None:
    """Generate multiple block-grouping problems.

    This keeps compatibility with the shared workflow API where
    output_folder and total_num_problems may be passed externally.

    :param output_folder: folder where generated PDDL files are written.
    :param total_num_problems: number of problems to generate.
    :param num_prev_instances: offset for generated pfile numbering.
    :param min_blocks: minimal number of blocks.
    :param max_blocks: maximal number of blocks.
    :param min_groups: minimal number of target groups.
    :param max_groups: maximal number of target groups.
    :param max_values: backwards-compatible alias for max_coord.
    :param max_coord: maximal coordinate value.
    """

    if max_coord is None:
        max_coord = max_values

    if min_blocks > max_blocks:
        raise ValueError(
            f"min_blocks must be <= max_blocks, got "
            f"{min_blocks} > {max_blocks}."
        )

    if min_groups > max_groups:
        raise ValueError(
            f"min_groups must be <= max_groups, got "
            f"{min_groups} > {max_groups}."
        )

    if max_coord <= 0:
        raise ValueError(f"max_coord must be positive, got {max_coord}.")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    start_index = num_prev_instances or 0

    for idx in range(total_num_problems):
        problem_index = start_index + idx

        num_blocks = random.randint(min_blocks, max_blocks)

        effective_max_groups = min(max_groups, num_blocks)
        effective_min_groups = min(min_groups, effective_max_groups)

        num_groups = random.randint(effective_min_groups, effective_max_groups)

        problem = generate_instance(
            instance_name=(
                f"instance_{max_coord}_{num_blocks}_{num_groups}_{problem_index + 1}"
            ),
            num_blocks=num_blocks,
            num_groups=num_groups,
            max_coord=max_coord,
        )

        with open(output_folder / f"pfile{problem_index}.pddl", "wt") as problem_file:
            problem_file.write(problem)


def main() -> None:
    args = parse_arguments()

    generate_multiple_problems(
        output_folder=args.output_path,
        total_num_problems=args.total_num_problems,
        num_prev_instances=args.num_prev_instances,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        min_groups=args.min_groups,
        max_groups=args.max_groups,
        max_coord=args.max_coord,
    )


if __name__ == "__main__":
    main()