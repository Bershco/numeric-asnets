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


def generate_partition(num_blocks: int, num_groups: int) -> list[list[int]]:
    """Split blocks into non-empty groups that define the target co-location classes."""
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
    """Render one block-grouping instance with unique initial coordinates."""
    template = get_problem_template(TEMPLATE_FILE_PATH)

    blocks = [f"b{i + 1}" for i in range(num_blocks)]
    positions = {}
    used_positions = set()
    initial_fluents = []
    for block in blocks:
        while True:
            position = (random.randint(1, max_coord), random.randint(1, max_coord))
            if position not in used_positions:
                used_positions.add(position)
                positions[block] = position
                break
        x_value, y_value = positions[block]
        initial_fluents.append(f"(= (x {block}) {x_value})")
        initial_fluents.append(f"(= (y {block}) {y_value})")

    groups = generate_partition(num_blocks, min(num_groups, num_blocks))
    goal_conditions = []
    for left_idx in range(num_blocks):
        for right_idx in range(left_idx + 1, num_blocks):
            left_block = blocks[left_idx]
            right_block = blocks[right_idx]
            same_group = any(left_idx in group and right_idx in group for group in groups)
            if same_group:
                goal_conditions.append(f"(= (x {left_block}) (x {right_block}))")
                goal_conditions.append(f"(= (y {left_block}) (y {right_block}))")
            else:
                goal_conditions.append(
                    f"(or (not (= (x {left_block}) (x {right_block}))) (not (= (y {left_block}) (y {right_block}))))"
                )

    template_mapping = {
        "instance_name": instance_name,
        "blocks": " ".join(blocks),
        "initial_fluents": "\n    ".join(initial_fluents),
        "max_coord": max_coord,
        "goal_conditions": "\n    ".join(goal_conditions),
    }
    return template.substitute(template_mapping)


def generate_multiple_problems(
        output_folder,
        total_num_problems,
        num_prev_instances=0,
        min_blocks=5,
        max_blocks=20,
        min_groups=2,
        max_groups=6,
        max_values=40,
        **_,
):
    """Generate a batch of block-grouping instances."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    start_index = num_prev_instances or 0

    for idx in range(total_num_problems):
        num_blocks = random.randint(min_blocks, max_blocks)
        num_groups = random.randint(min_groups, min(max_groups, num_blocks))
        problem = generate_instance(
            f"instance_{max_values}_{num_blocks}_{num_groups}_{start_index + idx + 1}",
            num_blocks=num_blocks,
            num_groups=num_groups,
            max_coord=max_values,
        )
        with open(output_folder / f"pfile{start_index + idx}.pddl", "wt") as problem_file:
            problem_file.write(problem)
