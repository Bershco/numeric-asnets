#!/usr/bin/python3

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


def generate_instance(instance_name: str, num_counters: int, max_int: int) -> str:
    """Generate a single functional-counters planning problem instance.

    The generated goal is a shuffled monotone chain of relational constraints:

        value(c_j) >= value(c_i) + 1

    :param instance_name: the name of the problem instance.
    :param num_counters: the number of counters in the problem.
    :param max_int: the maximal integer value.
    :return: the rendered PDDL problem string.
    """

    template = get_problem_template(TEMPLATE_FILE_PATH)

    # Make sure the max integer is large enough for an ordered chain.
    effective_max_int = max(max_int, num_counters * 2)

    counters = [f"c{i}" for i in range(num_counters)]

    # ---------------------------------------------------------
    # Initial values
    # ---------------------------------------------------------

    initial_values = [
        random.randint(0, effective_max_int)
        for _ in range(num_counters)
    ]

    counter_values = [
        f"(= (value c{i}) {initial_values[i]})"
        for i in range(num_counters)
    ]

    # Keep rate values initialized to 0, as in your original generator.
    rate_values = [
        f"(= (rate_value c{i}) 0)"
        for i in range(num_counters)
    ]

    # ---------------------------------------------------------
    # Goal generation
    # Structured but shuffled monotone counter chain
    # ---------------------------------------------------------

    step = 1

    order = list(range(num_counters))
    random.shuffle(order)

    goal_conditions = []

    for idx in range(num_counters - 1):
        i = order[idx]
        j = order[idx + 1]

        goal_conditions.append(
            f"(<= (+ (value c{i}) {step}) (value c{j}))"
        )

    # ---------------------------------------------------------
    # Template mapping
    # ---------------------------------------------------------

    template_mapping = {
        "instance_name": instance_name,
        "counters": " ".join(counters),
        "max_int": effective_max_int,
        "counter_values": "\n    ".join(counter_values),
        "rate_values": "\n    ".join(rate_values),
        "goal_conditions": "\n    ".join(goal_conditions),
    }

    return template.substitute(template_mapping)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate functional-counters planning instances."
    )

    parser.add_argument(
        "--min_counters",
        type=int,
        required=True,
        help="Minimal number of counters in each generated problem.",
    )

    parser.add_argument(
        "--max_counters",
        type=int,
        required=True,
        help="Maximal number of counters in each generated problem.",
    )

    parser.add_argument(
        "--max_int",
        type=int,
        required=True,
        help="Maximal integer value used in the generated problems.",
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
    min_counters: int = 2,
    max_counters: int = 20,
    max_int: int = 40,
    max_value: Optional[int] = None,
    **_,
) -> None:
    """Generate multiple functional-counters problems.

    This keeps compatibility with the shared workflow API where
    output_folder and total_num_problems may be passed externally.

    :param output_folder: folder where generated PDDL files are written.
    :param total_num_problems: number of problems to generate.
    :param num_prev_instances: offset for generated pfile numbering.
    :param min_counters: minimal number of counters.
    :param max_counters: maximal number of counters.
    :param max_int: maximal integer value.
    :param max_value: backwards-compatible alias for max_int.
    """

    if max_value is not None:
        max_int = max_value

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    start_index = num_prev_instances or 0

    for idx in range(total_num_problems):
        num_counters = random.randint(min_counters, max_counters)
        problem_index = start_index + idx

        problem = generate_instance(
            instance_name=f"instance_{problem_index + 2}",
            num_counters=num_counters,
            max_int=max_int,
        )

        with open(output_folder / f"pfile{problem_index}.pddl", "wt") as problem_file:
            problem_file.write(problem)


def main() -> None:
    args = parse_arguments()

    generate_multiple_problems(
        output_folder=args.output_path,
        total_num_problems=args.total_num_problems,
        num_prev_instances=args.num_prev_instances,
        min_counters=args.min_counters,
        max_counters=args.max_counters,
        max_int=args.max_int,
    )


if __name__ == "__main__":
    main()