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


def generate_instance(instance_name: str, num_counters: int, max_value: int) -> str:
    """Render a functional-counters instance with monotone ordering goals."""
    template = get_problem_template(TEMPLATE_FILE_PATH)
    counters = [f"c{i}" for i in range(num_counters)]
    counter_values = [f"(= (value {counter}) 0)" for counter in counters]
    rate_values = [f"(= (rate_value {counter}) 0)" for counter in counters]
    goal_conditions = [
        f"(<= (+ (value c{i}) 1) (value c{i + 1}))"
        for i in range(num_counters - 1)
    ]
    template_mapping = {
        "instance_name": instance_name,
        "counters": " ".join(counters),
        "max_int": max(max_value, num_counters * 2),
        "counter_values": "\n    ".join(counter_values),
        "rate_values": "\n    ".join(rate_values),
        "goal_conditions": "\n    ".join(goal_conditions),
    }
    return template.substitute(template_mapping)


def generate_multiple_problems(
        output_folder,
        total_num_problems,
        num_prev_instances=0,
        min_counters=2,
        max_counters=20,
        max_value=40,
        **_,
):
    """Generate a batch of fo-counters instances using the shared workflow API."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    start_index = num_prev_instances or 0
    for idx in range(total_num_problems):
        num_counters = random.randint(min_counters, max_counters)
        problem = generate_instance(
            f"instance_{start_index + idx + 2}",
            num_counters=num_counters,
            max_value=max_value,
        )
        with open(output_folder / f"pfile{start_index + idx}.pddl", "wt") as problem_file:
            problem_file.write(problem)
