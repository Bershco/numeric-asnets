#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType
import inspect
import csv
import re
from collections import defaultdict

DIFFICULTIES = ("easy", "medium", "hard")


DOMAINS = {
    "block-grouping": {
        "generator_relpath": Path("block-grouping") / "generator.py",
        "difficulties": {
            "easy": dict(
                min_blocks=5,
                max_blocks=10,
                min_groups=2,
                max_groups=3,
                max_values=10,
            ),
            "medium": dict(
                min_blocks=12,
                max_blocks=20,
                min_groups=3,
                max_groups=5,
                max_values=20,
            ),
            "hard": dict(
                min_blocks=25,
                max_blocks=40,
                min_groups=4,
                max_groups=8,
                max_values=40,
            ),
        },
    },

    "delivery": {
        "generator_relpath": Path("delivery") / "generator.py",
        "difficulties": {
            "easy": dict(
                min_locations=3,
                max_locations=4,
                min_packages=4,
                max_packages=8,
                max_capacity=4,
                max_distance=2,
                min_arms_per_bot=2,
                max_arms_per_bot=2,
            ),
            "medium": dict(
                min_locations=4,
                max_locations=5,
                min_packages=10,
                max_packages=20,
                max_capacity=7,
                max_distance=3,
                min_arms_per_bot=2,
                max_arms_per_bot=4,
            ),
            "hard": dict(
                min_locations=5,
                max_locations=6,
                min_packages=22,
                max_packages=42,
                max_capacity=9,
                max_distance=4,
                min_arms_per_bot=3,
                max_arms_per_bot=8,
            ),
        },
    },

    "drone": {
        "generator_relpath": Path("drone") / "generator.py",
        "difficulties": {
            "easy": dict(
                min_x=1,
                max_x=2,
                min_y=1,
                max_y=2,
                min_z=1,
                max_z=2,
            ),
            "medium": dict(
                min_x=2,
                max_x=4,
                min_y=2,
                max_y=4,
                min_z=2,
                max_z=3,
            ),
            "hard": dict(
                min_x=4,
                max_x=8,
                min_y=4,
                max_y=8,
                min_z=3,
                max_z=4,
            ),
        },
    },

    "fo-counters": {
        "generator_relpath": Path("fo-counters") / "generator.py",
        "difficulties": {
            "easy": dict(
                min_counters=2,
                max_counters=4,
                max_value=10,
            ),
            "medium": dict(
                min_counters=5,
                max_counters=8,
                max_value=20,
            ),
            "hard": dict(
                min_counters=9,
                max_counters=16,
                max_value=40,
            ),
        },
    },

    "mprime": {
        "generator_relpath": Path("mprime") / "generator.py",
        "difficulties": {
            "easy": dict(
                min_locations=4,
                max_locations=5,
                min_keys=1,
                max_keys=2,
                max_fuel=5,
            ),
            "medium": dict(
                min_locations=6,
                max_locations=7,
                min_keys=2,
                max_keys=3,
                max_fuel=7,
            ),
            "hard": dict(
                min_locations=8,
                max_locations=10,
                min_keys=3,
                max_keys=5,
                max_fuel=9,
            ),
        },
    },

    "rover": {
        "generator_relpath": Path("rover") / "generator.py",
        "difficulties": {
            "easy": dict(
                min_rovers=1,
                max_rovers=2,
                min_waypoints=4,
                max_waypoints=6,
                min_objectives=1,
                max_objectives=2,
                min_cameras=1,
                max_cameras=3,
                max_energy=50,
                traverse_keep_probability=0.35,
            ),
            "medium": dict(
                min_rovers=2,
                max_rovers=3,
                min_waypoints=6,
                max_waypoints=8,
                min_objectives=2,
                max_objectives=3,
                min_cameras=2,
                max_cameras=5,
                max_energy=80,
                traverse_keep_probability=0.35,
            ),
            "hard": dict(
                min_rovers=3,
                max_rovers=5,
                min_waypoints=8,
                max_waypoints=14,
                min_objectives=3,
                max_objectives=5,
                min_cameras=3,
                max_cameras=5,
                max_energy=120,
                traverse_keep_probability=0.40,
            ),
        },
    },

    "tpp": {
        "generator_relpath": Path("tpp") / "generator.py",
        "difficulties": {
            "easy": dict(
                min_markets=4,
                max_markets=5,
                min_products=2,
                max_products=3,
                max_cost=30,
                max_capacity=10,
                min_active_fraction=0.40,
                max_active_fraction=0.80,
            ),
            "medium": dict(
                min_markets=6,
                max_markets=12,
                min_products=3,
                max_products=8,
                max_cost=40,
                max_capacity=15,
                min_active_fraction=0.40,
                max_active_fraction=0.80,
            ),
            "hard": dict(
                min_markets=10,
                max_markets=18,
                min_products=6,
                max_products=12,
                max_cost=50,
                max_capacity=20,
                min_active_fraction=0.30,
                max_active_fraction=0.55,
            ),
        },
    },

    "zenotravel": {
        "generator_relpath": Path("zenotravel") / "generator.py",
        "difficulties": {
            "easy": dict(
                min_cities=3,
                max_cities=4,
                min_people=2,
                max_people=4,
                min_aircraft=1,
                max_aircraft=1,
                min_fuel=1,
                max_fuel=2500,
                min_capacity=100,
                max_capacity=7500,
                max_distance=1250,
            ),
            "medium": dict(
                min_cities=5,
                max_cities=6,
                min_people=4,
                max_people=6,
                min_aircraft=1,
                max_aircraft=2,
                min_fuel=1,
                max_fuel=4000,
                min_capacity=100,
                max_capacity=9000,
                max_distance=2000,
            ),
            "hard": dict(
                min_cities=8,
                max_cities=10,
                min_people=10,
                max_people=16,
                min_aircraft=2,
                max_aircraft=2,
                min_fuel=2500,
                max_fuel=5000,
                min_capacity=7000,
                max_capacity=10000,
                max_distance=500,
            ),
        },
    },
}

PFILE_RE = re.compile(r"^pfile(\d+)\.pddl$")


def parse_problem_index(problem_name: str) -> int:
    match = PFILE_RE.match(problem_name)

    if match is None:
        raise ValueError(
            f"Expected problem filename like pfile7.pddl, got {problem_name!r}."
        )

    return int(match.group(1))


def parse_status_set(raw: str) -> set[str]:
    return {
        item.strip()
        for item in raw.split(",")
        if item.strip()
    }


def problem_name_from_audit_row(row: dict[str, str]) -> str:
    if row.get("problem_path"):
        return Path(row["problem_path"]).name

    if row.get("problem_key"):
        return Path(row["problem_key"]).name

    raise ValueError(
        f"Audit row has neither problem_path nor problem_key: {row}"
    )


def load_regeneration_targets_from_audit_csv(
    *,
    audit_csv: Path,
    target_statuses: set[str],
) -> dict[tuple[str, str], set[int]]:
    """Return mapping: (domain, difficulty) -> pfile indices to regenerate."""

    targets: dict[tuple[str, str], set[int]] = defaultdict(set)

    with audit_csv.open(newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {"domain", "difficulty", "status"}

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"{audit_csv} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            status = row.get("status", "").strip()

            if status not in target_statuses:
                continue

            domain = row["domain"].strip()
            difficulty = row["difficulty"].strip()
            problem_name = problem_name_from_audit_row(row)
            problem_index = parse_problem_index(problem_name)

            targets[(domain, difficulty)].add(problem_index)

    return targets

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate validation instances for all synthetic domains except counters."
        )
    )

    parser.add_argument(
        "--problem-generator-root",
        type=Path,
        required=True,
        help=(
            "Path to the problem_generator directory, i.e. the directory "
            "containing block-grouping/, delivery/, drone/, etc."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root output directory for generated instances.",
    )

    parser.add_argument(
        "--instances-per-difficulty",
        type=int,
        default=10,
        help="Number of instances to generate per domain difficulty. Default: 10.",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete each target difficulty directory before generating. "
            "Use this if you want exactly N fresh files per difficulty."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files.",
    )

    parser.add_argument(
        "--regenerate-failed-from",
        type=Path,
        default=None,
        help=(
            "Optional audit CSV. If provided, regenerate only instances whose "
            "status appears in --regenerate-statuses. Uses domain/difficulty/pfileN "
            "from the CSV."
        ),
    )

    parser.add_argument(
        "--regenerate-statuses",
        default="PLANNER_FAIL",
        help=(
            "Comma-separated audit statuses to regenerate when using "
            "--regenerate-failed-from. Default: PLANNER_FAIL."
        ),
    )

    parser.add_argument(
        "--only-domain",
        default=None,
        help=(
            "Optional domain name to generate/regenerate, e.g. zenotravel. "
            "If omitted, all domains are processed."
        ),
    )

    parser.add_argument(
        "--only-difficulty",
        default=None,
        choices=["easy", "medium", "hard"],
        help=(
            "Optional difficulty to generate/regenerate. "
            "If omitted, all difficulties are processed."
        ),
    )

    return parser.parse_args()


def load_generator(generator_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, generator_path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generator module from {generator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

def validate_kwargs_are_explicit(
    *,
    domain_name: str,
    difficulty: str,
    generator_module: ModuleType,
    kwargs: dict,
) -> None:
    """Fail if a kwarg is only swallowed by **_ instead of explicitly declared."""

    signature = inspect.signature(generator_module.generate_multiple_problems)

    explicit_params = {
        name
        for name, param in signature.parameters.items()
        if param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }

    missing = sorted(
        key
        for key in kwargs
        if key not in explicit_params
    )

    if missing:
        raise TypeError(
            f"{domain_name}/{difficulty}: these kwargs are not explicit "
            f"parameters of generate_multiple_problems(): {missing}\n"
            f"Generator: {generator_module.__file__}\n"
            f"Explicit params: {sorted(explicit_params)}\n"
            f"This usually means they would be swallowed by **_ and ignored."
        )

def regenerate_specific_indices(
    *,
    generator_module,
    domain_name: str,
    difficulty: str,
    output_folder: Path,
    indices: set[int],
    kwargs: dict,
    dry_run: bool,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    for problem_index in sorted(indices):
        if dry_run:
            print(
                f"[DRY-RUN REGEN] {domain_name:15s} {difficulty:6s} "
                f"-> {output_folder / f'pfile{problem_index}.pddl'} | "
                f"num_prev_instances={problem_index} | kwargs={kwargs}"
            )
            continue

        generator_module.generate_multiple_problems(
            output_folder=output_folder,
            total_num_problems=1,
            num_prev_instances=problem_index,
            **kwargs,
        )

        print(
            f"[REGEN OK] {domain_name:15s} {difficulty:6s} "
            f"-> {output_folder / f'pfile{problem_index}.pddl'}"
        )

def generate_domain_difficulty(
    *,
    domain_name: str,
    difficulty: str,
    generator_module: ModuleType,
    output_dir: Path,
    total_num_problems: int,
    kwargs: dict,
    clean: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        print(
            f"[DRY-RUN] {domain_name:15s} {difficulty:6s} -> "
            f"{output_dir} | n={total_num_problems} | kwargs={kwargs}"
        )
        return

    if clean:
        clean_output_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    generator_module.generate_multiple_problems(
        output_folder=output_dir,
        total_num_problems=total_num_problems,
        num_prev_instances=0,
        **kwargs,
    )

    generated = sorted(output_dir.glob("*.pddl"))

    print(
        f"[OK] {domain_name:15s} {difficulty:6s} -> "
        f"{output_dir} | generated={len(generated)}"
    )


def main() -> None:
    args = parse_args()

    problem_generator_root = args.problem_generator_root.resolve()
    output_root = args.output_root.resolve()

    if args.instances_per_difficulty <= 0:
        raise ValueError(
            f"--instances-per-difficulty must be positive, got "
            f"{args.instances_per_difficulty}."
        )

    if args.regenerate_failed_from is not None and args.clean:
        raise ValueError(
            "--clean is not allowed together with --regenerate-failed-from, "
            "because it would delete solved instances and regenerate only failed ones."
        )

    if not problem_generator_root.exists():
        raise FileNotFoundError(
            f"problem_generator root does not exist: {problem_generator_root}"
        )

    regeneration_targets: dict[tuple[str, str], set[int]] | None = None

    if args.regenerate_failed_from is not None:
        target_statuses = parse_status_set(args.regenerate_statuses)

        regeneration_targets = load_regeneration_targets_from_audit_csv(
            audit_csv=args.regenerate_failed_from.resolve(),
            target_statuses=target_statuses,
        )

        print(f"Regenerate from CSV   : {args.regenerate_failed_from.resolve()}")
        print(f"Regenerate statuses   : {sorted(target_statuses)}")
        print(
            "Regenerate targets    : "
            f"{sum(len(v) for v in regeneration_targets.values())}"
        )

    # Needed because generators import problem_generator.common.
    sys.path.insert(0, str(problem_generator_root.parent))

    print(f"Problem generator root: {problem_generator_root}")
    print(f"Output root           : {output_root}")
    print(f"Instances/difficulty  : {args.instances_per_difficulty}")
    print(f"Clean first           : {args.clean}")
    print()

    for domain_name, domain_spec in DOMAINS.items():
        if args.only_domain is not None and domain_name != args.only_domain:
            continue
        generator_path = problem_generator_root / domain_spec["generator_relpath"]

        if not generator_path.exists():
            raise FileNotFoundError(
                f"Generator for domain {domain_name!r} does not exist: "
                f"{generator_path}"
            )

        module_name = f"generated_{domain_name.replace('-', '_')}_generator"
        generator_module = load_generator(generator_path, module_name)

        if not hasattr(generator_module, "generate_multiple_problems"):
            raise AttributeError(
                f"Generator {generator_path} has no generate_multiple_problems()"
            )

        for difficulty in DIFFICULTIES:
            if args.only_difficulty is not None and difficulty != args.only_difficulty:
                continue
            kwargs = domain_spec["difficulties"][difficulty]
            output_dir = output_root / domain_name / difficulty

            validate_kwargs_are_explicit(
                domain_name=domain_name,
                difficulty=difficulty,
                generator_module=generator_module,
                kwargs=kwargs,
            )

            if regeneration_targets is not None:
                indices = regeneration_targets.get((domain_name, difficulty), set())

                if not indices:
                    continue

                regenerate_specific_indices(
                    generator_module=generator_module,
                    domain_name=domain_name,
                    difficulty=difficulty,
                    output_folder=output_dir,
                    indices=indices,
                    kwargs=kwargs,
                    dry_run=args.dry_run,
                )

            else:
                generate_domain_difficulty(
                    generator_module=generator_module,
                    domain_name=domain_name,
                    difficulty=difficulty,
                    output_dir=output_dir,
                    total_num_problems=args.instances_per_difficulty,
                    kwargs=kwargs,
                    clean=args.clean,
                    dry_run=args.dry_run,
                )


if __name__ == "__main__":
    main()