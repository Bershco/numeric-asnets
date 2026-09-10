#!/usr/bin/env python3
"""Freeze exact FO Counters and Rover samples from Yarin's generators.

The code mirrors SPL-BGU/pddl-problem-generator at commit
b64e5d086117ebd5c1d53fbe9a9d93ae609a59fb.  The upstream rovergen binary is
supplied separately and checksum-verified rather than copied into this repo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
import subprocess
from pathlib import Path


UPSTREAM_COMMIT = "b64e5d086117ebd5c1d53fbe9a9d93ae609a59fb"
ROVERGEN_SHA256 = "9a82209b7e1b60a908c26a9ea599bde74d0981f644fb2d5a7d9d4412821573f2"
FREEZE_SEED = 20260910


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def align_rover_location_predicate(problem: str) -> str:
    """Rename only upstream Rover's location token ``at`` to numeric Rover's ``in``."""
    return re.sub(r"(?i)(\()at(\s)", r"\1in\2", problem)


def copy_base(root: Path, domain: str, destination: Path) -> None:
    source = root / "problems" / "numeric" / domain
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "domain.pddl", destination / "domain.pddl")
    train = destination / "train"
    train.mkdir(exist_ok=True)
    if domain == "fo-counters":
        for number in (2, 3, 4):
            shutil.copy2(source / "instances" / f"instance_{number}.pddl", train / f"instance_{number}.pddl")
    else:
        for number in range(1, 5):
            shutil.copy2(source / "problems" / f"pfile{number}.pddl", train / f"pfile{number}.pddl")


def fo_problem(index: int, counters: int, max_int: int, rng: random.Random) -> str:
    objects = " ".join(f"c{i}" for i in range(counters))
    initial = "\n    ".join(f"(= (value c{i}) {rng.randint(0, max_int)})" for i in range(counters))
    rates = "\n    ".join(f"(= (rate_value c{i}) 0)" for i in range(counters))
    goals = "\n    ".join(f"(<= (+ (value c{i}) 1) (value c{i + 1}))" for i in range(counters - 1))
    return f"""(define (problem yarin-fo-{index:02d})
  (:domain fo-counters)
  (:objects {objects} - counter)
  (:init
    (= (max_int) {max_int})
    {initial}
    {rates}
    (= (total-cost) 0))
  (:goal (and
    {goals}))
)
"""


def freeze_fo(root: Path, destination: Path, rng: random.Random, rows: list[dict[str, object]]) -> None:
    copy_base(root, "fo-counters", destination)
    external = destination / "external"
    external.mkdir(exist_ok=True)
    for index in range(20):
        counters = rng.randint(2, 21)
        path = external / f"pfile{index}.pddl"
        path.write_text(fo_problem(index, counters, 42, rng), encoding="utf-8")
        rows.append({"domain": "fo_counters", "index": index, "generator_seed": FREEZE_SEED,
                     "upstream_commit": UPSTREAM_COMMIT, "parameters": f"counters={counters};max_int=42",
                     "path": str(path), "sha256": sha256(path)})


def freeze_rover(root: Path, destination: Path, rovergen: Path, rng: random.Random,
                 rows: list[dict[str, object]]) -> None:
    if sha256(rovergen) != ROVERGEN_SHA256:
        raise ValueError("rovergen checksum does not match frozen upstream binary")
    copy_base(root, "rover", destination)
    external = destination / "external"
    external.mkdir(exist_ok=True)
    for index in range(20):
        parameters = {
            "prob_num": rng.randint(1, 1000), "native_seed": rng.randint(1, 4),
            "rovers": rng.randint(4, 8), "waypoints": rng.randint(2, 5),
            "objectives": rng.randint(1, 5), "cameras": rng.randint(1, 5),
        }
        command = [str(rovergen), str(parameters["prob_num"]), "-n",
                   str(parameters["native_seed"]), str(parameters["rovers"]),
                   str(parameters["waypoints"]), str(parameters["objectives"]),
                   str(parameters["cameras"])]
        output = subprocess.check_output(command, cwd=rovergen.parent).decode("utf-8")
        # Preserve upstream's export behavior: only the define line keeps case.
        rendered = "\n".join(line if "define" in line else line.lower() for line in output.splitlines()) + "\n"
        # The two domains use identical rover/waypoint location semantics but
        # different predicate names.  Leave at_lander/at_*_sample untouched.
        rendered = align_rover_location_predicate(rendered)
        path = external / f"pfile{index}.pddl"
        path.write_text(rendered, encoding="utf-8")
        rows.append({"domain": "rover", "index": index, "generator_seed": FREEZE_SEED,
                     "upstream_commit": UPSTREAM_COMMIT,
                     "parameters": ";".join(f"{key}={value}" for key, value in parameters.items())
                     + ";compatibility_rename=at_to_in",
                     "path": str(path), "sha256": sha256(path)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rovergen", type=Path)
    parser.add_argument("--domains", nargs="+", choices=("fo_counters", "rover"),
                        default=("fo_counters", "rover"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(FREEZE_SEED)
    rows: list[dict[str, object]] = []
    if "fo_counters" in args.domains:
        freeze_fo(args.repo_root, args.output_root / "fo-counters", rng, rows)
    if "rover" in args.domains:
        if args.rovergen is None:
            parser.error("--rovergen is required when rover is requested")
        freeze_rover(args.repo_root, args.output_root / "rover", args.rovergen.resolve(), rng, rows)
    manifest = args.output_root / "frozen_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {manifest} with {len(rows)} frozen instances")


if __name__ == "__main__":
    main()
