"""Generate, planner-screen, and freeze the MPrime Phase-C validation set.

The generator is deliberately independent of network and test scores.  It uses
only structural support observed in the fixed IPC test suite: object-count
strata, denser directed food graphs, heterogeneous numeric resources, and one
to three goals.  Every proposed goal has a constructed multi-step witness; an
instance is eligible for freezing only after ENHSP solves it and VAL certifies
the plan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import deque
from pathlib import Path


TIERS = (
    # foods, pleasures, pains, goals, target path lower bound, max locale
    ("easy", (5, 9), (2, 4), (4, 12), 1, 2, 9),
    ("medium", (10, 16), (4, 8), (10, 28), 2, 3, 12),
    ("hard", (17, 22), (7, 14), (20, 46), 3, 4, 15),
)
BASE_SEED = 2026091100
POOL_PER_TIER = 40


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def shortest_distances(nodes: list[str], edges: set[tuple[str, str]], source: str):
    distances = {node: 10**9 for node in nodes}
    distances[source] = 0
    queue = deque([source])
    outgoing: dict[str, list[str]] = {node: [] for node in nodes}
    for left, right in edges:
        outgoing[left].append(right)
    while queue:
        current = queue.popleft()
        for successor in outgoing[current]:
            if distances[successor] > distances[current] + 1:
                distances[successor] = distances[current] + 1
                queue.append(successor)
    return distances


def make_instance(tier_index: int, index: int) -> tuple[str, dict]:
    tier, food_range, pleasure_range, pain_range, goal_count, min_route, max_locale = TIERS[tier_index]
    seed = BASE_SEED + tier_index * 1000 + index
    rng = random.Random(seed)
    food_count = rng.randint(*food_range)
    pleasure_count = rng.randint(*pleasure_range)
    pain_count = rng.randint(*pain_range)
    foods = [f"f{i}" for i in range(food_count)]
    pleasures = [f"v{i}" for i in range(pleasure_count)]
    pains = [f"p{i}" for i in range(pain_count)]

    # A shuffled directed ring guarantees reachability.  Extra arcs reproduce
    # the roughly two-to-three outgoing edges per food seen in the IPC suite.
    order = foods[:]
    rng.shuffle(order)
    edges = {(order[i], order[(i + 1) % food_count]) for i in range(food_count)}
    target_edges = min(food_count * (food_count - 1), rng.randint(2 * food_count, 3 * food_count))
    while len(edges) < target_edges:
        left, right = rng.sample(foods, 2)
        edges.add((left, right))

    pleasure_positions = {pleasure: rng.choice(foods) for pleasure in pleasures}
    pain_sources = {pain: rng.choice(foods) for pain in pains}
    selected_goals: list[tuple[str, str, str, str, int]] = []
    available_pleasures = pleasures[:]
    available_pains = pains[:]
    rng.shuffle(available_pleasures)
    rng.shuffle(available_pains)
    for pain, pleasure in zip(available_pains, available_pleasures):
        source = pain_sources[pain]
        to_source = shortest_distances(foods, edges, pleasure_positions[pleasure])
        from_source = shortest_distances(foods, edges, source)
        eligible = [
            target for target in foods
            if target != source
            and to_source[source] + from_source[target] >= min_route
        ]
        if not eligible:
            continue
        target = rng.choice(eligible)
        selected_goals.append((pain, pleasure, source, target, to_source[source] + from_source[target]))
        if len(selected_goals) == goal_count:
            break
    if len(selected_goals) != goal_count:
        raise RuntimeError(f"could not construct {goal_count} goals for seed {seed}")

    cravings = {(pain, source) for pain, source in pain_sources.items()}
    cravings.update((pleasure, source) for pleasure, source in pleasure_positions.items())
    # Add background cravings without creating a direct goal witness.
    goal_pairs = {(pain, target) for pain, _, _, target, _ in selected_goals}
    feelings = pleasures + pains
    for feeling in feelings:
        if rng.random() < 0.55:
            candidate = rng.choice(foods)
            if (feeling, candidate) not in goal_pairs:
                cravings.add((feeling, candidate))

    locales = {food: rng.randint(0, max_locale) for food in foods}
    # Positive locale everywhere keeps the constructed feast routes executable;
    # a quarter of non-witness foods retain the test suite's zero-resource tail.
    witness_foods = {source for _, _, source, _, _ in selected_goals}
    witness_foods.update(pleasure_positions[pleasure] for _, pleasure, _, _, _ in selected_goals)
    for food in foods:
        if food in witness_foods:
            locales[food] = max(2, locales[food])
        elif rng.random() >= 0.25:
            locales[food] = max(1, locales[food])

    init = [f"(= (locale {food}) {locales[food]})" for food in foods]
    init += [f"(= (harmony {pleasure}) {rng.randint(1, 3)})" for pleasure in pleasures]
    init += [f"(eats {left} {right})" for left, right in sorted(edges)]
    init += [f"(craves {feeling} {food})" for feeling, food in sorted(cravings)]
    goals = [f"(craves {pain} {target})" for pain, _, _, target, _ in selected_goals]
    name = f"mprime-c-{tier_index}-{index}"
    text = (
        f"(define (problem {name}) (:domain mystery-prime-typed)\n"
        f"(:objects {' '.join(foods)} - food {' '.join(pleasures)} - pleasure "
        f"{' '.join(pains)} - pain)\n"
        f"(:init {' '.join(init)})\n"
        f"(:goal (and {' '.join(goals)})))\n"
    )
    metadata = {
        "tier": tier,
        "tier_index": tier_index,
        "index": index,
        "seed": seed,
        "foods": food_count,
        "pleasures": pleasure_count,
        "pains": pain_count,
        "edges": len(edges),
        "goals": goal_count,
        "minimum_constructed_route": min(item[4] for item in selected_goals),
        "zero_locale_foods": sum(value == 0 for value in locales.values()),
        "file": f"{name}.pddl",
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    return text, metadata


def generate(root: Path) -> None:
    if (root / "frozen_validation_manifest.csv").exists():
        raise FileExistsError("Phase-C validation set is already frozen")
    rows = []
    for tier_index in range(len(TIERS)):
        for index in range(POOL_PER_TIER):
            text, row = make_instance(tier_index, index)
            path = root / "candidates" / row["file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            # Hash the durable bytes, not the pre-write string.  On Windows,
            # Path.write_text may translate newlines; the frozen checksum must
            # describe exactly what is copied to and screened on the cluster.
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(row)
    write_csv(root / "candidates.csv", rows)
    protocol = {
        "version": "20260911-phase-c",
        "candidate_pool": len(rows),
        "selected_instances": 30,
        "per_tier": 10,
        "selection_order": "ascending predeclared seed; first ten ENHSP/VAL-certified per tier",
        "planner": "hmrp-ha-gbfs",
        "planner_timeout_seconds": 120,
        "minimum_certified_plan_lengths": [4, 6, 8],
        "network_or_test_scores_used_to_generate_or_select_instances": False,
        "structural_basis": "fixed IPC test-suite support only; no copied problem files",
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"generated={len(rows)}")


def screen(root: Path, repo: Path, task: int) -> None:
    sys.path.insert(0, str(repo))
    from problem_generator.audit_generated_instances_enhsp import load_enhsp

    enhsp, status, configs = load_enhsp(repo)
    rows = list(csv.DictReader((root / "candidates.csv").open()))
    selected, audited = [], []
    domain = repo / "problems/numeric/mprime/domain.pddl"
    for row in rows:
        if int(row["tier_index"]) != task:
            continue
        problem = root / "candidates" / row["file"]
        assert hashlib.sha256(problem.read_bytes()).hexdigest() == row["sha256"]
        planfile = root / "planner" / f"{problem.stem}.plan"
        planfile.parent.mkdir(exist_ok=True)
        try:
            result = enhsp(configs["hmrp-ha-gbfs"] + " -timeout 120").plan(str(domain), str(problem))
            plan = list(result.plan or [])
            planfile.write_text("\n".join(f"{i}: ({str(action).strip().strip('()')} )" for i, action in enumerate(plan)) + "\n")
            val = subprocess.run(
                ["/home/hersco/tools/VAL/build/bin/Validate", str(domain), str(problem), str(planfile)],
                capture_output=True, text=True, timeout=60,
            )
            val_log = planfile.with_suffix(".val.log")
            val_log.write_text(val.stdout + val.stderr)
            valid = "Plan valid" in val.stdout and result.status == status.SUCCESS
            row.update(
                planner_status=str(result.status), plan_length=len(plan), val_valid=valid,
                plan_file=str(planfile), val_log=str(val_log), job_id=os.environ.get("SLURM_JOB_ID", "local"),
            )
        except Exception as exc:
            row.update(
                planner_status=repr(exc), plan_length=0, val_valid=False,
                plan_file=str(planfile), val_log="", job_id=os.environ.get("SLURM_JOB_ID", "local"),
            )
        audited.append(row)
        write_csv(root / f"planner_task_{task}.csv", audited)
        if row["val_valid"] and int(row["plan_length"]) >= (4, 6, 8)[task]:
            selected.append(row)
        if len(selected) == 10:
            break
    if len(selected) != 10:
        raise RuntimeError(f"only {len(selected)}/10 candidates certified for tier {task}")
    write_csv(root / f"selected_task_{task}.csv", selected)
    print(f"CERTIFIED tier={task} count=10", flush=True)


def freeze(root: Path) -> None:
    rows = []
    for task in range(3):
        group = list(csv.DictReader((root / f"selected_task_{task}.csv").open()))
        assert len(group) == 10
        rows.extend(group)
    assert len(rows) == 30 and len({row["sha256"] for row in rows}) == 30
    for row in rows:
        assert hashlib.sha256((root / "candidates" / row["file"]).read_bytes()).hexdigest() == row["sha256"]
    write_csv(root / "frozen_validation_manifest.csv", rows)
    print("FROZEN 30 planner/VAL-certified Phase-C instances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "screen", "freeze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "generate":
        generate(args.root)
    elif args.mode == "screen":
        screen(args.root, args.repo, args.task)
    else:
        freeze(args.root)
