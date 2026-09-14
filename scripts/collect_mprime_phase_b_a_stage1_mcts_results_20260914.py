#!/usr/bin/env python3
"""Reconcile full and exact-recovery MPrime Stage-1 fixed-MCTS records."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import statistics
from pathlib import Path

DONE_RE = re.compile(r"\[EVAL INSTANCE\] completed number=(\d+).*?status=([^ ]+)")
TIMEOUT_RE = re.compile(r"\[EVAL INSTANCE\] timeout number=(\d+)")
JOB_RE = re.compile(r"(\d+)(?:_r\d+)?\.txt$")


def load_manifests(manifest_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = {}
    for mode in ("off", "on"):
        with (manifest_dir / f"manifest_{mode}.csv").open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                rows[(mode, row["seed"])] = row
    return rows


def source_uri(root: Path, path: Path, remote_root: str) -> str:
    return remote_root.rstrip("/") + "/" + path.relative_to(root).as_posix()


def parse_root(root: Path, remote_root: str, records: dict[tuple[str, str, int], dict]) -> None:
    if not root.exists():
        return
    for mode in ("off", "on"):
        mode_dir = root / mode
        if not mode_dir.exists():
            continue
        for seed_dir in mode_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            seed = seed_dir.name
            for path in seed_dir.glob("**/completion/*.jsonl"):
                with path.open(encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        row = json.loads(line)
                        records[(mode, seed, int(row["instance_number"]))] = {
                            "status": row["status"],
                            "elapsed_seconds": row.get("elapsed_seconds", ""),
                            "steps": row.get("steps", ""),
                            "instance_path": row.get("instance_path", ""),
                            "source_completion_record": source_uri(root, path, remote_root),
                            "source_attempt_log": "",
                            "source_job_id": "",
                        }
            for path in seed_dir.glob("**/attempts/*.txt"):
                match_job = JOB_RE.search(path.name)
                job_id = match_job.group(1) if match_job else ""
                with path.open(encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        match = TIMEOUT_RE.search(line)
                        if match:
                            key = (mode, seed, int(match.group(1)))
                            # A durable completion record wins over an older or
                            # duplicate timeout attempt for the same instance.
                            if key not in records:
                                records[key] = {
                                    "status": "timeout", "elapsed_seconds": 21600,
                                    "steps": "", "instance_path": "",
                                    "source_completion_record": "",
                                    "source_attempt_log": source_uri(root, path, remote_root),
                                    "source_job_id": job_id,
                                }
                            continue
                        match = DONE_RE.search(line)
                        if match:
                            key = (mode, seed, int(match.group(1)))
                            # Prefer the JSONL copy when available because it carries elapsed time.
                            if key not in records:
                                records[key] = {
                                    "status": match.group(2), "elapsed_seconds": "",
                                    "steps": "", "instance_path": "",
                                    "source_completion_record": "",
                                    "source_attempt_log": source_uri(root, path, remote_root),
                                    "source_job_id": job_id,
                                }
                            else:
                                records[key]["source_attempt_log"] = source_uri(root, path, remote_root)
                                records[key]["source_job_id"] = job_id


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_sign_flip(values: list[float]) -> float:
    observed = abs(statistics.mean(values))
    means = [
        abs(statistics.mean(sign * value for sign, value in zip(signs, values)))
        for signs in itertools.product((-1, 1), repeat=len(values))
    ]
    return sum(value >= observed - 1e-12 for value in means) / len(means)


def paired_ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2 or statistics.stdev(values) == 0:
        return mean, mean
    if len(values) != 10:
        raise ValueError("This declared paired analysis expects exactly ten seeds")
    margin = 2.2621571628540993 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - margin, mean + margin


def holm_for_new(existing: list[float], new_value: float) -> float:
    values = existing + [new_value]
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--root-uri", required=True)
    parser.add_argument("--recovery-root-uri", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rq-source", type=Path, required=True)
    args = parser.parse_args()
    manifests = load_manifests(args.manifest_dir)
    records: dict[tuple[str, str, int], dict] = {}
    parse_root(args.root, args.root_uri, records)
    parse_root(args.recovery_root, args.recovery_root_uri, records)

    per_instance = []
    per_seed = []
    for (mode, seed), manifest in sorted(manifests.items()):
        seed_rows = []
        for number in range(1, 21):
            record = records.get((mode, seed, number), {})
            elapsed = record.get("elapsed_seconds", "")
            success = record.get("status") == "success"
            row = {
                "value_head": mode, "seed": seed,
                "selected_epoch": manifest["selected_epoch"],
                "policy_score": manifest["selected_test_policy_score"],
                "instance_number": number,
                "status": record.get("status", "missing"),
                "success_30m": int(success and elapsed != "" and float(elapsed) <= 1800),
                "success_2h": int(success and elapsed != "" and float(elapsed) <= 7200),
                "success_6h": int(success),
                "elapsed_seconds": elapsed, "steps": record.get("steps", ""),
                "source_job_id": record.get("source_job_id", ""),
                "source_attempt_log": record.get("source_attempt_log", ""),
                "source_completion_record": record.get("source_completion_record", ""),
                "source_training_job_id": manifest["source_training_job_id"],
                "source_policy_job_id": manifest["source_policy_job_id"],
                "checkpoint": manifest["checkpoint"],
            }
            per_instance.append(row)
            seed_rows.append(row)
        per_seed.append({
            "value_head": mode, "seed": seed,
            "selected_epoch": manifest["selected_epoch"],
            "policy_score": manifest["selected_test_policy_score"],
            "mcts_30m": sum(row["success_30m"] for row in seed_rows),
            "mcts_2h": sum(row["success_2h"] for row in seed_rows),
            "mcts_6h": sum(row["success_6h"] for row in seed_rows),
            "classified": sum(row["status"] != "missing" for row in seed_rows),
            "timeouts": sum(row["status"] == "timeout" for row in seed_rows),
            "ordinary_unsolved": sum(row["status"] in {"unsolved", "finished_unsolved"} for row in seed_rows),
            "missing": sum(row["status"] == "missing" for row in seed_rows),
            "source_training_job_id": manifest["source_training_job_id"],
            "source_policy_job_id": manifest["source_policy_job_id"],
            "source_training_log": manifest["source_training_log"],
            "source_policy_log": manifest["source_policy_log"],
            "checkpoint": manifest["checkpoint"],
        })
    write_csv(args.output_dir / "per_instance_results.csv", per_instance)
    write_csv(args.output_dir / "per_seed_results.csv", per_seed)

    by_mode = {
        mode: {row["seed"]: row for row in per_seed if row["value_head"] == mode}
        for mode in ("off", "on")
    }
    with args.rq_source.open(newline="", encoding="utf-8-sig") as stream:
        prior_rq = list(csv.DictReader(stream))
    rq_rows = []
    estimands = [
        ("RQ2", "VH-off direct: MCTS - same-checkpoint policy", "off_direct"),
        ("RQ4", "VH-on direct: MCTS - same-checkpoint policy", "on_direct"),
        ("RQ4", "Cross-cell level: VH-on MCTS - parallel VH-off policy", "cross_cell"),
        ("RQ4", "VH interaction: VH-on MCTS benefit - VH-off MCTS benefit", "interaction"),
    ]
    for cutoff, column in (("30m", "mcts_30m"), ("2h", "mcts_2h"), ("6h", "mcts_6h")):
        for rq, estimand, kind in estimands:
            values = []
            baselines = []
            comparisons = []
            for seed in sorted(by_mode["off"], key=int):
                off = by_mode["off"][seed]
                on = by_mode["on"][seed]
                off_policy = float(off["policy_score"])
                on_policy = float(on["policy_score"])
                off_mcts = float(off[column])
                on_mcts = float(on[column])
                if kind == "off_direct":
                    baseline, comparison, value = off_policy, off_mcts, off_mcts - off_policy
                elif kind == "on_direct":
                    baseline, comparison, value = on_policy, on_mcts, on_mcts - on_policy
                elif kind == "cross_cell":
                    baseline, comparison, value = off_policy, on_mcts, on_mcts - off_policy
                else:
                    baseline = off_mcts - off_policy
                    comparison = on_mcts - on_policy
                    value = comparison - baseline
                baselines.append(baseline)
                comparisons.append(comparison)
                values.append(value)
            low, high = paired_ci(values)
            raw_p = exact_sign_flip(values)
            prior_p = [
                float(row["raw_p"]) for row in prior_rq
                if row["rq"] == rq and row["stage"] == "Stage 1"
                and row["estimand"] == estimand and row["cutoff"] == cutoff
            ]
            rq_rows.append({
                "rq": rq, "stage": "Stage 1", "estimand": estimand,
                "domain": "mprime", "cutoff": cutoff, "n": len(values),
                "capacity": 20, "baseline_mean": statistics.mean(baselines),
                "comparison_mean": statistics.mean(comparisons),
                "effect_solved": statistics.mean(values),
                "ci95_low_solved": low, "ci95_high_solved": high,
                "raw_p": raw_p, "holm_p_six_domain_extension": holm_for_new(prior_p, raw_p),
                "evidence_status": "complete_paired_stage1_extension",
                "row_level_provenance": str(args.output_dir / "per_seed_results.csv"),
                "multiplicity_status": "provisional_six_domain_stage1_extension",
            })
    write_csv(args.output_dir / "mprime_rq_extension.csv", rq_rows)
    print(json.dumps({
        "instances": len(per_instance),
        "classified": sum(row["status"] != "missing" for row in per_instance),
        "missing": sum(row["status"] == "missing" for row in per_instance),
        "successes_30m": sum(row["success_30m"] for row in per_instance),
        "successes_2h": sum(row["success_2h"] for row in per_instance),
        "successes_6h": sum(row["success_6h"] for row in per_instance),
    }, indent=2))


if __name__ == "__main__":
    main()
