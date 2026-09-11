#!/usr/bin/env python3
"""Consolidate the completed PW70 FO/Rover expansion with all-cutoff inference."""

from __future__ import annotations

import csv
import itertools
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "experiment_tracking"
P = T / "mcts_progressive_widening_cross_domain"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    critical = {9: 2.306, 10: 2.262}[len(values)]
    half = critical * statistics.stdev(values) / math.sqrt(len(values)) if statistics.stdev(values) else 0
    return mean - half, mean + half


def signflip(values: list[float]) -> float:
    observed = abs(statistics.mean(values))
    return sum(
        abs(statistics.mean(v * s for v, s in zip(values, signs))) >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=len(values))
    ) / (2 ** len(values))


def main() -> None:
    fixed = {
        (r["domain"], r["value_head"], r["seed"]): r
        for r in read(T / "stage1_policy_mcts_seed_cutoffs_latest.csv")
        if r["domain"] in {"fo_counters", "rover"}
    }
    rows = []
    for r in read(P / "pw70_confirmatory_seed_results_20260904.csv"):
        key = (r["domain"], r["value_head"], r["seed"])
        f = fixed[key]
        rows.append({
            **{k: r[k] for k in ("domain", "value_head", "seed")},
            "policy_score": r["policy_score"],
            "fixed_30m": f["mcts_30m"], "fixed_2h": f["mcts_2h"], "fixed_6h": f["mcts_6h"],
            "pw70_30m": r["pw70_30m_score"], "pw70_2h": r["pw70_2h_score"], "pw70_6h": r["pw70_6h_score"],
            "included_in_inference": "true", "evidence_status": "complete_declared_budget",
            "classified_instances": "20", "job_state": "COMPLETED", "pw_job_id": "historical",
            "source_policy_log": f["source_policy_log"], "source_fixed_log": r["source_fixed_log"],
            "source_pw_log": r["source_pw_log"],
        })
    new_jobs = {r["job_id"]: r for r in read(P / "pw70_ten_seed_jobs_20260907.csv")}
    dynamic = {r["job_id"]: r for r in read(T / "dynamic_experiment_jobs_latest.csv")}
    for job_id, meta in new_jobs.items():
        d = dynamic[job_id]
        key = (meta["domain"], meta["value_head"], meta["seed"])
        f = fixed[key]
        # A scheduler/OOM-terminal allocation is part of the declared-budget
        # result.  Keep the partially classified FO/off seed in the primary
        # descriptive and paired tables, but label it explicitly below so a
        # complete-allocation sensitivity analysis remains possible.
        include = True
        rows.append({
            "domain": key[0], "value_head": key[1], "seed": key[2],
            "policy_score": f["policy_score"],
            "fixed_30m": f["mcts_30m"], "fixed_2h": f["mcts_2h"], "fixed_6h": f["mcts_6h"],
            "pw70_30m": d["success_30m"], "pw70_2h": d["success_2h"], "pw70_6h": d["success_6h"],
            "included_in_inference": str(include).lower(),
            "evidence_status": (
                "oom_terminal_declared_budget"
                if d["state"] == "OUT_OF_MEMORY" and int(d["classified_instances"]) < 20
                else "complete_declared_budget"
            ),
            "classified_instances": d["classified_instances"],
            "job_state": d["state"], "pw_job_id": job_id,
            "source_policy_log": f["source_policy_log"], "source_fixed_log": f["source_mcts_log"],
            "source_pw_log": d["source_evaluation_log"],
        })
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["included_in_inference"] == "true":
            groups[(row["domain"], row["value_head"])].append(row)
    summary = []
    for (domain, vh), cell in sorted(groups.items()):
        item: dict[str, object] = {"domain": domain, "value_head": vh, "n": len(cell)}
        item = {"domain": domain, "value_head": vh, "n": len(cell),
                "n_oom_terminal_declared_budget": sum(r.get("evidence_status") == "oom_terminal_declared_budget" for r in cell),
                "policy_mean": statistics.mean(float(r["policy_score"]) for r in cell)}
        for cutoff in ("30m", "2h", "6h"):
            pw = [float(r[f"pw70_{cutoff}"]) for r in cell]
            item[f"pw70_mean_{cutoff}"] = statistics.mean(pw)
            for comparator, field in (("policy", "policy_score"), ("fixed", f"fixed_{cutoff}")):
                diffs = [x - float(r[field]) for x, r in zip(pw, cell)]
                low, high = ci(diffs)
                item[f"delta_vs_{comparator}_{cutoff}"] = statistics.mean(diffs)
                item[f"ci95_low_vs_{comparator}_{cutoff}"] = low
                item[f"ci95_high_vs_{comparator}_{cutoff}"] = high
                item[f"raw_p_vs_{comparator}_{cutoff}"] = signflip(diffs)
        item["status"] = "complete_n10_including_one_oom_terminal_endpoint" if any(
            r.get("evidence_status") == "oom_terminal_declared_budget" for r in cell
        ) else "complete"
        item["row_level_provenance"] = "experiment_tracking/mcts_progressive_widening_cross_domain/pw70_ten_seed_results_latest.csv"
        summary.append(item)
    for cutoff in ("30m", "2h", "6h"):
        for comparator in ("policy", "fixed"):
            key = f"raw_p_vs_{comparator}_{cutoff}"
            ordered = sorted(enumerate(summary), key=lambda x: float(x[1][key]))
            running = 0.0
            for rank, (index, row) in enumerate(ordered):
                running = max(running, min(1.0, float(row[key]) * (len(ordered) - rank)))
                summary[index][f"holm_p_vs_{comparator}_{cutoff}"] = running
    for path, data in ((P / "pw70_ten_seed_results_latest.csv", rows),
                       (P / "pw70_ten_seed_statistics_latest.csv", summary)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    print(f"wrote {len(rows)} seed rows and {len(summary)} cell rows")


if __name__ == "__main__":
    main()
