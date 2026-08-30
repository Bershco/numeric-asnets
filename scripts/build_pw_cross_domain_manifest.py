#!/usr/bin/env python3
"""Build the preregistered cross-domain Kmin=3 progressive-widening pilot."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiment_tracking" / "experiment_results.csv"
S2 = (
    ROOT / "experiment_tracking" / "mcts_counters_width_sensitivity"
    / "main_term_stage2_narrow_held_manifest.csv"
)
OUTPUT = (
    ROOT / "experiment_tracking" / "mcts_progressive_widening_cross_domain"
    / "pilot_manifest.csv"
)
SEEDS = {"1963100312", "2011206605"}
DOMAINS = {"block_grouping", "fo_counters", "rover", "counters"}
TEACHERS = {
    "block_grouping": "hadd-astar",
    "fo_counters": "hmrmax-astar",
    "rover": "hmrp-ha-gbfs",
    "counters": "hmrmax-astar",
}

FIELDS = [
    "experiment_id", "manifest_id", "domain", "value_head", "seed", "stage",
    "mainstream_lineage", "source_checkpoint", "source_training_job_id",
    "snapshot_epoch", "policy_score", "matched_fixed_width", "iterations",
    "pw_min_width", "pw_c", "pw_alpha", "terminal_safe", "workers", "cpus",
    "memory", "time_limit", "instance_timeout", "teacher", "status", "notes",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    rows: list[dict[str, str]] = []
    for src in read(RESULTS):
        if not (
            src["experiment_id"] == "MAIN-VAL"
            and src["task_type"] == "policy_eval"
            and src["stage"] == "stage1"
            and src["endpoint"] == "validation_selected"
            and src["domain"] in DOMAINS
            and src["seed"] in SEEDS
        ):
            continue
        iterations = "20" if src["domain"] in {"block_grouping", "counters"} else "70"
        fixed_width = "5" if iterations == "20" else "20"
        rows.append({
            "experiment_id": "MCTS-PW-CROSS-DOMAIN",
            "manifest_id": (
                f"pw-kmin3-{src['domain']}-{src['value_head']}-{src['seed']}-s1"
            ),
            "domain": src["domain"],
            "value_head": src["value_head"],
            "seed": src["seed"],
            "stage": "stage1",
            "mainstream_lineage": "MAIN-VAL",
            "source_checkpoint": src["checkpoint"],
            "source_training_job_id": src["source_training_job_id"],
            "snapshot_epoch": src["epoch"],
            "policy_score": src["score"],
            "matched_fixed_width": fixed_width,
            "iterations": iterations,
            "pw_min_width": "3",
            "pw_c": "0.6",
            "pw_alpha": "0.5",
            "terminal_safe": "true",
            "workers": "3",
            "cpus": "6",
            "memory": "120G",
            "time_limit": "3-00:00:00",
            "instance_timeout": "21600",
            "teacher": TEACHERS[src["domain"]],
            "status": "ready",
            "notes": (
                "Two-seed cross-domain pilot; compare full six-hour outcome and "
                "post-hoc deterministic 30-minute cutoff with matched policy/fixed search"
            ),
        })

    for src in read(S2):
        if src["seed"] not in SEEDS:
            continue
        rows.append({
            "experiment_id": "MCTS-PW-CROSS-DOMAIN",
            "manifest_id": f"pw-kmin3-counters-{src['value_head']}-{src['seed']}-s2",
            "domain": "counters",
            "value_head": src["value_head"],
            "seed": src["seed"],
            "stage": "stage2",
            "mainstream_lineage": "MAIN-TERM",
            "source_checkpoint": src["source_checkpoint_ref"],
            "source_training_job_id": src["source_training_job_id"],
            "snapshot_epoch": src["snapshot_epoch"],
            "policy_score": "",
            "matched_fixed_width": "5",
            "iterations": "20",
            "pw_min_width": "3",
            "pw_c": "0.6",
            "pw_alpha": "0.5",
            "terminal_safe": "true",
            "workers": "3",
            "cpus": "6",
            "memory": "120G",
            "time_limit": "3-00:00:00",
            "instance_timeout": "21600",
            "teacher": TEACHERS["counters"],
            "status": "ready",
            "notes": (
                "Stage-2 Counters extension matched to the held/active width-5/20 arm; "
                "report full and post-hoc 30-minute outcomes"
            ),
        })

    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 rows, got {len(rows)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["manifest_id"]))
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
