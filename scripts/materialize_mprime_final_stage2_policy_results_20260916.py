#!/usr/bin/env python3
"""Add final MPrime Stage-2 policy pairs to the canonical validation-led RQs."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
MCTS = TRACK / "mprime_phase_b_a_stage1_mcts_20260913"
SEARCH = TRACK / "mprime_final_stage2_search_20260916" / "manifest.csv"
OUT = TRACK / "mprime_final_stage2_search_20260916" / "policy_paired_seed_results.csv"
CANONICAL = TRACK / "policy_paired_seed_results.csv"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    stage1 = []
    for mode in ("off", "on"):
        stage1.extend(read(MCTS / f"manifest_{mode}.csv"))
    before = {(row["value_head"], row["seed"]): row for row in stage1}
    stage2_all = read(SEARCH)
    stage2 = {
        (row["value_head"], row["seed"]): row
        for row in stage2_all if row["search_method"] == "fixed"
    }
    if len(before) != 20 or len(stage2) != 20 or set(before) != set(stage2):
        raise RuntimeError("MPrime Stage-1/Stage-2 seed identities do not match 20/20")

    pairs: list[dict[str, str]] = []
    for mode, seed in sorted(before, key=lambda key: (key[0], int(key[1]))):
        s1, s2 = before[(mode, seed)], stage2[(mode, seed)]
        b = int(float(s1["selected_test_policy_score"]))
        a = int(float(s2["selected_test_policy_score"]))
        pairs.append({
            "experiment_id": "MAIN-VAL",
            "comparison": "S1 validation-selected -> S2 validation-selected",
            "domain": "mprime", "value_head": mode, "seed": seed,
            "before_score": str(b), "after_score": str(a), "difference": str(a - b),
            "before_training_job": s1["source_training_job_id"],
            "before_evaluation_job": s1["source_policy_job_id"],
            "before_log": s1["source_policy_log"],
            "after_training_job": s2["source_training_job_id"],
            "after_evaluation_job": s2["source_policy_job_id"],
            "after_log": s2["source_policy_log"],
        })
    write(OUT, pairs)

    canonical = [
        row for row in read(CANONICAL)
        if not (row["experiment_id"] == "MAIN-VAL" and row["domain"] == "mprime")
    ]
    canonical.extend(pairs)
    canonical.sort(key=lambda row: (
        row["experiment_id"], row["domain"], row["value_head"], int(row["seed"])
    ))
    write(CANONICAL, canonical)
    print(f"wrote {len(pairs)} MPrime pairs; canonical rows={len(canonical)}")


if __name__ == "__main__":
    main()
