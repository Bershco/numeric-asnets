#!/usr/bin/env python3
"""Materialize the terminal 20-seed MPrime PW70 result from local ledgers."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking/mprime_phase_b_a_stage1_pw70_20260913"
LEDGERS = TRACK / "local_completion_ledgers"
LOGS = TRACK / "local_stdout_logs"
FIXED = ROOT / "experiment_tracking/mprime_phase_b_a_stage1_mcts_20260913/results_20260914/per_seed_results.csv"
OUTPUT = TRACK / "results_final_20260915.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def identify_log(path: Path) -> tuple[str, str] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "[EVAL FINAL]" not in text:
        return None
    seed = re.search(r"\bseed=(\d+)", text)
    disabled = re.search(r"\bdisable_value_head=(True|False)", text)
    if not seed or not disabled:
        raise RuntimeError(f"cannot identify terminal log {path}")
    return ("off" if disabled.group(1) == "True" else "on", seed.group(1))


def main() -> None:
    fixed_rows = {(row["value_head"], row["seed"]): row for row in read_csv(FIXED)}
    log_by_cell: dict[tuple[str, str], Path] = {}
    for path in sorted(LOGS.glob("*.out")):
        identity = identify_log(path)
        if identity is not None:
            log_by_cell[identity] = path

    output: list[dict[str, object]] = []
    for mode in ("off", "on"):
        for ledger in sorted((LEDGERS / mode).glob("*.jsonl")):
            match = re.search(rf"-{mode}-(\d+)-e(\d+)-", ledger.name)
            if not match:
                raise RuntimeError(f"cannot identify ledger {ledger}")
            seed, epoch_text = match.groups()
            records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
            if any(record.get("status") != "success" for record in records):
                raise RuntimeError(f"unexpected non-success record in {ledger}")
            fixed = fixed_rows[(mode, seed)]
            log = log_by_cell[(mode, seed)]
            log_text = log.read_text(encoding="utf-8", errors="replace")
            final = re.findall(r"\[EVAL FINAL\] success=([0-9.]+)/20=", log_text)
            if not final or int(float(final[-1])) != len(records):
                raise RuntimeError(f"ledger/final-summary mismatch for {mode}/{seed}")
            job_id = log.stem
            remote_root = (
                "/home/hersco/training_new_domains/2026-09-13/"
                f"mprime_phase_b_a_stage1_pw70/{mode}/{seed}/full"
            )
            output.append({
                "value_head": mode,
                "seed": seed,
                "selected_epoch": int(epoch_text),
                "policy_score": float(fixed["policy_score"]),
                "fixed_30m": int(fixed["mcts_30m"]),
                "fixed_2h": int(fixed["mcts_2h"]),
                "fixed_6h": int(fixed["mcts_6h"]),
                "pw70_30m": sum(float(r["elapsed_seconds"]) <= 1800 for r in records),
                "pw70_2h": sum(float(r["elapsed_seconds"]) <= 7200 for r in records),
                "pw70_6h": sum(float(r["elapsed_seconds"]) <= 21600 for r in records),
                "pw70_final": len(records),
                "expected_instances": 20,
                "declared_result_status": "complete_terminal_20_instance_denominator",
                "job_id": job_id,
                "local_stdout_log": log.relative_to(ROOT).as_posix(),
                "remote_stdout_log": (
                    "/home/hersco/training_new_domains/2026-09-13/"
                    f"mprime_phase_b_a_stage1_pw70/slurm/{job_id}.out"
                ),
                "local_completion_ledger": ledger.relative_to(ROOT).as_posix(),
                "remote_completion_ledger": f"{remote_root}/completion/{ledger.name}",
                "fixed_seed_source": FIXED.relative_to(ROOT).as_posix(),
            })

    if len(output) != 20 or len({(r["value_head"], r["seed"]) for r in output}) != 20:
        raise RuntimeError("expected exactly 20 unique MPrime PW cells")
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(sorted(output, key=lambda r: (r["value_head"], int(r["seed"]))))
    print(OUTPUT)


if __name__ == "__main__":
    main()
