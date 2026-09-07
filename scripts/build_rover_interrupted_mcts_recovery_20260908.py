#!/usr/bin/env python3
"""Build an exact-only Rover MCTS interruption recovery manifest.

The builder re-reads every original allocation log on the cluster, verifies
that the requested instance numbers still lack both a completed and explicit
per-instance-timeout record, and extracts the original checkpoint/configuration.
It refuses to emit a manifest unless the audited scope remains exactly 21
lineages and 29 instance evaluations.
"""

from __future__ import annotations

import base64
import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "experiment_tracking"
SOURCE = TRACKING / "mcts_interruption_opportunity_jobs_20260907.csv"
CSV_OUT = TRACKING / "rover_interrupted_mcts_recovery_manifest_20260908.csv"
TSV_OUT = TRACKING / "rover_interrupted_mcts_recovery_manifest_20260908.tsv"
SSH = Path(r"C:\Windows\System32\OpenSSH\ssh.exe")


def remote_audit(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    specs = [{"job_id": row["job_id"], "log": row["source_log"]} for row in rows]
    program = r'''
import json
import re
from pathlib import Path
specs = json.loads(%r)
for spec in specs:
    path = Path(spec["log"])
    terminal = set()
    arglines = []
    with path.open(errors="replace") as handle:
        for line in handle:
            if "Arguments are: Namespace(" in line:
                arglines.append(line)
            match = re.search(r"\[EVAL INSTANCE\] (?:completed|skip completed|timeout) number=(\d+)", line)
            if match:
                terminal.add(int(match.group(1)))
    if not arglines:
        raise RuntimeError("missing Arguments line: " + str(path))
    def take(pattern, line):
        match = re.search(pattern, line)
        if not match:
            raise RuntimeError("missing argument %%r in %%s" %% (pattern, path))
        return match.group(1)
    patterns = (
        r"resume_from='([^']+)'", r"experiments_numeric\.architecture_2\.([a-z0-9_]+)-",
        r"disable_value_head=(True|False)", r"mcts_expansion_size=(\d+)",
        r"mcts_iterations=(\d+)", r"mcts_exploration_weight=([0-9.]+)",
        r"use_estimator=([0-9.]+)", r"mcts_heuristic='([^']+)'",
    )
    signatures = {tuple(take(pattern, line) for pattern in patterns) for line in arglines}
    if len(signatures) != 1:
        raise RuntimeError("configuration changed across retries: " + str(path) + " " + repr(signatures))
    checkpoint, architecture, disable_vh, width, iterations, puct, estimator, heuristic = signatures.pop()
    result = {
        "job_id": spec["job_id"],
        "checkpoint": checkpoint,
        "architecture": architecture,
        "disable_value_head": disable_vh,
        "mcts_expansion_size": int(width),
        "mcts_iterations": int(iterations),
        "mcts_exploration_weight": float(puct),
        "use_estimator": float(estimator),
        "mcts_heuristic": heuristic,
        "invocation_count": len(arglines),
        "terminal_numbers": sorted(terminal),
    }
    print(json.dumps(result))
''' % json.dumps(specs)
    payload = base64.b64encode(program.encode()).decode()
    command = f"python3 -c \"import base64;exec(base64.b64decode('{payload}'))\""
    result = subprocess.run([str(SSH), "uni-cluster", command], check=True, text=True, capture_output=True)
    return {row["job_id"]: row for row in map(json.loads, result.stdout.splitlines())}


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = [row for row in csv.DictReader(handle) if row["domain"] == "rover"]
    if len(source_rows) != 21:
        raise RuntimeError(f"refusing changed scope: expected 21 Rover lineages, got {len(source_rows)}")
    if sum(int(row["unclassified_opportunities"]) for row in source_rows) != 29:
        raise RuntimeError("refusing changed scope: expected exactly 29 instance opportunities")

    remote = remote_audit(source_rows)
    output: list[dict[str, object]] = []
    for index, row in enumerate(source_rows):
        original = remote[row["job_id"]]
        missing = [int(value) for value in row["missing_instance_numbers"].split(";")]
        terminal = set(map(int, original["terminal_numbers"]))
        rediscovered = [number for number in range(1, 21) if number not in terminal]
        if missing != rediscovered:
            raise RuntimeError(
                f"job {row['job_id']} audit changed: CSV={missing}, original_log={rediscovered}"
            )
        expected_vh_disabled = row["value_head"] == "off"
        if (original["disable_value_head"] == "True") != expected_vh_disabled:
            raise RuntimeError(f"job {row['job_id']} value-head mismatch")
        expected_architecture = "rover" if row["experiment_scope"].startswith("stage1_") else "rover_mcts"
        if original["architecture"] != expected_architecture:
            raise RuntimeError(
                f"job {row['job_id']} architecture={original['architecture']} expected={expected_architecture}"
            )
        exact = (
            original["mcts_expansion_size"] == 20
            and original["mcts_iterations"] in {70, 200}
            and original["mcts_exploration_weight"] == 0.1
            and original["use_estimator"] == 0.5
            and original["mcts_heuristic"] == "hadd-gbfs"
        )
        if not exact:
            raise RuntimeError(f"job {row['job_id']} does not match an approved source configuration: {original}")
        skip = [number for number in range(1, 21) if number not in missing]
        checkpoint = str(original["checkpoint"])
        scope = row["experiment_scope"]
        if original["mcts_iterations"] != 70 and scope.endswith("_20_70"):
            scope = scope.removesuffix("_20_70") + f"_20_{original['mcts_iterations']}"
        output.append({
            "array_index": index,
            "recovery_id": f"rover-recover-{row['job_id']}",
            "source_job_id": row["job_id"],
            "experiment_scope": scope,
            "value_head": row["value_head"],
            "seed": row["seed"],
            "architecture_module": original["architecture"],
            "missing_instance_numbers": ";".join(map(str, missing)),
            "skip_instance_numbers": ",".join(map(str, skip)),
            "missing_count": len(missing),
            "checkpoint": checkpoint,
            "checkpoint_b64": base64.b64encode(checkpoint.encode()).decode(),
            "width": 20,
            "iterations": original["mcts_iterations"],
            "puct": 0.1,
            "estimator": 0.5,
            "workers": 2,
            "requested_cpus": 4,
            "requested_memory": "160G",
            "time_limit": "13:00:00",
            "instance_timeout_seconds": 21600,
            "source_log": row["source_log"],
            "source_completion_ledger": row["completion_ledger"],
            "source_invocation_count": original["invocation_count"],
        })

    fields = list(output[0])
    for path, delimiter in ((CSV_OUT, ","), (TSV_OUT, "\t")):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(output)
    print(json.dumps({
        "lineages": len(output),
        "instances": sum(int(row["missing_count"]) for row in output),
        "csv": str(CSV_OUT),
        "tsv": str(TSV_OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
