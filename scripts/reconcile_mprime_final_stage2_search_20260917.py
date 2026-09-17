#!/usr/bin/env python3
"""Reconcile MPrime Stage-2 search outcomes and freeze exact recoveries.

Durable JSONL is authoritative for successes and ordinary completed failures.
Because the evaluator intentionally does not persist hard timeouts or worker
crashes, this script also parses immutable attempt/Slurm logs.  It never
classifies an OOM as a scientific terminal result.  An exact instance-scoped
OOM is retained only as an operational recovery hint; a job-level OOM cannot
even identify which of the concurrently active instances exhausted memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


MPRIME_TEST_INSTANCES = {
    1: "pfile01.pddl", 2: "pfile02.pddl", 3: "pfile04.pddl",
    4: "pfile05.pddl", 5: "pfile09.pddl", 6: "pfile13.pddl",
    7: "pfile14.pddl", 8: "pfile15.pddl", 9: "pfile16.pddl",
    10: "pfile18.pddl", 11: "pfile19.pddl", 12: "pfile20.pddl",
    13: "pfile21.pddl", 14: "pfile22.pddl", 15: "pfile23.pddl",
    16: "pfile24.pddl", 17: "pfile25.pddl", 18: "pfile26.pddl",
    19: "pfile27.pddl", 20: "pfile28.pddl",
}

COMPLETED_RE = re.compile(
    r"\[EVAL INSTANCE\] completed number=(?P<number>\d+) "
    r"path=(?P<path>\S+) status=(?P<status>success|unsolved) .*?"
    r"success=(?P<success>True|False) steps=(?P<steps>-?\d+)"
)
TIMEOUT_RE = re.compile(
    r"\[EVAL INSTANCE\] timeout number=(?P<number>\d+) "
    r"path=(?P<path>\S+) limit=(?P<limit>[0-9.]+)s"
)
CRASH_BLOCK_RE = re.compile(
    r"^(?P<header>\[EVAL INSTANCE\] crashed number=(?P<number>\d+) "
    r"path=(?P<path>\S+)[^\n]*)"
    r"(?P<body>(?:\n(?!\[EVAL (?:INSTANCE|FINAL)\]).*)*)",
    re.MULTILINE,
)
OOM_RE = re.compile(
    r"OutOfMemoryError|MemoryError|Cannot allocate memory|std::bad_alloc|"
    r"CUDA out of memory|Java heap space|oom-kill|Killed process",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Evidence:
    classification: str
    source_kind: str
    source_path: str
    detail: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def basename(path: str) -> str:
    return Path(path.replace("\\", "/")).name


def validate_instance(number: int, path: str) -> None:
    if number not in MPRIME_TEST_INSTANCES:
        raise ValueError(f"instance number outside 1-20: {number}")
    observed = basename(path)
    expected = MPRIME_TEST_INSTANCES[number]
    if observed != expected:
        raise ValueError(
            f"instance {number} path mismatch: observed {observed!r}, "
            f"expected {expected!r}"
        )


def evidence_from_ledger(path: Path, max_actions: int) -> dict[int, list[Evidence]]:
    evidence: dict[int, list[Evidence]] = defaultdict(list)
    if not path.is_file():
        return evidence
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            number = int(record["instance_number"])
            validate_instance(number, record["instance_path"])
            status = record["status"]
            steps = int(record["steps"])
            if status == "success" and bool(record["hit_goal"]):
                classification = "success"
            elif status == "finished_unsolved" and not bool(record["hit_goal"]):
                classification = "action_limit" if steps >= max_actions else "finished_unsolved"
            else:
                raise ValueError(
                    f"{path}:{line_number}: unsupported completion record {record}"
                )
            evidence[number].append(Evidence(
                classification, "completion_jsonl", str(path),
                f"line={line_number};status={status};steps={steps}",
            ))
    return evidence


def evidence_from_log(
    path: Path,
    *,
    declared_timeout: float,
    max_actions: int,
) -> dict[int, list[Evidence]]:
    evidence: dict[int, list[Evidence]] = defaultdict(list)
    if not path.is_file():
        return evidence
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in COMPLETED_RE.finditer(text):
        number = int(match["number"])
        validate_instance(number, match["path"])
        steps = int(match["steps"])
        classification = (
            "success" if match["success"] == "True"
            else "action_limit" if steps >= max_actions
            else "finished_unsolved"
        )
        evidence[number].append(Evidence(
            classification, "attempt_log_completed", str(path),
            f"status={match['status']};steps={steps}",
        ))
    for match in TIMEOUT_RE.finditer(text):
        number = int(match["number"])
        validate_instance(number, match["path"])
        limit = float(match["limit"])
        if abs(limit - declared_timeout) > 0.5:
            # Smoke or shortened timeouts are operational probes, not the
            # declared six-hour scientific classification.
            continue
        evidence[number].append(Evidence(
            "six_hour_timeout", "attempt_log_timeout", str(path),
            f"limit_seconds={limit:g}",
        ))
    for match in CRASH_BLOCK_RE.finditer(text):
        block = match.group(0)
        if not OOM_RE.search(block):
            continue
        number = int(match["number"])
        validate_instance(number, match["path"])
        evidence[number].append(Evidence(
            "instance_scoped_oom", "attempt_log_instance_oom", str(path),
            "explicit instance-scoped memory exhaustion; reported separately from search-unsolved",
        ))
    return evidence


def merge_evidence(
    ledger: dict[int, list[Evidence]],
    logs: list[dict[int, list[Evidence]]],
) -> tuple[
    dict[int, Evidence],
    dict[int, list[Evidence]],
    dict[int, list[Evidence]],
]:
    all_evidence: dict[int, list[Evidence]] = defaultdict(list)
    for source in [ledger, *logs]:
        for number, entries in source.items():
            all_evidence[number].extend(entries)
    terminal: dict[int, Evidence] = {}
    conflicts: dict[int, list[Evidence]] = {}
    oom_hints: dict[int, list[Evidence]] = {}
    for number, entries in all_evidence.items():
        oom_entries = [
            entry for entry in entries
            if entry.classification == "instance_scoped_oom"
        ]
        terminal_entries = [
            entry for entry in entries
            if entry.classification != "instance_scoped_oom"
        ]
        if oom_entries:
            oom_hints[number] = oom_entries
        if not terminal_entries:
            continue
        classes = {entry.classification for entry in terminal_entries}
        if len(classes) > 1:
            conflicts[number] = terminal_entries
            continue
        # Prefer the durable ledger, then the earliest deterministic path.
        terminal_entries.sort(key=lambda entry: (
            0 if entry.source_kind == "completion_jsonl" else 1,
            entry.source_path,
        ))
        terminal[number] = terminal_entries[0]
    return terminal, conflicts, oom_hints


def identity_paths(campaign_root: Path, row: dict[str, str]) -> tuple[Path, list[Path]]:
    identity_root = campaign_root / row["search_method"] / row["value_head"] / row["seed"]
    ledger = identity_root / "completion" / f"{row['manifest_id']}.jsonl"
    logs = sorted((identity_root / "attempts").glob("*.txt"))
    array_index = row["array_index"]
    logs.extend(sorted((campaign_root / "slurm").glob(f"*_{array_index}.out")))
    # Resolve duplicates if an attempt log was symlinked/copied into Slurm logs.
    unique = []
    seen = set()
    for path in logs:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return ledger, unique


def reconcile_identity(
    campaign_root: Path,
    row: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    max_actions = int(row["max_external_actions"])
    declared_timeout = float(row["instance_timeout_seconds"])
    expected = int(row["expected_test_instances"])
    if expected != 20:
        raise ValueError(f"{row['manifest_id']}: expected_test_instances must be 20")
    ledger_path, log_paths = identity_paths(campaign_root, row)
    ledger = evidence_from_ledger(ledger_path, max_actions)
    log_evidence = [
        evidence_from_log(
            path, declared_timeout=declared_timeout, max_actions=max_actions
        )
        for path in log_paths
    ]
    terminal, conflicts, oom_hints = merge_evidence(ledger, log_evidence)

    reconciliation = []
    recovery = []
    conflict_rows = []
    for number in range(1, expected + 1):
        base = {
            "array_index": row["array_index"],
            "manifest_id": row["manifest_id"],
            "search_method": row["search_method"],
            "value_head": row["value_head"],
            "seed": row["seed"],
            "selected_epoch": row["selected_epoch"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "instance_number": number,
            "instance_path": f"../problems/numeric/mprime/instances/{MPRIME_TEST_INSTANCES[number]}",
        }
        if number in conflicts:
            details = " | ".join(
                f"{entry.classification}@{entry.source_kind}:{entry.source_path}"
                for entry in conflicts[number]
            )
            reconciliation.append({
                **base, "scientific_state": "conflict_manual_review",
                "classification": "", "evidence_kind": "conflict",
                "evidence_path": "", "evidence_detail": details,
            })
            conflict_rows.append({**base, "conflict_detail": details})
        elif number in terminal:
            entry = terminal[number]
            reconciliation.append({
                **base, "scientific_state": "terminal",
                "classification": entry.classification,
                "evidence_kind": entry.source_kind,
                "evidence_path": entry.source_path,
                "evidence_detail": entry.detail,
            })
        else:
            oom_entries = oom_hints.get(number, [])
            if oom_entries:
                hint = sorted(oom_entries, key=lambda item: item.source_path)[0]
                scientific_state = "unclassified_after_instance_oom"
                evidence_kind = hint.source_kind
                evidence_path = hint.source_path
                evidence_detail = hint.detail
                recovery_reason = "instance_scoped_oom"
                recovery_memory_gib = 200
            else:
                scientific_state = "unclassified"
                evidence_kind = ""
                evidence_path = ""
                evidence_detail = ""
                recovery_reason = "no_terminal_instance_evidence"
                recovery_memory_gib = 120
            reconciliation.append({
                **base, "scientific_state": scientific_state,
                "classification": "", "evidence_kind": evidence_kind,
                "evidence_path": evidence_path,
                "evidence_detail": evidence_detail,
            })
            recovery.append({
                **base,
                "recovery_reason": recovery_reason,
                "recovery_memory_gib": recovery_memory_gib,
            })
    return reconciliation, recovery, conflict_rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--reconciliation-output", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--standard-recovery-manifest", type=Path, required=True)
    parser.add_argument("--oom-recovery-manifest", type=Path, required=True)
    parser.add_argument("--conflicts-output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_csv(args.manifest)
    if len(manifest) != 40:
        raise ValueError(f"expected 40 manifest rows, found {len(manifest)}")
    if len({row["manifest_id"] for row in manifest}) != 40:
        raise ValueError("manifest IDs are not unique")

    reconciliation = []
    recoveries = []
    conflicts = []
    for row in manifest:
        rec, missing, bad = reconcile_identity(args.campaign_root, row)
        reconciliation.extend(rec)
        recoveries.extend(missing)
        conflicts.extend(bad)
    for index, row in enumerate(recoveries):
        row["recovery_index"] = index
    standard_recoveries = [
        dict(row) for row in recoveries
        if int(row["recovery_memory_gib"]) == 120
    ]
    oom_recoveries = [
        dict(row) for row in recoveries
        if int(row["recovery_memory_gib"]) == 200
    ]
    for index, row in enumerate(standard_recoveries):
        row["recovery_index"] = index
    for index, row in enumerate(oom_recoveries):
        row["recovery_index"] = index

    reconciliation_fields = [
        "array_index", "manifest_id", "search_method", "value_head", "seed",
        "selected_epoch", "checkpoint_sha256", "instance_number", "instance_path",
        "scientific_state", "classification", "evidence_kind", "evidence_path",
        "evidence_detail",
    ]
    recovery_fields = [
        "recovery_index", "array_index", "manifest_id", "search_method",
        "value_head", "seed", "selected_epoch", "checkpoint_sha256",
        "instance_number", "instance_path", "recovery_reason",
        "recovery_memory_gib",
    ]
    conflict_fields = [
        "array_index", "manifest_id", "search_method", "value_head", "seed",
        "selected_epoch", "checkpoint_sha256", "instance_number", "instance_path",
        "conflict_detail",
    ]
    write_csv(args.reconciliation_output, reconciliation, reconciliation_fields)
    write_csv(args.recovery_manifest, recoveries, recovery_fields)
    write_csv(args.standard_recovery_manifest, standard_recoveries, recovery_fields)
    write_csv(args.oom_recovery_manifest, oom_recoveries, recovery_fields)
    write_csv(args.conflicts_output, conflicts, conflict_fields)

    counts: dict[str, int] = defaultdict(int)
    for row in reconciliation:
        key = row["classification"] or row["scientific_state"]
        counts[str(key)] += 1
    summary = {
        "manifest_identities": len(manifest),
        "expected_seed_instances": len(manifest) * 20,
        "terminal": sum(row["scientific_state"] == "terminal" for row in reconciliation),
        "unclassified": len(recoveries),
        "standard_recoveries": len(standard_recoveries),
        "oom_recoveries": len(oom_recoveries),
        "conflicts": len(conflicts),
        "classification_counts": dict(sorted(counts.items())),
        "recovery_manifest": str(args.recovery_manifest),
        "assumptions": {
            "timeout": "terminal only for exact instance/path and declared 21600-second cap",
            "oom": "never a terminal scientific outcome; explicit instance-scoped OOM is recovered at 200 GiB, while job-level OOM is not attributed across active instances",
            "worker_death": "unclassified without a persisted result or explicit instance-scoped terminal marker",
            "conflict": "never recovered automatically; requires manual scientific review",
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 3 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
