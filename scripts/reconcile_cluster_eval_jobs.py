#!/usr/bin/env python3
"""Create a compact, provenance-linked ledger from terminal evaluation jobs."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


FINAL_RE = re.compile(r"\[EVAL FINAL\]\s+success=([0-9.]+)/([0-9.]+)")
VAL_VALID_RE = re.compile(r"\[VAL\]\s+VAL-valid plans\s*:\s*(\d+)")
VAL_INVALID_RE = re.compile(r"\[VAL\]\s+VAL-invalid plans\s*:\s*(\d+)")
INSTANCE_RE = re.compile(
    r"\[EVAL INSTANCE\] completed number=(\d+) path=(\S+) "
    r"status=(\S+) elapsed=([0-9.]+)s success=([0-9.]+) steps=(\d+)"
)
NAME_RE = re.compile(
    r"Ev_(?P<domain>[^_]+(?:_[^_]+)*)_(?P=domain)_mcts_orig_"
    r"(?P<vh>novh|vh).*?_s(?P<seed>\d+)_K0_(?P<tag>[^_]+)_src"
    r"(?P<src>\d+)_e(?P<epoch>\d+)"
)
BEST_RE = re.compile(
    r"\[VALIDATION\] New best reached! .*?iteration (\d+) .*?"
    r"snapshot name: (snapshot_\d+_[^\]]+)"
)
LAST_RE = re.compile(r"Last valid checkpoint is (.+/snapshot_(\d+)_[^\s]+)")

FIELDS = [
    "job_id", "job_name", "domain", "value_head", "seed", "tag",
    "source_training_job_id", "snapshot_epoch", "analysis_roles",
    "slurm_state", "exit_code", "elapsed", "successes", "total",
    "val_valid", "val_invalid", "classified_instances",
    "successful_instance_runtime_seconds", "source_evaluation_log",
    "source_training_log",
]


def sacct(*, ids: list[str] | None = None, start: str | None = None) -> list[dict[str, str]]:
    command = ["sacct", "-X", "-P", "-n"]
    if ids:
        command += ["-j", ",".join(ids)]
    if start:
        command += ["-S", start]
    command += [
        "-o", "JobIDRaw,JobName%180,State,ExitCode,Elapsed,StdOut%1000"
    ]
    text = subprocess.check_output(command, text=True)
    rows = []
    for line in text.splitlines():
        parts = line.split("|", 5)
        if len(parts) != 6 or not parts[0].isdigit():
            continue
        rows.append(dict(zip(
            ("job_id", "job_name", "state", "exit_code", "elapsed", "stdout"),
            parts,
        )))
    return rows


def read_ids(path: Path, field: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [row[field] for row in csv.DictReader(stream) if row.get(field)]


def terminal_training_metadata(source_ids: set[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for offset in range(0, len(source_ids), 200):
        chunk = sorted(source_ids)[offset:offset + 200]
        for row in sacct(ids=chunk):
            log = Path(row["stdout"].replace("%j", row["job_id"]))
            if not log.is_file():
                continue
            text = log.read_text(encoding="utf-8", errors="replace")
            best = BEST_RE.findall(text)
            last = LAST_RE.findall(text)
            result[row["job_id"]] = {
                "selected_epoch": best[-1][0] if best else "",
                "final_epoch": last[-1][1] if last else "",
                "log": str(log),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ids-csv", type=Path)
    source.add_argument("--sacct-start")
    parser.add_argument("--id-field", default="job_id")
    parser.add_argument("--job-name-contains", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-training-roles", action="store_true")
    args = parser.parse_args()

    if args.ids_csv:
        ids = read_ids(args.ids_csv, args.id_field)
        account = []
        for offset in range(0, len(ids), 200):
            account.extend(sacct(ids=ids[offset:offset + 200]))
    else:
        account = sacct(start=args.sacct_start)
    if args.job_name_contains:
        account = [
            row for row in account
            if args.job_name_contains in row["job_name"]
        ]

    parsed = []
    source_ids: set[str] = set()
    for row in account:
        match = NAME_RE.search(row["job_name"])
        if not match:
            print(f"[SKIP] unparsed job name: {row['job_name']}")
            continue
        source_ids.add(match.group("src"))
        parsed.append((row, match))
    training = terminal_training_metadata(source_ids) if args.include_training_roles else {}

    output = []
    for row, match in parsed:
        log = Path(row["stdout"].replace("%j", row["job_id"]))
        if not log.is_file():
            print(f"[SKIP] missing log: {log}")
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        final = FINAL_RE.findall(text)
        valid = VAL_VALID_RE.findall(text)
        invalid = VAL_INVALID_RE.findall(text)
        instances = INSTANCE_RE.findall(text)
        successes = int(float(final[-1][0])) if final else sum(
            float(entry[4]) == 1.0 for entry in instances)
        total = int(float(final[-1][1])) if final else len(instances)
        epoch = str(int(match.group("epoch")))
        roles = []
        train = training.get(match.group("src"), {})
        if args.include_training_roles:
            roles.append("stage2_policy_curve")
            if epoch == train.get("selected_epoch"):
                roles.append("stage2_validation_selected_policy")
            if epoch == train.get("final_epoch"):
                roles.append("stage2_final_policy")
        tag = match.group("tag")
        if tag == "SR10M":
            roles.append("main_val_stage2_selected_mcts")
        elif tag == "SR10TCM":
            roles.append("main_term_stage2_selected_mcts")
        elif tag == "SR10LONGM":
            roles.append("long_training_final_mcts")
        success_times = [
            entry[3] for entry in instances if float(entry[4]) == 1.0
        ]
        output.append({
            "job_id": row["job_id"], "job_name": row["job_name"],
            "domain": match.group("domain"),
            "value_head": "off" if match.group("vh") == "novh" else "on",
            "seed": match.group("seed"), "tag": tag,
            "source_training_job_id": match.group("src"),
            "snapshot_epoch": epoch, "analysis_roles": ";".join(roles),
            "slurm_state": row["state"].split()[0],
            "exit_code": row["exit_code"], "elapsed": row["elapsed"],
            "successes": successes, "total": total,
            "val_valid": valid[-1] if valid else "",
            "val_invalid": invalid[-1] if invalid else "",
            "classified_instances": len(instances),
            "successful_instance_runtime_seconds": ";".join(success_times),
            "source_evaluation_log": str(log),
            "source_training_log": train.get("log", ""),
        })
    output.sort(key=lambda row: (
        row["domain"], row["value_head"], int(row["seed"]),
        int(row["snapshot_epoch"]), int(row["job_id"])
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(output)
    print(f"rows={len(output)} output={args.output}")


if __name__ == "__main__":
    main()
