#!/usr/bin/env python3
"""Materialize every-five and endpoint policy work from a Stage-2 ledger."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


BEST_RE = re.compile(
    r"\[VALIDATION\] New best(?: reached!)?.*?"
    r"(?:iteration\s+|iter_num=)(\d+).*?"
    r"(?:snapshot name:\s*|snapshot_name=)(snapshot_\d+_[^\s\]]+)"
)
LAST_RE = re.compile(r"Last valid checkpoint is (.+/snapshot_\d+_[^\s]+)")
TEACHERS = {
    "delivery": "hadd-astar",
    "tpp": "hadd-astar",
    "zenotravel": "hadd-gbfs",
    "mprime": "hmrp-ha-gbfs",
}
FIELDS = [
    "manifest_id", "task_type", "domain", "value_head", "seed", "stage",
    "status", "teacher", "source_checkpoint_ref", "source_training_job_id",
    "snapshot_epoch", "analysis_roles", "training_state", "training_log",
]


def read(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def accounting(job_ids: list[str]) -> dict[str, tuple[str, Path]]:
    if not job_ids:
        return {}
    text = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
         "-o", "JobIDRaw,State,StdOut%1000"], text=True
    )
    result: dict[str, tuple[str, Path]] = {}
    for line in text.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3 and parts[0].isdigit():
            result[parts[0]] = (
                parts[1].split()[0], Path(parts[2].replace("%j", parts[0]))
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--domain")
    parser.add_argument("--branch")
    parser.add_argument("--exclude-reused", action="store_true")
    parser.add_argument("--role-prefix", required=True)
    args = parser.parse_args()

    rows = read(args.ledger)
    selected: list[dict[str, str]] = []
    for row in rows:
        domain = row.get("domain") or args.domain or ""
        if args.domain and domain != args.domain:
            continue
        if args.branch and row.get("branch") != args.branch:
            continue
        if args.exclude_reused and row.get("reuse_tuning_job"):
            continue
        row = dict(row); row["domain"] = domain
        selected.append(row)
    ids = [row["slurm_job_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate Stage-2 training job IDs after filtering")
    info = accounting(ids)

    output: list[dict[str, str]] = []
    terminal = 0
    for row in selected:
        job_id = row["slurm_job_id"]
        state, log = info.get(job_id, ("", Path()))
        if state in {"RUNNING", "PENDING", "REQUEUED", "COMPLETING"}:
            continue
        if not log.is_file():
            raise RuntimeError(f"terminal job {job_id} has no readable log: {log}")
        text = log.read_text(encoding="utf-8", errors="replace")
        last = LAST_RE.findall(text)
        if not last:
            raise RuntimeError(f"terminal job {job_id} has no valid checkpoint: {log}")
        snapshot_dir = Path(last[-1]).parent
        candidates: dict[int, list[Path]] = {}
        for entry in snapshot_dir.iterdir():
            match = re.fullmatch(r"snapshot_(\d+)_.+", entry.name)
            if match:
                candidates.setdefault(int(match.group(1)), []).append(entry)
        best = BEST_RE.findall(text)
        if not best:
            raise RuntimeError(f"terminal job {job_id} has no validation-selected checkpoint")
        selected_epoch = int(best[-1][0])
        final_epoch = max(candidates)
        epochs = {epoch for epoch in candidates if epoch % 5 == 0}
        epochs.update({selected_epoch, final_epoch})
        terminal += 1
        for epoch in sorted(epochs):
            paths = candidates.get(epoch, [])
            if len(paths) != 1:
                raise RuntimeError(f"job {job_id} epoch {epoch}: expected one snapshot, got {paths}")
            roles = [f"{args.role_prefix}_learning_curve"]
            if epoch == selected_epoch:
                roles.append(f"{args.role_prefix}_validation_selected_policy")
            if epoch == final_epoch:
                roles.append(f"{args.role_prefix}_final_policy")
            output.append({
                "manifest_id": f"{row['manifest_id']}-policy-e{epoch:04d}",
                "task_type": "policy_eval", "domain": row["domain"],
                "value_head": row["value_head"], "seed": row["seed"],
                "stage": "stage2", "status": "ready",
                "teacher": TEACHERS[row["domain"]],
                "source_checkpoint_ref": str(paths[0]),
                "source_training_job_id": job_id, "snapshot_epoch": str(epoch),
                "analysis_roles": ";".join(roles), "training_state": state,
                "training_log": str(log),
            })
    manifest_ids = [row["manifest_id"] for row in output]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise RuntimeError("duplicate policy manifest IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(output)
    print(f"selected_lineages={len(selected)} terminal={terminal} policy_rows={len(output)}")


if __name__ == "__main__":
    main()
