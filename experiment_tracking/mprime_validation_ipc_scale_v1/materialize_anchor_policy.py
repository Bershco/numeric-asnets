#!/usr/bin/env python3
"""Materialize policy curves for terminal corrected-MPrime anchor lineages."""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path


CAMPAIGN = Path("/home/hersco/training_new_domains/2026-08-28/mprime_corrected_downstream")
SUBMISSIONS = CAMPAIGN / "anchor_tuning_submissions.tsv"
OUTPUT = CAMPAIGN / "policy_ready.csv"
BEST_RE = re.compile(
    r"\[VALIDATION\] New best(?: reached!)?.*?"
    r"(?:iteration\s+|iter_num=)(\d+)"
)
LAST_RE = re.compile(r"Last valid checkpoint is (.+/snapshot_\d+_[^\s]+)")
FIELDS = [
    "manifest_id", "domain", "value_head", "seed", "anchor", "stage",
    "status", "teacher", "source_checkpoint_ref", "source_training_job_id",
    "snapshot_epoch", "analysis_roles", "training_state", "training_log",
]


def read(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def main() -> None:
    submitted = read(SUBMISSIONS, "\t")
    ids = [row["slurm_job_id"] for row in submitted]
    accounting: dict[str, tuple[str, Path]] = {}
    text = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(ids),
         "-o", "JobIDRaw,State,StdOut%1000"], text=True
    )
    for line in text.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3 and parts[0].isdigit():
            accounting[parts[0]] = (parts[1].split()[0], Path(parts[2].replace("%j", parts[0])))

    output: list[dict[str, str]] = []
    terminal = 0
    for row in submitted:
        job_id = row["slurm_job_id"]
        state, log = accounting.get(job_id, ("", Path()))
        if state in {"RUNNING", "PENDING", "REQUEUED", "COMPLETING"} or not log.is_file():
            continue
        body = log.read_text(encoding="utf-8", errors="replace")
        last = LAST_RE.findall(body)
        if not last:
            print(f"[SKIP] job={job_id} state={state} no valid checkpoint")
            continue
        snapshot_dir = Path(last[-1]).parent
        candidates: dict[int, list[Path]] = {}
        for entry in snapshot_dir.iterdir():
            match = re.fullmatch(r"snapshot_(\d+)_.+", entry.name)
            if match:
                candidates.setdefault(int(match.group(1)), []).append(entry)
        if not candidates:
            continue
        terminal += 1
        final_epoch = max(candidates)
        best = BEST_RE.findall(body)
        selected_epoch = int(best[-1]) if best else None
        epochs = {epoch for epoch in candidates if epoch % 5 == 0}
        epochs.add(final_epoch)
        if selected_epoch is not None:
            epochs.add(selected_epoch)
        for epoch in sorted(epochs):
            paths = candidates.get(epoch, [])
            if len(paths) != 1:
                raise RuntimeError(f"job {job_id} epoch {epoch}: expected one snapshot, got {paths}")
            roles = ["mprime_anchor_policy_curve"]
            if epoch == selected_epoch:
                roles.append("anchor_validation_selected_policy")
            if epoch == final_epoch:
                roles.append("anchor_final_policy")
            output.append({
                "manifest_id": f"{row['manifest_id']}-policy-e{epoch:04d}",
                "domain": "mprime", "value_head": row["value_head"], "seed": row["seed"],
                "anchor": row["anchor"], "stage": "stage2_tuning", "status": "ready",
                "teacher": row["teacher"], "source_checkpoint_ref": str(paths[0]),
                "source_training_job_id": job_id, "snapshot_epoch": str(epoch),
                "analysis_roles": ";".join(roles), "training_state": state,
                "training_log": str(log),
            })
    if len({row["manifest_id"] for row in output}) != len(output):
        raise RuntimeError("duplicate manifest IDs")
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(output)
    print(f"terminal_lineages={terminal} policy_rows={len(output)} output={OUTPUT}")


if __name__ == "__main__":
    main()
