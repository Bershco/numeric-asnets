#!/usr/bin/env python3
"""Materialize corrected MPrime policy curves and schedule the anchor gate."""

from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path


ROOT = Path("/home/hersco/training_new_domains")
SUBMITTER = ROOT / "submit_training.sh"
WORK = ROOT / "2026-08-28/mprime_corrected_downstream"
LEDGER = WORK / "policy_submissions.tsv"
MANIFEST = WORK / "policy_manifest.csv"
GATE = WORK / "mprime_anchor_gate.sbatch"
JOB_RE = re.compile(r"\[OK \] job=\s*(\d+)")
BEST_RE = re.compile(r"New best reached!.*snapshot name: (snapshot_\d+_[^\]]+)")
SNAP_RE = re.compile(r"snapshot_(\d+)_")
LAST_CHECKPOINT_RE = re.compile(
    r"^Last valid checkpoint is (.+?)/snapshots/snapshot_\d+_[^\s]+$",
    re.MULTILINE,
)
TRAINING = [
    (20618716, "off", 1963100312), (20618717, "off", 2011206605),
    (20618718, "off", 1073581256), (20618719, "off", 1239739722),
    (20618720, "off", 1472491096), (20618721, "off", 534933607),
    (20618722, "off", 2082152039), (20618723, "off", 1510771779),
    (20618724, "off", 923500475), (20618725, "off", 1972442430),
    (20618726, "on", 1963100312), (20618727, "on", 2011206605),
    (20618728, "on", 1073581256), (20618729, "on", 1239739722),
    (20618730, "on", 1472491096), (20618731, "on", 534933607),
    (20618732, "on", 2082152039), (20618733, "on", 1510771779),
    (20618734, "on", 923500475), (20618735, "on", 1972442430),
]


def accounting(job: int) -> tuple[str, Path]:
    text = subprocess.check_output([
        "sacct", "-X", "-n", "-j", str(job),
        "--format=State,StdOut", "-P",
    ], text=True).strip().splitlines()[0]
    state, stdout = text.split("|", 1)
    return state, Path(stdout.replace("%j", str(job)))


def submit_policy(row: dict[str, str]) -> str:
    cmd = [
        str(SUBMITTER), "--dom-mprime", "--original-only",
        "--domain-architecture", "policy", "--eval-from", row["checkpoint"],
        "--seed", row["seed"], "--workers", "10", "--jpddl-max-heap", "4g",
        "--time", "04:00:00", "--mem", "20G", "--cpus", "6",
        "--output-subdir", "mprime_corrected_stage1_policy",
        "--job-suffix", f"MPCP_src{row['training_job']}_e{int(row['epoch']):04d}",
    ]
    if row["value_head"] == "off":
        cmd.append("--vh-off")
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout)
    ids = JOB_RE.findall(result.stdout)
    if len(ids) != 1:
        raise RuntimeError(result.stdout)
    return ids[0]


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    tasks: dict[tuple[int, int], dict[str, str]] = {}
    for job, vh, seed in TRAINING:
        state, log = accounting(job)
        if state not in {"COMPLETED", "TIMEOUT"}:
            raise RuntimeError(f"training job {job} ended as {state}")
        text = log.read_text(encoding="utf-8", errors="replace")
        roots = [f"{root}/snapshots"
                 for root in LAST_CHECKPOINT_RE.findall(text)]
        if not roots:
            raise RuntimeError(
                f"job {job}: no 'Last valid checkpoint' snapshot marker")
        snapshot_dir = Path(roots[-1])
        snapshots = []
        for path in snapshot_dir.glob("snapshot_*"):
            match = SNAP_RE.search(path.name)
            if match and path.is_dir():
                snapshots.append((int(match.group(1)), path))
        if not snapshots:
            raise RuntimeError(f"job {job}: no snapshots")
        best_names = BEST_RE.findall(text)
        if not best_names:
            raise RuntimeError(f"job {job}: no validation-selected snapshot marker")
        selected = snapshot_dir / best_names[-1]
        if not selected.is_dir():
            raise RuntimeError(f"job {job}: selected snapshot missing: {selected}")
        final_epoch, final = max(snapshots)
        selected_epoch = int(SNAP_RE.search(selected.name).group(1))
        chosen = {epoch: path for epoch, path in snapshots if epoch % 5 == 0}
        chosen[selected_epoch] = selected
        chosen[final_epoch] = final
        for epoch, checkpoint in sorted(chosen.items()):
            roles = ["curve"] if epoch % 5 == 0 else []
            if epoch == selected_epoch:
                roles.append("selected")
            if epoch == final_epoch:
                roles.append("final")
            tasks[(job, epoch)] = {
                "manifest_id": f"mprime-{vh}-{seed}-s1-e{epoch:04d}",
                "training_job": str(job), "value_head": vh, "seed": str(seed),
                "epoch": str(epoch), "checkpoint": str(checkpoint),
                "roles": ";".join(roles),
            }

    fields = ["manifest_id", "training_job", "value_head", "seed", "epoch",
              "checkpoint", "roles"]
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(tasks.values())

    existing = {}
    if LEDGER.exists():
        with LEDGER.open(newline="", encoding="utf-8") as stream:
            existing = {row["manifest_id"]: row for row in csv.DictReader(stream, delimiter="\t")}
    ledger_fields = fields + ["slurm_job_id"]
    selected_jobs = []
    with LEDGER.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ledger_fields, delimiter="\t", lineterminator="\n")
        if stream.tell() == 0:
            writer.writeheader()
        for row in tasks.values():
            if row["manifest_id"] in existing:
                job_id = existing[row["manifest_id"]]["slurm_job_id"]
            else:
                job_id = submit_policy(row)
                writer.writerow({**row, "slurm_job_id": job_id}); stream.flush(); os.fsync(stream.fileno())
            if "selected" in row["roles"].split(";"):
                selected_jobs.append(job_id)
    if len(selected_jobs) != 20:
        raise RuntimeError(f"expected 20 selected endpoint evaluations, found {len(selected_jobs)}")
    dependency = "afterany:" + ":".join(selected_jobs)
    result = subprocess.run([
        "sbatch", f"--dependency={dependency}", str(GATE),
    ], check=True, text=True, capture_output=True)
    print(f"policy_tasks={len(tasks)} selected=20 gate={result.stdout.strip()}")


if __name__ == "__main__":
    main()
