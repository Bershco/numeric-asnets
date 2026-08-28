#!/usr/bin/env python3
"""Idempotently submit SAFE-CONTEXT and binding-Horizon manifests."""

import argparse
import base64
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

TRACKING = Path(__file__).resolve().parent
RESULTS = TRACKING / "experiment_results.csv"


def checkpoints():
    found = {}
    with RESULTS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not (
                row["experiment_id"] == "MAIN-VAL"
                and row["task_type"] == "policy_eval"
                and row["stage"] == "stage1"
                and row["endpoint"] == "validation_selected"
                and row["checkpoint"]
            ):
                continue
            key = (row["domain"], row["value_head"], row["seed"])
            previous = found.setdefault(key, row["checkpoint"])
            if previous != row["checkpoint"]:
                raise RuntimeError(f"Conflicting checkpoints for {key}")
    return found


def submit(campaign, dry_run=False):
    cfg = {
        "context": (
            TRACKING / "mcts_safe_context/instrumentation_manifest.csv",
            TRACKING / "mcts_safe_context/evaluate_context.sbatch",
            TRACKING / "mcts_safe_context/submissions.tsv",
        ),
        "horizon": (
            TRACKING / "mcts_horizon_binding/manifest.csv",
            TRACKING / "mcts_horizon_binding/evaluate_horizon.sbatch",
            TRACKING / "mcts_horizon_binding/submissions.tsv",
        ),
        "context-full": (
            TRACKING / "mcts_safe_context/full_manifest.csv",
            TRACKING / "mcts_safe_context/evaluate_full.sbatch",
            TRACKING / "mcts_safe_context/full_submissions.tsv",
        ),
    }[campaign]
    manifest, sbatch, ledger = cfg
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    existing = set()
    if ledger.exists():
        with ledger.open(newline="", encoding="utf-8") as stream:
            existing = {
                row["manifest_id"]
                for row in csv.DictReader(stream, delimiter="\t")
            }
    resolved = checkpoints()
    fields = ("manifest_id", "slurm_job_id", "submitted_at")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if stream.tell() == 0:
            writer.writeheader()
        for idx, row in enumerate(rows, 1):
            if campaign == "context":
                manifest_id = f"context-{row['domain']}-{row['value_head']}-{row['seed']}"
            elif campaign == "context-full":
                manifest_id = f"context-full-drone-{row['value_head']}-{row['seed']}-{row['arm']}"
            else:
                manifest_id = f"horizon-drone-{row['value_head']}-{row['seed']}-{row['arm']}"
            if manifest_id in existing:
                continue
            key = (row["domain"], row["value_head"], row["seed"])
            checkpoint = resolved.get(key)
            if checkpoint is None:
                raise RuntimeError(f"No authoritative checkpoint for {key}")
            exports = [
                "ALL",
                "CHECKPOINT_B64=" + base64.b64encode(
                    checkpoint.encode()).decode(),
                f"SEED={row['seed']}",
                f"VALUE_HEAD={row['value_head']}",
                f"MANIFEST_ID={manifest_id}",
            ]
            if campaign == "context":
                restrict = "0,29,58" if row["domain"] == "counters" else "0,9,19"
                exports.extend((f"DOMAIN={row['domain']}", f"RESTRICT={restrict}"))
            else:
                exports.append(f"ARM={row['arm']}")
            output_root = (
                "/home/hersco/training_new_domains/2026-08-28/"
                + ("mcts_safe_context" if campaign == "context"
                   else ("mcts_safe_context_full" if campaign == "context-full"
                         else "mcts_horizon_binding")))
            if dry_run:
                print(
                    f"DRY_RUN|{manifest_id}|checkpoint={checkpoint}|"
                    f"sbatch={sbatch}")
                continue
            result = subprocess.run([
                "sbatch", *( ["--hold"] if campaign == "context-full" else [] ),
                f"--job-name={manifest_id}",
                f"--output={output_root}/%x_%j.out",
                "--export=" + ",".join(exports), str(sbatch),
            ], check=True, text=True, capture_output=True)
            job_id = result.stdout.strip().split()[-1]
            writer.writerow({
                "manifest_id": manifest_id,
                "slurm_job_id": job_id,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            })
            stream.flush()
            print(f"SUBMITTED|{manifest_id}|{job_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "campaign", choices=("context", "context-full", "horizon"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    submit(args.campaign, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
