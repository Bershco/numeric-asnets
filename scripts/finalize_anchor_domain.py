#!/usr/bin/env python3
"""Audit a completed ANCHOR-4 domain, freeze winners, and launch confirmation.

The script is intended for a small Slurm controller after all tuning lineages
for one domain are terminal.  It reconstructs each lineage from the original
job plus its quota continuation, selects the coefficient by two-seed mean
validation AUC (peak and final coverage as tie-breaks), and submits only the
sixteen held-out-seed Stage-2 confirmation jobs.  The two winning tuning
lineages are reused.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/hersco/training_new_domains")
MAIN = ROOT / "2026-08-25/completed_domains_anchor_tuning"
RECOVERY = ROOT / "2026-08-26/quota_recovery"
CAMPAIGN = ROOT / "2026-08-27/anchor_domain_finalizers"
SUBMITTER = ROOT / "submit_training.sh"
RESULTS = CAMPAIGN / "experiment_results.csv"
MAIN_MANIFEST = MAIN / "anchor_tuning_stage2_training.csv"
MAIN_LEDGER = MAIN / "anchor_tuning_submissions.tsv"
CONT_MANIFEST = RECOVERY / "failed_anchor_training_continuations.csv"
CONT_LEDGER = RECOVERY / "failed_anchor_training_continuation_submissions.tsv"
TUNING_SEEDS = {"1963100312", "2011206605"}
DOMAINS = {
    "delivery": "hadd-astar",
    "tpp": "hadd-astar",
    "zenotravel": "hadd-gbfs",
}
RATE_RE = re.compile(r"\[VALIDATION\] Current network validation success rate: ([0-9.]+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
JOB_RE = re.compile(r"\[OK \] job=\s*(\d+)")


def read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def accounting(job_ids: list[str]) -> dict[str, tuple[str, Path]]:
    result: dict[str, tuple[str, Path]] = {}
    for start in range(0, len(job_ids), 200):
        text = subprocess.check_output([
            "sacct", "-X", "-n", "-P", "-j", ",".join(job_ids[start:start + 200]),
            "-o", "JobIDRaw,State,StdOut%1000",
        ], text=True)
        for line in text.splitlines():
            job, state, stdout = line.split("|", 2)
            if job.isdigit():
                result[job] = (state, Path(stdout.replace("%j", job)))
    return result


def validation_rates(path: Path, fresh_phase: bool) -> list[float]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    if fresh_phase:
        marker = "[VALIDATION STATE] Starting a new trainer phase"
        if marker not in text:
            raise RuntimeError(f"missing fresh-phase validation marker: {path}")
        text = text.rsplit(marker, 1)[1]
    return [float(value) for value in RATE_RE.findall(text)]


def freeze(domain: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    main_rows = {
        row["manifest_id"]: row for row in read(MAIN_MANIFEST)
        if row["domain"] == domain and row["seed"] in TUNING_SEEDS
    }
    if len(main_rows) != 28:
        raise RuntimeError(f"{domain}: expected 28 tuning lineages, got {len(main_rows)}")
    main_jobs = {
        row["manifest_id"]: row["slurm_job_id"] for row in read(MAIN_LEDGER, "\t")
        if row["manifest_id"] in main_rows
    }
    cont_spec = {
        row["original_manifest_id"]: row for row in read(CONT_MANIFEST)
        if row["original_manifest_id"] in main_rows
    }
    original_by_continuation = {
        row["continuation_id"]: row["original_manifest_id"]
        for row in cont_spec.values()
    }
    cont_jobs = {
        original_by_continuation[row["continuation_id"]]: row["slurm_job_id"]
        for row in read(CONT_LEDGER, "\t")
        if row["continuation_id"] in original_by_continuation
    }
    all_jobs = sorted(set(main_jobs.values()) | set(cont_jobs.values()), key=int)
    info = accounting(all_jobs)
    evidence: list[dict[str, str]] = []
    grouped: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for manifest_id, row in sorted(main_rows.items()):
        original_job = main_jobs[manifest_id]
        original_state, original_log = info[original_job]
        final_job = cont_jobs.get(manifest_id, original_job)
        final_state, final_log = info[final_job]
        if final_state in {"RUNNING", "PENDING"}:
            raise RuntimeError(f"{manifest_id}: final job is still {final_state}")
        if not original_log.is_file() or not final_log.is_file():
            raise RuntimeError(f"{manifest_id}: missing source log")
        rates = validation_rates(original_log, fresh_phase=True)
        if final_job != original_job:
            rates += validation_rates(final_log, fresh_phase=False)
        if not rates:
            raise RuntimeError(f"{manifest_id}: no validation rates")
        auc = sum(rates) / len(rates)
        peak = max(rates)
        final = rates[-1]
        grouped[(row["value_head"], row["anchor"])].append((auc, peak, final))
        evidence.append({
            "manifest_id": manifest_id, "domain": domain,
            "value_head": row["value_head"], "seed": row["seed"],
            "anchor": row["anchor"], "original_job_id": original_job,
            "final_job_id": final_job, "original_state": original_state,
            "final_state": final_state, "validation_points": str(len(rates)),
            "validation_auc": f"{auc:.9f}", "peak_validation": f"{peak:.9f}",
            "final_validation": f"{final:.9f}",
            "original_log": str(original_log), "final_log": str(final_log),
        })
    winners: dict[str, str] = {}
    for vh in ("off", "on"):
        candidates = []
        for (candidate_vh, anchor), values in grouped.items():
            if candidate_vh != vh:
                continue
            if len(values) != 2:
                raise RuntimeError(f"{domain}/{vh}/anchor={anchor}: expected two seeds")
            means = tuple(sum(value[i] for value in values) / 2 for i in range(3))
            candidates.append((means, -float(anchor), anchor))
        if len(candidates) != 7:
            raise RuntimeError(f"{domain}/{vh}: expected seven coefficients")
        winners[vh] = max(candidates)[2]
    return evidence, winners


def submit_confirmation(domain: str, winners: dict[str, str], dry_run: bool) -> list[dict[str, str]]:
    sources = [
        row for row in read(RESULTS)
        if row["experiment_id"] == "PRESERVE-4" and row["domain"] == domain
        and row["task_type"] == "policy_eval" and row["stage"] == "stage1"
        and row["endpoint"] == "validation_selected" and row["seed"] not in TUNING_SEEDS
    ]
    if len(sources) != 16 or len({(r["value_head"], r["seed"]) for r in sources}) != 16:
        raise RuntimeError(f"{domain}: expected sixteen unique held-out Stage-1 sources")
    ledger_path = CAMPAIGN / f"{domain}_stage2_confirmation_submissions.tsv"
    existing = {row["manifest_id"] for row in read(ledger_path, "\t")} if ledger_path.exists() else set()
    submitted: list[dict[str, str]] = []
    for source in sorted(sources, key=lambda r: (r["value_head"], int(r["seed"]))):
        vh, seed, anchor = source["value_head"], source["seed"], winners[source["value_head"]]
        manifest_id = f"preserve4-{domain}-{vh}-{seed}-stage2-a{anchor.replace('.', 'p')}"
        if manifest_id in existing:
            continue
        command = [
            str(SUBMITTER), f"--dom-{domain}", "--original-only",
            "--domain-architecture", "mcts", "--seed", seed,
            "--workers", "3", "--jpddl-max-heap", "4g", "--time", "3-00:00:00",
            "--mem", "48G", "--cpus", "6", "--train-from", source["checkpoint"],
            "--use-estimator", "0.5", "--exploration-weight", "0.1",
            "--override-tree-sampling", "0", "--mcts-expansion-size", "20",
            "--mcts-iterations", "0", "--policy-anchor-kl-coeff", anchor,
            "--max-opt-epochs", "100", "--supervised-lr", "0.0003",
            "--job-suffix", f"P4S2A{anchor.replace('.', 'p')}_src{source['source_training_job_id']}",
            "--output-subdir", f"preserve4_{domain}_stage2_confirmation",
        ]
        if vh == "off":
            command.append("--vh-off")
        if dry_run:
            command.append("--dry-run")
        env = os.environ.copy(); env["ENHSP_CONFIG_OVERRIDE"] = DOMAINS[domain]
        output = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if output.returncode:
            raise RuntimeError(output.stdout)
        if dry_run:
            continue
        ids = JOB_RE.findall(output.stdout)
        if len(ids) != 1:
            raise RuntimeError(output.stdout)
        record = {
            "manifest_id": manifest_id, "domain": domain, "value_head": vh,
            "seed": seed, "anchor": anchor,
            "source_training_job_id": source["source_training_job_id"],
            "snapshot_epoch": source["epoch"], "slurm_job_id": ids[0],
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": source["checkpoint"],
        }
        new_file = not ledger_path.exists()
        with ledger_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record), delimiter="\t", lineterminator="\n")
            if new_file: writer.writeheader()
            writer.writerow(record); stream.flush(); os.fsync(stream.fileno())
        existing.add(manifest_id); submitted.append(record)
    return submitted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=DOMAINS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    evidence, winners = freeze(args.domain)
    evidence_path = CAMPAIGN / f"{args.domain}_anchor_evidence.csv"
    with evidence_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evidence[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(evidence)
    print(f"[FROZEN] domain={args.domain} winners={winners} evidence={evidence_path}")
    submitted = submit_confirmation(args.domain, winners, args.dry_run)
    print(f"[CONFIRMATION] domain={args.domain} submitted={len(submitted)} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
