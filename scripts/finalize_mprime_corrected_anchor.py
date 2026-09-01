#!/usr/bin/env python3
"""Freeze corrected MPrime anchors and submit both Stage-2 branches.

This intentionally ignores the obsolete training-log validation scores.  It
uses the frozen IPC-scale validation rescore artifacts.  An anchor may be
frozen before every point is present only when its AUC lower bound is strictly
above every competitor's best possible AUC upper bound.
"""

from __future__ import annotations

import csv
import os
import re
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parents[1] / "experiment_tracking/mprime_validation_ipc_scale_v1"
ROOT = Path("/home/hersco/training_new_domains")
RESCORE = ROOT / "2026-08-31/mprime_anchor_corrected_validation/rescore"
AUDIT = HERE / "validation_test_checkpoint_audit.csv"
TUNING_LEDGER = HERE / "anchor_tuning_submissions.tsv"
FROZEN = HERE / "anchor_selection_corrected_frozen.csv"
STAGE2_LEDGER = HERE / "stage2_submissions_corrected.tsv"
SUBMITTER = ROOT / "submit_training.sh"
ANCHORS = ("0", "0.03", "0.3", "1", "3", "10", "30")
TUNING_SEEDS = {"1963100312", "2011206605"}
JOB_RE = re.compile(r"\[OK \] job=\s*(\d+)")


def read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def corrected_metrics() -> tuple[dict[str, str], list[dict[str, str]]]:
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_seed: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    pattern = re.compile(r"mprime-corrected-anchor-(off|on)-(\d+)-a(.+)")
    for directory in RESCORE.iterdir():
        match = pattern.fullmatch(directory.name)
        if not match:
            continue
        vh, seed, anchor = match.groups()
        anchor = anchor.replace("p", ".")
        for summary in directory.glob("epoch_*_val.csv"):
            epoch = int(re.search(r"epoch_(\d+)_val", summary.name).group(1))
            rows = read(summary)
            if len(rows) > 30:
                raise RuntimeError(f"{summary}: more than 30 validation instances: {len(rows)}")
            # The validator CSV records candidate plans.  Instances for which
            # inference printed no plan are absent and conservatively score 0.
            score = sum(int(row["val_valid"]) for row in rows) / 30
            samples[(vh, anchor)].append(score)
            by_seed[(vh, anchor, seed)].append((epoch, score))

    evidence: list[dict[str, str]] = []
    winners: dict[str, str] = {}
    for vh in ("off", "on"):
        bounds: dict[str, tuple[float, float]] = {}
        for anchor in ANCHORS:
            values = samples[(vh, anchor)]
            if len(values) > 42:
                raise RuntimeError(f"{vh}/{anchor}: more than 42 points")
            lower = sum(values) / 42
            upper = (sum(values) + (42 - len(values))) / 42
            bounds[anchor] = (lower, upper)
            seed_groups = [v for key, v in by_seed.items() if key[:2] == (vh, anchor)]
            observed_auc = statistics.mean(values)
            peak = statistics.mean(max(score for _, score in group) for group in seed_groups)
            final = statistics.mean(max(group)[1] for group in seed_groups)
            evidence.append({
                "value_head": vh, "anchor": anchor, "points": str(len(values)),
                "expected_points": "42", "observed_auc": f"{observed_auc:.6f}",
                "auc_lower_bound": f"{lower:.6f}", "auc_upper_bound": f"{upper:.6f}",
                "mean_peak": f"{peak:.6f}", "mean_final": f"{final:.6f}",
                "rescore_root": str(RESCORE),
            })
        winner = max(ANCHORS, key=lambda a: (bounds[a][0], -float(a)))
        competitor_upper = max(bounds[a][1] for a in ANCHORS if a != winner)
        if bounds[winner][0] <= competitor_upper:
            raise RuntimeError(
                f"{vh}: winner not mathematically locked; {winner} lower={bounds[winner][0]:.6f} "
                f"competitor upper={competitor_upper:.6f}"
            )
        winners[vh] = winner

    with FROZEN.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evidence[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(evidence)
    print(f"FROZEN|off={winners['off']}|on={winners['on']}|evidence={FROZEN}")
    return winners, evidence


def sources(role: str) -> list[dict[str, str]]:
    rows = [row for row in read(AUDIT) if role in row["roles"].split(";")]
    if len(rows) != 20 or len({(row["value_head"], row["seed"]) for row in rows}) != 20:
        raise RuntimeError(f"{role}: expected 20 unique sources, got {len(rows)}")
    return rows


def submit(winners: dict[str, str]) -> None:
    tuning_jobs = {row["manifest_id"]: row["slurm_job_id"] for row in read(TUNING_LEDGER, "\t")}
    existing = set()
    if STAGE2_LEDGER.exists():
        existing = {row["manifest_id"] for row in read(STAGE2_LEDGER, "\t")}
    fields = ["manifest_id", "branch", "value_head", "seed", "anchor", "source_training_job",
              "source_epoch", "source_checkpoint", "reuse_tuning_job", "slurm_job_id", "submitted_at"]
    with STAGE2_LEDGER.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if stream.tell() == 0:
            writer.writeheader()
        for branch, role in (("validation_led", "selected"), ("terminal_led", "final")):
            for source in sorted(sources(role), key=lambda row: (row["value_head"], int(row["seed"]))):
                vh, seed, anchor = source["value_head"], source["seed"], winners[source["value_head"]]
                manifest_id = f"mprime-{branch}-{vh}-{seed}-a{anchor.replace('.', 'p')}-corrected"
                if manifest_id in existing:
                    continue
                if branch == "validation_led" and seed in TUNING_SEEDS:
                    tuning_id = f"mprime-corrected-anchor-{vh}-{seed}-a{anchor.replace('.', 'p')}"
                    job_id = tuning_jobs[tuning_id]
                    reused = job_id
                    print(f"REUSED|{manifest_id}|{job_id}")
                else:
                    command = [str(SUBMITTER), "--dom-mprime", "--original-only",
                        "--domain-architecture", "mcts", "--seed", seed, "--workers", "3",
                        "--jpddl-max-heap", "4g", "--time", "3-00:00:00", "--mem", "48G",
                        "--cpus", "6", "--train-from", source["checkpoint"], "--use-estimator", "0.5",
                        "--exploration-weight", "0.1", "--override-tree-sampling", "0",
                        "--mcts-expansion-size", "20", "--mcts-iterations", "0",
                        "--policy-anchor-kl-coeff", anchor, "--max-opt-epochs", "100",
                        "--supervised-lr", "0.0003", "--job-suffix",
                        f"MPEXT6{branch[0].upper()}A{anchor.replace('.', 'p')}_CORR_src{source['training_job']}",
                        "--output-subdir", f"mprime_{branch}_stage2"]
                    if vh == "off":
                        command.append("--vh-off")
                    env = os.environ.copy()
                    env["ENHSP_CONFIG_OVERRIDE"] = "hmrp-ha-gbfs"
                    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(result.stdout)
                    ids = JOB_RE.findall(result.stdout)
                    if len(ids) != 1:
                        raise RuntimeError(result.stdout)
                    job_id, reused = ids[0], ""
                    print(f"SUBMITTED|{manifest_id}|{job_id}")
                writer.writerow({"manifest_id": manifest_id, "branch": branch, "value_head": vh,
                    "seed": seed, "anchor": anchor, "source_training_job": source["training_job"],
                    "source_epoch": source["epoch"], "source_checkpoint": source["checkpoint"],
                    "reuse_tuning_job": reused, "slurm_job_id": job_id,
                    "submitted_at": datetime.now(timezone.utc).isoformat()})
                stream.flush()
                os.fsync(stream.fileno())
                existing.add(manifest_id)


def main() -> None:
    winners, _ = corrected_metrics()
    submit(winners)


if __name__ == "__main__":
    main()
