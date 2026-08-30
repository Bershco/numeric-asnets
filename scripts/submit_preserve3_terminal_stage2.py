#!/usr/bin/env python3
"""Submit the true terminal-led Stage-2 branch for the three stable domains."""

from __future__ import annotations

import csv, os, re, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/hersco/training_new_domains")
REPO = Path("/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context")
RESULTS = REPO / "experiment_tracking/experiment_results.csv"
MANIFEST = REPO / "experiment_tracking/four_domain_preservation/terminal_stage2_manifest.csv"
LEDGER = REPO / "experiment_tracking/four_domain_preservation/terminal_stage2_submissions.tsv"
SUBMITTER = ROOT / "submit_training.sh"
ANCHORS = {("delivery", "off"): "30", ("delivery", "on"): "30",
           ("tpp", "off"): "3", ("tpp", "on"): "10",
           ("zenotravel", "off"): "30", ("zenotravel", "on"): "0.3"}
TEACHERS = {"delivery": "hadd-astar", "tpp": "hadd-astar", "zenotravel": "hadd-gbfs"}
JOB_RE = re.compile(r"\[OK \] job=\s*(\d+)")

def read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))

def build_manifest() -> list[dict[str, str]]:
    rows = [r for r in read(RESULTS) if r["experiment_id"] == "PRESERVE-4"
            and r["task_type"] == "policy_eval" and r["stage"] == "stage1"
            and r["endpoint"] == "final" and r["domain"] in TEACHERS]
    if len(rows) != 60 or len({(r["domain"], r["value_head"], r["seed"]) for r in rows}) != 60:
        raise RuntimeError(f"expected 60 unique final endpoints, got {len(rows)}")
    out = []
    for r in sorted(rows, key=lambda x: (x["domain"], x["value_head"], int(x["seed"]))):
        anchor = ANCHORS[(r["domain"], r["value_head"])]
        out.append({"manifest_id": f"preserve3-term-{r['domain']}-{r['value_head']}-{r['seed']}-a{anchor.replace('.', 'p')}",
                    "experiment_id": "PRESERVE-3-TERM", "domain": r["domain"],
                    "value_head": r["value_head"], "seed": r["seed"], "anchor": anchor,
                    "source_training_job_id": r["source_training_job_id"],
                    "source_epoch": r["epoch"], "source_checkpoint": r["checkpoint"],
                    "status": "ready", "teacher": TEACHERS[r["domain"]]})
    fields = list(out[0])
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(out)
    return out

def main() -> None:
    rows = build_manifest()
    existing = set()
    if LEDGER.exists(): existing = {r["manifest_id"] for r in read(LEDGER, "\t")}
    fields = list(rows[0]) + ["slurm_job_id", "submitted_at"]
    with LEDGER.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if stream.tell() == 0: writer.writeheader()
        for row in rows:
            if row["manifest_id"] in existing: continue
            command = [str(SUBMITTER), f"--dom-{row['domain']}", "--original-only",
                "--domain-architecture", "mcts", "--seed", row["seed"], "--workers", "3",
                "--jpddl-max-heap", "4g", "--time", "3-00:00:00", "--mem", "48G",
                "--cpus", "6", "--train-from", row["source_checkpoint"], "--use-estimator", "0.5",
                "--exploration-weight", "0.1", "--override-tree-sampling", "0",
                "--mcts-expansion-size", "20", "--mcts-iterations", "0",
                "--policy-anchor-kl-coeff", row["anchor"], "--max-opt-epochs", "100",
                "--supervised-lr", "0.0003", "--job-suffix",
                f"P3TERMS2A{row['anchor'].replace('.', 'p')}_src{row['source_training_job_id']}",
                "--output-subdir", f"preserve3_terminal_{row['domain']}_stage2"]
            if row["value_head"] == "off": command.append("--vh-off")
            env = os.environ.copy(); env["ENHSP_CONFIG_OVERRIDE"] = row["teacher"]
            result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(result.stdout)
            ids = JOB_RE.findall(result.stdout)
            if len(ids) != 1: raise RuntimeError(result.stdout)
            record = dict(row); record.update(slurm_job_id=ids[0], submitted_at=datetime.now(timezone.utc).isoformat())
            writer.writerow(record); stream.flush(); os.fsync(stream.fileno())
            print(f"SUBMITTED|{row['manifest_id']}|{ids[0]}")

if __name__ == "__main__": main()
