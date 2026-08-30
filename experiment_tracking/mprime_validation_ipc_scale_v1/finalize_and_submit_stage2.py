#!/usr/bin/env python3
"""Freeze MPrime anchors and submit validation-led plus terminal-led Stage 2."""

from __future__ import annotations

import csv, os, re, subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/hersco/training_new_domains")
MANIFEST = HERE / "anchor_tuning_manifest.csv"
LEDGER = HERE / "anchor_tuning_submissions.tsv"
AUDIT = HERE / "validation_test_checkpoint_audit.csv"
FROZEN = HERE / "anchor_selection_frozen.csv"
STAGE2_LEDGER = HERE / "stage2_submissions.tsv"
SUBMITTER = ROOT / "submit_training.sh"
TUNING_SEEDS = {"1963100312", "2011206605"}
RATE_RE = re.compile(r"\[VALIDATION\] Current network validation success rate: ([0-9.]+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
JOB_RE = re.compile(r"\[OK \] job=\s*(\d+)")

def read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))

def accounting(ids: list[str]) -> dict[str, tuple[str, Path]]:
    text = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", ",".join(ids),
                                    "-o", "JobIDRaw,State,StdOut%1000"], text=True)
    out = {}
    for line in text.splitlines():
        job, state, stdout = line.split("|", 2)
        if job.isdigit(): out[job] = (state, Path(stdout.replace("%j", job)))
    return out

def freeze() -> dict[str, str]:
    specs = read(MANIFEST); jobs = {r["manifest_id"]: r["slurm_job_id"] for r in read(LEDGER, "\t")}
    if len(specs) != 28 or len(jobs) != 28: raise RuntimeError("MPrime tuning ledger is not complete")
    info = accounting(list(jobs.values())); grouped = defaultdict(list); evidence = []
    for row in specs:
        job = jobs[row["manifest_id"]]; state, log = info[job]
        if state in {"RUNNING", "PENDING"}: raise RuntimeError(f"{job} still {state}")
        text = ANSI_RE.sub("", log.read_text(encoding="utf-8", errors="replace"))
        marker = "[VALIDATION STATE] Starting a new trainer phase"
        if marker not in text: raise RuntimeError(f"missing phase marker: {log}")
        rates = [float(x) for x in RATE_RE.findall(text.rsplit(marker, 1)[1])]
        if not rates: raise RuntimeError(f"no validation rates: {log}")
        metrics = (sum(rates) / len(rates), max(rates), rates[-1])
        grouped[(row["value_head"], row["anchor"])].append(metrics)
        evidence.append({"manifest_id": row["manifest_id"], "job_id": job, "state": state,
                         "value_head": row["value_head"], "seed": row["seed"], "anchor": row["anchor"],
                         "validation_points": len(rates), "auc": metrics[0], "peak": metrics[1],
                         "final": metrics[2], "source_log": str(log)})
    winners = {}
    for vh in ("off", "on"):
        ranked = []
        for (candidate_vh, anchor), values in grouped.items():
            if candidate_vh != vh: continue
            if len(values) != 2: raise RuntimeError(f"{vh}/{anchor}: expected two tuning seeds")
            means = tuple(sum(v[i] for v in values) / 2 for i in range(3))
            ranked.append((means, -float(anchor), anchor))
        if len(ranked) != 7: raise RuntimeError(f"{vh}: expected seven coefficients")
        winners[vh] = max(ranked)[2]
    with FROZEN.open("w", newline="", encoding="utf-8") as stream:
        fields = list(evidence[0]); writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(evidence)
    print(f"FROZEN|off={winners['off']}|on={winners['on']}|evidence={FROZEN}")
    return winners

def sources(role: str) -> list[dict[str, str]]:
    rows = read(AUDIT)
    out = [r for r in rows if role in r["roles"].split(";")]
    if len(out) != 20 or len({(r["value_head"], r["seed"]) for r in out}) != 20:
        raise RuntimeError(f"{role}: expected 20 sources, got {len(out)}")
    return out

def submit(winners: dict[str, str]) -> None:
    tuning = read(MANIFEST); tuning_jobs = {r["manifest_id"]: r["slurm_job_id"] for r in read(LEDGER, "\t")}
    existing = set()
    if STAGE2_LEDGER.exists(): existing = {r["manifest_id"] for r in read(STAGE2_LEDGER, "\t")}
    fields = ["manifest_id", "branch", "value_head", "seed", "anchor", "source_training_job",
              "source_epoch", "source_checkpoint", "reuse_tuning_job", "slurm_job_id", "submitted_at"]
    with STAGE2_LEDGER.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if stream.tell() == 0: writer.writeheader()
        for branch, role in (("validation_led", "selected"), ("terminal_led", "final")):
            for src in sorted(sources(role), key=lambda r: (r["value_head"], int(r["seed"]))):
                vh, seed, anchor = src["value_head"], src["seed"], winners[src["value_head"]]
                mid = f"mprime-{branch}-{vh}-{seed}-a{anchor.replace('.', 'p')}"
                if mid in existing: continue
                if branch == "validation_led" and seed in TUNING_SEEDS:
                    tune_id = f"mprime-corrected-anchor-{vh}-{seed}-a{anchor.replace('.', 'p')}"
                    record = {"manifest_id": mid, "branch": branch, "value_head": vh, "seed": seed,
                              "anchor": anchor, "source_training_job": src["training_job"],
                              "source_epoch": src["epoch"], "source_checkpoint": src["checkpoint"],
                              "reuse_tuning_job": tuning_jobs[tune_id], "slurm_job_id": tuning_jobs[tune_id],
                              "submitted_at": datetime.now(timezone.utc).isoformat()}
                    writer.writerow(record); stream.flush(); existing.add(mid)
                    print(f"REUSED|{mid}|{tuning_jobs[tune_id]}"); continue
                command = [str(SUBMITTER), "--dom-mprime", "--original-only", "--domain-architecture", "mcts",
                    "--seed", seed, "--workers", "3", "--jpddl-max-heap", "4g", "--time", "3-00:00:00",
                    "--mem", "48G", "--cpus", "6", "--train-from", src["checkpoint"], "--use-estimator", "0.5",
                    "--exploration-weight", "0.1", "--override-tree-sampling", "0", "--mcts-expansion-size", "20",
                    "--mcts-iterations", "0", "--policy-anchor-kl-coeff", anchor, "--max-opt-epochs", "100",
                    "--supervised-lr", "0.0003", "--job-suffix",
                    f"MPEXT6{branch[0].upper()}A{anchor.replace('.', 'p')}_src{src['training_job']}",
                    "--output-subdir", f"mprime_{branch}_stage2"]
                if vh == "off": command.append("--vh-off")
                env = os.environ.copy(); env["ENHSP_CONFIG_OVERRIDE"] = "hmrp-ha-gbfs"
                result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if result.returncode: raise RuntimeError(result.stdout)
                ids = JOB_RE.findall(result.stdout)
                if len(ids) != 1: raise RuntimeError(result.stdout)
                record = {"manifest_id": mid, "branch": branch, "value_head": vh, "seed": seed,
                          "anchor": anchor, "source_training_job": src["training_job"],
                          "source_epoch": src["epoch"], "source_checkpoint": src["checkpoint"],
                          "reuse_tuning_job": "", "slurm_job_id": ids[0],
                          "submitted_at": datetime.now(timezone.utc).isoformat()}
                writer.writerow(record); stream.flush(); os.fsync(stream.fileno()); existing.add(mid)
                print(f"SUBMITTED|{mid}|{ids[0]}")

def main() -> None:
    winners = freeze(); submit(winners)

if __name__ == "__main__": main()
