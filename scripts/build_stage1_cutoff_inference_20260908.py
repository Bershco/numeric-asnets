#!/usr/bin/env python3
"""Build preferred Stage-1 policy/MCTS inference at all declared cutoffs."""

from __future__ import annotations

import base64
import csv
import itertools
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "experiment_tracking"
SSH = [
    r"C:\Windows\System32\OpenSSH\ssh.exe", "-F",
    r"C:\Users\roeeh\.ssh\config", "uni-cluster",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def signflip(values: list[float]) -> float:
    observed = abs(statistics.mean(values))
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(values)):
        if abs(statistics.mean(v * s for v, s in zip(values, signs))) >= observed - 1e-12:
            extreme += 1
    return extreme / (2 ** len(values))


def ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2 or statistics.stdev(values) == 0:
        return mean, mean
    # Exact two-sided 95% Student-t criticals for the seed counts used here.
    critical = {8: 2.365, 9: 2.306, 10: 2.262}.get(len(values), 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half, mean + half


def holm(rows: list[dict[str, object]], cutoff: str) -> None:
    keyed = sorted(enumerate(rows), key=lambda item: float(item[1][f"raw_p_{cutoff}"]))
    running = 0.0
    total = len(keyed)
    for rank, (index, row) in enumerate(keyed):
        running = max(running, min(1.0, float(row[f"raw_p_{cutoff}"]) * (total - rank)))
        rows[index][f"holm_p_{cutoff}"] = running


def main() -> None:
    preferred = {
        ("block_grouping", "off"): (5, 20), ("block_grouping", "on"): (5, 20),
        ("counters", "off"): (5, 20), ("counters", "on"): (5, 20),
    }
    jobs = []
    for row in read(TRACKING / "stage1_mcts_results.csv"):
        key = (row["domain"], row["value_head"])
        wanted = preferred.get(key, (20, 70))
        if int(row["width"]) == wanted[0] and int(row["iterations"]) == wanted[1]:
            jobs.append(row)
    for row in read(
        TRACKING / "mcts_counters_width_sensitivity" / "stage1_narrow_terminal_results.csv"
    ):
        jobs.append({
            "domain": "counters", "value_head": row["value_head"], "seed": row["seed"],
            "width": "5", "iterations": "20", "mcts_job_id": row["job_id"],
            "source_evaluation_log": row["source_evaluation_log"],
            "completion_manifest_local": "",
        })
    paired_rows = [
        row for row in read(TRACKING / "mcts_paired_seed_results.csv")
        if row["experiment_id"] == "MAIN-VAL"
    ]
    policy = {
        (row["domain"], row["value_head"], row["seed"]): float(row["policy_score"])
        for row in paired_rows
    }
    policy_logs = {
        (row["domain"], row["value_head"], row["seed"]): row["policy_log"]
        for row in paired_rows
    }
    paths = [row["source_evaluation_log"] for row in jobs]
    remote = r'''
import glob,json,re
from pathlib import Path
paths=json.loads(%r)
pat=re.compile(r"\[EVAL INSTANCE\] completed number=(\d+) path=(\S+) status=(\S+) elapsed=([0-9.]+)s success=([0-9.]+) steps=(\d+)")
out={}
for raw in paths:
    records={}
    path=Path(raw)
    job_id=path.name.split("_",1)[0]
    ledger_paths = [
        path.parent/".resume_state"/f"{job_id}.eval_completed.jsonl",
        path.parent/"completion"/f"{job_id}.jsonl",
    ]
    ledger_paths.extend(Path(x) for x in glob.glob(
        f"/home/hersco/training_new_domains/2026-08-27/*/.resume_state/{job_id}.eval_completed.jsonl"
    ))
    for ledger in ledger_paths:
        if not ledger.is_file(): continue
        for line in ledger.open(errors="replace"):
            try: item=json.loads(line)
            except json.JSONDecodeError: continue
            number=item.get("instance_number",item.get("number"))
            if number is None: continue
            candidate=(1.0 if item.get("hit_goal",item.get("success",False)) else 0.0,
                       float(item.get("elapsed_seconds",item.get("elapsed",0.0)) or 0.0))
            prior=records.get(int(number))
            if prior is None or candidate[0]>prior[0] or (candidate[0]==prior[0] and candidate[1]<prior[1]):
                records[int(number)]=candidate
    if path.is_file():
        for line in path.open(errors="replace"):
            m=pat.search(line)
            if not m: continue
            number=int(m.group(1)); candidate=(float(m.group(5)),float(m.group(4)))
            prior=records.get(number)
            if prior is None or candidate[0]>prior[0] or (candidate[0]==prior[0] and candidate[1]<prior[1]):
                records[number]=candidate
    good=[elapsed for success,elapsed in records.values() if success==1.0]
    out[raw]={"recorded":len(records),"m30":sum(x<=1800 for x in good),"m2":sum(x<=7200 for x in good),"m6":sum(x<=21600 for x in good)}
print(json.dumps(out,separators=(",",":")))
''' % json.dumps(paths)
    encoded = base64.b64encode(remote.encode()).decode()
    proc = subprocess.run(SSH + [f'python3 -c "import base64;exec(base64.b64decode(\'{encoded}\'))"'],
                          check=True, text=True, capture_output=True)
    timing = json.loads(proc.stdout)
    seed_rows = []
    for job in jobs:
        key = (job["domain"], job["value_head"], job["seed"])
        item = timing[job["source_evaluation_log"]]
        seed_rows.append({
            "experiment_id": "MAIN-VAL", "domain": key[0], "value_head": key[1],
            "seed": key[2], "search": f"width{job['width']}_sim{job['iterations']}",
            "policy_score": policy[key], "mcts_30m": item["m30"], "mcts_2h": item["m2"],
            "mcts_6h": item["m6"], "mcts_job_id": job["mcts_job_id"],
            "source_policy_log": policy_logs[key],
            "source_mcts_log": job["source_evaluation_log"],
            "source_completion_ledger": job.get("completion_manifest_local", ""),
        })
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        groups[(row["domain"], row["value_head"])].append(row)
    summary = []
    for (domain, vh), rows in sorted(groups.items()):
        item: dict[str, object] = {
            "experiment_id": "MAIN-VAL", "domain": domain, "value_head": vh,
            "search": rows[0]["search"], "n": len(rows),
            "policy_mean": statistics.mean(float(r["policy_score"]) for r in rows),
        }
        for label in ("30m", "2h", "6h"):
            scores = [float(r[f"mcts_{label}"]) for r in rows]
            diffs = [score - float(row["policy_score"]) for score, row in zip(scores, rows)]
            low, high = ci(diffs)
            item.update({
                f"mcts_mean_{label}": statistics.mean(scores),
                f"change_{label}": statistics.mean(diffs),
                f"ci95_low_{label}": low, f"ci95_high_{label}": high,
                f"raw_p_{label}": signflip(diffs),
            })
        item["row_level_provenance"] = "experiment_tracking/stage1_policy_mcts_seed_cutoffs_latest.csv"
        summary.append(item)
    for label in ("30m", "2h", "6h"):
        holm(summary, label)
    for path, rows in (
        (TRACKING / "stage1_policy_mcts_seed_cutoffs_latest.csv", seed_rows),
        (TRACKING / "stage1_policy_mcts_all_cutoff_statistics_latest.csv", summary),
    ):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"seed_rows": len(seed_rows), "summary_rows": len(summary)}))


if __name__ == "__main__":
    main()
