#!/usr/bin/env python3
"""Build branch-aware Stage-2 policy/MCTS inference at all time cutoffs."""

from __future__ import annotations

import base64
import csv
import itertools
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "experiment_tracking"
SSH = [r"C:\Windows\System32\OpenSSH\ssh.exe", "-F", r"C:\Users\roeeh\.ssh\config", "uni-cluster"]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def signflip(values: list[float]) -> float:
    observed = abs(statistics.mean(values))
    return sum(abs(statistics.mean(v * s for v, s in zip(values, signs))) >= observed - 1e-12
               for signs in itertools.product((-1, 1), repeat=len(values))) / 2 ** len(values)


def ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    half = 2.262 * statistics.stdev(values) / math.sqrt(10) if statistics.stdev(values) else 0.0
    return mean - half, mean + half


def key(exp: str, domain: str, vh: str, seed: str) -> tuple[str, str, str, str]:
    return exp, domain, vh, seed


def main() -> None:
    policy_rows = read(T / "policy_paired_seed_results.csv")
    policy = {key(r["experiment_id"], r["domain"], r["value_head"], r["seed"]): r for r in policy_rows}
    rows: dict[tuple[str, str, str, str], dict[str, object]] = {}

    # Generic historical rows used for terminal BG/Rover and both Drone branches.
    generic_cells = {
        ("MAIN-TERM", "block_grouping"), ("MAIN-TERM", "rover"),
        ("MAIN-VAL", "drone"), ("MAIN-TERM", "drone"),
    }
    historical = [r for r in read(T / "stage2_mcts_historical_log_audit_20260902.csv")
                  if (r["experiment_id"], r["domain"]) in generic_cells]
    for r in historical:
        if (r["experiment_id"], r["domain"], r["value_head"], r["seed"]) == (
            "MAIN-TERM", "block_grouping", "on", "2082152039"
        ):
            r["candidate_mcts_logs"] = (
                "/home/hersco/training_new_domains/2026-09-03/"
                "statistical_replication_stage2_mcts_eval/"
                "20891018_Ev_block_grouping_block_grouping_mcts_orig_vh_"
                "e.5_c.1_s2082152039_K0_SR10M_src20489404_e0062.txt"
            )
    paths = [r["candidate_mcts_logs"].split(";")[0] for r in historical]
    remote = r'''
import glob,json,re
from pathlib import Path
paths=json.loads(%r)
pat=re.compile(r"\[EVAL INSTANCE\] completed number=(\d+) path=(\S+) status=(\S+) elapsed=([0-9.]+)s success=([0-9.]+) steps=(\d+)")
out={}
for raw in paths:
 records={}; path=Path(raw); m=re.match(r"(\d+)",path.name); jid=m.group(1) if m else ""
 ledgers=[]
 for pattern in (f"/home/hersco/training_new_domains/*/*/.resume_state/{jid}.eval_completed.jsonl",f"/home/hersco/training_new_domains/*/*/completion/{jid}.jsonl"):
  ledgers += [Path(x) for x in glob.glob(pattern)]
 for ledger in ledgers:
  for line in ledger.open(errors="replace"):
   try: item=json.loads(line)
   except json.JSONDecodeError: continue
   num=item.get("instance_number",item.get("number"))
   if num is None: continue
   cand=(1.0 if item.get("hit_goal",item.get("success",False)) else 0.0,float(item.get("elapsed_seconds",item.get("elapsed",0)) or 0))
   old=records.get(int(num))
   if old is None or cand[0]>old[0] or (cand[0]==old[0] and cand[1]<old[1]): records[int(num)]=cand
 if path.is_file():
  for line in path.open(errors="replace"):
   hit=pat.search(line)
   if not hit: continue
   num=int(hit.group(1)); cand=(float(hit.group(5)),float(hit.group(4))); old=records.get(num)
   if old is None or cand[0]>old[0] or (cand[0]==old[0] and cand[1]<old[1]): records[num]=cand
 good=[t for success,t in records.values() if success==1]
 out[raw]={"classified":len(records),"m30":sum(t<=1800 for t in good),"m2":sum(t<=7200 for t in good),"m6":sum(t<=21600 for t in good),"ledgers":";".join(map(str,ledgers))}
print(json.dumps(out,separators=(",",":")))
''' % json.dumps(paths)
    encoded = base64.b64encode(remote.encode()).decode()
    timing = json.loads(subprocess.run(SSH + [f'python3 -c "import base64;exec(base64.b64decode(\'{encoded}\'))"'],
                                       check=True, text=True, capture_output=True).stdout)
    for r in historical:
        k = key(r["experiment_id"], r["domain"], r["value_head"], r["seed"])
        path = r["candidate_mcts_logs"].split(";")[0]; t = timing[path]; p = policy[k]
        rows[k] = {"experiment_id": k[0], "stage2_branch": "validation_led" if k[0] == "MAIN-VAL" else "terminal_led",
                   "domain": k[1], "value_head": k[2], "seed": k[3], "search": "narrow5/20" if k[1] == "block_grouping" else "normal20/70",
                   "policy_score": p["after_score"], "mcts_30m": t["m30"], "mcts_2h": t["m2"], "mcts_6h": t["m6"],
                   "classified_instances": t["classified"], "evidence_status": "complete_declared_budget",
                   "mcts_job_id": re.match(r"(\d+)", Path(path).name).group(1), "source_policy_log": p["after_log"],
                   "source_mcts_log": path, "source_completion_ledger": t["ledgers"]}

    def add_special(exp: str, domain: str, records: list[dict[str, str]], fields: tuple[str, str, str],
                    vh_field: str, seed_field: str, job_field: str, log_field: str, search: str) -> None:
        for r in records:
            if vh_field == "job_name":
                name = r["job_name"]
                vh = "off" if "orig_novh" in name else "on"
                seed = re.search(r"_s(\d+)_", name).group(1)
            else:
                vh, seed = r[vh_field], r[seed_field]
            k = key(exp, domain, vh, seed); p = policy[k]
            rows[k] = {"experiment_id": exp, "stage2_branch": "validation_led" if exp == "MAIN-VAL" else "terminal_led",
                       "domain": domain, "value_head": k[2], "seed": k[3], "search": search,
                       "policy_score": p["after_score"], "mcts_30m": r[fields[0]], "mcts_2h": r[fields[1]], "mcts_6h": r[fields[2]],
                       "classified_instances": r.get("classified_instances", r.get("recorded_instances", "")),
                       "evidence_status": "complete_declared_budget", "mcts_job_id": r[job_field],
                       "source_policy_log": p["after_log"], "source_mcts_log": r[log_field],
                       "source_completion_ledger": r.get("completion_record_path", r.get("completion_record", ""))}

    add_special("MAIN-VAL", "block_grouping", read(T / "bg_validation_mcts_provenance_20260905.csv"),
                ("success_30m", "success_2h", "success_6h"), "job_name", "job_name", "job_id", "log_path", "narrow5/20")

    add_special("MAIN-VAL", "rover", read(T / "rover_validation_seed_results_20260906.csv"),
                ("mcts_30m", "mcts_2h", "mcts_6h"), "value_head", "seed", "job_id", "mcts_log", "normal20/70")
    add_special("MAIN-VAL", "counters", read(T / "mcts_counters_width_sensitivity" / "stage2_narrow_matched_10seed.csv"),
                ("narrow_30m", "narrow_2h", "narrow_6h"), "value_head", "seed", "job_id", "source_log", "narrow5/20")
    add_special("MAIN-TERM", "counters", read(T / "counters_terminal_stage2_narrow_seed_results_latest.csv"),
                ("mcts_30m", "mcts_2h", "mcts_6h"), "value_head", "seed", "mcts_job_id", "source_mcts_log", "narrow5/20")
    add_special("MAIN-TERM", "fo_counters", read(T / "fo_terminal_mcts_provenance_20260905.csv"),
                ("success_30m", "success_2h", "success_6h"), "job_name", "job_name", "job_id", "log_path", "normal20/70")

    seed_rows = sorted(rows.values(), key=lambda r: (r["stage2_branch"], r["domain"], r["value_head"], int(r["seed"])))
    complete_cells = {
        (r["stage2_branch"], r["domain"], r["value_head"])
        for r in read(T / "stage2_policy_mcts_comparison_by_branch_latest.csv")
        if r["n_complete_fixed_budget"] == "10"
    }
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows: groups[(row["stage2_branch"], row["domain"], row["value_head"])].append(row)
    summary = []
    for (branch, domain, vh), cell in sorted(groups.items()):
        if len(cell) != 10 or (branch, domain, vh) not in complete_cells: continue
        out: dict[str, object] = {"stage2_branch": branch, "domain": domain, "value_head": vh,
                                  "search": cell[0]["search"], "n": 10,
                                  "policy_mean": statistics.mean(float(r["policy_score"]) for r in cell)}
        for cutoff in ("30m", "2h", "6h"):
            vals = [float(r[f"mcts_{cutoff}"]) for r in cell]; diffs = [v - float(r["policy_score"]) for v, r in zip(vals, cell)]
            lo, hi = ci(diffs); out.update({f"mcts_mean_{cutoff}": statistics.mean(vals), f"change_{cutoff}": statistics.mean(diffs),
                                             f"ci95_low_{cutoff}": lo, f"ci95_high_{cutoff}": hi, f"raw_p_{cutoff}": signflip(diffs)})
        out["row_level_provenance"] = "experiment_tracking/stage2_policy_mcts_seed_cutoffs_latest.csv"; summary.append(out)
    for cutoff in ("30m", "2h", "6h"):
        ordered = sorted(enumerate(summary), key=lambda x: float(x[1][f"raw_p_{cutoff}"])); running = 0.0
        for rank, (idx, row) in enumerate(ordered):
            running = max(running, min(1.0, float(row[f"raw_p_{cutoff}"]) * (len(ordered) - rank)))
            summary[idx][f"holm_p_{cutoff}_interim_complete_cells"] = running
    for path, data in ((T / "stage2_policy_mcts_seed_cutoffs_latest.csv", seed_rows), (T / "stage2_policy_mcts_all_cutoff_statistics_latest.csv", summary)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    print(json.dumps({"seed_rows": len(seed_rows), "complete_cells": len(summary)}))


if __name__ == "__main__": main()
