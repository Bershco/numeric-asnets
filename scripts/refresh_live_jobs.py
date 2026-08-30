#!/usr/bin/env python3
"""Refresh the replaceable local Slurm live-job snapshot."""
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiment_tracking" / "live_jobs.csv"
# Use the single documented Windows-profile alias. It pins the intended
# cluster identity file and avoids the recurring wrong-profile/host-key issue.
HOST = "uni-cluster"


def classify(name: str, reason: str) -> str:
    if name == "MPRIME_ANCHOR_POLICY_REFRESH":
        return "MPRIME-ANCHOR-POLICY-CONTROLLER"
    if name == "MPRIME_FINALIZE_STAGE2":
        return "MPRIME-STAGE2-FINALIZER"
    if name == "P4_DELIVERY_S2_POLICY_REFRESH":
        return "PRESERVE-DELIVERY-S2-POLICY-CONTROLLER"
    if name == "P4_TPP_S2_POLICY_REFRESH":
        return "PRESERVE-TPP-S2-POLICY-CONTROLLER"
    if name.startswith("context-full-"):
        return "MCTS-SAFE-CONTEXT"
    if name.startswith("horizon-drone-") or name == "HORIZON_POSTHOC_VAL":
        return "MCTS-HORIZON-BINDING"
    if name.startswith("horizon-counters-"):
        return "MCTS-HORIZON-COUNTERS"
    if name.startswith("pw-kmin3-"):
        return "MCTS-PW-CROSS-DOMAIN"
    if name.startswith("pw70-kmin3-"):
        return "MCTS-PW70-CROSS-DOMAIN"
    if "P3TERMS2" in name:
        return "PRESERVE-3-TERM"
    if "MPEXT6V" in name:
        return "MAIN-EXT6-MPRIME"
    if "MPEXT6T" in name:
        return "MAIN-TERM-EXT6-MPRIME"
    if "MPCAT" in name:
        return "MPRIME-ANCHOR"
    if "MPATPOL" in name:
        return "MPRIME-ANCHOR-POLICY"
    if name.startswith("Re-Tr_delivery_"):
        return "PRESERVE-DELIVERY-S2"
    if name.startswith("Ev_delivery_"):
        return "PRESERVE-DELIVERY-S2-POLICY"
    if name.startswith("Re-Tr_tpp_"):
        return "PRESERVE-TPP-S2"
    if name.startswith("Ev_tpp_"):
        return "PRESERVE-TPP-S2-POLICY"
    if name.startswith("Ev_counters_") and "SR10M" in name:
        return "MCTS-WIDTH-COUNTERS-S2"
    if name.startswith("Ev_drone_") and "SR10M" in name:
        return "MAIN-VAL-S2-MCTS"
    if name.startswith("Ev_rover_"):
        return "MCTS-RESOURCE-ROVER"
    if name.startswith("Ev_fo_counters_"):
        return "MCTS-RESOURCE-FO-HELD"
    if name.startswith("HORIZON_"):
        return "MCTS-HORIZON"
    if name.startswith("SAFE3_"):
        return "MCTS-SAFE"
    if name.startswith("MPRIME_VAL_V1_RESCORE"):
        return "MPRIME-VAL"
    if name.startswith("MPRIME_POLICY_FINALIZER"):
        return "MPRIME-VAL"
    if name.startswith("STORAGE_POST_COMPACT"):
        return "STORAGE-AUDIT"
    if name.startswith("ANCHOR_DOMAIN_FINALIZER"):
        return "ANCHOR-4"
    if name.startswith("P4_ZENO_S2_POLICY_CTRL"):
        return "PRESERVE-4"
    if "MPRIME_VAL_V1_S1" in name:
        return "MPRIME-VAL"
    if "P4S2" in name:
        return "PRESERVE-4"
    if name.startswith("pw-safe-drone-"):
        return "MCTS-PW-SAFE-SENSITIVITY"
    if "pw-drone-" in name:
        return "MCTS-PW"
    if "block_grouping" in name and "mcts" in name:
        return "MCTS-WIDTH"
    if "CD4AT" in name:
        return "ANCHOR-4"
    if reason == "JobHeldUser" and "mcts" in name:
        return "MCTS-HELD"
    if name.startswith("Ev_") and "mcts" in name:
        return "MAIN-VAL"
    if name.startswith("Ev_"):
        return "POLICY-PIPELINE"
    if name.startswith(("Tr_", "Re-Tr_")):
        return "TRAINING-PIPELINE"
    return "OTHER"


def main() -> None:
    command = (
        "date -Is; squeue -h -u hersco "
        "-o '%i|%T|%r|%M|%l|%C|%m|%j'"
    )
    result = subprocess.run(["ssh", HOST, command], check=True, text=True,
                            stdout=subprocess.PIPE).stdout.splitlines()
    timestamp = result[0]
    rows = []
    for line in result[1:]:
        parts = line.split("|", 7)
        if len(parts) != 8:
            continue
        job, state, reason, elapsed, limit, cpus, memory, name = parts
        rows.append({
            "snapshot_time": timestamp,
            "experiment_id": classify(name, reason),
            "job_id": job,
            "state": state,
            "reason": reason,
            "elapsed": elapsed,
            "time_limit": limit,
            "cpus": cpus,
            "memory": memory,
            "job_name": name,
        })
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"snapshot={timestamp} jobs={len(rows)}")


if __name__ == "__main__":
    main()
