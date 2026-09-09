#!/usr/bin/env python3
"""Cache compact epoch-level adaptive-KL progress through canonical SSH."""

from __future__ import annotations

import base64
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiment_tracking" / "anchor_kl_control_progress_latest.csv"
SSH = [
    r"C:\Windows\System32\OpenSSH\ssh.exe",
    "-F",
    r"C:\Users\roeeh\.ssh\config",
    "uni-cluster",
]
JOBS = [
    {
        "role": "catastrophic_outlier",
        "seed": "1972442430",
        "job_id": "21144388",
        "source_epoch": 15,
        "constant_job_id": "20684881",
        "path": "/home/hersco/training_new_domains/2026-09-08/anchor_kl_control/21144388_anchor-kl-tpp-off-bad-adaptive-retry1.txt",
    },
    {
        "role": "stable_control",
        "seed": "1963100312",
        "job_id": "21144389",
        "source_epoch": 3,
        "constant_job_id": "20553944",
        "path": "/home/hersco/training_new_domains/2026-09-08/anchor_kl_control/21144389_anchor-kl-tpp-off-control-adaptive-retry1.txt",
    },
]


def main() -> None:
    remote = r'''
import json, re
from pathlib import Path

jobs = json.loads(%r)
scalar = re.compile(r"/train/(policy_anchor_kl_(?:loss|post_update|coeff_before|coeff_after|controller_adjustments|target))\s*:\s*([-+0-9.eE]+)")
final = re.compile(r"\[EVAL FINAL\]\s+success=([0-9.]+)/([0-9.]+)")
rows = []
for job in jobs:
    current = {}
    epoch_number = 0
    with Path(job["path"]).open(errors="replace") as stream:
        for line in stream:
            match = scalar.search(line)
            if match:
                current[match.group(1)] = float(match.group(2))
            match = final.search(line)
            if match and "policy_anchor_kl_post_update" in current:
                rows.append({
                    **job,
                    "stage2_epoch": epoch_number,
                    "cumulative_epoch": job["source_epoch"] + epoch_number,
                    "validation_successes": float(match.group(1)),
                    "validation_total": float(match.group(2)),
                    **current,
                })
                current = {}
                epoch_number += 1
print(json.dumps(rows, separators=(",", ":")))
''' % json.dumps(JOBS)
    payload = base64.b64encode(remote.encode()).decode()
    command = f"python3 -c \"import base64;exec(base64.b64decode('{payload}'))\""
    result = subprocess.run(SSH + [command], check=True, text=True, capture_output=True)
    rows = json.loads(result.stdout)
    fields = [
        "role", "seed", "job_id", "constant_job_id", "stage2_epoch",
        "cumulative_epoch", "validation_successes", "validation_total",
        "policy_anchor_kl_loss", "policy_anchor_kl_post_update",
        "policy_anchor_kl_coeff_before", "policy_anchor_kl_coeff_after",
        "policy_anchor_kl_controller_adjustments", "policy_anchor_kl_target",
        "path",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(OUT)}))


if __name__ == "__main__":
    main()
