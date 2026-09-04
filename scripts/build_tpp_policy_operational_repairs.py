#!/usr/bin/env python3
"""Build exact TPP policy retries and the one-segment TPP/on override."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "experiment_tracking" / "four_domain_preservation"
SOURCE = TRACKING / "terminal_stage2_tpp_off_policy_retry_20260903.csv"
RETRY = TRACKING / "terminal_stage2_tpp_off_policy_retry2_20260904.csv"
OVERRIDE = TRACKING / "terminal_stage2_tpp_on_terminal_override_20260904.csv"
EXCLUDED = "ise-cpu128-03,ise-cpu-intl-13"

with SOURCE.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    rows = list(reader)
    fields = list(reader.fieldnames or [])
if len(rows) != 2:
    raise RuntimeError(f"expected two TPP/off retry rows, found {len(rows)}")
if "excluded_nodes" not in fields:
    fields.append("excluded_nodes")
for row in rows:
    row["manifest_id"] = row["manifest_id"].replace("-retry1", "-retry2")
    row["excluded_nodes"] = EXCLUDED
with RETRY.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)

override_fields = [
    "manifest_id", "snapshot_segments", "selected_epoch", "final_epoch",
    "training_logs", "reason",
]
override_row = {
    "manifest_id": "preserve3-term-tpp-on-1073581256-a10",
    "snapshot_segments": (
        "/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/experiment-results/"
        "experiments_numeric.domain.tpp-experiments_numeric.architecture_2.tpp_mcts-"
        "2026-08-31T19:46:57.380631/P[domain,pfile1,pfile2,pfile4]-"
        "S[0.0003,50,enhsp-hadd-astar]-MO[]-T[518400]-04921381-1247c09/snapshots@0"
    ),
    "selected_epoch": "6",
    "final_epoch": "83",
    "training_logs": (
        "/home/hersco/training_new_domains/2026-08-31/preserve3_terminal_tpp_stage2/"
        "20755755_Re-Tr_tpp_tpp_mcts_orig_vh_e.5_c.1_s1073581256_K0_"
        "P3TERMS2A10_src20523010.txt"
    ),
    "reason": (
        "Job reached its 72-hour allocation during epoch84 after writing snapshot83. "
        "The ordinary log epilogue was absent, so the original controller could not "
        "discover an otherwise valid one-segment checkpoint directory."
    ),
}
with OVERRIDE.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=override_fields, lineterminator="\n")
    writer.writeheader(); writer.writerow(override_row)
print(f"wrote {RETRY}")
print(f"wrote {OVERRIDE}")
