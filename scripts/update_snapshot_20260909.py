#!/usr/bin/env python3
"""Publish the 9 September snapshot after adaptive-KL deployment."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "experiment_tracking"
STAMP = "2026-09-09T00:43:18+03:00"


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[:left] + replacement.rstrip() + "\n\n" + text[right:]


def number(value: str, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def cutoff_cell(row: dict[str, str], suffix: str) -> str:
    mean = number(row[f"mcts_mean_{suffix}"])
    change = float(row[f"change_{suffix}"])
    low = number(row[f"ci95_low_{suffix}"])
    high = number(row[f"ci95_high_{suffix}"])
    raw = number(row[f"raw_p_{suffix}"], 4)
    holm = number(row[f"holm_p_{suffix}_interim_complete_cells"], 4)
    return f"{mean}; {change:+.2f} [{low},{high}]; {raw}/{holm}"


queue = read_csv(TRACKING / "cluster_workload_latest.csv")
assert len(queue) == 59
assert all(row["state"] == "RUNNING" for row in queue)
running_cpus = sum(int(row["cpus"]) for row in queue)
running_gib = sum(float(row["memory_gib"]) for row in queue)
assert running_cpus == 248 and running_gib == 1776.0

groups: list[dict[str, str]] = []
for experiment in sorted({row["experiment"] for row in queue}):
    rows = [row for row in queue if row["experiment"] == experiment]
    groups.append(
        {
            "experiment": experiment,
            "running": str(len(rows)),
            "cpus": str(sum(int(row["cpus"]) for row in rows)),
            "memory_gib": str(int(sum(float(row["memory_gib"]) for row in rows))),
            "job_ids": ";".join(row["job_id"] for row in rows),
            "source": "experiment_tracking/cluster_workload_latest.csv",
        }
    )
write_csv(TRACKING / "live_experiment_status_latest.csv", groups)

registry_updates = {
    "ANCHOR-KL-CONTROL": {
        "role": "training-diagnostic",
        "status": "two-adaptive-jobs-running",
        "results_file": "experiment_tracking/anchor_kl_control_live_progress_20260909.csv",
        "held_reason": "",
        "next_action": (
            "Retries 21144388 and 21144389 are running after compute smoke 21144340 passed. "
            "Original submissions 21144210/21144211 failed before training because the clean "
            "worktree lacked ignored native operator _asnet_ops_impl.so; the isolated checkout "
            "now links the checksum-verified production build."
        ),
    },
    "MPRIME-VAL-ADEQUACY": {
        "status": "live-60-lineage-full-rescore-71.2-percent",
        "next_action": (
            "1608/2260 checkpoint-replicates complete at 2026-09-09 00:43 IDT; seven of60 "
            "lineages complete and52 tasks running. Continue idempotently after current allocations "
            "for any remaining tails; do not submit MPrime Stage2/MCTS before rankings are frozen."
        ),
    },
    "TPP-CATASTROPHIC-MCTS": {
        "status": "three-terminal-one-running-final-instance",
        "next_action": (
            "Fixed narrow4/20 OOM; PW20 10/20 OOM; PW70 7/20 OOM-terminal endpoint; fixed normal4/20 "
            "after19 classified with one exact instance active and at most six hours remaining."
        ),
    },
    "MCTS-PW70-CROSS-DOMAIN": {
        "status": "live-one-job-tail-48-of59-classified",
        "next_action": (
            "Counters Stage1/off seed2011206605 remains >=21/59 with48 classified and about13h "
            "of allocation left; a resumable tail is likely."
        ),
    },
    "MCTS-PW-COUNTERS-DIVERGENCE": {
        "status": "pw20-complete-pw70-two-terminal-one-live-52-of59",
        "next_action": (
            "Seed534933607 PW70 remains >=22/59 with52 classified and under six hours of allocation "
            "left; exact unfinished instances may require a bounded continuation."
        ),
    },
    "MCTS-LEGACY-FO": {
        "status": "live-one-job-final-instance",
        "next_action": (
            "Final terminal-led VH-on seed remains5/20 after19 durably classified; one exact "
            "instance is active and must classify within six hours."
        ),
    },
    "ROVER-MCTS-INTERRUP-REC": {
        "status": "27-of29-timeouts-final-two-active",
        "next_action": (
            "Twenty-seven recovered opportunities are confirmed six-hour timeouts. Job21107687_7 "
            "is running the final two independent instances; no aggregate score has changed yet."
        ),
    },
}
for name in ("experiments.csv", "experiment_registry.csv"):
    rows = read_csv(TRACKING / name)
    for row in rows:
        row.update(registry_updates.get(row["experiment_id"], {}))
    write_csv(TRACKING / name, rows)

manifest = read_csv(TRACKING / "anchor_kl_control_tpp_screen_20260908.csv")
retry_jobs = {"1972442430": "21144388", "1963100312": "21144389"}
for row in manifest:
    row["status"] = "running_retry1"
    row["slurm_job_id"] = retry_jobs[row["seed"]]
    row["output_log"] = row["output_log"].replace("%j", retry_jobs[row["seed"]]).replace(
        ".txt", "-retry1.txt"
    )
write_csv(TRACKING / "anchor_kl_control_tpp_screen_20260908.csv", manifest)

source = (TRACKING / "status_latest.md").read_text(encoding="utf-8")
source = re.sub(r"^# Complete experiment snapshot — .*?$", f"# Complete experiment snapshot — {STAMP}", source, count=1, flags=re.M)
source = source.replace(
    "This report joins a fresh Slurm query with bounded reads of every live MCTS log,",
    "This report joins the 00:43 IDT Slurm query with bounded reads of every live MCTS log,",
)

workload = f"""## Current workload

| State | Jobs | CPUs | Requested RAM |
|---|---:|---:|---:|
| Running | 59 | {running_cpus} | {running_gib:,.0f} GiB |
| Ordinary pending | 0 | 0 | 0 GiB |
| Slurm-held | 0 | 0 | 0 GiB |

| Live experiment | Running | CPU / RAM | Current evidence | Remaining bound / expectation |
|---|---:|---:|---|---|
| MPrime validation adequacy Phase B | 52 | 208 / 1,040 GiB | 1,608/2,260 checkpoint-replicates complete (71.2%); all 60 lineages represented, 7 complete | Current tasks have roughly 15–23h to allocation limits; aggregate throughput suggests another ~13h, but slow lineage tails may require another idempotent continuation |
| Adaptive KL-control TPP/off | 2 | 12 / 96 GiB | First adaptive checkpoints exist: outlier validation 22/30 with post-update KL .0484; stable control 30/30 with KL .0189; coefficient remains at floor 3 | Existing comparable runs took ~16h for the stable seed and ~30h for the outlier; 72h hard cap |
| TPP catastrophic-seed MCTS | 1 | 6 / 120 GiB | Fixed-normal 4/20 after 19 classified; policy 9/20 | One instance remains; <=6h |
| FO terminal-led Stage-2 MCTS tail | 1 | 6 / 120 GiB | 5/20 after 19 durably classified | One instance remains; <=6h |
| PW70 correction tail | 1 | 6 / 120 GiB | Counters S1/off seed 2011206605: >=21/59; 48 classified | ~13h allocation remains; exact continuation likely if the tail does not fit |
| Counters exact-snapshot PW70 recovery | 1 | 6 / 120 GiB | Seed 534933607: >=22/59; 52 classified | <6h allocation remains; exact continuation likely |
| Rover exact-instance recovery | 1 | 4 / 160 GiB | 27/29 classified, all as six-hour timeouts; final two active | <=4.3h from this snapshot |

There are no ordinary-pending or deliberately held Slurm jobs. The two adaptive
jobs are the only newly released work in this refresh. The current allocation is
well below the approximately 6 TiB effective user ceiling, but no other held
design has an equally unambiguous activation decision; those remain a user
priority choice rather than being silently released.
"""
source = replace_section(source, "## Current workload", "## Stage-1 validation-selected", workload)

comparison = {
    (row["stage2_branch"], row["domain"], row["value_head"]): row
    for row in read_csv(TRACKING / "stage2_policy_mcts_comparison_by_branch_latest.csv")
}
stats = read_csv(TRACKING / "stage2_policy_mcts_all_cutoff_statistics_latest.csv")
domain_order = {name: index for index, name in enumerate(
    ["block_grouping", "drone", "fo_counters", "rover", "counters"]
)}
stats.sort(key=lambda row: (domain_order[row["domain"]], row["stage2_branch"], row["value_head"]))
stage2_lines = [
    "## Stage-2 policy versus fixed MCTS",
    "",
    "Each cutoff cell is `MCTS mean; paired change [95% CI]; raw/Holm p`.",
    "Holm is computed across the 17 currently complete domain/VH/branch cells at",
    "each cutoff. It remains interim until FO/on terminal-led becomes the 18th complete cell.",
    "",
    "| Domain/VH | Branch | Search | n | Policy | 30 minutes | 2 hours | 6 hours | Visible conclusion |",
    "|---|---|---|---:|---:|---|---|---|---|",
]
labels = {"block_grouping": "BG", "drone": "Drone", "fo_counters": "FO", "rover": "Rover", "counters": "Counters"}
for row in stats:
    key = (row["stage2_branch"], row["domain"], row["value_head"])
    conclusion = comparison[key]["conclusion"]
    stage2_lines.append(
        "| " + " | ".join([
            f"{labels[row['domain']]}/{row['value_head']}", row["stage2_branch"].replace("_led", ""),
            row["search"], row["n"], number(row["policy_mean"]), cutoff_cell(row, "30m"),
            cutoff_cell(row, "2h"), cutoff_cell(row, "6h"), conclusion,
        ]) + " |"
    )
live = comparison[("terminal_led", "fo_counters", "on")]
stage2_lines.append(
    f"| FO/on | terminal | {live['search']} | {live['n_scheduler_terminal']} terminal + {live['n_live']} live | "
    f"{live['policy_mean']} | {live['mcts_30m']} lower bound | {live['mcts_2h']} lower bound | "
    f"{live['mcts_6h']} lower bound | {live['conclusion']} |"
)
stage2_lines.extend([
    "",
    "**Current conclusion:** after the single 17-cell family correction, Drone/on is",
    "significantly improved in both branches and FO/off is significantly improved in its",
    "terminal-led branch at all three cutoffs (Holm p=.0332). Counters/on terminal-led",
    "has a large positive mean but no longer survives this broader correction at six hours",
    "(Holm p=.1094). Block Grouping is negative, Rover is small/neutral, and FO/on remains live.",
])
source = replace_section(
    source, "## Stage-2 policy versus fixed MCTS", "## PW70 ten-seed FO/Rover expansion",
    "\n".join(stage2_lines),
)

source = source.replace("1,599 (70.8%)", "1,608 (71.2%)")
source = source.replace("1,599/2,260", "1,608/2,260")
source = source.replace("six lineages are complete", "seven lineages are complete")
source = source.replace("six lineages complete", "seven lineages complete")
source = source.replace("PW70 live at 50/59 classified", "PW70 live at 52/59 classified")
source = source.replace("Phase B is 70.8% complete", "Phase B is 71.2% complete")

adaptive = """### Adaptive KL-control screen

The adaptive-only screen is now live. Existing constant-anchor runs are reused,
so only two new training jobs are required:

| Role | Seed | Constant evidence | Adaptive job | Resources | State |
|---|---:|---|---:|---:|---|
| Catastrophic outlier | 1972442430 | Stage-2 epoch 0 fell from 20/20 to 10/20; selected endpoint 9/20 | 21144388 | 6 CPU / 48 GiB | Epoch 0 saved; validation 22/30; post-update KL .0484 |
| Stable control | 1963100312 | Stable constant-anchor training | 21144389 | 6 CPU / 48 GiB | Epoch 0 saved; validation 30/30; post-update KL .0189 |

The first submissions, `21144210` and `21144211`, failed before training after
about one minute because a fresh Git worktree does not contain the ignored
compiled TensorFlow operator `_asnet_ops_impl.so`. The isolated checkout now
links the checksum-verified production build. Strengthened compute smoke
`21144340` imported that operator, ran all five controller tests, verified both
CLI options and completed successfully. The retry ledger explicitly links each
new job to its failed precursor, source Stage-1 training job, checkpoint and log.

The decision criterion remains: adaptive control should prevent or materially
reduce the outlier's first-update collapse without damaging the stable seed.
The current jobs are training-only; matched policy evaluation is materialized
after their checkpoints exist.

The first realized KL values are below the target band, so the controller made
no upward adjustment; because coefficient 3 is the declared floor, it also did
not reduce protection. These are validation diagnostics, not test-policy scores.
"""
source = replace_section(source, "### Adaptive KL-control screen", "## Best demonstrated result by domain", adaptive)

held = """## Design-held experiments in priority order

| Priority | Experiment | Activation condition |
|---:|---|---|
| 1 | MCTS-PW-30M | Optional fresh whole-job efficiency validation; post-hoc cutoff coverage is already defensible |
| 2 | MCTS-PW-PATHBATCH | Freeze nonstandard multiple-expansion update semantics |
| 3 | MCTS-RESOURCE | Use only for unresolved OOM/lifecycle endpoints |
| 4 | ACT-HISTORY-ABLATION | Separate fresh Stage-1/Stage-2 campaign on Drone, Counters and TPP |
| 5 | LONG-DRONE selected MCTS | Reprioritize the remaining side question |
| 6 | STOP-ORIG | Finalize compatibility manifest |
| 7 | PUCT-EST | Run only after higher-priority evidence closes |
| 8 | MCTS-SAFE2 | Require nonzero cutoffs and demonstrated cross-horizon statistic contamination |

`ANCHOR-KL-CONTROL` is no longer held; it is the two-job live experiment above.
"""
source = replace_section(source, "## Design-held experiments in priority order", "## Provenance contract", held)
source = source.replace(
    "- Best demonstrated domain configurations: `best_configuration_by_domain_latest.csv`.",
    "- Best demonstrated domain configurations: `best_configuration_by_domain_latest.csv`.\n"
    "- Adaptive KL frozen rows: `anchor_kl_control_tpp_screen_20260908.csv`.\n"
    "- Adaptive KL initial and retry job/log lineage: `anchor_kl_control_submissions_20260909.tsv` and `anchor_kl_control_retry_submissions_20260909.tsv`.",
)

preserve_replacements = {
    "| Domain/VH | S1 selected | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Conclusion |":
        "| Domain/VH | S1 selected | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Raw p | Conclusion |",
    "|---|---:|---:|---:|---:|---|---|": "|---|---:|---:|---:|---:|---|---:|---|",
    "| Delivery/off | 19.8 | 19.6 | 19.50 | 20.0 | -.2 [-1.01,.61] | Essentially preserved |":
        "| Delivery/off | 19.8 | 19.6 | 19.50 | 20.0 | -.2 [-1.01,.61] | 1.0 | Essentially preserved |",
    "| Delivery/on | 19.2 | 19.6 | 19.50 | 20.0 | +.4 [-.37,1.17] | Preserved |":
        "| Delivery/on | 19.2 | 19.6 | 19.50 | 20.0 | +.4 [-.37,1.17] | .5 | Preserved |",
    "| TPP/off | 20.0 | 18.9* | 18.625* | 20.0 | -1.1 [-3.59,1.39] | Not uniform |":
        "| TPP/off | 20.0 | 18.9* | 18.625* | 20.0 | -1.1 [-3.59,1.39] | 1.0 | Not uniform |",
    "| TPP/on | 20.0 | 19.5 | 19.375 | 20.0 | -.5 [-1.63,.63] | Essentially preserved |":
        "| TPP/on | 20.0 | 19.5 | 19.375 | 20.0 | -.5 [-1.63,.63] | 1.0 | Essentially preserved |",
    "| Zenotravel/off | 20.0 | 20.0 | 20.0 | 20.0 | 0 [0,0] | Preserved |":
        "| Zenotravel/off | 20.0 | 20.0 | 20.0 | 20.0 | 0 [0,0] | 1.0 | Preserved |",
    "| Zenotravel/on | 20.0 | 19.9 | 19.875 | 20.0 | -.1 [-.33,.13] | Preserved |":
        "| Zenotravel/on | 20.0 | 19.9 | 19.875 | 20.0 | -.1 [-.33,.13] | 1.0 | Preserved |",
    "| Domain/VH | S1 final | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Conclusion |":
        "| Domain/VH | S1 final | S2 all 10 | Held-out 8 | Tuning 2 | Change [95% CI] | Raw p | Conclusion |",
    "| Delivery/off | 14.1 | 15.4 | 14.50 | 19.0 | +1.3 [-2.52,5.12] | Preserved relative to source |":
        "| Delivery/off | 14.1 | 15.4 | 14.50 | 19.0 | +1.3 [-2.52,5.12] | .496 | Preserved relative to source |",
    "| Delivery/on | 16.7 | 18.4 | 18.625 | 17.5 | +1.7 [-.91,4.31] | Positive mean |":
        "| Delivery/on | 16.7 | 18.4 | 18.625 | 17.5 | +1.7 [-.91,4.31] | .188 | Positive mean |",
    "| TPP/off | 17.1 | 17.5 | 16.875 | 20.0 | +.4 [-2.13,2.93] | Preserved on average |":
        "| TPP/off | 17.1 | 17.5 | 16.875 | 20.0 | +.4 [-2.13,2.93] | 1.0 | Preserved on average |",
    "| TPP/on | 18.9 | 18.3 | 17.875 | 20.0 | -.6 [-3.77,2.57] | No reliable change |":
        "| TPP/on | 18.9 | 18.3 | 17.875 | 20.0 | -.6 [-3.77,2.57] | 1.0 | No reliable change |",
    "| Zenotravel/off | 20.0 | 19.8 | 19.875 | 19.5 | -.2 [-.50,.10] | Essentially preserved |":
        "| Zenotravel/off | 20.0 | 19.8 | 19.875 | 19.5 | -.2 [-.50,.10] | .5 | Essentially preserved |",
    "| Zenotravel/on | 19.8 | 20.0 | 20.0 | 20.0 | +.2 [-.10,.50] | Preserved |":
        "| Zenotravel/on | 19.8 | 20.0 | 20.0 | 20.0 | +.2 [-.10,.50] | .5 | Preserved |",
}
for old, new in preserve_replacements.items():
    source = source.replace(old, new)
source = source.replace(
    "The complete branch-aware table is above. Current firm claims are: strong\n"
    "Drone/on, FO/off and Counters/on gains; terminal-led BG loss; small Rover\n"
    "gains; and one still-live FO/on cell. MPrime enters only after Phase B freezes\n"
    "checkpoints.",
    "The branch-aware all-cutoff table above is authoritative. Under one interim Holm\n"
    "family across the 17 complete cells, Drone/on (both branches) and FO/off terminal-led\n"
    "remain significant at six hours. Counters/on terminal-led is a large positive mean\n"
    "that does not survive the broader family correction; terminal-led BG is negative,\n"
    "Rover is small/neutral, and FO/on remains live. MPrime enters only after Phase B\n"
    "freezes defensible checkpoints.",
)

(TRACKING / "status_latest.md").write_text(source, encoding="utf-8")
(TRACKING / "status_20260909_latest.md").write_text(source, encoding="utf-8")

provenance = read_csv(TRACKING / "snapshot_provenance_index_latest.csv")
provenance = [row for row in provenance if row["artifact"] not in {
    "status_20260909_latest.md", "anchor_kl_control_submissions_20260909.tsv",
    "anchor_kl_control_retry_submissions_20260909.tsv", "anchor_kl_control_live_progress_20260909.csv",
}]
provenance.extend([
    {
        "artifact": "status_20260909_latest.md",
        "scope": "Complete narrative snapshot",
        "authoritative_as_of": STAMP,
        "row_level_source": "All rows below",
        "notes": "Current workload live experiments completed RQs held designs and best-domain table",
    },
    {
        "artifact": "anchor_kl_control_submissions_20260909.tsv",
        "scope": "Adaptive KL initial submissions",
        "authoritative_as_of": STAMP,
        "row_level_source": "Direct Slurm job ids source checkpoints and stdout paths",
        "notes": "Both initial jobs failed before training due to missing ignored native operator in clean worktree",
    },
    {
        "artifact": "anchor_kl_control_retry_submissions_20260909.tsv",
        "scope": "Adaptive KL corrected retries",
        "authoritative_as_of": STAMP,
        "row_level_source": "retry_of;job_id;source_training_job_id;source_checkpoint;output_log",
        "notes": "Both corrected jobs running after strengthened compute smoke21144340",
    },
    {
        "artifact": "anchor_kl_control_live_progress_20260909.csv",
        "scope": "Adaptive KL first-update progress",
        "authoritative_as_of": "2026-09-09T00:49:30+03:00",
        "row_level_source": "Direct source_training_log and Slurm job id",
        "notes": "Validation and KL diagnostics only; no test-policy result yet",
    },
])
write_csv(TRACKING / "snapshot_provenance_index_latest.csv", provenance)

print(f"published status_latest.md and status_20260909_latest.md at {STAMP}")
