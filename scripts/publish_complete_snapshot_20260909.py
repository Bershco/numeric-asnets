#!/usr/bin/env python3
"""Publish the current 9 September RQ-first snapshot and provenance indexes."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
STAMP = "2026-09-09T10:46:40+03:00"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def update_registry() -> None:
    updates = {
        "ANCHOR-KL-CONTROL": {
            "status": "two-adaptive-jobs-running-controller-noop-so-far",
            "results_file": "experiment_tracking/anchor_kl_control_progress_latest.csv",
            "next_action": (
                "Jobs21144388/21144389 reached Stage2 epochs34/52. Coefficient remains3 in every epoch "
                "and controller adjustments remain zero; evaluate matched test checkpoints before any efficacy claim."
            ),
        },
        "MPRIME-VAL-ADEQUACY": {
            "status": "live-60-lineage-full-rescore-89.1-percent",
            "next_action": (
                "2014/2260 checkpoint-replicates complete;25/60 lineages complete and33 tasks active. "
                "Finish or resume only missing points, compare replicate rankings, then freeze checkpoints/anchors."
            ),
        },
        "TPP-CATASTROPHIC-MCTS": {
            "status": "completed-four-search-arms",
            "next_action": (
                "Policy9/20; fixed narrow4; PW20 5/10/10; fixed normal4; PW70 4/5/7 at30m/2h/6h. "
                "Only PW20 recovers one net policy success."
            ),
        },
        "MCTS-PW70-CROSS-DOMAIN": {
            "status": "live-one-job-tail-52-of59-classified",
            "next_action": (
                "Only Counters Stage1/off seed2011206605 remains active: >=21/59 after52 classified; "
                "allocation hard bound approximately3.4h."
            ),
        },
        "MCTS-PW-COUNTERS-DIVERGENCE": {
            "status": "completed-six-job-focused-screen",
            "next_action": (
                "All PW20/PW70 exact-snapshot arms terminal. Neither widening mode fully recovers the policy; "
                "PW20 seed923500475 gives the strongest partial recovery at48/59."
            ),
        },
        "MCTS-LEGACY-FO": {
            "status": "live-one-job-final-instance",
            "next_action": (
                "Final terminal-led VH-on seed remains5/20 after19 classified and14 explicit timeouts; "
                "one exact instance is active with approximately2.5h allocation bound."
            ),
        },
        "ROVER-MCTS-INTERRUP-REC": {
            "status": "completed-29-of29-opportunities-timeout",
            "next_action": "All29 formerly unclassified instances reached the declared six-hour timeout; Rover aggregates are unchanged.",
        },
    }
    for filename in ("experiments.csv", "experiment_registry.csv"):
        rows = read_csv(TRACK / filename)
        for row in rows:
            row.update(updates.get(row["experiment_id"], {}))
        write_csv(TRACK / filename, rows)


def workload_tables() -> tuple[str, list[dict[str, object]]]:
    queue = read_csv(TRACK / "cluster_workload_latest.csv")
    by_state: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_exp: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in queue:
        by_state[row["state"]].append(row)
        by_exp[row["experiment"]].append(row)
    lines = [
        "## Current workload",
        "",
        "| State | Jobs | CPUs | Requested RAM |",
        "|---|---:|---:|---:|",
    ]
    for state, label in (("RUNNING", "Running"), ("PENDING", "Ordinary pending"), ("HELD", "Slurm-held")):
        rows = by_state.get(state, [])
        lines.append(
            f"| {label} | {len(rows)} | {sum(int(r['cpus']) for r in rows)} | "
            f"{sum(float(r['memory_gib']) for r in rows):,.0f} GiB |"
        )
    details: list[dict[str, object]] = []
    for exp, rows in sorted(by_exp.items()):
        details.append({
            "experiment": exp,
            "state": rows[0]["state"],
            "jobs": len(rows),
            "cpus": sum(int(r["cpus"]) for r in rows),
            "memory_gib": int(sum(float(r["memory_gib"]) for r in rows)),
            "job_ids": ";".join(r["job_id"] for r in rows),
            "row_level_provenance": "experiment_tracking/cluster_workload_latest.csv",
        })
    write_csv(TRACK / "live_experiment_status_latest.csv", details)
    lines += [
        "",
        "| Live experiment | Jobs | CPU / RAM | Latest evidence | Realistic timing |",
        "|---|---:|---:|---|---|",
        "| MPrime validation adequacy Phase B | 33 | 132 / 660 GiB | 2,014/2,260 checkpoint-replicates (89.1%); 25/60 lineages complete | Aggregate throughput suggests ~6h; older allocations have under6h hard bounds, so a small idempotent tail may still be required |",
        "| Adaptive KL-control TPP/off | 2 | 12 / 96 GiB | Outlier through epoch37; stable control through epoch53; zero coefficient changes | Stable control ~8h to epoch100; outlier ~17h at current rates; both may stop earlier |",
        "| FO Counters terminal-led S2 MCTS tail | 1 | 6 / 120 GiB | 5/20, 19/20 classified, 14 explicit timeouts | One instance; <=2.5h allocation bound |",
        "| PW70 cross-domain correction tail | 1 | 6 / 120 GiB | Counters S1/off seed2011206605: >=21/59, 52 classified | <=3.4h allocation bound |",
        "",
        "There are no ordinary-pending or Slurm-held jobs. Design-held experiments appear later and do not occupy the scheduler.",
    ]
    return "\n".join(lines), details


def adaptive_summary() -> list[dict[str, object]]:
    rows = read_csv(TRACK / "anchor_kl_control_progress_latest.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["job_id"]].append(row)
    output = []
    for job, group in sorted(grouped.items()):
        group.sort(key=lambda row: int(row["stage2_epoch"]))
        output.append({
            "job_id": job,
            "role": group[0]["role"],
            "seed": group[0]["seed"],
            "epochs_recorded": len(group),
            "latest_stage2_epoch": group[-1]["stage2_epoch"],
            "validation_first": f"{float(group[0]['validation_successes']):.0f}/{float(group[0]['validation_total']):.0f}",
            "validation_latest": f"{float(group[-1]['validation_successes']):.0f}/{float(group[-1]['validation_total']):.0f}",
            "validation_min": int(min(float(r["validation_successes"]) for r in group)),
            "validation_max": int(max(float(r["validation_successes"]) for r in group)),
            "max_post_update_kl": f"{max(float(r['policy_anchor_kl_post_update']) for r in group):.6f}",
            "coefficient_values": ";".join(sorted({r["policy_anchor_kl_coeff_after"] for r in group})),
            "controller_adjustments": int(max(float(r["policy_anchor_kl_controller_adjustments"]) for r in group)),
            "target_kl": group[0]["policy_anchor_kl_target"],
            "source_log": group[0]["path"],
        })
    write_csv(TRACK / "anchor_kl_control_summary_latest.csv", output)
    return output


def experiment_catalog(live: list[dict[str, object]]) -> list[dict[str, object]]:
    live_map = {row["experiment"]: row for row in live}
    id_to_live = {
        "ANCHOR-KL-CONTROL": "Adaptive KL-control TPP/off screen",
        "MPRIME-VAL-ADEQUACY": "MPrime validation adequacy Phase B full rescore",
        "MCTS-LEGACY-FO": "FO Counters terminal-led Stage-2 MCTS",
        "MCTS-PW70-CROSS-DOMAIN": "PW70 cross-domain correction",
    }
    rows = []
    for row in read_csv(TRACK / "experiments.csv"):
        lrow = live_map.get(id_to_live.get(row["experiment_id"], ""), {})
        status = row["status"]
        if lrow:
            category = "live"
        elif "held" in status:
            category = "held_design"
        else:
            category = "completed_or_inactive"
        rows.append({
            "experiment_id": row["experiment_id"],
            "display_name": row["display_name"],
            "category": category,
            "status": status,
            "purpose": row["primary_question"],
            "configuration": row["configuration_summary"],
            "results_file": row["results_file"],
            "manifest_path": row["manifest_path"],
            "live_jobs": lrow.get("jobs", 0),
            "live_cpu": lrow.get("cpus", 0),
            "live_memory_gib": lrow.get("memory_gib", 0),
            "live_job_ids": lrow.get("job_ids", ""),
            "next_action_or_conclusion": row["next_action"],
        })
    write_csv(TRACK / "experiment_catalog_latest.csv", rows)
    return rows


def update_best() -> None:
    rows = read_csv(TRACK / "best_configuration_by_domain_latest.csv")
    for row in rows:
        if row["domain"] == "mprime":
            row["status"] = "provisional_phase_b_2014_of_2260"
            row["conclusion"] = (
                "Current policy result remains provisional; Phase B is89.1% complete and may change checkpoint/anchor selection"
            )
    write_csv(TRACK / "best_configuration_by_domain_latest.csv", rows)


def write_rq_index() -> None:
    rows = [
        {
            "research_question": "RQ1",
            "question": "Does MCTS-guided Stage-2 training improve VH-off policy coverage?",
            "status": "complete_original_five_domain_family",
            "headline": "No domain survives five-domain Holm correction; Counters has a large but noisy terminal-led mean.",
            "detail_file": "experiment_tracking/experiment_statistics.csv",
            "row_level_provenance": "experiment_tracking/policy_paired_seed_results.csv",
        },
        {
            "research_question": "RQ2",
            "question": "Does inference-time MCTS improve coverage?",
            "status": "stage1_complete_stage2_one_fo_on_tail",
            "headline": "Drone/on and FO Counters show robust gains; Block Grouping and Counters/off show that MCTS is not uniformly safe.",
            "detail_file": "experiment_tracking/stage1_policy_mcts_all_cutoff_statistics_latest.csv;experiment_tracking/stage2_policy_mcts_all_cutoff_statistics_latest.csv",
            "row_level_provenance": "experiment_tracking/stage1_policy_mcts_seed_cutoffs_latest.csv;experiment_tracking/stage2_policy_mcts_seed_cutoffs_latest.csv",
        },
        {
            "research_question": "RQ3",
            "question": "Does the value head improve Stage-2 refinement?",
            "status": "complete_original_five_domain_family",
            "headline": "Only terminal-led Block Grouping is significant after correction, and the effect is harmful.",
            "detail_file": "experiment_tracking/experiment_statistics.csv",
            "row_level_provenance": "experiment_tracking/policy_paired_seed_results.csv",
        },
        {
            "research_question": "RQ4",
            "question": "Does the value head alter the benefit of inference-time MCTS?",
            "status": "stage2_one_fo_on_tail_mprime_pending_phase_b",
            "headline": "The clearest interaction is the large Drone/on MCTS gain; MPrime enters only after Phase B freezes defensible checkpoints.",
            "detail_file": "experiment_tracking/stage2_policy_mcts_all_cutoff_statistics_latest.csv",
            "row_level_provenance": "experiment_tracking/stage2_policy_mcts_seed_cutoffs_latest.csv",
        },
    ]
    write_csv(TRACK / "rq_results_latest.csv", rows)


def extract(text: str, start: str, end: str | None) -> str:
    left = text.index(start)
    right = len(text) if end is None else text.index(end, left)
    return text[left:right].strip()


update_registry()
workload, live = workload_tables()
adaptive = adaptive_summary()
catalog = experiment_catalog(live)
update_best()
write_rq_index()

source = (TRACK / "status_20260909_latest.md").read_text(encoding="utf-8")
stage1 = extract(source, "## Stage-1 validation-selected", "## Stage-2 policy").replace("## ", "#### ", 1)
stage2 = extract(source, "## Stage-2 policy", "### RQ3").replace("## ", "#### ", 1)
pw = extract(source, "## PW70 ten-seed", "## Completed preservation").replace("## ", "### ", 1)
best = extract(source, "## Best demonstrated", "## Provenance contract").replace("## ", "### ", 1)
best = best.replace("Phase B is 71.2% complete", "Phase B is 89.1% complete")
best = best.replace("Phase B is 88.3% complete", "Phase B is 89.1% complete")
rq1 = extract(source, "### RQ1", "### RQ2")
rq3 = extract(source, "### RQ3", "### RQ4")
preserve = extract(source, "## Completed preservation", "## Other completed").replace("## ", "### ", 1).replace("### PRESERVE", "#### PRESERVE")
completed = extract(source, "## Other completed", "## Design-held").replace("## ", "### ", 1)
held = extract(source, "## Design-held", "## Complete experiment catalog").replace("## ", "### ", 1)
provenance = extract(source, "## Provenance contract", None)

adaptive_text = f"""#### ANCHOR-KL-CONTROL — adaptive KL screen

Purpose: test whether controlling actual policy drift prevents TPP/off seed
1972442430 from collapsing after the first Stage-2 update without damaging a
stable control seed.

| Role | Job | Epochs recorded | Validation first -> latest (range) | Max post-update KL | Coefficient | Adjustments |
|---|---:|---:|---|---:|---:|---:|
| Catastrophic outlier | {adaptive[0]['job_id']} | {adaptive[0]['epochs_recorded']} | {adaptive[0]['validation_first']} -> {adaptive[0]['validation_latest']} ({adaptive[0]['validation_min']}–{adaptive[0]['validation_max']}/30) | {adaptive[0]['max_post_update_kl']} | {adaptive[0]['coefficient_values']} | {adaptive[0]['controller_adjustments']} |
| Stable control | {adaptive[1]['job_id']} | {adaptive[1]['epochs_recorded']} | {adaptive[1]['validation_first']} -> {adaptive[1]['validation_latest']} ({adaptive[1]['validation_min']}–{adaptive[1]['validation_max']}/30) | {adaptive[1]['max_post_update_kl']} | {adaptive[1]['coefficient_values']} | {adaptive[1]['controller_adjustments']} |

The intervention is currently a **controller no-op**: coefficient 3 has never
changed. The upper trigger is approximately .17145 (target .1143 x tolerance
1.5), while the largest post-update KL observed is below .090. Values below the
lower band cannot reduce the coefficient because 3 is the configured floor.
The live runs therefore add diagnostics but have not yet tested a different
training objective from the constant-anchor baseline. Test-policy scores are
still required; validation alone cannot establish repair.
"""

live_experiments = """### Live and recently completed experiment updates

#### MPRIME-VAL-ADEQUACY Phase B

Two independently frozen harder validation replicates are being evaluated on
all 1,130 saved checkpoints: **2,014/2,260 results (89.1%)**, all 60 lineages
represented, 25 lineages complete, 33 jobs active. No new training is involved.
The next decision is whether both replicates agree on checkpoint and anchor
rankings. Only then should MPrime Stage-2 and its eventual 40 MCTS comparisons
be released.

""" + adaptive_text + """
#### Remaining MCTS tails

| Experiment | Current result | What remains | Consequence |
|---|---|---|---|
| FO terminal-led S2/on | >=5/20 at every cutoff, 19 classified | One instance, <=2.5h allocation bound | Final eighteenth Stage-2 cell and final Holm family update |
| PW70 correction, Counters S1/off seed2011206605 | >=21/59, 52 classified | Seven classifications, <=3.4h allocation bound | Finishes the corrected PW70 narrow-domain screen |

#### Diagnostics completed since the prior report

| Experiment | 30m / 2h / 6h | Conclusion |
|---|---|---|
| TPP catastrophic seed: policy | 9 / 9 / 9 | Collapsed Stage-2 policy baseline |
| Fixed narrow 5/20 | 4 / 4 / 4 | Worse than policy; OOM after18 classified |
| PW20 | 5 / 10 / 10 | Only arm to improve, by one plan at2h/6h; OOM after19 classified |
| Fixed normal 20/70 | 4 / 4 / 4 | Complete20-instance run; no recovery |
| PW70 | 4 / 5 / 7 | OOM after12 classified; no recovery |
| Rover interrupted-instance recovery | 0 / 0 / 0 new plans | All29 exact opportunities became six-hour timeouts; aggregate Rover scores unchanged |
| Counters exact-snapshot widening | see focused ledger | Neither PW20 nor PW70 restores all policy successes; PW20 seed923500475 reaches48/59 and is the strongest partial recovery |
"""

learning = """## Learning curves — locally cached and immediately re-plottable

The source rows, aggregate rows, figures and direct training/evaluation-log
mapping are frozen under `experiment_tracking/learning_curves/latest/`.

| View | File | Use |
|---|---|---|
| RQ1 + RQ3 combined | `by_rq_rq1_rq3.png` / `.svg` | Advisor overview; five domains x two RQs |
| RQ1 only | `by_rq_rq1.png` / `.svg` | Stage-2 policy refinement without VH interaction |
| RQ3 only | `by_rq_rq3.png` / `.svg` | Value-head comparison |
| Per domain | `by_domain_<domain>.svg` | One-domain discussion or slide |
| Tidy source | `rq1_rq3_policy_curve_source.csv` | Fast restyling/replotting without cluster access |
| Aggregate source | `rq1_rq3_five_domain_learning_curve_aggregates.csv` | Mean/median/best/min-max curves |
| Provenance | `learning_curve_provenance.csv` | Direct training job, evaluation job and log paths |

The figures support the numerical conclusions below: Stage-2 does not provide a
consistent policy-only gain across domains; Block Grouping shows the clearest
negative VH interaction, while Counters is high-variance and should never be
summarized only by its mean.
"""

catalog_lines = [
    "### Complete experiment catalog",
    "",
    "The row-level catalog is `experiment_catalog_latest.csv`. The table below keeps every registered experiment visible while using the detailed sections above for scores.",
    "",
    "| Experiment | State | What it tested | Result / next decision |",
    "|---|---|---|---|",
]
for row in catalog:
    state = row["category"].replace("_", " ")
    purpose = row["purpose"].replace("|", "/")
    result = row["next_action_or_conclusion"].replace("|", "/")
    catalog_lines.append(f"| {row['experiment_id']} | {state} | {purpose} | {result} |")

report = "\n\n".join([
    f"# Complete RQ and experiment snapshot — {STAMP}",
    "This report uses the fresh live scheduler capture only for jobs that were live in the prior report. Completed results come from locally cached authoritative ledgers. `>=` always denotes an active-run lower bound.",
    workload,
    "## Research questions",
    "### RQ1 — does MCTS-guided Stage-2 training improve VH-off policy?\n\n" + rq1.split("\n", 1)[1],
    "### RQ2 — does inference-time MCTS improve coverage?\n\nThe complete Stage-1 and branch-aware Stage-2 tables below answer this question at 30m, 2h and 6h. The p-values are paired exact sign-flip tests; Holm correction is recomputed separately at each cutoff.\n\n" + stage1 + "\n\n" + stage2,
    "### RQ3 — does the value head improve Stage-2 refinement?\n\n" + rq3.split("\n", 1)[1],
    "### RQ4 — does the value head alter the benefit of inference-time MCTS?\n\nThe Stage-2 table under RQ2 is also the authoritative RQ4 evidence. The strongest reproducible pattern is Drone/on: large Holm-significant gains in both validation- and terminal-led branches. FO/off is also significant in the terminal-led branch. Block Grouping is negative, Rover is small/neutral, Counters is variable, FO/on is awaiting one final instance, and MPrime awaits Phase B checkpoint freezing.",
    learning,
    "## Experiments",
    live_experiments,
    pw,
    preserve,
    completed,
    held,
    "\n".join(catalog_lines),
    best,
    provenance,
]) + "\n"

for target in (TRACK / "status_latest.md", TRACK / "status_20260909_latest.md", TRACK / "advisor_report_20260909_current.md"):
    target.write_text(report, encoding="utf-8")

index = read_csv(TRACK / "snapshot_provenance_index_latest.csv")
new_artifacts = [
    ("advisor_report_20260909_current.md", "Complete RQ-first report", "All referenced local authoritative ledgers"),
    ("experiment_catalog_latest.csv", "All registered experiments", "experiments.csv plus cluster_workload_latest.csv"),
    ("anchor_kl_control_summary_latest.csv", "Adaptive KL job summary", "anchor_kl_control_progress_latest.csv and direct source logs"),
    ("learning_curves/latest/rq1_rq3_policy_curve_source.csv", "Replottable learning-curve source", "Direct training/evaluation job and log columns"),
    ("learning_curves/latest/learning_curve_provenance.csv", "Learning-curve provenance", "Direct training/evaluation jobs and logs"),
    ("rq_results_latest.csv", "RQ-level result index", "Direct detail and row-level provenance files"),
]
known = {row["artifact"] for row in index}
for artifact, scope, row_source in new_artifacts:
    if artifact not in known:
        index.append({
            "artifact": artifact,
            "scope": scope,
            "authoritative_as_of": STAMP,
            "row_level_source": row_source,
            "notes": "Locally cached; no cluster query required for re-rendering",
        })
for row in index:
    if row["artifact"] in {item[0] for item in new_artifacts}:
        row["authoritative_as_of"] = STAMP
write_csv(TRACK / "snapshot_provenance_index_latest.csv", index)

print(f"published complete RQ-first snapshot at {STAMP}")
