#!/usr/bin/env python3
"""Build a durable, RQ-aligned advisor audit from authoritative experiment ledgers."""

from __future__ import annotations

import csv
import itertools
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
OUT = TRACK / "advisor_audit_20260830"
T_CRIT_95 = 2.2621571628540993  # df=9


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        x = values[0] if values else math.nan
        return x, x
    margin = T_CRIT_95 * stdev(values) / math.sqrt(len(values))
    return mean(values) - margin, mean(values) + margin


def signflip_p(values: list[float]) -> float:
    if not values:
        return math.nan
    observed = abs(mean(values))
    extreme = 0
    total = 1 << len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(mean([v * s for v, s in zip(values, signs)]))
        if permuted + 1e-12 >= observed:
            extreme += 1
    return extreme / total


def holm(raw: list[float]) -> list[float]:
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    adjusted = [1.0] * len(raw)
    running = 0.0
    m = len(raw)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * raw[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def grouped_stats(group_id: str, estimand: str, rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        values = row.pop("_values")
        low, high = ci95(values)
        result.append(
            {
                "rq_group": group_id,
                "estimand": estimand,
                **row,
                "n": len(values),
                "mean_difference": round(mean(values), 6),
                "ci95_low": round(low, 6),
                "ci95_high": round(high, 6),
                "raw_signflip_p": round(signflip_p(values), 8),
            }
        )
    adjusted = holm([float(row["raw_signflip_p"]) for row in result])
    for row, adj in zip(result, adjusted):
        row["holm_p_within_rq"] = round(adj, 8)
        row["significant_raw_0_05"] = float(row["raw_signflip_p"]) < 0.05
        row["significant_holm_0_05"] = adj < 0.05
    return result


def build_rq_results() -> list[dict]:
    policy = read_csv(TRACK / "policy_paired_seed_results.csv")
    mcts = read_csv(TRACK / "mcts_paired_seed_results.csv")
    domains = ["block_grouping", "drone", "fo_counters", "rover", "counters"]
    pmap = defaultdict(dict)
    for row in policy:
        if row["experiment_id"] in {"MAIN-VAL", "MAIN-TERM"}:
            pmap[(row["experiment_id"], row["domain"], row["value_head"])][row["seed"]] = row

    output: list[dict] = []
    for experiment_id, label in (
        ("MAIN-VAL", "validation-selected S1 to validation-selected S2"),
        ("MAIN-TERM", "terminal S1 to validation-selected S2"),
    ):
        rq1_rows = []
        rq3_rows = []
        for domain in domains:
            off = pmap[(experiment_id, domain, "off")]
            on = pmap[(experiment_id, domain, "on")]
            shared = sorted(set(off) & set(on), key=int)
            off_values = [float(off[s]["difference"]) for s in shared]
            did_values = [float(on[s]["difference"]) - float(off[s]["difference"]) for s in shared]
            rq1_rows.append({"experiment_id": experiment_id, "domain": domain, "value_head": "off", "_values": off_values})
            rq3_rows.append({"experiment_id": experiment_id, "domain": domain, "value_head": "on-minus-off", "_values": did_values})
        output += grouped_stats(f"RQ1-{experiment_id}", label, rq1_rows)
        output += grouped_stats(
            f"RQ3-{experiment_id}",
            f"difference-in-differences: VH-on change minus VH-off change ({label})",
            rq3_rows,
        )

    mmap = defaultdict(dict)
    for row in mcts:
        if row["experiment_id"] == "MAIN-VAL" and row["comparison"] == "S1 policy -> S1 MCTS":
            mmap[(row["domain"], row["value_head"])][row["seed"]] = row
    for vh, rq in (("off", "RQ2-STAGE1-SECONDARY"), ("on", "RQ4-STAGE1-SECONDARY")):
        rows = []
        for domain in domains:
            seed_rows = mmap[(domain, vh)]
            values = [float(seed_rows[s]["difference"]) for s in sorted(seed_rows, key=int)]
            rows.append({"experiment_id": "MAIN-VAL", "domain": domain, "value_head": vh, "_values": values})
        output += grouped_stats(rq, "Stage-1 validation-selected policy to MCTS", rows)
    return output


def memory_gib(text: str) -> float:
    text = text.strip().upper()
    if text.endswith("G"):
        return float(text[:-1])
    if text.endswith("M"):
        return float(text[:-1]) / 1024.0
    return 0.0


def build_workload() -> list[dict]:
    jobs = read_csv(TRACK / "live_jobs.csv")
    grouped: dict[tuple[str, str, str], dict] = {}
    for row in jobs:
        hold = "held" if row["reason"] == "JobHeldUser" else "ordinary"
        key = (row["experiment_id"], row["state"], hold)
        item = grouped.setdefault(
            key,
            {
                "snapshot_time": row["snapshot_time"],
                "experiment_id": row["experiment_id"],
                "state": row["state"],
                "queue_class": hold,
                "jobs": 0,
                "cpus": 0,
                "requested_ram_gib": 0.0,
                "dominant_reason": row["reason"],
            },
        )
        item["jobs"] += 1
        item["cpus"] += int(row["cpus"])
        item["requested_ram_gib"] += memory_gib(row["memory"])
    return sorted(grouped.values(), key=lambda x: (x["state"], x["experiment_id"]))


STORY_HOLES = [
    ("STAT-RQ", "all mainstream", "Existing Holm correction pooled ten VH/domain tests and RQ3 difference-in-differences was absent.", "Recompute exact sign-flip tests with Holm over five domains within each RQ; report RQ3 as paired difference-in-differences.", "analysis-only", "closed by rq_results.csv"),
    ("SIX-DOMAIN-FAMILY", "MAIN-EXT6-MPRIME", "MPrime joined the imperfect-domain group only after the corrected validation audit, so its Stage-2 effect is absent from the original five-domain confirmatory family.", "Preserve the preregistered five-domain Holm results and add a separately labelled six-domain extension after MPrime Stage-2 confirmation; existing per-domain CIs/raw p-values do not change, but six-family Holm p-values may stay equal or increase.", "training/evaluation plus analysis", "dependency-live"),
    ("RQ2-PRIMARY", "MAIN-VAL-S2-MCTS", "The paper-plan primary inference comparison is Stage-2 policy versus MCTS, but most present evidence is Stage-1 secondary analysis.", "Finish the submitted Drone gap, then release remaining domain-matched Stage-2 endpoint MCTS in a resource-prioritized order.", "new compute", "open/live"),
    ("TERM-FIDELITY", "MAIN-TERM", "Terminal checkpoints reuse runs whose stopping was still influenced by validation; this is not strict original-paper early stopping.", "Run STOP-ORIG with training-success stopping and terminal selection, first as a three-seed pilot.", "new training", "held"),
    ("MPRIME-RANK", "MPRIME-VAL", "Corrected validation removes epoch-1 bias but has weak checkpoint-ranking agreement with test coverage.", "Report rank agreement and regret; consider a larger preregistered validation ensemble without using test scores to tune it.", "analysis plus possible policy eval", "partly closed"),
    ("RESOURCE-CENSOR", "MCTS-RESOURCE", "OOM/time-limited MCTS rows are conservative lower bounds and confound algorithm with worker concurrency/memory.", "Deploy lifecycle-safe two-worker/160-GiB resumptions and report both fixed-budget lower bounds and resource sensitivity.", "new compute", "held"),
    ("SEARCH-CONFIG", "MCTS-WIDTH", "Search width differs by domain; a single cross-domain MCTS average would be invalid.", "Keep domain-specific comparisons explicit; use width 5/20 for Block Grouping and Counters, width 20/70 elsewhere.", "reporting", "closed by design"),
    ("PW-POWER", "MCTS-PW-CROSS-DOMAIN", "Two-seed cross-domain PW is exploratory and cannot support final statistical claims.", "Expand only promising domain/arm combinations to five matched seeds after current two-seed screen.", "conditional compute", "live"),
    ("HORIZON-BIND", "MCTS-HORIZON", "The original Drone horizon pilot had zero cutoffs and could not test the mechanism.", "Finish fresh aware/unaware 750-action pairs; if cutoffs remain sparse, move to cutoff-rich Counters instances with preregistered limits.", "new compute", "live"),
    ("SAFE-INCOMPLETE", "MCTS-SAFE", "Terminal masking repaired two of four failures; two still wandered into failure.", "Use SAFE-CONTEXT for history-aliasing, retain complete decision diagnostics, and classify remaining failure modes.", "new compute", "live overlap"),
    ("CONTEXT-MEM", "MCTS-SAFE-CONTEXT", "Contextual nodes can multiply node count and memory (diagnostic maximum already >5x).", "Gate full conclusions on peak RSS, node multiplier, collision rate, and matched coverage/runtime—not coverage alone.", "instrumentation", "live"),
    ("CYCLE-ABLATION", "MCTS-SAFE-CONTEXT", "Physical-state cycle detection may reject a beneficial revisit with a different action history.", "Keep current defensible physical-cycle rule, but add a small contextual-cycle ablation only if collision witnesses show useful revisits.", "conditional design", "open"),
    ("POLICY-TIME", "MCTS-PW-30M", "The deterministic 30-minute recensoring is valid descriptive evidence but is not yet a fresh operational hard-cap experiment.", "After the two-seed screen selects promising cells, run fresh same-commit fixed/PW jobs with a hard 30-minute instance cap; this is cheap enough to close directly.", "new compute", "held-ready"),
    ("S2-SYNERGY", "MAIN-VAL-S2-MCTS", "Training-plus-search synergy is not yet estimable for most domains because Stage-2 MCTS is held/incomplete.", "Complete selected-checkpoint Stage-2 MCTS; keep Stage-2-final MCTS excluded unless protocol changes.", "new compute", "open"),
    ("TUNING-HOLDOUT", "PRESERVE-4", "Two of ten seeds select the anchor coefficient, leaving only eight unbiased confirmation seeds per VH.", "Report tuning winners separately from the eight held-out seeds; never present all ten as fully held-out.", "analysis", "open/live"),
    ("PAPER-VARIANCE", "all", "Published paper scores are single aggregates without seed-level variance.", "Show our mean/CI and descriptive difference only; do not calculate a p-value against the paper number.", "reporting", "closed"),
    ("BUILD-VARIATION", "MCTS-HORIZON", "Historical and fresh MCTS processes are not bitwise reproducible across code/lifecycle builds.", "Use fresh aware and unaware arms from the same commit for causal claims.", "new compute", "closed in binding design"),
    ("CONTINUATION-STATE", "LONG-DRONE", "Legacy continuations reset their validation baseline, overstating continuation improvements.", "Use restored trainer-state metadata and label legacy continuation evidence as compatibility-limited.", "code/reporting", "fix implemented; historical caveat"),
    ("LOW-N", "all live", "Non-significance at n=2-5 is not evidence of no effect.", "Label exploratory rows; freeze confirmatory inference only after the preregistered matched seed count is terminal.", "reporting/compute", "open/live"),
]


IMPROVEMENTS = {
    "MAIN-VAL": ("Use RQ-aligned Holm families and paired difference-in-differences for the value-head question.", 5, 5, "high interpretive benefit", "STAT-RQ analysis; no separate experiment"),
    "MAIN-TERM": ("Add strict training-success stopping replication rather than treating terminal snapshots as perfect paper replication.", 5, 5, "medium-high scientific benefit", "STOP-ORIG"),
    "PRESERVE-4": ("Finish Delivery/TPP and report coefficient-tuning seeds separately from eight held-out confirmation seeds.", 5, 5, "high", "ANCHOR-4"),
    "MPRIME-VAL": ("Use a larger frozen validation ensemble or rank-robust selection objective, without test-guided tuning.", 4, 5, "medium", "none"),
    "MAIN-EXT6-MPRIME": ("Finish tuning and ten-seed Stage-2 confirmation, then publish both five-domain confirmatory and six-domain extension families.", 5, 5, "high", "MPRIME-VAL and ANCHOR-4"),
    "MAIN-TERM-EXT6-MPRIME": ("Reuse the validation-led per-VH anchor coefficients and release all twenty final-checkpoint lineages after coefficient freeze.", 5, 5, "high", "MAIN-EXT6-MPRIME and MPRIME-ANCHOR"),
    "ANCHOR-4": ("Automate terminal training -> policy curve -> coefficient freeze -> confirmation submission.", 5, 4, "high operational benefit", "PRESERVE-4 controller chain"),
    "MCTS-WIDTH": ("Report domain-specific width choices and conservative interrupted-job lower bounds side by side.", 5, 5, "high", "MCTS-PW-CROSS-DOMAIN"),
    "MCTS-PW": ("Raise minimum width to three while keeping policy ordering and SAFE behavior.", 4, 5, "observed moderate recovery", "already tested by MCTS-PW-SAFE"),
    "MCTS-PW-SAFE": ("Expand only promising Kmin=3 domain arms from two to five seeds.", 4, 5, "medium", "MCTS-PW-CROSS-DOMAIN"),
    "MCTS-PW-PATHBATCH": ("Preflight backup semantics and memory before any grid; use one seed/instance first.", 2, 4, "uncertain", "none"),
    "MCTS-SAFE": ("Separate node statistics by action-history context while retaining physical-state cycle detection.", 5, 5, "potentially high", "MCTS-SAFE-CONTEXT"),
    "MCTS-HORIZON": ("Stratify instances by actual cutoff opportunity; migrate to Counters if Drone remains nonbinding.", 3, 5, "medium if cutoffs occur", "MCTS-SAFE2 is different and held"),
    "MCTS-SAFE-CONTEXT": ("Make peak RSS, context multiplier, prediction disagreement and coverage co-primary diagnostics.", 5, 5, "high correctness value", "current instrumentation"),
    "MCTS-SAFE2": ("Estimate state-by-horizon multiplication before behavior change; keep unsent until memory is bounded.", 2, 4, "uncertain correctness benefit", "MCTS-HORIZON"),
    "MAIN-VAL-S2-MCTS": ("Finish the sixteen Drone gaps, then release fast/high-information domains before resource-heavy ones.", 5, 5, "high RQ coverage", "none"),
    "MCTS-PW-CROSS-DOMAIN": ("Use the two-seed screen to select domains for a five-seed confirmation rather than expanding every arm.", 4, 5, "medium-high efficiency", "MCTS-PW-SAFE"),
    "MCTS-PW-30M": ("Run fresh 30-minute capped fixed/PW comparisons on promising screened cells, including Counters.", 5, 5, "high advisor relevance", "post-hoc 30-minute analysis already exists"),
    "MCTS-RESOURCE": ("Resume with two workers/160 GiB after lifecycle deployment and preserve fixed-budget lower bounds.", 5, 5, "high reliability", "MCTS-LEGACY-ROVER"),
    "MCTS-LEGACY-ROVER": ("Replace fragile three-worker/120-GiB releases with the documented resource-safe continuation.", 5, 5, "high completion probability", "MCTS-RESOURCE"),
    "LONG-DRONE": ("Restore lineage validation state and compare terminal—not peak—policy under the same budget.", 3, 4, "medium", "continuation-state fix"),
    "STOP-ORIG": ("Pilot strict original training-success stopping on three matched seeds before a full campaign.", 5, 5, "high fidelity", "MAIN-TERM"),
    "PUCT-EST": ("Use a small paired factorial pilot instead of an open-ended coefficient sweep.", 2, 4, "uncertain", "BG-HIST historical sensitivity"),
    "ENHSP-LEAF": ("Archive as exploratory and do not expand unless main estimands identify an estimator bottleneck.", 1, 2, "low", "none"),
    "BG-HIST": ("Retain provenance only; prevent historical width-5/3 results from re-entering primary tables.", 1, 3, "high reporting hygiene", "MCTS-WIDTH"),
}


def build_improvements() -> list[dict]:
    experiments = read_csv(TRACK / "experiments.csv")
    rows = []
    for exp in experiments:
        suggestion = IMPROVEMENTS.get(exp["experiment_id"], ("Reconcile status and define a matched estimand before expansion.", 3, 3, "unknown", "none"))
        rows.append(
            {
                "experiment_id": exp["experiment_id"],
                "display_name": exp["display_name"],
                "current_status": exp["status"],
                "suggested_improvement": suggestion[0],
                "priority_1_to_5": suggestion[1],
                "relevance_1_to_5": suggestion[2],
                "predicted_benefit": suggestion[3],
                "already_present_or_overlap": suggestion[4],
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rq = build_rq_results()
    workload = build_workload()
    improvements = build_improvements()
    write_csv(OUT / "rq_results.csv", rq)
    write_csv(OUT / "cluster_workload.csv", workload)
    write_csv(
        OUT / "story_holes.csv",
        [
            {
                "hole_id": x[0],
                "experiment_scope": x[1],
                "story_hole": x[2],
                "defensible_cover": x[3],
                "work_type": x[4],
                "status": x[5],
            }
            for x in STORY_HOLES
        ],
    )
    write_csv(OUT / "improvement_priorities.csv", improvements)
    (OUT / "README.md").write_text(
        "# Advisor audit package — 2026-08-30\n\n"
        "This directory is the durable source for the advisor status presentation. "
        "It joins the authoritative experiment ledgers without copying raw logs.\n\n"
        "- `rq_results.csv`: paired seed-level RQ estimands, exact sign-flip p-values, and the original five-domain Holm correction within each RQ. MPrime will be added only to a separately labelled six-domain extension after its Stage-2 confirmation.\n"
        "- `cluster_workload.csv`: exact Slurm job/CPU/requested-memory snapshot grouped by experiment and state.\n"
        "- `story_holes.csv`: threats, missing links, and defensible closure strategies.\n"
        "- `improvement_priorities.csv`: at least one improvement for every registered experiment, scored for priority/relevance and cross-checked for overlap.\n"
        "- `current_status.md` and `experiment_status.csv`: the human-readable and machine-readable live/held experiment snapshot, including estimates and dependencies.\n"
        "- `numeric_asnets_advisor_status_20260830.pptx`: advisor-ready presentation (generated separately).\n\n"
        "Raw provenance remains in `../experiment_results.csv`, `../policy_paired_seed_results.csv`, "
        "`../mcts_paired_seed_results.csv`, `../live_jobs.csv`, and experiment-specific subdirectories.\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rq)} RQ rows, {len(workload)} workload rows, {len(STORY_HOLES)} holes, {len(improvements)} improvements")


if __name__ == "__main__":
    main()
