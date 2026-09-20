#!/usr/bin/env python3
"""Exact synchronized-sign-flip maxT sensitivity analysis for primary RQs.

The ten canonical seed IDs are shared across domains.  One sign is therefore
drawn per seed and applied to every domain simultaneously, preserving observed
cross-domain dependence.  This is a post-hoc sensitivity analysis alongside
Holm, not a replacement selected because it yields smaller adjusted p-values.
"""

from __future__ import annotations

import csv
import itertools
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
OUT = TRACK / "westfall_young_rq_sensitivity_20260920"
PRIMARY = TRACK / "advisor_followup_20260910" / "rq_primary_validation_led.csv"
DOMAINS = ["block_grouping", "drone", "fo_counters", "rover", "counters", "mprime"]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def policy_vectors() -> dict[tuple[str, str, str, str], dict[str, dict[str, float]]]:
    rows = [r for r in read(TRACK / "policy_paired_seed_results.csv")
            if r["experiment_id"] == "MAIN-VAL"]
    m1 = read(TRACK / "mprime_phase_b_a_stage1_mcts_20260913" /
              "results_20260914" / "per_seed_results.csv")
    m2 = read(TRACK / "mprime_final_stage2_search_20260916" /
              "final_per_seed_results_20260918.csv")
    before = {(r["value_head"], r["seed"]): float(r["policy_score"]) for r in m1}
    for r in m2:
        b = before[(r["value_head"], r["seed"])]
        rows.append({"domain": "mprime", "value_head": r["value_head"],
                     "seed": r["seed"], "difference": str(float(r["stage2_policy"]) - b)})
    by = {(r["domain"], r["value_head"], r["seed"]): float(r["difference"]) for r in rows}
    seeds = sorted({r["seed"] for r in rows if r["domain"] == "block_grouping"})
    result: dict[tuple[str, str, str, str], dict[str, dict[str, float]]] = {}
    for domain in DOMAINS:
        result.setdefault(("RQ1", "Stage 2", "endpoint",
                           "VH-off: Stage 2 policy - Stage 1 policy"), {})[domain] = {
            s: by[(domain, "off", s)] for s in seeds}
        result.setdefault(("RQ3", "Stage 2", "endpoint",
                           "VH-on direct: Stage 2 policy - Stage 1 policy"), {})[domain] = {
            s: by[(domain, "on", s)] for s in seeds}
        result.setdefault(("RQ3", "Stage 2", "endpoint",
                           "VH interaction: VH-on refinement - VH-off refinement"), {})[domain] = {
            s: by[(domain, "on", s)] - by[(domain, "off", s)] for s in seeds}
    return result


def search_vectors() -> dict[tuple[str, str, str, str], dict[str, dict[str, float]]]:
    result: dict[tuple[str, str, str, str], dict[str, dict[str, float]]] = {}
    m1_path = TRACK / "mprime_phase_b_a_stage1_mcts_20260913" / "results_20260914" / "per_seed_results.csv"
    m2_path = TRACK / "mprime_final_stage2_search_20260916" / "final_per_seed_results_20260918.csv"
    for stage, source in (("Stage 1", TRACK / "stage1_policy_mcts_seed_cutoffs_latest.csv"),
                          ("Stage 2", TRACK / "stage2_policy_mcts_seed_cutoffs_latest.csv")):
        rows = read(source)
        if stage == "Stage 2":
            rows = [r for r in rows if r.get("stage2_branch") == "validation_led"
                    and r["domain"] != "fo_counters"]
        if stage == "Stage 1":
            rows += [{"domain": "mprime", "value_head": r["value_head"], "seed": r["seed"],
                      "policy_score": r["policy_score"], "mcts_30m": r["mcts_30m"],
                      "mcts_2h": r["mcts_2h"], "mcts_6h": r["mcts_6h"]}
                     for r in read(m1_path)]
        else:
            rows += [{"domain": "mprime", "value_head": r["value_head"], "seed": r["seed"],
                      "policy_score": r["stage2_policy"], "mcts_30m": r["fixed_30m"],
                      "mcts_2h": r["fixed_2h"], "mcts_6h": r["fixed_6h"]}
                     for r in read(m2_path)]
            off = read(TRACK / "advisor_followup_20260910" /
                       "fo_stage2_validation_vh_off_exact_seed_results_20260913.csv")
            on = read(TRACK / "advisor_followup_20260910" /
                      "fo_stage2_validation_vh_on_exact_seed_results_20260912.csv")
            rows += [{"domain": "fo_counters", "value_head": "off", "seed": r["seed"],
                      "policy_score": r["policy_score"], "mcts_30m": r["mcts_30m"],
                      "mcts_2h": r["mcts_2h"], "mcts_6h": r["mcts_6h"]} for r in off]
            rows += [{"domain": "fo_counters", "value_head": "on", "seed": r["seed"],
                      "policy_score": r["policy_vh_on"], "policy_off": r["policy_vh_off"],
                      "mcts_30m": r["mcts_30m"], "mcts_2h": r["mcts_2h"],
                      "mcts_6h": r["mcts_6h"]} for r in on]
        by = {(r["domain"], r["value_head"], r["seed"]): r for r in rows}
        seeds = sorted({r["seed"] for r in rows if r["domain"] == "block_grouping"})
        for cutoff in ("30m", "2h", "6h"):
            for domain in DOMAINS:
                offv = {}
                onv = {}
                cross = {}
                interaction = {}
                for seed in seeds:
                    off = by[(domain, "off", seed)]
                    on = by[(domain, "on", seed)]
                    offv[seed] = float(off[f"mcts_{cutoff}"]) - float(off["policy_score"])
                    onv[seed] = float(on[f"mcts_{cutoff}"]) - float(on["policy_score"])
                    off_policy = (float(on["policy_off"]) if domain == "fo_counters" and
                                  stage == "Stage 2" else float(off["policy_score"]))
                    cross[seed] = float(on[f"mcts_{cutoff}"]) - off_policy
                    interaction[seed] = onv[seed] - offv[seed]
                result.setdefault(("RQ2", stage, cutoff,
                                   "VH-off direct: MCTS - same-checkpoint policy"), {})[domain] = offv
                result.setdefault(("RQ4", stage, cutoff,
                                   "VH-on direct: MCTS - same-checkpoint policy"), {})[domain] = onv
                result.setdefault(("RQ4", stage, cutoff,
                                   "Cross-cell level: VH-on MCTS - parallel VH-off policy"), {})[domain] = cross
                result.setdefault(("RQ4", stage, cutoff,
                                   "VH interaction: VH-on MCTS benefit - VH-off MCTS benefit"), {})[domain] = interaction
    return result


def tstat(values: list[float]) -> float:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return 0.0
    sd = statistics.stdev(values)
    if sd <= 1e-15:
        return math.inf if abs(mean) > 1e-15 else 0.0
    return abs(mean) / (sd / math.sqrt(len(values)))


def main() -> None:
    vectors = policy_vectors()
    vectors.update(search_vectors())
    primary = {(r["rq"], r["stage"], r["cutoff"], r["estimand"], r["domain"]): r
               for r in read(PRIMARY)}
    output: list[dict[str, object]] = []
    for family, domains in sorted(vectors.items()):
        if sorted(domains) != sorted(DOMAINS):
            raise RuntimeError(f"{family}: incomplete domains {sorted(domains)}")
        seeds = sorted(next(iter(domains.values())))
        if any(sorted(v) != seeds for v in domains.values()):
            raise RuntimeError(f"{family}: seed alignment differs")
        observed = {d: tstat([domains[d][s] for s in seeds]) for d in DOMAINS}
        permuted: list[dict[str, float]] = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(seeds)):
            permuted.append({d: tstat([domains[d][s] * sign for s, sign in zip(seeds, signs)])
                             for d in DOMAINS})
        raw = {d: sum(p[d] >= observed[d] - 1e-12 for p in permuted) / len(permuted)
               for d in DOMAINS}
        single = {d: sum(max(p.values()) >= observed[d] - 1e-12 for p in permuted) / len(permuted)
                  for d in DOMAINS}
        ordered = sorted(DOMAINS, key=lambda d: observed[d], reverse=True)
        step: dict[str, float] = {}
        running = 0.0
        for rank, domain in enumerate(ordered):
            remaining = ordered[rank:]
            pvalue = sum(max(p[d] for d in remaining) >= observed[domain] - 1e-12
                         for p in permuted) / len(permuted)
            running = max(running, pvalue)
            step[domain] = running
        rq, stage, cutoff, estimand = family
        for domain in DOMAINS:
            canonical = primary[(rq, stage, cutoff, estimand, domain)]
            values = [domains[domain][s] for s in seeds]
            output.append({
                "rq": rq, "stage": stage, "cutoff": cutoff, "estimand": estimand,
                "domain": domain, "n": len(seeds),
                "paired_differences": ";".join(f"{v:g}" for v in values),
                "mean_difference": statistics.fmean(values),
                "studentized_abs_t": observed[domain],
                "canonical_raw_mean_signflip_p": canonical["raw_p"],
                "canonical_holm_p": canonical["holm_p"],
                "westfall_raw_studentized_p": raw[domain],
                "westfall_young_single_step_maxT_p": single[domain],
                "westfall_young_step_down_maxT_p": step[domain],
                "holm_significant_0_05": float(canonical["holm_p"]) < .05,
                "single_step_maxT_significant_0_05": single[domain] < .05,
                "step_down_maxT_significant_0_05": step[domain] < .05,
                "permutations": len(permuted),
                "seed_alignment": ";".join(seeds),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "westfall_young_maxT_results.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(output)
    print(f"wrote {len(output)} rows across {len(vectors)} families to {path}")


if __name__ == "__main__":
    main()
