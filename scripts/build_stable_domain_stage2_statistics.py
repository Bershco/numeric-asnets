#!/usr/bin/env python3
"""Build the authoritative ten-seed stable-domain Stage-2 paired statistics.

The seed-level inputs are intentionally explicit and retain source-log pointers.
Tuning-seed endpoint evidence that was produced by the September endpoint pass is
joined to the held-out reconciliation rather than being hidden in an aggregate.
"""

from __future__ import annotations

import csv
import itertools
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiment_tracking"
OUT_DIR = TRACK / "four_domain_preservation"
PAIR_OUT = OUT_DIR / "stable_domain_stage2_seed_pairs_20260902.csv"
STAT_OUT = OUT_DIR / "stable_domain_stage2_statistics_20260902.csv"
TUNING = {1963100312, 2011206605}
T95 = {10: 2.262157}


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def exact_sign_flip(values: list[float]) -> float:
    observed = abs(sum(values))
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(values)):
        if abs(sum(sign * value for sign, value in zip(signs, values))) >= observed - 1e-12:
            extreme += 1
    return extreme / (2 ** len(values))


def ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2 or statistics.stdev(values) == 0:
        return mean, mean
    half = T95[len(values)] * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half, mean + half


def main() -> None:
    s1 = {}
    for row in rows(TRACK / "experiment_results.csv"):
        if (row["experiment_id"] == "PRESERVE-4" and row["stage"] == "stage1"
                and row["endpoint"] == "validation_selected"
                and row["domain"] in {"delivery", "tpp", "zenotravel"}):
            key = (row["domain"], row["value_head"], int(row["seed"]))
            s1[key] = row

    s2 = {}
    for domain in ("delivery", "tpp"):
        path = OUT_DIR / f"{domain}_stage2_policy_reconciliation_20260831.csv"
        for row in rows(path):
            if "stage2_validation_selected_policy" not in row["analysis_roles"]:
                continue
            key = (domain, row["value_head"], int(row["seed"]))
            row = dict(row)
            row["source_evaluation_log"] = row.get("source_log", "")
            s2[key] = row

    for row in rows(OUT_DIR / "zenotravel_stage2_policy_results.csv"):
        if "stage2_validation_selected_policy" not in row["analysis_roles"]:
            continue
        key = ("zenotravel", row["value_head"], int(row["seed"]))
        row = dict(row)
        row["score"] = row["successes"]
        s2[key] = row

    # Endpoint pass used only where the original reconciliation did not contain
    # a tuning seed. Scores and literal paths were checked against Slurm logs.
    tuning_rows = {
        ("delivery", "off", 1963100312): (20, 20835064, "/home/hersco/training_new_domains/2026-09-01/preserve4_delivery_stage2_policy_eval/20835064_Ev_delivery_delivery_mcts_orig_novh_c.1_s1963100312_K0_P4DELIVERYS2P_TUNINGSEL_src20553890_e0003.txt"),
        ("delivery", "off", 2011206605): (20, 20835065, "/home/hersco/training_new_domains/2026-09-01/preserve4_delivery_stage2_policy_eval/20835065_Ev_delivery_delivery_mcts_orig_novh_c.1_s2011206605_K0_P4DELIVERYS2P_TUNINGSEL_src20553897_e0001.txt"),
        ("delivery", "on", 1963100312): (20, 20835066, "/home/hersco/training_new_domains/2026-09-01/preserve4_delivery_stage2_policy_eval/20835066_Ev_delivery_delivery_mcts_orig_vh_c.1_s1963100312_K0_P4DELIVERYS2P_TUNINGSEL_src20553876_e0001.txt"),
        ("delivery", "on", 2011206605): (20, 20835067, "/home/hersco/training_new_domains/2026-09-01/preserve4_delivery_stage2_policy_eval/20835067_Ev_delivery_delivery_mcts_orig_vh_c.1_s2011206605_K0_P4DELIVERYS2P_TUNINGSEL_src20595750_e0038.txt"),
        ("tpp", "off", 1963100312): (20, 20835062, "/home/hersco/training_new_domains/2026-09-01/preserve4_tpp_stage2_policy_eval/20835062_Ev_tpp_tpp_mcts_orig_novh_c.1_s1963100312_K0_P4TPPS2P_TUNINGSEL_src20553944_e0001.txt"),
        ("tpp", "off", 2011206605): (20, 20594461, "/home/hersco/training_new_domains/2026-08-26/completed_domains_anchor_tuning_policy_eval/20594461_Ev_tpp_tpp_orig_novh_c.1_s2011206605_SR10CD4ATP_src20553951_e0000.txt"),
        ("tpp", "on", 1963100312): (20, 20576315, "/home/hersco/training_new_domains/2026-08-26/completed_domains_anchor_tuning_policy_eval/20576315_Ev_tpp_tpp_orig_vh_c.1_s1963100312_SR10CD4ATP_src20553931_e0090.txt"),
        ("tpp", "on", 2011206605): (20, 20594354, "/home/hersco/training_new_domains/2026-08-26/completed_domains_anchor_tuning_policy_eval/20594354_Ev_tpp_tpp_orig_vh_c.1_s2011206605_SR10CD4ATP_src20553938_e0000.txt"),
        ("tpp", "on", 923500475): (20, 20763718, "/home/hersco/training_new_domains/2026-08-31/preserve4_tpp_stage2_policy_eval/20763718_Ev_tpp_tpp_mcts_orig_vh_c.1_s923500475_K0_P4TPPS2P_src20684884_e0036.txt"),
        ("tpp", "on", 1073581256): (15, 20794985, "/home/hersco/training_new_domains/2026-08-31/preserve4_tpp_stage2_policy_eval/20794985_Ev_tpp_tpp_mcts_orig_vh_c.1_s1073581256_K0_P4TPPS2P_src20684885_e0002.txt"),
        ("tpp", "on", 2082152039): (20, 20821331, "/home/hersco/training_new_domains/2026-09-01/preserve4_tpp_stage2_policy_eval/20821331_Ev_tpp_tpp_mcts_orig_vh_c.1_s2082152039_K0_P4TPPS2P_src20684890_e0079.txt"),
    }
    for key, (score, job_id, log) in tuning_rows.items():
        s2[key] = {"score": str(score), "slurm_job_id": str(job_id),
                   "source_evaluation_log": log, "source_training_job_id": ""}

    pair_fields = [
        "domain", "value_head", "seed", "seed_role", "stage1_selected",
        "stage2_selected", "change", "stage1_evaluation_log",
        "stage1_training_log", "stage2_evaluation_log", "stage2_training_job_id",
        "evidence_note",
    ]
    pair_rows = []
    for key in sorted(s1):
        if key not in s2:
            raise RuntimeError(f"missing Stage-2 endpoint: {key}")
        domain, value_head, seed = key
        left, right = s1[key], s2[key]
        left_score, right_score = float(left["score"]), float(right["score"])
        direct = str(right.get("source_evaluation_log", "")).startswith("/home/")
        pair_rows.append({
            "domain": domain,
            "value_head": value_head,
            "seed": seed,
            "seed_role": "tuning" if seed in TUNING else "held_out",
            "stage1_selected": left_score,
            "stage2_selected": right_score,
            "change": right_score - left_score,
            "stage1_evaluation_log": left["source_evaluation_log"],
            "stage1_training_log": left["source_training_log"],
            "stage2_evaluation_log": right.get("source_evaluation_log", ""),
            "stage2_training_job_id": right.get("source_training_job_id", ""),
            "evidence_note": "direct endpoint log" if direct else "companion aggregate/training provenance; endpoint path retained in source registry",
        })

    with PAIR_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(pair_rows)

    stats = []
    for domain in ("delivery", "tpp", "zenotravel"):
        for value_head in ("off", "on"):
            cell = [r for r in pair_rows if r["domain"] == domain and r["value_head"] == value_head]
            changes = [float(r["change"]) for r in cell]
            lo, hi = ci95(changes)
            held = [float(r["stage2_selected"]) for r in cell if r["seed_role"] == "held_out"]
            tuning = [float(r["stage2_selected"]) for r in cell if r["seed_role"] == "tuning"]
            stats.append({
                "domain": domain, "value_head": value_head, "n": len(cell),
                "stage1_selected_mean": statistics.mean(float(r["stage1_selected"]) for r in cell),
                "stage2_selected_mean_all10": statistics.mean(float(r["stage2_selected"]) for r in cell),
                "stage2_selected_mean_heldout8": statistics.mean(held),
                "stage2_selected_mean_tuning2": statistics.mean(tuning),
                "mean_change": statistics.mean(changes), "ci95_low": lo, "ci95_high": hi,
                "raw_sign_flip_p": exact_sign_flip(changes),
                "holm_p": "", "note": "TPP/off: nine seeds=20/20; seed 1972442430=9/20" if (domain, value_head) == ("tpp", "off") else "",
                "seed_ledger": str(PAIR_OUT.relative_to(ROOT)).replace("\\", "/"),
            })

    order = sorted(range(len(stats)), key=lambda i: float(stats[i]["raw_sign_flip_p"]))
    running = 0.0
    m = len(stats)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (m - rank) * float(stats[index]["raw_sign_flip_p"]))
        running = max(running, adjusted)
        stats[index]["holm_p"] = running

    fields = list(stats[0])
    with STAT_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stats)

    print(f"wrote {len(pair_rows)} seed pairs and {len(stats)} statistics rows")


if __name__ == "__main__":
    main()
