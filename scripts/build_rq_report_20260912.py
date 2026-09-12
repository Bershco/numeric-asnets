#!/usr/bin/env python3
"""Build the validation-led RQ report from canonical local result CSVs."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiment_tracking" / "advisor_followup_20260910"
PRIMARY = OUT / "rq_primary_validation_led.csv"
PW2 = OUT / "rq2_pw70_branch_latest.csv"
PW4 = OUT / "rq4_pw70_branch_latest.csv"
REPORT = OUT / "rq_report_validation_led_20260912.md"

DOMAIN_LABEL = {
    "block_grouping": "Block Grouping",
    "drone": "Drone",
    "fo_counters": "FO Counters",
    "rover": "Rover",
    "counters": "Counters",
}
DOMAIN_ORDER = list(DOMAIN_LABEL)
CUTOFF_ORDER = {"30m": 0, "2h": 1, "6h": 2, "endpoint": 0}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: str, digits: int = 2) -> str:
    if value in ("", None):
        return "—"
    x = float(value)
    if math.isnan(x):
        return "—"
    text = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def effect_cell(row: dict[str, str], partial: bool = False) -> str:
    prefix = "≥" if partial else ""
    mean = prefix + number(row["comparison_mean"], 2)
    effect = prefix + ("+" if float(row["effect_solved"]) > 0 else "") + number(row["effect_solved"], 2)
    if partial or not row.get("ci95_low_solved"):
        return f"{mean}; {effect}; CI/p pending"
    ci = f"[{number(row['ci95_low_solved'])}, {number(row['ci95_high_solved'])}]"
    raw = number(row.get("raw_p", ""), 4)
    holm = number(row.get("holm_p", ""), 4)
    return f"{mean}; {effect} {ci}; p={raw}/{holm}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


primary = read_rows(PRIMARY)
pw2 = read_rows(PW2)
pw4 = read_rows(PW4)

rq1 = [r for r in primary if r["rq"] == "RQ1"]
rq3_direct = [r for r in primary if r["rq"] == "RQ3" and r["estimand"].startswith("VH-on direct")]
rq3_inter = [r for r in primary if r["rq"] == "RQ3" and "interaction" in r["estimand"]]

rq1_map = {r["domain"]: r for r in rq1}
rq3d_map = {r["domain"]: r for r in rq3_direct}
rq3i_map = {r["domain"]: r for r in rq3_inter}

rq1_rows = []
for domain in DOMAIN_ORDER:
    r = rq1_map[domain]
    rq1_rows.append([
        DOMAIN_LABEL[domain], number(r["baseline_mean"]), number(r["comparison_mean"]),
        f"{number(r['effect_solved'])} [{number(r['ci95_low_solved'])}, {number(r['ci95_high_solved'])}]",
        f"{number(r['raw_p'], 4)} / {number(r['holm_p'], 4)}",
    ])

rq2 = [r for r in primary if r["rq"] == "RQ2"]
rq2_group = defaultdict(dict)
for r in rq2:
    rq2_group[(r["stage"], r["domain"])][r["cutoff"]] = r
rq2_rows = []
for stage in ("Stage 1", "Stage 2"):
    for domain in DOMAIN_ORDER:
        group = rq2_group.get((stage, domain), {})
        if not group:
            continue
        first = next(iter(group.values()))
        row = [stage, DOMAIN_LABEL[domain], "Narrow 5/20" if domain in {"block_grouping", "counters"} else "Normal 20/70", number(first["baseline_mean"])]
        for cutoff in ("30m", "2h", "6h"):
            r = group[cutoff]
            partial = r["evidence_status"] != "complete_paired_inference"
            row.append(effect_cell(r, partial))
        rq2_rows.append(row)

rq3_rows = []
for domain in DOMAIN_ORDER:
    d, i = rq3d_map[domain], rq3i_map[domain]
    rq3_rows.append([
        DOMAIN_LABEL[domain], f"{number(d['baseline_mean'])} → {number(d['comparison_mean'])}",
        f"{number(d['effect_solved'])} [{number(d['ci95_low_solved'])}, {number(d['ci95_high_solved'])}]; p={number(d['raw_p'],4)}/{number(d['holm_p'],4)}",
        number(i["baseline_mean"]),
        f"{number(i['effect_solved'])} [{number(i['ci95_low_solved'])}, {number(i['ci95_high_solved'])}]; p={number(i['raw_p'],4)}/{number(i['holm_p'],4)}",
    ])

rq4 = [r for r in primary if r["rq"] == "RQ4"]
rq4_group = defaultdict(dict)
for r in rq4:
    rq4_group[(r["stage"], r["domain"], r["estimand"])][r["cutoff"]] = r

def rq4_table(match: str) -> str:
    rows = []
    keys = [key for key in rq4_group if match in key[2]]
    for stage in ("Stage 1", "Stage 2"):
        for domain in DOMAIN_ORDER:
            candidates = [key for key in keys if key[0] == stage and key[1] == domain]
            if not candidates:
                if stage == "Stage 2" and domain == "fo_counters" and match == "VH interaction":
                    rows.append([stage, DOMAIN_LABEL[domain], "Partial", "Indeterminate; recovery live", "Indeterminate", "Indeterminate"])
                continue
            group = rq4_group[candidates[0]]
            first = next(iter(group.values()))
            row = [stage, DOMAIN_LABEL[domain], number(first["baseline_mean"])]
            for cutoff in ("30m", "2h", "6h"):
                r = group[cutoff]
                partial = r["evidence_status"] != "complete_paired_inference"
                row.append(effect_cell(r, partial))
            rows.append(row)
    return md_table(["Stage", "Domain", "Baseline", "30m: mean; Δ [95% CI]; raw/Holm p", "2h", "6h"], rows)


pw2_rows = []
for r in sorted(pw2, key=lambda x: (DOMAIN_ORDER.index(x["domain"]), CUTOFF_ORDER[x["cutoff"]])):
    pw2_rows.append([
        DOMAIN_LABEL[r["domain"]], r["cutoff"], r["n"], number(r["vh_off_policy_mean"]),
        number(r["vh_off_fixed_mcts_mean"]), number(r["vh_off_pw70_mean"]),
        f"{number(r['pw70_minus_policy'])} [{number(r['ci95_low'])}, {number(r['ci95_high'])}]",
        f"{number(r['raw_p'], 4)} / {number(r['holm_p'], 4)}",
    ])

pw4_rows = []
for r in sorted(pw4, key=lambda x: (DOMAIN_ORDER.index(x["domain"]), CUTOFF_ORDER[x["cutoff"]], x["estimand"])):
    pw4_rows.append([
        DOMAIN_LABEL[r["domain"]], r["cutoff"], r["estimand"].replace("VH-on ", "on ").replace("parallel VH-off", "off"),
        f"off policy {number(r['vh_off_policy_mean'])}; on policy {number(r['vh_on_policy_mean'])}; off PW {number(r['vh_off_pw70_mean'])}; on PW {number(r['vh_on_pw70_mean'])}",
        f"{number(r['effect'])} [{number(r['ci95_low'])}, {number(r['ci95_high'])}]",
        f"{number(r['raw_p'], 4)} / {number(r['holm_p'], 4)}",
    ])

text = f"""# Validation-led RQ report — 12 September 2026

This is the primary thesis view. Terminal-led campaigns are excluded. Fixed-search 30-minute and two-hour figures are deterministic cutoffs of the same six-hour runs, not separate reruns. Block Grouping and Counters use narrow fixed search (5 retained children, 20 simulations); Drone, FO Counters and Rover use normal fixed search (20 children, 70 simulations). Counts are solved test instances; Counters has 59 instances and the other domains have 20.

Holm correction is applied separately within each RQ × stage × cutoff × estimand family. RQ2 Stage-2 VH-off and the RQ4 Stage-2 interaction are provisional four-domain families while FO/VH-off is partial. RQ4 Stage-2 direct and cross-cell comparisons are final five-domain families because FO/VH-on is exact. PW uses separate two-domain families and is never pooled with fixed search.

MPrime is not yet admitted to RQ1/RQ3: Phase C selected Phase-B replicate A as the validator, but the existing Stage-2 networks were trained from the superseded Stage-1 selections. A clean MPrime result requires anchor rescoring and 20 new validation-led Stage-2 lineages.

## RQ1 — Does Stage-2 training improve policy coverage without a value head?

{md_table(['Domain', 'Stage-1 policy', 'Stage-2 policy', 'Δ [95% CI]', 'Raw / Holm p'], rq1_rows)}

**Conclusion:** Stage 2 does not produce a Holm-significant VH-off policy improvement. FO Counters has a raw decline; Counters has a positive but highly variable mean.

![RQ1 paired Stage-2 effect](rq1_stage2_training_vh_off.png)

## RQ2 — Does inference-time search improve coverage without a value head?

### Fixed search

{md_table(['Stage', 'Domain', 'Search', 'Policy', '30m: MCTS; Δ [95% CI]; raw/Holm p', '2h', '6h'], rq2_rows)}

**Conclusion:** FO Counters is the clear fixed-search success; Drone and Rover improve modestly. Block Grouping needs the longer budget to approach parity, while Counters can be harmed. The live FO Stage-2 VH-off exact-instance recovery leaves only one outcome unresolved; until it terminates, that RQ2 row remains a lower bound and has no CI or p-value.

![RQ2 raw policy and fixed-search means](rq2_raw_means_by_stage.png)

![RQ2 paired fixed-search effects](rq2_mcts_vh_off.png)

### Progressive widening (PW70; Stage 1 confirmation)

{md_table(['Domain', 'Cutoff', 'n', 'Policy', 'Fixed MCTS', 'PW70', 'PW−policy [95% CI]', 'Raw / Holm p'], pw2_rows)}

**Conclusion:** PW70 gives large, corrected-significant FO Counters gains already at 30 minutes. Rover is approximately fixed-search parity, without a significant policy gain. PW is meaningful RQ2 evidence, but this ten-seed confirmation exists only after Stage 1 and only for FO Counters/Rover.

![RQ2/RQ4 PW70 confirmation](rq2_rq4_pw70_final.png)

## RQ3 — Does the value head improve Stage-2 policy refinement?

The direct column answers whether VH-on Stage 2 improves its own VH-on Stage-1 policy. The interaction asks whether that refinement is better than the parallel VH-off refinement.

{md_table(['Domain', 'VH-on raw S1 → S2', 'VH-on direct Δ [95% CI]; raw/Holm p', 'VH-off Δ', 'Interaction Δ [95% CI]; raw/Holm p'], rq3_rows)}

**Conclusion:** Neither the direct VH-on changes nor the interactions show a corrected-significant benefit. Block Grouping is the clearest harmful tendency; the DiD does not hide a beneficial VH-on result.

![RQ3 raw means and interaction](rq3_raw_means_and_interaction.png)

![RQ3 direct VH-on Stage-2 effect](rq3_value_head_training.png)

## RQ4 — Does the value head change the benefit of MCTS inference?

These three estimands are deliberately separate.

### A. VH-on MCTS versus its own VH-on policy

{rq4_table('VH-on direct')}

### B. VH-on MCTS versus the parallel VH-off policy

{rq4_table('parallel VH-off policy')}

### C. Difference in MCTS benefit: VH-on minus VH-off

{rq4_table('VH interaction')}

**Conclusion:** Drone is the robust RQ4 success: VH-on materially increases MCTS usefulness. FO Counters VH-on is now exact and benefits from MCTS, but the VH interaction remains indeterminate until the live VH-off instance recovery ends. Other domains do not show a reliable value-head interaction.

![RQ4 raw six-hour levels](rq4_raw_means_6h_by_stage.png)

![RQ4 direct effect](rq4_direct.png)

![RQ4 comparison against parallel VH-off policy](rq4_cross_cell.png)

![RQ4 interaction](rq4_interaction.png)

### PW70 contribution to RQ4 (Stage 1 confirmation)

{md_table(['Domain', 'Cutoff', 'Estimand', 'Raw means', 'Effect [95% CI]', 'Raw / Holm p'], pw4_rows)}

**Conclusion:** PW preserves the distinction seen with fixed search: FO Counters has strong search gains, but VH-off benefits at least as much; Rover shows parity-scale, non-significant effects. PW therefore strengthens RQ2 more than RQ4.

## Results still required

1. **FO Counters Stage-2 fixed MCTS:** one exact instance is live. It changes only one seed from 5/20 to at most 6/20 and the VH-off mean from 6.1 to at most 6.2.
2. **MPrime:** the first smoke (`21221744`) exposed a missing deployed validator directory before scientific work; its dependent jobs were automatically cancelled. The corrected chain is smoke `21222348`, array `21222349[0-27]`, finalizer `21222350`. After manual coefficient review, train 20 validation-led Stage-2 lineages from the new Stage-1 selections, evaluate policy curves/endpoints, then run matched fixed MCTS if MPrime is to enter RQ2/RQ4.
3. **Counters tie-break:** the targeted three-instance causal screen is complete; a multi-seed confirmation is still needed before changing the default evaluator.

Canonical evidence files: [`rq_primary_validation_led.csv`](rq_primary_validation_led.csv), [`rq2_raw_means_validation_led.csv`](rq2_raw_means_validation_led.csv), [`rq3_raw_means_validation_led.csv`](rq3_raw_means_validation_led.csv), [`rq4_raw_means_validation_led.csv`](rq4_raw_means_validation_led.csv), [`rq2_pw70_branch_latest.csv`](rq2_pw70_branch_latest.csv), and [`rq4_pw70_branch_latest.csv`](rq4_pw70_branch_latest.csv). Their row-level job/log routes are indexed in [`../../result_csv_provenance_index_latest.csv`](../../result_csv_provenance_index_latest.csv).
"""

REPORT.write_text(text, encoding="utf-8")
print(REPORT)
