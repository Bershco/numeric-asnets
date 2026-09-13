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
    value = f"{mean}; {effect} {ci}; p={raw}/{holm}"
    return f"**{value}**" if float(row.get("holm_p", "nan")) < 0.05 else value


def effect_with_p(row: dict[str, str]) -> str:
    value = (
        f"{number(row['effect_solved'])} "
        f"[{number(row['ci95_low_solved'])}, {number(row['ci95_high_solved'])}]; "
        f"p={number(row['raw_p'],4)}/{number(row['holm_p'],4)}"
    )
    return f"**{value}**" if float(row["holm_p"]) < 0.05 else value


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
        (f"**{number(r['raw_p'], 4)} / {number(r['holm_p'], 4)}**"
         if float(r["holm_p"]) < 0.05 else
         f"{number(r['raw_p'], 4)} / {number(r['holm_p'], 4)}"),
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
        effect_with_p(d),
        number(i["baseline_mean"]),
        effect_with_p(i),
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
for domain in ("fo_counters", "rover"):
    group = {r["cutoff"]: r for r in pw2 if r["domain"] == domain}
    r6 = group["6h"]
    effects, pvalues = [], []
    for cutoff in ("30m", "2h", "6h"):
        r = group[cutoff]
        effect = f"{number(r['pw70_minus_policy'])} [{number(r['ci95_low'])}, {number(r['ci95_high'])}]"
        pvalue = f"{number(r['raw_p'], 4)}/{number(r['holm_p'], 4)}"
        if float(r["holm_p"]) < 0.05:
            effect, pvalue = f"**{effect}**", f"**{pvalue}**"
        effects.append(effect)
        pvalues.append(pvalue)
    pw2_rows.append([
        DOMAIN_LABEL[domain], r6["n"], number(r6["vh_off_policy_mean"]),
        " / ".join(number(group[c]["vh_off_fixed_mcts_mean"]) for c in ("30m", "2h", "6h")),
        " / ".join(number(group[c]["vh_off_pw70_mean"]) for c in ("30m", "2h", "6h")),
        " / ".join(effects), " / ".join(pvalues),
    ])

pw4_rows = []
for domain in ("fo_counters", "rover"):
    estimands = sorted({r["estimand"] for r in pw4 if r["domain"] == domain})
    for estimand in estimands:
        group = {r["cutoff"]: r for r in pw4 if r["domain"] == domain and r["estimand"] == estimand}
        r6 = group["6h"]
        effects, pvalues = [], []
        for cutoff in ("30m", "2h", "6h"):
            r = group[cutoff]
            effect = f"{number(r['effect'])} [{number(r['ci95_low'])}, {number(r['ci95_high'])}]"
            pvalue = f"{number(r['raw_p'], 4)}/{number(r['holm_p'], 4)}"
            if float(r["holm_p"]) < 0.05:
                effect, pvalue = f"**{effect}**", f"**{pvalue}**"
            effects.append(effect)
            pvalues.append(pvalue)
        pw4_rows.append([
            DOMAIN_LABEL[domain], estimand.replace("VH-on ", "on ").replace("parallel VH-off", "off"),
            (f"off policy {number(r6['vh_off_policy_mean'])}; on policy {number(r6['vh_on_policy_mean'])}; "
             f"off PW {' / '.join(number(group[c]['vh_off_pw70_mean']) for c in ('30m', '2h', '6h'))}; "
             f"on PW {' / '.join(number(group[c]['vh_on_pw70_mean']) for c in ('30m', '2h', '6h'))}"),
            " / ".join(effects), " / ".join(pvalues),
        ])

text = f"""# Validation-led RQ report — 13 September 2026

This is the primary thesis view. Terminal-led campaigns are excluded. Fixed-search 30-minute and two-hour figures are deterministic cutoffs of the same six-hour runs, not separate reruns. Block Grouping and Counters use narrow fixed search (5 retained children, 20 simulations); Drone, FO Counters and Rover use normal fixed search (20 children, 70 simulations). Counts are solved test instances; Counters has 59 instances and the other domains have 20.

Effects are seed-paired mean differences; confidence intervals are paired t-intervals; raw p-values are two-sided exact sign-flip tests; Holm correction is applied separately within each RQ × stage × cutoff × estimand family. All five-domain fixed-search families are now complete. PW uses separate two-domain families and is never pooled with fixed search. **Bold entries are Holm-significant at .05; raw-only significance is not bolded.**

MPrime is not yet admitted to RQ1/RQ3: Phase C selected Phase-B replicate A as the validator, but exactly 16 existing Stage-2 networks were trained from superseded Stage-1 selections. Four more are only reuse candidates until the frozen coefficient, checkpoint hash, code and complete configuration are matched. A clean result therefore requires 16-20 new validation-led Stage-2 lineages. This is separate from the Stage-1 fixed-MCTS audit, which found 0/20 reusable search evaluations.

## RQ1 — Does Stage-2 training improve policy coverage without a value head?

{md_table(['Domain', 'Stage-1 policy', 'Stage-2 policy', 'Δ [95% CI]', 'Raw / Holm p'], rq1_rows)}

**Conclusion:** Stage 2 does not produce a Holm-significant VH-off policy improvement. FO Counters has a raw decline; Counters has a positive but highly variable mean.

![RQ1 paired Stage-2 effect](rq1_stage2_training_vh_off.png)

## RQ2 — Does inference-time search improve coverage without a value head?

### Fixed search

All fixed-search rows below use the historical final action-index tie-break. The live policy-prior candidate is a separate causal experiment and has not changed any RQ value in this table.

{md_table(['Stage', 'Domain', 'Search', 'Policy', '30m: MCTS; Δ [95% CI]; raw/Holm p', '2h', '6h'], rq2_rows)}

**Conclusion:** FO Counters is the clear fixed-search success at both stages and all three cutoffs; Drone and Rover improve modestly. Block Grouping needs the longer budget to approach parity, while Counters can be harmed. The final FO Stage-2 recovery timed out on its exact six-hour instance budget, so the exact mean remains 6.1/20 and the previously partial row is now final.

![RQ2 raw policy and fixed-search means](rq2_raw_means_by_stage.png)

![RQ2 paired fixed-search effects](rq2_mcts_vh_off.png)

### Progressive widening (PW70; Stage 1 confirmation)

{md_table(['Domain', 'n', 'Policy', 'Fixed MCTS 30m / 2h / 6h', 'PW70 30m / 2h / 6h', 'PW−policy [95% CI] at 30m / 2h / 6h', 'Raw/Holm p at 30m / 2h / 6h'], pw2_rows)}

**Conclusion:** PW70 gives large, corrected-significant FO Counters gains already at 30 minutes. Rover is approximately fixed-search parity, without a significant policy gain. These were the only cells promoted to ten seeds: the eight-seed Drone Kmin=3 extension and two-seed corrected Block Grouping screens lost fixed-search coverage, while the five-seed Counters confirmation did not establish a reliable advantage. The earlier accidental PW20 Block Grouping/Counters screen remains documented separately and is never pooled with PW70.

| Screen not promoted | Method | n | Policy | Fixed 30m / 2h / 6h | PW 30m / 2h / 6h | Decision |
|---|---|---:|---:|---:|---:|---|
| Drone, Kmin=3 | PW70 | 8 | 7.0/20 | 10.25 / 10.5 / 10.5 | 9.5 / 9.5 / 9.5 | Better runtime tail, but lost eight matched fixed successes |
| Block Grouping/off | PW20 | 2 | 16.5/20 | 11.0 / 13.5 / 15.0 narrow | 11.5 / 15.0 / 15.0 | Fixed parity only at 6h |
| Block Grouping/on | PW20 | 2 | 17.0/20 | 11.5 / 13.5 / 17.0 narrow | 12.5 / 17.0 / 18.0 | No runtime gain; tiny final mean gain |
| Block Grouping/off | PW70 | 2 | 16.5/20 | 11.0 / 13.5 / 15.0 narrow | 10.5 / 11.5 / 13.0 | Lost fixed coverage |
| Block Grouping/on | PW70 | 2 | 17.0/20 | 11.5 / 13.5 / 17.0 narrow | 9.0 / 11.5 / 13.5 | Lost fixed coverage |
| Counters S1/off | PW20 | 2 | 18.0/59 | 21.5 / 21.5 / 21.5 narrow | 20.5 / 20.5 / 20.5 | Slightly below fixed |
| Counters S1/on | PW20 | 2 | 5.0/59 | 13.5 / 13.5 / 13.5 narrow | 12.5 / 12.5 / 12.5 | Slightly below fixed |
| Counters S2/off | PW20 | 2 | 49.0/59 | ≥37.0 / ≥41.5 / 44.5 narrow | 46.0 / 49.5 / 49.5 | Restored policy mean in this screen |
| Counters S2/on | PW20 | 2 | 5.0/59 | 16.5 / 17.5 / 17.5 narrow | 15.0 / 15.0 / 15.0 | Below fixed |
| Counters S2/off | PW70 | 5 | 37.8/59 | 34.6 / 36.4 / 36.4 narrow | 29.4 / 33.6 / 36.4 | Fixed parity only at 6h; below policy |
| Counters S2/on | PW70 | 5 | 32.4/59 | 27.4 / 34.2 / 35.6 narrow | 22.0 / 28.4 / 31.4 | Below policy and fixed |

![RQ2/RQ4 PW70 confirmation](rq2_rq4_pw70_final.png)

## RQ3 — Does the value head improve Stage-2 policy refinement?

The direct column answers whether VH-on Stage 2 improves its own VH-on Stage-1 policy. The interaction asks whether that refinement is better than the parallel VH-off refinement.

{md_table(['Domain', 'VH-on raw S1 → S2', 'VH-on direct Δ [95% CI]; raw/Holm p', 'VH-off Δ', 'Interaction Δ [95% CI]; raw/Holm p'], rq3_rows)}

**Conclusion:** Neither the direct VH-on changes nor the interactions show a corrected-significant benefit. Block Grouping is the clearest harmful tendency; the DiD does not hide a beneficial VH-on result.

![RQ3 raw means and interaction](rq3_raw_means_and_interaction.png)

![RQ3 direct VH-on Stage-2 effect](rq3_value_head_training.png)

## RQ4 — Does the value head change the benefit of MCTS inference?

These three estimands are deliberately separate.

As in RQ2, these fixed-search values retain the historical action-index tie-break; the live policy-prior experiment is reported separately until its same-build confirmation completes.

### A. VH-on MCTS versus its own VH-on policy

{rq4_table('VH-on direct')}

### B. VH-on MCTS versus the parallel VH-off policy

{rq4_table('parallel VH-off policy')}

### C. Difference in MCTS benefit: VH-on minus VH-off

{rq4_table('VH interaction')}

**Conclusion:** Drone is the robust RQ4 success: VH-on materially increases MCTS usefulness. FO Counters benefits strongly from search in both VH modes, but its exact interaction is not significant—VH-on does not add a reliable extra gain there. Other domains do not show a reliable value-head interaction.

![RQ4 raw six-hour levels](rq4_raw_means_6h_by_stage.png)

![RQ4 direct effect](rq4_direct.png)

![RQ4 comparison against parallel VH-off policy](rq4_cross_cell.png)

![RQ4 interaction](rq4_interaction.png)

### PW70 contribution to RQ4 (Stage 1 confirmation)

{md_table(['Domain', 'Estimand', 'Raw means (policy; PW 30m / 2h / 6h)', 'Effect [95% CI] at 30m / 2h / 6h', 'Raw/Holm p at 30m / 2h / 6h'], pw4_rows)}

**Conclusion:** PW preserves the distinction seen with fixed search: FO Counters has strong search gains. Its observed VH-off benefit is no smaller than its VH-on benefit, but that interaction is not significant. Rover shows parity-scale, non-significant effects. PW therefore strengthens RQ2 more than RQ4.

## Results still required

1. **MPrime anchor:** 466/588 checkpoint validations are durable. Six recovery tasks are running; the original attempts have left the queue. Recheck `21233927` is dependency-pending and submits only still-missing identities before an analysis-only finalizer. Stage-2 training remains gated on complete curves and manual coefficient review. Sixteen old lineages definitely require retraining; four remain exact-identity reuse candidates.
2. **Counters tie-break:** four strict-confirmation tasks are terminal and 16 are running. The two completed matched seed pairs are exact ties: seed `1073581256` scores 59/59 under both rules, and seed `2011206605` scores 20/59 under both rules. This is neutral interim evidence, not a ten-seed conclusion. The targeted pilot remains action-ID 0/3, Q 0/3 and policy-prior 3/3 by two hours, all VAL-valid. Its VH-on 0/3 arm is not a positive control because the exact VH-on policy solved none of those targets.
3. **Block Grouping tie-break transfer:** both same-build four-target arms are running. Each has classified the first three targets as ordinary unsolved outcomes (0/3); the fourth remains active. The targets are genuine policy-success/MCTS failures--four of the 16 such VH-off losses across ten seeds--but no transfer rescue has yet appeared.
4. **MPrime Stage-1 search:** all 20 canonical fixed-MCTS evaluations are running as arrays `21237328`/`21237329`. Current durable successes are lower bounds of at least 6.8/20 VH-off and 7.2/20 VH-on; no seed is terminal, so no CI, cutoff comparison or RQ conclusion is yet valid. A two-seed Stage-1 PW70 screen can follow after the fixed comparator becomes interpretable; Stage-2 MCTS/PW still requires corrected Stage-2 endpoints.

Canonical evidence files: [`rq_primary_validation_led.csv`](rq_primary_validation_led.csv), [`rq2_raw_means_validation_led.csv`](rq2_raw_means_validation_led.csv), [`rq3_raw_means_validation_led.csv`](rq3_raw_means_validation_led.csv), [`rq4_raw_means_validation_led.csv`](rq4_raw_means_validation_led.csv), [`rq2_pw70_branch_latest.csv`](rq2_pw70_branch_latest.csv), and [`rq4_pw70_branch_latest.csv`](rq4_pw70_branch_latest.csv). Their row-level job/log routes are indexed in [`../../result_csv_provenance_index_latest.csv`](../../result_csv_provenance_index_latest.csv).
"""

REPORT.write_text(text, encoding="utf-8")
print(REPORT)
