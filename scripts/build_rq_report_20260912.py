#!/usr/bin/env python3
"""Build the validation-led RQ report from canonical local result CSVs."""

from __future__ import annotations

import csv
import itertools
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiment_tracking" / "advisor_followup_20260910"
PRIMARY = OUT / "rq_primary_validation_led.csv"
PW2 = OUT / "rq2_pw70_branch_latest.csv"
PW4 = OUT / "rq4_pw70_branch_latest.csv"
PRESERVE = ROOT / "experiment_tracking" / "four_domain_preservation" / "stable_domain_stage2_statistics_20260902.csv"
PRESERVE_SEEDS = ROOT / "experiment_tracking" / "four_domain_preservation" / "stable_domain_stage2_seed_pairs_20260902.csv"
REPORT = OUT / "rq_report_validation_led_20260912.md"

DOMAIN_LABEL = {
    "block_grouping": "Block Grouping",
    "drone": "Drone",
    "fo_counters": "FO Counters",
    "rover": "Rover",
    "counters": "Counters",
    "mprime": "MPrime",
}
DOMAIN_ORDER = list(DOMAIN_LABEL)
POLICY_DOMAIN_ORDER = list(DOMAIN_ORDER)
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
preserve = read_rows(PRESERVE)
preserve_seeds = read_rows(PRESERVE_SEEDS)


def paired_stats(diffs: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.mean(diffs)
    critical = {8: 2.364624251, 10: 2.262157163}.get(len(diffs), 1.96)
    half = critical * statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    observed = abs(mean)
    extreme = sum(
        abs(statistics.mean(sign * value for sign, value in zip(signs, diffs))) >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=len(diffs))
    )
    return mean, mean - half, mean + half, extreme / (2 ** len(diffs))


def holm_values(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * raw[key])
        adjusted[key] = min(1.0, running)
    return adjusted

rq1 = [r for r in primary if r["rq"] == "RQ1"]
rq3_direct = [r for r in primary if r["rq"] == "RQ3" and r["estimand"].startswith("VH-on direct")]
rq3_inter = [r for r in primary if r["rq"] == "RQ3" and "interaction" in r["estimand"]]

rq1_map = {r["domain"]: r for r in rq1}
rq3d_map = {r["domain"]: r for r in rq3_direct}
rq3i_map = {r["domain"]: r for r in rq3_inter}

rq1_rows = []
for domain in POLICY_DOMAIN_ORDER:
    if domain not in rq1_map:
        continue
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
for domain in POLICY_DOMAIN_ORDER:
    if domain not in rq3d_map or domain not in rq3i_map:
        continue
    d, i = rq3d_map[domain], rq3i_map[domain]
    rq3_rows.append([
        DOMAIN_LABEL[domain], f"{number(d['baseline_mean'])} → {number(d['comparison_mean'])}",
        effect_with_p(d),
        number(i["baseline_mean"]),
        effect_with_p(i),
    ])

preserve_rq1_rows = []
preserve_rq3_rows = []
preserve_heldout_off = {}
preserve_interactions = {}
for domain in ("delivery", "tpp", "zenotravel"):
    heldout = [r for r in preserve_seeds if r["domain"] == domain and r["value_head"] == "off" and r["seed_role"] == "held_out"]
    preserve_heldout_off[domain] = (
        statistics.mean(float(r["stage1_selected"]) for r in heldout),
        statistics.mean(float(r["stage2_selected"]) for r in heldout),
        *paired_stats([float(r["change"]) for r in heldout]),
    )
    by_seed = defaultdict(dict)
    for row in preserve_seeds:
        if row["domain"] == domain:
            by_seed[row["seed"]][row["value_head"]] = float(row["change"])
    preserve_interactions[domain] = paired_stats([
        pair["on"] - pair["off"] for pair in by_seed.values()
    ])
heldout_holm = holm_values({domain: values[-1] for domain, values in preserve_heldout_off.items()})
interaction_holm = holm_values({domain: values[-1] for domain, values in preserve_interactions.items()})
for domain in ("delivery", "tpp", "zenotravel"):
    off = next(r for r in preserve if r["domain"] == domain and r["value_head"] == "off")
    on = next(r for r in preserve if r["domain"] == domain and r["value_head"] == "on")
    label = {"delivery": "Delivery", "tpp": "TPP", "zenotravel": "Zenotravel"}[domain]
    h_s1, h_s2, h_delta, h_low, h_high, h_raw = preserve_heldout_off[domain]
    h_holm = heldout_holm[domain]
    preserve_rq1_rows.append([
        label, "10 (8 held-out + 2 tuning)", number(off["stage1_selected_mean"]),
        number(off["stage2_selected_mean_all10"]), f"{number(h_s1)} → {number(h_s2)}",
        f"{number(h_delta)} [{number(h_low)}, {number(h_high)}]; p={number(h_raw,4)}/{number(h_holm,4)}",
        f"{number(off['mean_change'])} [{number(off['ci95_low'])}, {number(off['ci95_high'])}]",
        f"{number(off['raw_sign_flip_p'],4)} / {number(off['holm_p'],4)}",
        off["note"] or "Preserved on average",
    ])
    off_change = float(off["mean_change"])
    on_change = float(on["mean_change"])
    inter, inter_low, inter_high, inter_raw = preserve_interactions[domain]
    inter_holm = interaction_holm[domain]
    preserve_rq3_rows.append([
        label, f"{number(on['stage1_selected_mean'])} → {number(on['stage2_selected_mean_all10'])}",
        f"{number(on['mean_change'])} [{number(on['ci95_low'])}, {number(on['ci95_high'])}]; p={number(on['raw_sign_flip_p'],4)}/{number(on['holm_p'],4)}",
        number(off_change),
        f"{number(inter)} [{number(inter_low)}, {number(inter_high)}]; p={number(inter_raw,4)}/{number(inter_holm,4)}",
        "Exploratory post-hoc interaction; separate 3-domain family",
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


pw_domains = [domain for domain in ("fo_counters", "rover", "mprime") if any(r["domain"] == domain for r in pw2)]
pw2_rows = []
for domain in pw_domains:
    group = {r["cutoff"]: r for r in pw2 if r["domain"] == domain}
    r6 = group["6h"]
    effects, pvalues, fixed_effects, fixed_pvalues = [], [], [], []
    for cutoff in ("30m", "2h", "6h"):
        r = group[cutoff]
        effect = f"{number(r['pw70_minus_policy'])} [{number(r['ci95_low'])}, {number(r['ci95_high'])}]"
        pvalue = f"{number(r['raw_p'], 4)}/{number(r['holm_p'], 4)}"
        if float(r["holm_p"]) < 0.05:
            effect, pvalue = f"**{effect}**", f"**{pvalue}**"
        effects.append(effect)
        pvalues.append(pvalue)
        fixed_effect = f"{number(r['pw70_minus_fixed'])} [{number(r['pw70_minus_fixed_ci95_low'])}, {number(r['pw70_minus_fixed_ci95_high'])}]"
        fixed_pvalue = f"{number(r['pw70_minus_fixed_raw_p'], 4)}/{number(r['pw70_minus_fixed_holm_p'], 4)}"
        if float(r["pw70_minus_fixed_holm_p"]) < 0.05:
            fixed_effect, fixed_pvalue = f"**{fixed_effect}**", f"**{fixed_pvalue}**"
        fixed_effects.append(fixed_effect)
        fixed_pvalues.append(fixed_pvalue)
    pw2_rows.append([
        DOMAIN_LABEL[domain], r6["n"], number(r6["vh_off_policy_mean"]),
        " / ".join(number(group[c]["vh_off_fixed_mcts_mean"]) for c in ("30m", "2h", "6h")),
        " / ".join(number(group[c]["vh_off_pw70_mean"]) for c in ("30m", "2h", "6h")),
        " / ".join(effects), " / ".join(pvalues),
        " / ".join(fixed_effects), " / ".join(fixed_pvalues),
    ])

pw4_rows = []
pw4_fixed_rows = []
for domain in pw_domains:
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
    direct_group = {r["cutoff"]: r for r in pw4 if r["domain"] == domain and r["estimand"] == "VH-on PW70 - VH-on policy"}
    fixed_effects, fixed_pvalues = [], []
    for cutoff in ("30m", "2h", "6h"):
        r = direct_group[cutoff]
        effect = f"{number(r['pw70_minus_fixed'])} [{number(r['pw70_minus_fixed_ci95_low'])}, {number(r['pw70_minus_fixed_ci95_high'])}]"
        pvalue = f"{number(r['pw70_minus_fixed_raw_p'],4)}/{number(r['pw70_minus_fixed_holm_p'],4)}"
        if float(r["pw70_minus_fixed_holm_p"]) < 0.05:
            effect, pvalue = f"**{effect}**", f"**{pvalue}**"
        fixed_effects.append(effect)
        fixed_pvalues.append(pvalue)
    pw4_fixed_rows.append([
        DOMAIN_LABEL[domain],
        " / ".join(number(direct_group[c]["vh_on_fixed_mcts_mean"]) for c in ("30m", "2h", "6h")),
        " / ".join(number(direct_group[c]["vh_on_pw70_mean"]) for c in ("30m", "2h", "6h")),
        " / ".join(fixed_effects), " / ".join(fixed_pvalues),
    ])

text = f"""# Validation-led RQ report — 16 September 2026

This is the primary thesis view. Terminal-led campaigns are excluded. Fixed-search 30-minute and two-hour figures are deterministic cutoffs of the same six-hour runs, not separate reruns. Block Grouping and Counters use narrow fixed search (5 retained children, 20 simulations); Drone, FO Counters and Rover use normal fixed search (20 children, 70 simulations). Counts are solved test instances; Counters has 59 instances and the other domains have 20.

Effects are seed-paired mean differences; confidence intervals are paired t-intervals; raw p-values are two-sided exact sign-flip tests; Holm correction is applied separately within each RQ × stage × cutoff × estimand family. Stage-1 fixed-search families now include six domains with final MPrime results; Stage 2 contains the five completed validation-led domains. PW uses separate families and is never pooled with fixed search. **Bold entries are Holm-significant at .05; raw-only significance is not bolded.**

All twenty clean MPrime Stage-2 training lineages are complete, but their 420 checkpoints are being rescored on the frozen Phase-B-A validator. MPrime therefore remains absent from RQ1/RQ3 and the Stage-2 portions of RQ2/RQ4 until that independent selector freezes the endpoints. Its completed Stage-1 fixed/PW results remain valid.

## RQ1 — Does Stage-2 training improve policy coverage without a value head?

{md_table(['Domain', 'Stage-1 policy', 'Stage-2 policy', 'Δ [95% CI]', 'Raw / Holm p'], rq1_rows)}

**Conclusion:** Stage 2 does not produce a Holm-significant VH-off policy improvement. FO Counters has a raw decline; Counters has a positive but highly variable mean.

### RQ1 extension: PRESERVE-3 validation-led domains

These stable-domain cells are reported separately from the primary five-domain multiplicity family. The paired held-out eight-seed contrast is the cleaner confirmation view; the all-ten means and inference are also shown so the two tuning seeds are not hidden.

{md_table(['Domain', 'n', 'Stage-1 all 10', 'Stage-2 all 10', 'Held-out 8: S1 → S2', 'Held-out Δ [95% CI]; raw/Holm p', 'All-10 Δ [95% CI]', 'All-10 raw/Holm p', 'Conclusion'], preserve_rq1_rows)}

**Extension conclusion:** Delivery and Zenotravel are preserved. TPP is preserved in nine of ten VH-off seeds, but one predeclared held-out seed collapses to 9/20; that outlier is a real seed-specific failure and is not hidden by the 18.9 mean.

![RQ1 paired Stage-2 effect](rq1_stage2_training_vh_off.png)

![PRESERVE-3 validation-selected seed robustness](../advisor_meeting_20260910/after_review/04_preserve3_validation_seed_robustness.png)

## RQ2 — Does inference-time search improve coverage without a value head?

### Fixed search

All fixed-search rows below use the historical final action-index tie-break. The live policy-prior candidate is a separate causal experiment and has not changed any RQ value in this table.

{md_table(['Stage', 'Domain', 'Search', 'Policy', '30m: MCTS; Δ [95% CI]; raw/Holm p', '2h', '6h'], rq2_rows)}

**Conclusion:** FO Counters is the clear fixed-search success at both stages and all three cutoffs; Drone and Rover improve modestly. Block Grouping needs the longer budget to approach parity, while Counters can be harmed. The final FO Stage-2 recovery timed out on its exact six-hour instance budget, so the exact mean remains 6.1/20 and the previously partial row is now final.

![RQ2 raw policy and fixed-search means](rq2_raw_means_by_stage.png)

![RQ2 paired fixed-search effects](rq2_mcts_vh_off.png)

### Progressive widening (PW70; Stage 1 confirmation)

{md_table(['Domain', 'n', 'Policy', 'Fixed MCTS 30m / 2h / 6h', 'PW70 30m / 2h / 6h', 'PW−policy [95% CI] at 30m / 2h / 6h', 'Raw/Holm p', 'PW−fixed [95% CI] at 30m / 2h / 6h', 'Raw/Holm p'], pw2_rows)}

**Conclusion:** PW70 gives large, corrected-significant VH-off FO Counters gains already at 30 minutes. Rover is approximately fixed-search parity, without a significant VH-off policy gain. MPrime VH-off PW70 significantly beats fixed search at every cutoff and exceeds its policy descriptively by 2h/6h. The corresponding VH-on results are reported under RQ4. These were the only cells promoted to ten seeds: the eight-seed Drone Kmin=3 extension and two-seed corrected Block Grouping screens lost fixed-search coverage, while the five-seed Counters confirmation did not establish a reliable advantage. The earlier accidental PW20 Block Grouping/Counters screen remains documented separately and is never pooled with PW70.

The following are **descriptive exploratory screens**, not confirmatory families. CIs/tests were intentionally withheld because these small cells selected which branches to promote; they are not forgotten results.

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

Screen provenance: `../mcts_progressive_widening_cross_domain/comparative_summary_20260901_1022.csv`, `../mcts_progressive_widening_cross_domain/comparative_summary_20260901_2304.csv`, `../mcts_progressive_widening_cross_domain/pw70_confirmation_summary_20260906.csv`, and `../mcts_progressive_widening_sensitivity/kmin3_runtime_summary.csv`.

![RQ2/RQ4 PW70 confirmation](rq2_rq4_pw70_final.png)

## RQ3 — Does the value head improve Stage-2 policy refinement?

The direct column answers whether VH-on Stage 2 improves its own VH-on Stage-1 policy. The interaction asks whether that refinement is better than the parallel VH-off refinement.

{md_table(['Domain', 'VH-on raw S1 → S2', 'VH-on direct Δ [95% CI]; raw/Holm p', 'VH-off Δ', 'Interaction Δ [95% CI]; raw/Holm p'], rq3_rows)}

**Conclusion:** Neither the direct VH-on changes nor the interactions show a corrected-significant benefit. Block Grouping is the clearest harmful tendency; the DiD does not hide a beneficial VH-on result.

### RQ3 extension: PRESERVE-3 validation-led domains

{md_table(['Domain', 'VH-on raw S1 → S2', 'VH-on direct Δ [95% CI]; raw/Holm p', 'VH-off Δ', 'Observed interaction', 'Scope'], preserve_rq3_rows)}

**Extension conclusion:** The stable domains do not supply evidence that the value head improves refinement. Their near-ceiling scores primarily test preservation; TPP/off's single catastrophic seed is investigated separately as an optimization-path failure.

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

Direct VH-on PW70 versus the corresponding VH-on fixed-search arm:

{md_table(['Domain', 'VH-on fixed 30m / 2h / 6h', 'VH-on PW70 30m / 2h / 6h', 'PW−fixed [95% CI] at 30m / 2h / 6h', 'Raw/Holm p'], pw4_fixed_rows)}

**Conclusion:** PW preserves the distinction seen with fixed search: FO Counters and MPrime have strong search gains, while Rover shows parity-scale, non-significant effects. MPrime VH-on PW reaches 18.5/20 at six hours, but its VH-on-versus-VH-off benefit interaction is not significant. PW therefore strengthens RQ2 more than RQ4.

## Results still required

1. **MPrime Stage 2:** all twenty clean lineages completed from the final Phase-B-A Stage-1 checkpoints: ten VH-off with anchor 30 and ten VH-on with anchor 10. The ordinary training validator saturated and cannot select the final endpoints. A twenty-lineage/420-checkpoint Phase-B-A rescore is live; MPrime enters RQ1/RQ3 and Stage-2 RQ2/RQ4 only after that independent selection and matched endpoint evaluation.
2. **Counters tie-break:** nine original strict-confirmation tasks completed, eight are running and three ended OOM. Recovery array `21308619[4,6,14]` requests 200 GiB per task and reuses the exact original durable ledgers; it therefore runs only the remaining 49 unclassified identities rather than repeating 177. Dependency controller `21319149` is now queued and will fail closed unless all action-ID ledgers reach exactly 59 terminal identities; it then traces only the exact policy-success/action-ID-failure union under action-ID and policy-prior with complete root vectors. The targeted causal pilot remains action-ID 0/3, Q 0/3 and policy-prior 3/3, all VAL-valid; its VH-on 0/3 arm is not a positive control because that policy solved none of the three targets.
3. **Block Grouping tie-break transfer:** both same-build four-target arms completed 0/4. Every outcome was an ordinary 10,000-action failure, not a timeout. Policy-prior tie-breaking therefore did not transfer to this selected seed, and the fixed-search branch will not be expanded from this negative screen.
4. **MPrime Stage-1 fixed search:** all 20 canonical evaluations are complete and now extend RQ2/RQ4. VH-off scores are 13.0 / 14.7 / 15.7 at 30m / 2h / 6h versus policy 16.3; VH-on scores are 13.3 / 15.1 / 16.0 versus policy 15.7.
5. **MPrime Stage-1 PW70:** complete for all twenty identities. VH-off is 15.7 / 17.6 / 17.7 and VH-on is 15.5 / 17.9 / 18.5 at 30m / 2h / 6h. It now appears in RQ2/RQ4 above.
6. **Block Grouping PW70 trace:** all four exact policy-success/historical-PW-timeout traces completed and timed out again. Three first divergences had a unique maximum-visit winner; the fourth had a three-way maximum tie but the policy action was not among it. The predeclared tie-transfer gate is negative: arbitrary action-ID tie-breaking is not the demonstrated cause of these PW losses.
7. **TPP first-update causal chain:** the complete crossover is bad checkpoint × stable replay = 3/20 and stable checkpoint × bad replay = 20/20, versus diagonal 10/20 and 20/20. Starting-checkpoint susceptibility is necessary in this frozen pair; replay identity changes severity. The obsolete coefficient-doubling attempt failed because Adam-normalized proposals did not shrink. Corrected fixed-anchor/LR-backtracking treatment completed at 20/20 for both the catastrophic seed and stable control; the bad seed triggered 11 backtracks and the stable control 4. This prevents the selected collapse without damaging the control, but is a two-seed mechanism screen outside the primary RQ family; retries resample dropout, so it is not an exact deterministic learning-rate-rescaling experiment.

Canonical evidence files: [`rq_primary_validation_led.csv`](rq_primary_validation_led.csv), [`rq2_raw_means_validation_led.csv`](rq2_raw_means_validation_led.csv), [`rq3_raw_means_validation_led.csv`](rq3_raw_means_validation_led.csv), [`rq4_raw_means_validation_led.csv`](rq4_raw_means_validation_led.csv), [`rq2_pw70_branch_latest.csv`](rq2_pw70_branch_latest.csv), and [`rq4_pw70_branch_latest.csv`](rq4_pw70_branch_latest.csv). Their row-level job/log routes are indexed in [`../result_csv_provenance_index_latest.csv`](../result_csv_provenance_index_latest.csv).
"""

# Keep volatile scheduler state out of the reproducible numerical report.  The
# canonical live status file is refreshed independently from Slurm.
text = text.split("## Results still required", 1)[0] + """## Live execution status

The numerical tables above are frozen until a complete new matched family is
available.  Current training, evaluation, recovery and controller state is
maintained in [`../status_latest.md`](../status_latest.md), so rebuilding this
report cannot resurrect an obsolete scheduler snapshot.

Canonical evidence files: [`rq_primary_validation_led.csv`](rq_primary_validation_led.csv), [`rq2_raw_means_validation_led.csv`](rq2_raw_means_validation_led.csv), [`rq3_raw_means_validation_led.csv`](rq3_raw_means_validation_led.csv), [`rq4_raw_means_validation_led.csv`](rq4_raw_means_validation_led.csv), [`rq2_pw70_branch_latest.csv`](rq2_pw70_branch_latest.csv), and [`rq4_pw70_branch_latest.csv`](rq4_pw70_branch_latest.csv). Their row-level job/log routes are indexed in [`../result_csv_provenance_index_latest.csv`](../result_csv_provenance_index_latest.csv).
"""

REPORT.write_text(text, encoding="utf-8")
print(REPORT)
