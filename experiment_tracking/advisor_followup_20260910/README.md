# Advisor follow-up notes — 10 September 2026

This package records the decisions and follow-up questions from the 10 September
advisor meeting. It was built from local authoritative result and provenance
ledgers; no cluster refresh was needed. No MCTS job was submitted.

## 1. Primary analysis is validation-led

**Decision:** validation-led is the only primary branch from this date onward.
Completed terminal-led results remain available as an archived sensitivity
analysis, but are excluded from primary plots, tables, training continuations,
policy evaluations, and MCTS evaluations.

This is not a major methodological mistake. Validation-led answers the
deployable question: choose a Stage-1 source and a Stage-2 checkpoint without
looking at the test set. Terminal-led asks a different robustness question and
approximately doubles the factorial campaign. Keeping both as co-primary
branches makes the narrative harder to interpret and spends substantial compute
on an estimand we would not deploy.

The genuine risk is that a validation distribution may rank checkpoints poorly
or saturate. The correct remedy is to audit and improve validation—as is already
being done for MPrime Phase B—not to keep a terminal-led branch indefinitely.
Completed terminal-led evidence is not deleted. Its rows, jobs and logs remain
traceable through `terminal_led_archive_index.csv`.

## 2. Presentation contract: one section per RQ

The primary statistical unit is the matched training seed. Confidence intervals
are paired 95% t-intervals. P-values are two-sided exact sign-flip tests and are
Holm-adjusted across domains within each RQ/stage/cutoff/estimand family.

### RQ1 — Does MCTS-guided Stage-2 training improve policy coverage without a value head?

The direct estimand is `Stage-2 VH-off policy − Stage-1 VH-off policy`, using
validation-selected checkpoints at both stages.

| Domain | Stage 1 | Stage 2 | Change [95% CI] | Raw / Holm p | Answer |
|---|---:|---:|---:|---:|---|
| Block Grouping | 16.3/20 | 16.0/20 | −0.3 [−1.20, 0.60] | .625 / 1.000 | No reliable change |
| Drone | 5.9/20 | 6.7/20 | +0.8 [−1.27, 2.87] | .504 / 1.000 | No reliable change |
| FO Counters | 4.2/20 | 2.9/20 | −1.3 [−2.37, −0.23] | .047 / .234 | Raw decline; not family-significant |
| Rover | 4.0/20 | 4.0/20 | 0.0 [0.0, 0.0] | 1.000 / 1.000 | No change |
| Counters | 32.5/59 | 36.9/59 | +4.4 [−17.44, 26.24] | .660 / 1.000 | Positive mean, extreme seed variance |

**RQ1 answer:** no domain provides Holm-significant evidence that Stage-2
training improves the VH-off policy. Counters is promising in mean but too
variable to support a reliable claim.

### RQ2 — Does inference-time MCTS improve coverage without a value head?

The direct estimand is `VH-off MCTS − the same checkpoint under policy-only
inference`. This deliberately excludes VH-on; that belongs to RQ4.

Each cutoff cell is `MCTS mean; change [95% CI]; Holm p`.

| Stage/domain | Policy | 30 minutes | 2 hours | 6 hours | Answer |
|---|---:|---|---|---|---|
| S1 Block Grouping | 16.3 | 11.6; −4.7 [−5.53,−3.87]; .010 | 14.8; −1.5 [−2.68,−0.32]; .125 | 15.4; −0.9 [−1.88,0.08]; .223 | Search needs the long budget merely to approach parity |
| S1 Drone | 5.9 | 6.9; +1.0 [0.05,1.95]; .188 | 6.9; +1.0 [0.05,1.95]; .148 | 6.9; +1.0 [0.05,1.95]; .223 | Small positive mean, not corrected-significant |
| S1 FO Counters | 4.2 | 7.5; +3.3 [2.04,4.56]; .016 | 7.8; +3.6 [2.20,5.00]; .020 | 7.8; +3.6 [2.20,5.00]; .020 | Strong significant gain, mostly realized by 30m |
| S1 Rover | 4.0 | 4.8; +0.8 [0.06,1.54]; .188 | 5.0; +1.0 [0.25,1.75]; .125 | 5.0; +1.0 [0.25,1.75]; .125 | Positive mean, not corrected-significant |
| S1 Counters | 32.5 | 24.9; −7.6 [−18.52,3.32]; .203 | 25.6; −6.9 [−17.95,4.15]; .250 | 25.7; −6.8 [−17.73,4.13]; .250 | Negative, highly variable |
| S2 Block Grouping | 16.0 | 11.4; −4.6 [−5.37,−3.83]; .008 | 15.0; −1.0 [−2.01,0.01]; .375 | 15.7; −0.3 [−0.89,0.29]; 1.000 | Neutral only at the full budget |
| S2 Drone | 6.7 | 7.5; +0.8 [−0.58,2.18]; .719 | 7.7; +1.0 [−0.43,2.43]; .391 | 7.7; +1.0 [−0.43,2.43]; .586 | Positive but not significant |
| S2 Rover | 4.0 | 4.5; +0.5 [−0.01,1.01]; .375 | 4.5; +0.5 [−0.01,1.01]; .375 | 4.5; +0.5 [−0.01,1.01]; .500 | Small neutral-positive effect |
| S2 Counters | 36.9 | 34.9; −2.0 [−6.93,2.93]; .719 | 36.7; −0.2 [−2.87,2.47]; 1.000 | 36.7; −0.2 [−2.87,2.47]; 1.000 | Aggregate parity hides important policy-only losses |
| S2 FO Counters | — | — | — | — | Validation-led cell is incomplete: five historical endpoint jobs remain missing |

**RQ2 answer:** MCTS is domain-dependent. It is strongly useful for Stage-1
FO Counters, modestly positive for Drone/Rover, unsafe at short budgets in Block
Grouping, and can be harmful in Counters. It is not a universal inference
replacement.

### RQ3 — Does the value head improve Stage-2 refinement?

Two views are mandatory:

1. **VH-on direct:** `Stage-2 VH-on policy − Stage-1 VH-on policy`.
2. **Parallel-cell interaction:** `(VH-on Stage-2 change) − (VH-off Stage-2 change)`.

| Domain | VH-on S1 → S2 | VH-on direct [95% CI]; Holm p | VH-off change | VH interaction [95% CI]; Holm p |
|---|---:|---|---:|---|
| Block Grouping | 15.9 → 12.8 | −3.1 [−5.16,−1.04]; .078 | −0.3 | −2.8 [−4.61,−0.99]; .088 |
| Drone | 5.1 → 5.0 | −0.1 [−1.34,1.14]; 1.000 | +0.8 | −0.9 [−2.88,1.08]; 1.000 |
| FO Counters | 3.7 → 3.1 | −0.6 [−1.50,0.30]; 1.000 | −1.3 | +0.7 [−0.26,1.66]; 1.000 |
| Rover | 3.8 → 3.9 | +0.1 [−0.31,0.51]; 1.000 | 0.0 | +0.1 [−0.31,0.51]; 1.000 |
| Counters | 18.6 → 21.8 | +3.2 [−6.15,12.55]; 1.000 | +4.4 | −1.2 [−25.87,23.47]; 1.000 |

**RQ3 answer:** Stage-2 training with the value head has no corrected-significant
benefit. Block Grouping shows the clearest harmful tendency. The direct table
confirms that the difference-in-differences result is not hiding a general
positive VH-on refinement effect.

### RQ4 — Does the value head change the benefit of inference-time MCTS?

Two views are mandatory:

1. **VH-on direct:** `VH-on MCTS − the same VH-on checkpoint's policy score`.
2. **Parallel-cell interaction:** `(VH-on MCTS benefit) − (VH-off MCTS benefit)`.

Each cell is `VH-on direct / VH interaction`, in solved-instance units. Full
95% CIs and raw/Holm p-values are in `rq_primary_validation_led.csv`.

| Stage/domain | 30 minutes | 2 hours | 6 hours | Answer |
|---|---:|---:|---:|---|
| S1 Block Grouping | −3.9 / +0.8 | −1.9 / −0.4 | +0.3 / +1.2 | VH-on eventually reaches parity; interaction not significant |
| S1 Drone | +4.9 / +3.9 | +5.3 / +4.3 | +5.3 / +4.3 | Large direct gain; interaction is also Holm-significant at all cutoffs |
| S1 FO Counters | +1.6 / −1.7 | +2.0 / −1.6 | +2.0 / −1.6 | MCTS helps VH-on, but less than VH-off |
| S1 Rover | +0.6 / −0.2 | +0.6 / −0.4 | +0.6 / −0.4 | Small direct gain; no VH interaction |
| S1 Counters | +1.7 / +9.3 | +3.5 / +10.4 | +3.9 / +10.7 | Large noisy contrast because VH-off MCTS regresses |
| S2 Block Grouping | −2.7 / +1.9 | −2.2 / −1.2 | −0.2 / +0.1 | No full-budget benefit or interaction |
| S2 Drone | +5.9 / +5.1 | +6.2 / +5.2 | +6.2 / +5.2 | Large significant direct gain and interaction at every cutoff |
| S2 Rover | +0.5 / 0.0 | +0.6 / +0.1 | +0.6 / +0.1 | Small direct gain; no VH interaction |
| S2 Counters | +0.8 / +2.8 | +4.6 / +4.8 | +5.3 / +5.5 | Positive but highly variable; not significant |
| S2 FO Counters | — | — | — | Awaiting five missing validation-led endpoint evaluations |

At six hours, the Stage-2 Drone interaction is +5.2 plans, 95% CI
[3.24, 7.16], Holm p=.008. The direct VH-on gain is +6.2 plans,
95% CI [4.33, 8.07], Holm p=.008. This is the clearest RQ4 result.

**RQ4 answer:** the value head materially increases the benefit of MCTS in
Drone, but not generally. FO Counters gains from MCTS in both modes, with the
larger gain actually occurring without the value head. Counters has a large but
unstable interaction because its VH-off search often degrades a strong policy.

## 3. External generator comparison

The referenced repository overlaps with only three of the nine thesis domains:

| Thesis domain | Yarin file | Mapping | Ours / Yarin score | Distribution distance | Verdict |
|---|---|---|---:|---:|---|
| FO Counters | `counters_generator.py` | Direct | 8.0 / 5.5 | 3/10 | Closest match; suitable after adding seeds and our audit |
| Rover | `rovers_generator.py` | Direct | 8.5 / 6.0 | 8/10 | Strong external distribution shift; useful after freezing rovergen version/command |
| Zenotravel | `zenotravel_generator.py` | Direct | 8.5 / 3.5 | 7/10 | Raw output is not evaluation-ready because of indexing/reproducibility defects |
| FO Counters variant | `complex_counters_generator.py` | Related, not same distribution | 8.0 / 4.5 | 9/10 | Future OOD stress test; do not pool with FO Counters |
| Other six domains | none | No overlap | — | — | No comparison available from this repository |

Scores are engineering/readiness scores, not claims about scientific truth:
validity/solvability safeguards (2), reproducibility (2), difficulty control
(2), schema/diversity (2), and audit/provenance (2). Distance is 0 for the same
sampling process and 10 for only sharing a domain name/schema.

Important mapping correction: Yarin's `counters_generator.py` declares
`fo-counters-rnd`; it maps to **FO Counters**, not the separate `fn-counters`
domain called Counters in the thesis.

The strongest low-cost empirical follow-up is not retraining. First freeze one
external test set per overlapping domain, run our static and ENHSP audit, then
evaluate existing Stage-1 validation-selected checkpoints. A three-seed screen
requires 18 policy-evaluation tasks. Expand to all ten seeds only if the external
distribution materially changes a conclusion.

Detailed code-level findings and source links are in `generator_comparison.csv`.

## 4. Git/reproducibility publication

Advisor-facing artifacts should be published to the dedicated repository:

<https://github.com/Bershco/numeric-asnets-thesis-artifacts>

The main development repository remains:

<https://github.com/Bershco/numeric-asnets>

The artifact repository should contain the selected source revision, compact
result tables, job/command manifests, log provenance, checksums, plotting and
statistics scripts, and the already Git-LFS-managed runtime image/checkpoints.
It should not duplicate every multi-gigabyte raw log. The exact publication map
is in `reproducibility_publication_manifest.csv`; every statistical CSV points
to a seed-level ledger containing training/evaluation job IDs and original log
paths.

## 5. Can historical logs tell us how many external actions occurred by 30 minutes?

Not exactly for ordinary historical runs. They record final elapsed time and
final step count, but not the elapsed wall time at each external action. Dividing
final steps by final runtime would assume constant throughput, which is false as
the retained search tree and successor-generation cost change.

The evaluation instrumentation is now prepared to record, for every opted-in
root decision:

- external step and elapsed wall time;
- raw network argmax/probability;
- selected action and its raw probability;
- every expanded child's visits, visit share, prior, Q and U;
- visit entropy and top-one/top-two visit margin.

`summarize_mcts_visit_distribution.py` converts one six-hour trace into exact
cumulative 30-minute, two-hour and six-hour summaries. Therefore no separate
30-minute rerun is required.

## 6. Why Counters MCTS can be worse than policy

The current evidence is stronger than a generic speculation:

- In the reconciled Stage-2 VH-off subset, there are 12 instances solved by the
  policy but not by fixed narrow MCTS.
- Eleven diverge from the policy at the **first external action** and then reach
  exactly 10,000 actions unsolved.
- Zero of those eleven is explained by the six-hour instance timeout.
- All printed plans elsewhere in the audit are VAL-valid.

The likely mechanism is an early search-induced policy displacement:

1. Policy-only inference takes the network argmax directly.
2. MCTS chooses the argmax of root visit counts after only 20 simulations over
   up to five retained children.
3. Twenty visits make the action ranking coarse. If Q values are close or weak,
   small exploration/prior differences can decide the visit winner.
4. Once the first chosen action differs, the policy is evaluated on off-policy
   Counters states. Repeated small deviations can create a long oscillating
   trajectory that exhausts 10,000 actions even though the original policy had
   a successful path.

This makes the advisors' “insufficient visit evidence” hypothesis plausible,
but it is not yet proven by the old logs. The proposed diagnostic measures the
visit margin, entropy, Q/U balance, policy rank of the selected action, and the
first point where MCTS departs from the policy.

## 7. What “30-minute versus six-hour distribution” should mean

The evaluator does **not** spend six hours deepening one root. Every external
action receives a fresh fixed-budget MCTS search, then the environment advances
and search is re-rooted. Consequently, the defensible comparison is between
distributions across root decisions observed by each elapsed-time milestone.

For 30m, 2h and 6h, report:

- number of external actions/root searches completed;
- policy-argmax versus MCTS-selected disagreement rate;
- selected action's policy rank;
- top-one visit share and top-one/top-two visit margin;
- visit entropy/effective concentration;
- Jensen-Shannon divergence between policy priors and visit shares;
- Q range, U range and whether U or Q dominates the winning action;
- child count, goal discovery and final outcome.

The same six-hour run supplies all three milestones, allowing paired within-run
comparisons without process/build confounding.

## 8. Minimal job count implied by these notes

### Required new diagnostic work

| Job | Slurm jobs | CPUs/job | RAM/job | Wall time/job | Expected use |
|---|---:|---:|---:|---:|---|
| Counters visit audit, VH-off failures | 1 | 2 | 120 GiB | 24h | Three instances sequentially; ≤18h evaluation plus overhead |
| Counters visit audit, matched VH-on control | 1 | 2 | 120 GiB | 24h | Three instances sequentially; ≤18h evaluation plus overhead |
| **Required total** | **2** | **4 concurrent** | **240 GiB concurrent** | — | About 40 CPU-hours and at most 2,400 GiB-hours if both use 20h |

One worker is intentional: it prevents interleaved per-action traces and worker
scheduling from confounding the timing distribution. The full 120 GiB is retained
rather than assuming memory scales linearly with worker count.

The two jobs are fully predeclared in `counters_visit_audit_manifest.csv`. Both
use seed 1963100312 and instances 51, 55 and 59. In the VH-off evidence, policy
inference solved these instances in 2,103, 2,385 and 2,683 external actions,
whereas narrow MCTS diverged at the first action and reached 10,000 actions.
The VH-on job uses the same instances and seed as a parallel-cell control.

### Optional or pre-existing work

| Scope | Tasks | Submit now? | Reason |
|---|---:|---|---|
| Three-seed external-generator screen | 18 policy tasks in 6 arrays | No | First repair/audit external Zenotravel and freeze manifests |
| Ten-seed external-generator confirmation | 60 policy tasks in 6 arrays | No | Only if the screen changes conclusions |
| Existing validation-led FO Stage-2 MCTS gap | 5 MCTS jobs | No | Needed eventually for complete Stage-2 RQ2/RQ4, but not created by these meeting notes |

Thus the meeting notes themselves imply **two new MCTS jobs**, not dozens. If
the optional generator-bias screen is approved later, the minimal first stage is
18 policy-only tasks. No terminal-led continuation belongs in either total.

## Files

- `rq_primary_validation_led.csv`: all RQ1–RQ4 direct and parallel-cell
  estimands, all cutoffs, CIs, raw/Holm p-values and provenance.
- `rq1_stage2_training_vh_off.*`, `rq2_mcts_vh_off.*`,
  `rq3_value_head_training.*`, `rq4_value_head_mcts.*`: RQ-separated plots.
- `terminal_led_archive_index.csv`: retained terminal-led evidence and the
  explicit no-continuation rule.
- `generator_comparison.csv`: generator mapping, scores, distance and code links.
- `proposed_job_plan.csv`: exact job counts/resources/time gates.
- `counters_visit_audit_manifest.csv`: exact two-arm Counters diagnostic
  configuration, checkpoint sources, instances and historical log provenance.
- `reproducibility_publication_manifest.csv`: Git publication map.
