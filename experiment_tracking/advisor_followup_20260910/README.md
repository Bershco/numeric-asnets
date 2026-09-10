# Advisor follow-up notes — 10 September 2026

This package records the decisions and follow-up questions from the 10 September
advisor meeting. It uses local authoritative result/provenance ledgers plus a
targeted live-cluster refresh. The current two-job Counters diagnostic, three
minimal FO Counters recoveries and two-lineage MPrime tail are recorded below.

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

The 30-minute, two-hour and six-hour columns are cumulative instances solved by
those **per-instance wall-clock cutoffs within the same declared six-hour
evaluation**, not separate runs and not whole-job elapsed time. Tables use
solved-instance counts (20 instances per domain, except Counters with 59); the
forest plots use percentage points so domains remain visually comparable.

Each cutoff cell is `MCTS mean; change [95% CI]; Holm p`.
Because the FO Counters Stage-2 cell is still censored, Stage-2 RQ2/RQ4 Holm
values are provisional adjustments among the four complete domains. The final
five-domain family will be frozen only after the FO recovery completes.

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
| S2 FO Counters | 2.9 | ≥6.0; ≥+3.1 | ≥6.0; ≥+3.1 | ≥6.0; ≥+3.1 | All ten VH-off identities ran; eight are complete and two are partial lower bounds |

**RQ2 answer:** MCTS is domain-dependent. It is strongly useful for Stage-1
FO Counters and modestly positive for Drone/Rover. In Block Grouping it is
harmful at 30 minutes and only approaches parity by six hours. Counters has
negative descriptive means and documented policy-success losses, but no
significant average effect. MCTS is not a universal inference replacement.

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

Three views are mandatory:

1. **VH-on direct:** `VH-on MCTS − the same VH-on checkpoint's policy score`.
2. **Parallel-cell interaction:** `(VH-on MCTS benefit) − (VH-off MCTS benefit)`.
3. **Cross-cell level check:** `VH-on MCTS − parallel VH-off policy`. This is
   not a causal value-head effect, but it verifies that a reported VH-on MCTS
   gain is not merely an artefact of comparing against a degraded VH-on policy.

Each complete cell is `VH-on direct / VH interaction / cross-cell level`, in
solved-instance units. Full 95% CIs and raw/Holm p-values are in
`rq_primary_validation_led.csv`.

| Stage/domain | 30 minutes | 2 hours | 6 hours | Answer |
|---|---:|---:|---:|---|
| S1 Block Grouping | −3.9 / +0.8 / −4.3 | −1.9 / −0.4 / −2.3 | +0.3 / +1.2 / −0.1 | VH-on eventually reaches policy parity; no reliable interaction |
| S1 Drone | +4.9 / +3.9 / +4.1 | +5.3 / +4.3 / +4.5 | +5.3 / +4.3 / +4.5 | Large direct gain and interaction; also exceeds VH-off policy |
| S1 FO Counters | +1.6 / −1.7 / +1.1 | +2.0 / −1.6 / +1.5 | +2.0 / −1.6 / +1.5 | MCTS helps VH-on, but VH-off gains more |
| S1 Rover | +0.6 / −0.2 / +0.4 | +0.6 / −0.4 / +0.4 | +0.6 / −0.4 / +0.4 | Small direct gain; no reliable interaction |
| S1 Counters | +1.7 / +9.3 / −12.2 | +3.5 / +10.4 / −10.4 | +3.9 / +10.7 / −10.0 | Positive direct effect but still below the stronger VH-off policy |
| S2 Block Grouping | −2.7 / +1.9 / −5.9 | −2.2 / −1.2 / −5.4 | −0.2 / +0.1 / −3.4 | No MCTS benefit; remains below VH-off policy |
| S2 Drone | +5.9 / +5.1 / +4.2 | +6.2 / +5.2 / +4.5 | +6.2 / +5.2 / +4.5 | Large direct gain and interaction; cross-cell gain is positive but not Holm-significant |
| S2 Rover | +0.5 / 0.0 / +0.4 | +0.6 / +0.1 / +0.5 | +0.6 / +0.1 / +0.5 | Small direct gain; no reliable interaction |
| S2 Counters | +0.8 / +2.8 / −14.3 | +4.6 / +4.8 / −10.5 | +5.3 / +5.5 / −9.8 | Positive direct effect but still descriptively below VH-off policy |
| S2 FO Counters | ≥+2.2 / indeterminate / ≥+2.4 | ≥+2.3 / indeterminate / ≥+2.5 | ≥+2.3 / indeterminate / ≥+2.5 | Both direct and cross-cell lower bounds are positive; interaction awaits exact recovery |

At six hours, the Stage-2 Drone interaction is +5.2 plans, 95% CI
[3.24, 7.16], Holm p=.008. The direct VH-on gain is +6.2 plans,
95% CI [4.33, 8.07], Holm p=.008. This is the clearest RQ4 result.

**RQ4 answer:** the value head materially increases the benefit of MCTS in
Drone, but not generally. At Stage 1, FO Counters gains in both modes and the
VH-off gain is larger. Stage-2 FO lower bounds are positive in both modes, but
their interaction is not identifiable until the three exact recoveries finish.
Counters has a large but unstable interaction because its VH-off search often
degrades a strong policy; its VH-on MCTS level remains below VH-off policy.

## 3. External generator comparison

The active comparison is limited to FO Counters and Rover and measures the
actual PDDL distributions, not filenames. `generator_distribution_instances.csv`
contains every parsed test/validation/frozen instance.

| Domain/distribution | Main size | Other structural evidence | Distance from test | Interpretation |
|---|---:|---|---:|---|
| FO test | 2–21 counters; mean 11.50 | all initial values zero; `max_int=2n`; ordered chain | 0/10 | Reference |
| FO thesis validation | 2–16; mean 7.53 | 95.9% initial values nonzero; tiered max; shuffled chain | 6.8/10 | Smaller and structurally different from test |
| FO Yarin frozen | 2–20; mean 8.85 | 98.5% initial values nonzero; fixed max 42; ordered chain | 5.4/10 | Closer goal order/range, but still a strong initial-state shift |
| Rover test | 1–8 rovers; 4–25 waypoints | means 3.75 and 9.50; graph grows through suite | 0/10 | Reference |
| Rover thesis validation | 1–5; 4–20 | means 2.60 and 9.03; connected/reachable safeguards | 3.2/10 | Good central overlap; under-covers largest test tail |
| Rover Yarin frozen | 1–4 rovers; 4–8 waypoints | means 2.40 and 6.50; visible edges mean 28.8; traverse edges mean 25.2 | qualitative strong shift | Fewer objects and smaller graphs; normalized traversal density is only somewhat below thesis validation |

The earlier decimal “distance” scores were an undocumented judgment scale and
are not used as quantitative evidence. The reproducible comparison is the raw
object-count, state/topology, goal-construction and support/tail data in
`generator_distribution_instances.csv` and `generator_distribution_summary.csv`.
Engineering readiness remains a separate score in `generator_comparison.csv`.

Important mapping correction: Yarin's `counters_generator.py` declares
`fo-counters-rnd`; it maps to **FO Counters**, not the separate `fn-counters`
domain called Counters in the thesis.

Yarin's Rover generator uses the standard IPC location predicate `at`, while
this numeric Rover domain calls the same rover/waypoint predicate `in`.  The
frozen external set applies only this token-boundary rename.  Objects,
topology, numeric energy, initial facts and goals remain generator output, so
the translation removes a vocabulary incompatibility without narrowing the
intended distribution shift.

The strongest low-cost empirical follow-up is not retraining. Freeze one
external set for FO Counters and Rover, run the static audit, then evaluate
existing Stage-1 validation-selected checkpoints. A three-seed, two-VH screen
requires 12 policy-evaluation tasks. Expand to all ten seeds only if the external
distribution materially changes a conclusion.

Detailed code-level findings and source links are in `generator_comparison.csv`.

The completed three-seed policy screen is:

| Domain | VH | Thesis test | External frozen set | Paired change [95% CI] | Exact p | Reading |
|---|---|---:|---:|---:|---:|---|
| FO Counters | off | 3.33/20 | 5.67/20 | +2.33 [−1.46, 6.13] | .250 | Encouraging shift, underpowered |
| FO Counters | on | 3.67/20 | 5.00/20 | +1.33 [−6.25, 8.91] | .750 | Uncertain |
| Rover | off | 4.00/20 | 0.33/20 | −3.67 [−5.10, −2.23] | .250 | Large descriptive loss |
| Rover | on | 4.00/20 | 1.33/20 | −2.67 [−4.10, −1.23] | .250 | Large descriptive loss |

With only three paired seeds, a two-sided exact sign-flip test cannot be below
.25. The confidence intervals describe seed variability, but no confirmatory
significance claim is made.

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

Not exactly for ordinary historical runs. This limitation applies to runs made
before the new instrumentation. They record final elapsed time and
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
cumulative 30-minute, two-hour and six-hour summaries. Therefore future opted-in
six-hour runs need no separate 30-minute rerun. Historical ordinary logs cannot
be retroactively upgraded to exact step-at-30m traces; they support final step
counts and success-by-cutoff only.

## 6. Why Counters MCTS can be worse than policy

The current evidence is stronger than a generic speculation:

- Across all ten validation-led Stage-2 VH-off seeds, there are 12 instances
  solved by policy but not by fixed narrow MCTS: eleven ordinary unsolved
  trajectories and one unclassified instance from an interrupted allocation.
- The detailed trajectory reconstruction covers those twelve: the eleven
  classified failures diverge from policy on the **first external action** and
  then reach exactly 10,000 actions unsolved.
- Across the parallel ten VH-on seeds, there are another 13 policy-success/MCTS-
  failure instances, all ordinary unsolved. MCTS nevertheless improves the
  VH-on aggregate because it also adds more new successes elsewhere.
- No policy-success loss in either mode is explained by a six-hour per-instance
  timeout. All printed plans elsewhere in the audit are VAL-valid.

The global seed-level join and every original policy/MCTS job path are frozen in
`counters_policy_mcts_failure_global.csv`; the trajectory-level actions for the
VH-off failure cells remain in the earlier audit.

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

Specifically, the two proposed jobs distinguish three explanations: (a) the
network policy strongly prefers the successful action but 20 coarse visits
select another action; (b) the network itself is already ambiguous or wrong at
the root; or (c) Q/value evidence deliberately overturns the policy. The matched
VH-on arm tests whether the pattern is specific to the failing VH-off cell.

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
| Counters visit audit, VH-off failures | 1 | 2 | 120 GiB | 24h | Live as 21178320; three instances sequentially; ≤18h evaluation plus overhead |
| Counters visit audit, matched VH-on control | 1 | 2 | 120 GiB | 24h | Live as 21178321; three instances sequentially; ≤18h evaluation plus overhead |
| **Required total** | **2** | **4 concurrent** | **240 GiB concurrent** | — | About 80 CPU-hours and at most 4,800 GiB-hours if both use 20h |

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
| Three-seed FO/Rover external-generator screen | 12 policy tasks across the successful FO and Rover arrays | Complete | Results and original logs are in `yarin_external_results_latest.csv` |
| Ten-seed FO/Rover external-generator confirmation | 28 additional policy tasks | No | Only if the screen changes conclusions; reuses the first 12 |
| FO Stage-2 validation-led exact recovery | 3 minimal MCTS jobs | Live | Jobs 21178377/79/80 run only the 42 instances missing from partial jobs 20943885/845/846. |

Thus the meeting notes produced **two diagnostic MCTS jobs** and, after the
historical FO audit proved three identities genuinely partial, **three minimal
completion jobs**. The external-generator screen is already complete. No
terminal-led continuation belongs in any of these totals.

## Files

- `rq_primary_validation_led.csv`: all RQ1–RQ4 direct and parallel-cell
  estimands, all cutoffs, CIs, raw/Holm p-values and provenance.
- `rq1_stage2_training_vh_off.*`, `rq2_mcts_vh_off.*`,
  `rq3_value_head_training.*`, `rq4_value_head_mcts.*`: RQ-separated plots.
- `rq4_direct.*`, `rq4_cross_cell.*`, `rq4_interaction.*`: readable RQ4
  subplots for the three required estimands; the combined RQ4 figure remains
  available for overview use.
- `terminal_led_archive_index.csv`: retained terminal-led evidence and the
  explicit no-continuation rule.
- `generator_comparison.csv`: generator mapping, scores, distance and code links.
- `proposed_job_plan.csv`: exact job counts/resources/time gates.
- `counters_visit_audit_manifest.csv`: exact two-arm Counters diagnostic
  configuration, checkpoint sources, instances and historical log provenance.
- `counters_policy_mcts_failure_global.csv`: all twenty validation-led Stage-2
  Counters seed pairs, global policy-success/MCTS-failure counts and direct log
  provenance.
- `reproducibility_publication_manifest.csv`: Git publication map.

## 9. Targeted live update — 10 September, 21:52 IDT

- Adaptive-KL outlier `21144388` remains live (6 CPU, 48 GiB) with 98/100
  updates logged; the stable control is complete at 100/100. The controller has
  not changed the coefficient from 3, so this run currently tests another
  constant-anchor trajectory rather than a genuinely different KL regime.
- Adaptive-KL stable control `21144389` completed all 100 epochs. The outlier's
  epoch-0 test result remained 10/20, so the controller did not repair the first
  update because it did not intervene.
- MPrime Phase B has 2,243/2,260 checkpoint-replicate results.
  Seventeen exact evaluations remain across `stage1-on-1972442430` and
  `validation_led-off-1472491096`. Fifty-eight of sixty lineages are complete.
- Array `21178405[15,42]` durably added two results and then failed because the
  evaluator imported `post_training` from the wrong checkout. The repaired
  exact tail `21178598[15,42]` is running with the intended isolated checkout
  first on `PYTHONPATH`. Array `21178356` first stopped
  because the clean checkout lacked the two generated domain modules; after the
  modules were copied and checksum-verified, `21178385` exposed concatenated
  multiprocess plan/output lines that the validator could not parse. The parser
  was repaired and smoke-tested before the current resubmission. Neither failed
  attempt repeated checkpoint inference that was already complete.
- FO/off PW70 recovery job `21157787` correctly evaluated `instance_15.pddl`
  with one worker and the full 120 GiB, but timed out unsolved at 21,603.5 s.
  The affected seed therefore remains 7/20 and the ten-seed mean remains
  8.40/20.
- External-generator preparation job `21175928` completed in 70 seconds and
  froze 20 FO Counters plus 20 Rover instances.  The first policy array
  `21175932` failed before inference because the isolated worktree lacked the
  ignored compiled TensorFlow operator; no scientific result was produced.
  After linking the production-compatible operator, array `21175984` ran:
  its six FO Counters cells completed.  Its six Rover cells exposed the upstream
  `at` versus numeric-domain `in` vocabulary mismatch and failed before
  evaluation.  The vocabulary-only repair is commit `44397bc8`; Rover-only
  preparation job `21176013` and replacement array `21176016[6-11]` then both
  completed successfully.  No MCTS or retraining was involved.
- The six FO Counters screen cells subsequently completed in 2m58s–4m50s.
  On the three matched seeds, VH-off changed from 3.33/20 on the thesis test
  set to 5.67/20 externally; VH-on changed from 3.67/20 to 5.00/20.  This is
  useful distribution-sensitivity screening, not a confirmatory claim: the
  external set differs in initial values and size sampling, and n=3 per mode.
  Per-seed scores and both original/external log paths are frozen in
  `yarin_external_results_latest.csv`.
- The six Rover screen cells completed in 2m32s–5m19s.  VH-off changed from
  4.00/20 on the thesis test set to 0.33/20 externally; VH-on changed from
  4.00/20 to 1.33/20.  This is a large three-seed distribution-shift signal,
  on an external set with fewer objects and smaller graphs. Its normalized
  traversal density is only somewhat below thesis validation, so this screen
  does not identify graph sparsity as the cause. It is not evidence that the
  generator is defective or a ten-seed estimate.
- Counters visit-audit jobs `21178320` and `21178321` are live. They run only
  instances 51, 55 and 59, with one worker, 2 CPU, 120 GiB and a 24-hour
  allocation each. They log priors, visits, Q/U and selected actions at every
  root for exact 30m/2h/6h distribution summaries.
- The three FO validation-led Stage-2 records were rechecked and are genuinely
  partial: jobs `20943885`, `20945845` and `20945846` leave 15, 13 and 14
  instances unclassified. Corrected minimal jobs `21178377`, `21178379` and
  `21178380` run only those 42 instances. The first submission attempts stopped
  after roughly 90 seconds because old completion ledgers encoded a different
  test-order signature; those attempts produced no result and the corrected
  jobs use explicit skip lists instead.

## 10. MPrime Phase-B interim scientific result

The two independent harder validation replicates remove the old Stage-2
30/30 saturation, but their checkpoint-ranking quality is still limited.
Values below use only fully evaluated lineages; the final two can change the
Stage-1/on and Stage-2/off rows slightly.

| Branch | VH | Complete | Replicate rank agreement | Validation–test rank agreement | Mean selection regret |
|---|---|---:|---:|---:|---:|
| Stage 1 | off | 10/10 | .547 | .308 | 1.90 plans |
| Stage 1 | on | 9/10 | .710 | .509 | 1.67 plans |
| Validation-led Stage 2 | off | 9/10 | .275 | .131 | 1.67 plans |
| Validation-led Stage 2 | on | 10/10 | .241 | .018 | 2.30 plans |

Here, replicate agreement is the mean within-lineage Spearman correlation
between the two frozen sets across saved checkpoints. Validation–test agreement
is the same rank correlation against retrospective test-policy scores.
Selection regret is the number of test instances lost by the validation-chosen
checkpoint relative to the retrospectively best saved checkpoint; it measures
selector weakness, not a deployable oracle. The interim conclusion is that the
new sets solve saturation but Stage-2 ranking, especially VH-on, remains weak.
Row-level checkpoint, job and log provenance is in
`mprime_validation_phase_b_20260906/phase_b_checkpoint_scores_latest.csv`.
