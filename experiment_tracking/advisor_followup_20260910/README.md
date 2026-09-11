# Advisor follow-up notes — refreshed 11 September 2026

This package records the decisions and follow-up questions from the 10 September
advisor meeting. It uses local authoritative result/provenance ledgers plus a
targeted live-cluster refresh. MPrime Phase B and both adaptive-KL training jobs
are now complete. The forty-task MPrime Phase-C validation audit and the six
matched Counters root-tie-break tasks are live; this document does not infer
their outcomes before their logs are terminal.

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

The primary RQ tables cover the five imperfect domains: Block Grouping, Drone,
FO Counters, Rover and Counters. The three ceiling-level PRESERVE-3 domains are
reported separately as preservation evidence, and MPrime remains outside the
primary RQs until its live Phase-C validation audit selects a defensible
checkpoint rule.

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

Search configuration is fixed by domain: Block Grouping and Counters use the
declared **narrow 5 children / 20 simulations** comparison; Drone, FO Counters
and Rover use **normal 20 children / 70 simulations**.

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
| S2 Counters | 36.9 | 34.9; −2.0 [−6.93,2.93]; .719 | 36.7; −0.2 [−2.87,2.47]; .969 | 36.7; −0.2 [−2.87,2.47]; 1.000 | Aggregate parity hides important policy-only losses |
| S2 FO Counters | 2.9 | ≥6.0; ≥+3.1 | ≥6.0; ≥+3.1 | ≥6.0; ≥+3.1 | All ten VH-off identities ran; eight are complete and two are partial lower bounds |

**RQ2 answer:** MCTS is domain-dependent. It is strongly useful for Stage-1
FO Counters and modestly positive for Drone/Rover. In Block Grouping it is
harmful at 30 minutes and only approaches parity by six hours. Counters has
negative descriptive means and documented policy-success losses, but no
significant average effect. MCTS is not a universal inference replacement.

#### RQ2 progressive-widening extension

The selected ten-seed Stage-1 extension uses `Kmin=3`, maximum width 20 and 70
simulations; fixed comparisons are normal 20/70.

| Domain | n | Policy | Fixed 30m / 2h / 6h | PW70 30m / 2h / 6h | PW − policy at 6h [95% CI] | Raw / Holm p |
|---|---:|---:|---:|---:|---:|---:|
| FO Counters | 10 | 4.2 | 7.5 / 7.8 / 7.8 | 8.4 / 8.4 / 8.4 | +4.2 [3.26,5.14] | .002 / .004 |
| Rover | 10 | 4.0 | 4.8 / 5.0 / 5.0 | 4.7 / 4.7 / 4.7 | +0.7 [0.02,1.38] | .125 / .125 |

PW strengthens RQ2 in FO Counters: the full gain is already visible by 30
minutes. Rover reaches approximate fixed-search parity without a reliable
policy improvement. These cells were expanded after screening; their Holm
correction is a separate two-domain PW family, not pristine globally
predeclared confirmation.

### RQ3 — Does the value head improve Stage-2 refinement?

Two views are mandatory:

1. **VH-on direct:** `Stage-2 VH-on policy − Stage-1 VH-on policy`.
2. **Parallel-cell interaction:** `(VH-on Stage-2 change) − (VH-off Stage-2 change)`.

| Domain | VH-off S1 → S2 | VH-off direct [95% CI]; Holm p | VH-on S1 → S2 | VH-on direct [95% CI]; Holm p | VH interaction [95% CI]; Holm p |
|---|---:|---|---:|---|---|
| Block Grouping | 16.3 → 16.0 | −0.3 [−1.20,0.60]; 1.000 | 15.9 → 12.8 | −3.1 [−5.16,−1.04]; .078 | −2.8 [−4.61,−0.99]; .088 |
| Drone | 5.9 → 6.7 | +0.8 [−1.27,2.87]; 1.000 | 5.1 → 5.0 | −0.1 [−1.34,1.14]; 1.000 | −0.9 [−2.88,1.08]; 1.000 |
| FO Counters | 4.2 → 2.9 | −1.3 [−2.37,−0.23]; .234 | 3.7 → 3.1 | −0.6 [−1.50,0.30]; 1.000 | +0.7 [−0.26,1.66]; .813 |
| Rover | 4.0 → 4.0 | 0.0 [0,0]; 1.000 | 3.8 → 3.9 | +0.1 [−0.31,0.51]; 1.000 | +0.1 [−0.31,0.51]; 1.000 |
| Counters | 32.5 → 36.9 | +4.4 [−17.44,26.24]; 1.000 | 18.6 → 21.8 | +3.2 [−6.15,12.55]; 1.000 | −1.2 [−25.87,23.47]; 1.000 |

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

The raw levels below prevent those effects from hiding the actual coverage.
Each MCTS entry is `30m / 2h / 6h`; Stage 2 is validation-led only.

As in RQ2, Block Grouping and Counters use narrow 5/20 MCTS; Drone, FO Counters
and Rover use normal 20/70 MCTS.

| Stage/domain | VH-off policy | VH-off MCTS | VH-on policy | VH-on MCTS |
|---|---:|---:|---:|---:|
| S1 Block Grouping | 16.3 | 11.6 / 14.8 / 15.4 | 15.9 | 12.0 / 14.0 / 16.2 |
| S1 Drone | 5.9 | 6.9 / 6.9 / 6.9 | 5.1 | 10.0 / 10.4 / 10.4 |
| S1 FO Counters | 4.2 | 7.5 / 7.8 / 7.8 | 3.7 | 5.3 / 5.7 / 5.7 |
| S1 Rover | 4.0 | 4.8 / 5.0 / 5.0 | 3.8 | 4.4 / 4.4 / 4.4 |
| S1 Counters | 32.5 | 24.9 / 25.6 / 25.7 | 18.6 | 20.3 / 22.1 / 22.5 |
| S2 Block Grouping | 16.0 | 11.4 / 15.0 / 15.7 | 12.8 | 10.1 / 10.6 / 12.6 |
| S2 Drone | 6.7 | 7.5 / 7.7 / 7.7 | 5.0 | 10.9 / 11.2 / 11.2 |
| S2 FO Counters | 2.9 | ≥6.0 / ≥6.0 / ≥6.0 | 3.1 | ≥5.3 / ≥5.4 / ≥5.4 |
| S2 Rover | 4.0 | 4.5 / 4.5 / 4.5 | 3.9 | 4.4 / 4.5 / 4.5 |
| S2 Counters | 36.9 | 34.9 / 36.7 / 36.7 | 21.8 | 22.6 / 26.4 / 27.1 |

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

At 30 minutes, the Stage-2 Block Grouping interaction is provisionally +1.9
plans, 95% CI [0.76, 3.04], Holm p=.047 among the four currently complete
domains. This means the value head attenuates some of the short-budget loss; it
does **not** make the raw Block Grouping MCTS level beneficial. The five-domain
Holm value waits for FO Counters recovery.

At six hours, the Stage-2 Drone interaction is +5.2 plans, 95% CI
[3.24, 7.16], provisional Holm p=.008 among the four complete domains. The
direct VH-on gain is +6.2 plans, 95% CI [4.33, 8.07], provisional Holm p=.008.
This is the clearest RQ4 result; final five-domain adjustment awaits FO.

**RQ4 answer:** the value head materially increases the benefit of MCTS in
Drone, but not generally. At Stage 1, FO Counters gains in both modes and the
VH-off gain is larger. Stage-2 FO lower bounds are positive in both modes, but
their interaction is not identifiable until the three exact recoveries finish.
Counters has a large but unstable interaction because its VH-off search often
degrades a strong policy; its VH-on MCTS level remains below VH-off policy.
The older combined RQ4 panel is retained for appendix use only; the three
estimand-specific plots are the primary presentation figures.

#### RQ4 progressive-widening extension

| Domain | n | VH-off policy / PW70 | VH-on policy / PW70 | VH-on direct [95% CI]; Holm p | VH-on PW70 − VH-off policy [95% CI]; Holm p | Interaction [95% CI]; Holm p |
|---|---:|---:|---:|---:|---:|---:|
| FO Counters | 10 | 4.2 / 8.4 | 3.7 / 7.3 | +3.6 [2.70,4.50]; .004 | +3.1 [2.18,4.02]; .004 | −0.6 [−1.87,0.67]; .828 |
| Rover | 10 | 4.0 / 4.7 | 3.8 / 4.6 | +0.8 [−0.08,1.68]; .125 | +0.6 [−0.09,1.29]; .125 | +0.1 [−0.88,1.08]; 1.000 |

FO PW70 improves both VH modes, but the interaction is not significant: PW is
an RQ2 success there, not evidence that the value head specifically enables
PW. The full 30m/2h/6h RQ4 PW estimands remain in
`rq4_pw70_branch_latest.csv`.

## 3. External generator comparison

The active comparison is limited to FO Counters and Rover and measures the
actual PDDL distributions, not filenames. `generator_distribution_instances.csv`
contains every parsed test/validation/frozen instance.

| Domain/distribution | Main size | Other structural evidence | Interpretation |
|---|---:|---|---|
| FO test | 2–21 counters; mean 11.50 | all initial values zero; `max_int=2n`; ordered chain | Reference |
| FO thesis validation | 2–16; mean 7.53 | 95.9% initial values nonzero; tiered max; shuffled chain | Smaller and structurally different from test |
| FO Yarin frozen | 2–20; mean 8.85 | 98.5% initial values nonzero; fixed max 42; ordered chain | Closer goal order/range, but still a strong initial-state shift |
| Rover test | 1–8 rovers; 4–25 waypoints | means 3.75 and 9.50; graph grows through suite | Reference |
| Rover thesis validation | 1–5; 4–20 | means 2.60 and 9.03; connected/reachable safeguards | Good central overlap; under-covers largest test tail |
| Rover Yarin frozen | 1–4 rovers; 4–8 waypoints | means 2.40 and 6.50; visible edges mean 28.8; traverse edges mean 25.2 | Fewer objects and smaller graphs; normalized traversal density is only somewhat below thesis validation |

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

**Generator conclusion.** The thesis generators are defensible as the primary
training/validation generators: they are reproducible, produce valid instances,
and expose explicit difficulty controls. They are not unbiased replicas of the
fixed test distributions. In particular, FO Counters validation usually starts
with nonzero counter values while the test suite starts at zero, and Rover
validation under-covers the largest test instances. Yarin's generators are also
valid and useful, but as independent distribution-shift/stress generators—not as
drop-in replacements. Yarin's FO generator is somewhat closer in counter-count
range and goal order yet retains the same initial-value mismatch; Yarin's Rover
instances are markedly smaller. The two sources are therefore not similar
enough to pool. There is no need to rerun the thesis campaign on Yarin's sets,
because the primary RQs are evaluated on the fixed test suite; the external
screen should be reported as robustness evidence and a limitation check.

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

The Counters audit instrumentation now records, for every opted-in root
decision:

- external step and elapsed wall time;
- raw network argmax/probability;
- selected action and its raw probability;
- every expanded child's visits, visit share, prior, Q and U;
- visit entropy and top-one/top-two visit margin.

`summarize_mcts_visit_distribution.py` converts one six-hour trace into exact
cumulative 30-minute, two-hour and six-hour summaries. It has now done so for
both complete VH-off and VH-on arms. Therefore future opted-in
six-hour runs need no separate 30-minute rerun. Historical ordinary logs cannot
be retroactively upgraded to exact step-at-30m traces; they support final step
counts and success-by-cutoff only.

## 6. Why Counters MCTS can be worse than policy

The current evidence is stronger than a generic speculation:

- Across all ten validation-led Stage-2 VH-off seeds, there are 12 instances
  solved by policy but not by fixed narrow MCTS: eleven ordinary unsolved
  trajectories and one unclassified instance from an interrupted allocation.
- The new instrumented VH-off audit targets three representative losses. They
  first diverge from policy after 881, 993 and 1,105 external actions, then
  reach exactly 10,000 actions unsolved. The older reconstruction's
  “first-action divergence” claim was incorrect.
- Across the parallel ten VH-on seeds, there are another 13 policy-success/MCTS-
  failure instances, all ordinary unsolved. MCTS nevertheless improves the
  VH-on aggregate because it also adds more new successes elsewhere.
- No policy-success loss in either mode is explained by a six-hour per-instance
  timeout. All printed plans elsewhere in the audit are VAL-valid.

The global seed-level join and every original policy/MCTS job path are frozen in
`counters_policy_mcts_failure_global.csv`; the trajectory-level actions for the
VH-off failure cells remain in the earlier audit.

The observed mechanism is a late-onset search-induced policy displacement:

1. Policy-only inference takes the network argmax directly.
2. MCTS chooses the argmax of root visit counts after only 20 simulations over
   up to five retained children.
3. At each first observed divergence, all five child Q values are effectively
   equal, the maximum visit count is tied 9–9, and the selected action has
   policy rank 4. The network prior itself is nearly flat, so discrete visit
   counts and tie-breaking can override a slightly better prior.
4. Once a chosen action differs, the policy is evaluated on off-policy
   Counters states. Repeated small deviations can create a long oscillating
   trajectory that exhausts 10,000 actions even though the original policy had
   a successful path.

This directly supports the advisors' “insufficient visit evidence” hypothesis
for the three audited VH-off trajectories, but does not establish that it
explains every Counters loss. The still-live VH-on control tests whether the
same signature is specific to the failing VH-off cell.

Specifically, the two audit arms distinguish three explanations: (a) the
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
| Counters visit audit, VH-off failures | 1 | 2 | 120 GiB | 24h | Complete as 21178320; all three instances reached 10,000 actions |
| Counters visit audit, matched VH-on control | 1 | 2 | 120 GiB | 24h | Complete as 21178321; all three instances have exact 30m/2h/6h landmarks |
| **Required total** | **2** | **4 concurrent** | **240 GiB concurrent** | — | About 80 CPU-hours and at most 4,800 GiB-hours if both use 20h |

One worker is intentional: it prevents interleaved per-action traces and worker
scheduling from confounding the timing distribution. The full 120 GiB is retained
rather than assuming memory scales linearly with worker count.

The two jobs are fully predeclared in `counters_visit_audit_manifest.csv`. Both
use seed 1963100312 and instances 51, 55 and 59. In the VH-off evidence, policy
inference solved these instances in 2,103, 2,385 and 2,683 external actions,
whereas narrow MCTS first diverged only after 881, 993 and 1,105 external
actions, then reached 10,000 actions. The earlier “first action” description was
wrong and is superseded by the instrumented trace.
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
- `rq2_raw_means_validation_led.csv`, `rq3_raw_means_validation_led.csv`, and
  `rq4_raw_means_validation_led.csv`: raw policy/MCTS levels underlying those
  effects, with job/log provenance.
- Main advisor reading order: `rq2_raw_means_by_stage.*` then
  `rq2_mcts_vh_off.*`; `rq3_raw_means_and_interaction.*` then
  `rq3_value_head_training.*`; `rq4_raw_means_6h_by_stage.*` then the three
  readable `rq4_direct.*`, `rq4_cross_cell.*`, and `rq4_interaction.*` panels.
- `rq1_stage2_training_vh_off.*`, `rq2_mcts_vh_off.*`,
  `rq3_value_head_training.*`, `rq4_value_head_mcts.*`: RQ-separated plots.
- `rq4_direct.*`, `rq4_cross_cell.*`, `rq4_interaction.*`: readable RQ4
  subplots for the three required estimands; the combined RQ4 figure is retained
  for appendix use only.
- `terminal_led_archive_index.csv`: retained terminal-led evidence and the
  explicit no-continuation rule.
- `generator_comparison.csv`: generator mapping, scores, distance and code links.
- `proposed_job_plan.csv`: exact job counts/resources/time gates.
- `counters_visit_audit_manifest.csv`: exact two-arm Counters diagnostic
  configuration, checkpoint sources, instances and historical log provenance.
- `counters_policy_mcts_failure_global.csv`: all twenty validation-led Stage-2
  Counters seed pairs, global policy-success/MCTS-failure counts and direct log
  provenance.
- `fo_stage2_validation_recovery_progress_latest.csv`: per-recovery success,
  timeout and operational-failure progress with direct source logs.
- `reproducibility_publication_manifest.csv`: Git publication map.

## 9. Targeted live update — 11 September, 10:47 IDT

- Adaptive-KL jobs `21144388` and `21144389` both completed 100 updates. The
  catastrophic-seed arm ended at 21/30 validation coverage and the stable arm
  at 30/30. Neither arm ever changed coefficient 3: all 200 updates remained
  below the target controller's intervention threshold. This is a repeated
  constant-anchor trajectory, not evidence for or against adaptive KL. Test
  policy evaluation is still required before comparing coverage with the
  original catastrophic run.
- MPrime Phase B is complete: **2,260/2,260** checkpoint-replicate evaluations
  and **60/60** lineages. The final repaired tails completed successfully.
  Earlier retries exposed missing generated domain modules, concatenated
  multiprocess output and wrong-checkout import precedence; all three defects
  were repaired without repeating already complete checkpoint inference.
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
- Counters visit-audit jobs `21178320` and `21178321` are complete. They run only
  instances 51, 55 and 59, with one worker, 2 CPU, 120 GiB and a 24-hour
  allocation each. They log priors, visits, Q/U and selected actions at every
  root for exact 30m/2h/6h distribution summaries. In the complete VH-off arm,
  the first policy/MCTS disagreements occur at steps 881, 993 and 1,105—not at
  step zero. At each first disagreement all child Q values are effectively
  equal, the highest visit count is tied 9–9, and the chosen action is policy
  rank 4. This directly supports coarse integer visits/tie-breaking overriding
  a slightly better but nearly flat policy prior. By six hours, disagreement
  rates grow to .57–.72 after search has entered an off-policy trajectory. The
  completed VH-on arm also shows high disagreement (.72, .77 and .79 at its
  six-hour landmarks), so coarse visit distributions are not exclusively a
  VH-off mechanism.
- The three FO validation-led Stage-2 records were rechecked and are genuinely
  partial: jobs `20943885`, `20945845` and `20945846` leave 15, 13 and 14
  instances unclassified. Per-instance timeouts were followed by worker-cleanup
  failures (`survived SIGKILL`), and the allocations later exhausted their
  Slurm limits. Only already persisted outcomes entered the durable ledger;
  “unclassified” does not mean “unsolved.” Corrected minimal jobs `21178377`, `21178379` and
  `21178380` run only those 42 instances. The first submission attempts stopped
  after roughly 90 seconds because old completion ledgers encoded a different
  test-order signature; those attempts produced no result and the corrected
  jobs use explicit skip lists instead.

## 10. MPrime Phase-B final scientific result

The two independent harder validation replicates remove the old Stage-2
30/30 saturation, but their checkpoint-ranking quality is still limited.
| Branch | VH | Complete | Replicate rank agreement | Validation–test rank agreement | Mean selection regret |
|---|---|---:|---:|---:|---:|
| Stage 1 | off | 10/10 | .547 | .308 | 1.90 plans |
| Stage 1 | on | 10/10 | .729 | .490 | 1.60 plans |
| Validation-led Stage 2 | off | 10/10 | .266 | .162 | 1.60 plans |
| Validation-led Stage 2 | on | 10/10 | .241 | .018 | 2.30 plans |

Here, replicate agreement is the mean within-lineage Spearman correlation
between the two frozen sets across saved checkpoints. Validation–test agreement
is the same rank correlation against retrospective test-policy scores.
Selection regret is the number of test instances lost by the validation-chosen
checkpoint relative to the retrospectively best saved checkpoint; it measures
selector weakness, not a deployable oracle. The final conclusion is that the
new sets solve saturation but Stage-2 ranking, especially VH-on, remains weak.
Combining the replicates does not reliably fix selection: it improves mean
regret for Stage-2/VH-off (1.6 versus 1.7 and 2.2) but worsens it for
Stage-2/VH-on (2.3 versus 1.6 and 1.9). A third set sampled from the same design
is therefore not automatically justified. Any Phase C should alter structural
support and test a frozen, small union of candidate checkpoints instead of
rerunning all 1,130 checkpoints. Row-level checkpoint, job and log provenance is in
`mprime_validation_phase_b_20260906/phase_b_checkpoint_scores_latest.csv`.

## 11. Progressive widening inside RQ2 and RQ4

Progressive widening is supporting evidence for the inference questions, not a
fifth research question. The expanded ten-seed branch uses validation-selected
Stage-1 checkpoints, 70 simulations, `Kmin=3`, maximum width 20 and the same
ten matched seeds as policy/fixed search.

### RQ2 — PW70 without the value head

| Domain | n | Policy | Fixed 30m / 2h / 6h | PW70 30m / 2h / 6h | PW70 − policy at 6h [95% CI] | Raw / Holm p | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| FO Counters | 10 | 4.2 | 7.5 / 7.8 / 7.8 | 8.4 / 8.4 / 8.4 | +4.2 [3.26, 5.14] | .002 / .004 | Strong significant gain; already realized by 30m |
| Rover | 10 | 4.0 | 4.8 / 5.0 / 5.0 | 4.7 / 4.7 / 4.7 | +0.7 [0.02, 1.38] | .125 / .125 | Positive mean; approximate fixed parity |

FO/off is final at 8.40/20. Its 7/20 seed is an OOM-terminal declared-budget
endpoint, not a pending partial result: the one unclassified instance counts
unsuccessful and its dedicated six-hour recovery also timed out. FO/on is also
final at 7.30/20 and contains two OOM-terminal declared-budget endpoints (18/20
and 19/20 instances classified); their unclassified instances likewise count
unsuccessful. These endpoints remain in the predeclared ten-seed means/tests.

### RQ4 — PW70 with the value head

| Domain | n | VH-off policy / PW70 | VH-on policy / PW70 | VH-on direct [95% CI]; Holm p | VH-on PW70 − VH-off policy [95% CI]; Holm p | Interaction [95% CI]; Holm p |
|---|---:|---:|---:|---:|---:|---:|
| FO Counters | 10 | 4.2 / 8.4 | 3.7 / 7.3 | +3.6 [2.70, 4.50]; .004 | +3.1 [2.18, 4.02]; .004 | −0.6 [−1.87, 0.67]; .828 |
| Rover | 10 | 4.0 / 4.7 | 3.8 / 4.6 | +0.8 [−0.08, 1.68]; .125 | +0.6 [−0.09, 1.29]; .125 | +0.1 [−0.88, 1.08]; 1.0 |

Both VH modes benefit in FO, and no value-head interaction is detectable at
this sample size (interaction −0.6 [−1.87, 0.67]). PW therefore strengthens RQ2 in FO Counters,
while RQ4 remains chiefly the fixed-MCTS Drone result.

Supporting lower-resolution PW evidence remains visible rather than being
discarded: the Drone `Kmin=3` arm retained 9.5/20 versus policy 7.0 and fixed
10.5 while sharply reducing the successful-instance runtime tail; two-seed
PW20 screens showed isolated Counters recoveries; five-seed corrected PW70 in
Counters reached fixed parity only for VH-off at six hours; and the collapsed
TPP seed improved only under PW20 (9 to 10/20). These screens diagnose scope
and failure modes but are not pooled with the ten-seed FO/Rover inference.

Replottable tables are `rq2_pw70_branch_latest.csv` and
`rq4_pw70_branch_latest.csv`; both point to the 40-row seed-level provenance
file. The compact raw-means plot is `rq2_rq4_pw70_final.svg`.

## 12. Counters tie-breaking audit and proposed rule

The current external-action rule is not Q-based. The root target is normalized
edge visits and `ArgmaxPolicy` applies NumPy argmax, so a tied maximum selects
the earliest global action index. Q is used during PUCT selection, but not to
resolve the final visit-count tie. Exact PUCT-score ties likewise retain the
earliest stored child.

The offline audit enumerates raw logged-child visit ties, but it does not log
the complete post-search eligibility mask or whether goal chasing overrides
ordinary root argmax. Its aggregate 11.2–15.2% tie rates and 64–79% possible
changes are therefore **diagnostic upper bounds**, not valid predictions of
how often the real action pipeline would change. The three headline first
divergence roots remain credible: steps 881, 993 and 1,105 were manually
checked, had tied maximum visits and effectively equal Q, and did not involve a
known-goal override. At those roots, current action-order tie-breaking selected
policy-rank-four actions while the proposed prior tie-break selects the network
argmax.

Recommended candidate rule:

1. maximize visits;
2. among tied actions, use Q only when the tied Q range exceeds a frozen
   numerical tolerance;
3. when Q is effectively equal, choose the largest root network prior;
4. use stable action ID only as a final exact tie-break.

This preserves MCTS evidence while replacing arbitrary action ordering with
the trained policy exactly when search value evidence is uninformative. It is
deterministic and adds no simulations. Risks are that the prior can itself be
wrong, the Q tolerance must be predeclared, and the rule cannot fix non-tied
search divergence. Goal-chasing and any active eligibility/safety mask must
retain precedence; the optional MCTS-SAFE terminal mask is not enabled as a
treatment in this experiment.

The current result is mechanistic, not causal: an offline change cannot reveal
the downstream trajectory that the alternative action would create. A causal
three-way contest is now submitted: historical action-index tie-breaking,
sign-correct Q tie-breaking, and policy-prior tie-breaking, each under matched
VH-off/VH-on arms on the same three instances and narrow 5/20 budget. Compute
smoke `21185508` passed all 23 tests and released all six tasks in array
`21185509[0-5]`; the first smoke attempt
`21185504` failed only because its test module path resolved to the installed
package, and correctly cancelled dependent array `21185505` before inference.
Offline counterfactual per-root and summary provenance is in
`counters_tie_break_counterfactual_latest.csv` and its
`counters_tie_break_counterfactual_latest_summary.csv` companion; submission
provenance is in `counters_tie_break_3way_manifest.csv`.

## 13. MPrime Phase C submission

Phase C is now submitted. It screens 120 structurally redesigned candidates in
three independent planner/VAL jobs, freezes thirty certified problems, runs a
compute preflight, then evaluates a maximum of twelve predeclared checkpoints
per primary lineage. Deduplication reduced the work from the 480 upper bound to
347 actual evaluations across 40 independently schedulable tasks: 133 repeated
references were removed when the two top-five lists overlapped or selected and
final endpoints already appeared in those lists. No distinct checkpoint was
dropped. The initial
screen `21183472[0-2]` exposed and safely failed on a cross-platform
newline/checksum mismatch before planning. After post-write hash verification,
screen `21183496[0-2]` raced the slow replacement transfer and repeated the
same safe pre-planning failure. Remote checksums were then verified before
screen `21183533[0-2]` and dependent gate `21183534` were submitted.
All three screens then certified 10/10. The finalizer froze the set, preflight
`21183541` completed successfully in 1:21:45, and all forty tasks in rescore
array `21183542[0-39]` were running at the targeted 11 September 18:48 IDT
check.
Its deployed checkpoint manifest contains no test-score column, but its ranges
were designed from aggregate fixed-test structural support. It is therefore a
transductive validator audit whose eventual adoption must be disclosed as
test-informed, not an independent confirmatory validation result. The full
design and checkpoint/job/log provenance are in
`mprime_validation_phase_c_20260911/README.md` and
`mprime_validation_phase_c_20260911/checkpoint_candidates.csv`.

## 14. Evidence kept outside the four primary RQ tables

The RQ focus does not erase smaller, negative or mechanistic experiments. The
following remain part of the thesis evidence map but should usually appear as
diagnostics or appendix material rather than be pooled with the confirmatory
five-domain tables.

| Experiment | Resolution | Result retained | Why it remains relevant |
|---|---|---|---|
| PRESERVE-3 validation-led | Delivery, TPP and Zenotravel; ten seeds × two VH modes | Delivery essentially preserved; Zenotravel preserved; TPP/off has nine 20/20 seeds and one audited 9/20 collapse | Separates robust ceiling preservation from a rare but severe Stage-2 failure |
| ANCHOR-4 | Two tuning seeds × two VH modes × seven coefficients | Frozen coefficients for Delivery/TPP/Zenotravel; MPrime anchor 10 won the corrected 588-point audit | Documents how Stage-2 policy anchoring was selected without test-set tuning; validation adequacy remains a separate question |
| MCTS-SAFE | Four targeted Drone failures | Repaired two of four | Shows dead-end-aware action masking can repair specific search failures, but is not a general coverage estimate |
| SAFE-CONTEXT | Ten matched Drone pairs | Contextual nodes changed off by −0.4 and on by −3.4 plans; not Holm-significant | Rejects contextual node splitting as the default despite real prediction aliasing |
| Binding Horizon | Ten matched pairs | Mean change 0, CI [−0.89, 0.89], p=1; essentially no binding cutoffs | The implemented constraint was a non-result, not evidence that finite-horizon reasoning is useless |
| Counters remaining horizon | Two seeds × two VH modes × aware/unaware | Zero aware-minus-unaware change and zero cutoffs | Confirms the horizon mechanism was also nonbinding in long Counters trajectories |
| Determinism audit | Two six-run instances | CPU families changed internal checksums but not chosen actions | Rules out casual “process timing” explanations for the historical Horizon discrepancy |
| Counters narrow/width | Ten seeds per cell | S2/off reaches policy parity by 2h; S2/on positive but variable; severe S1 seed regressions exist | Establishes the concrete visit/tie diagnostic now under study |
| Drone PW sensitivity | Eight to ten jobs depending arm | Large tree/runtime reduction; `Kmin=3` recovered most but not all fixed coverage | Important efficiency frontier even though it was not confirmatory across domains |
| Counters PW screens | Two- and five-seed cells | Isolated recoveries; no robust overall superiority | Prevents overgeneralizing the FO/Rover PW success |
| TPP catastrophic seed | One targeted seed, four search arms | Only PW20 improved 9→10; other search arms failed to recover | Shows the Stage-2 collapse is not merely a slightly misranked policy |
| Adaptive KL | Bad seed plus stable control | Controller never left minimum coefficient 3 | Treatment was not activated; the run cannot test adaptive KL efficacy |
| External generators | Three seeds × two modes × two domains | FO transfers descriptively; Rover drops sharply | Demonstrates distribution sensitivity and motivates generator documentation |
| MPrime Phase B/C | 1,130 full Phase-B checkpoints; 347 Phase-C candidates | Phase B removes saturation but ranks Stage 2 weakly; Phase C live | Validation adequacy is a methodological result needed before MPrime joins primary RQs |
| Rover interrupted-instance recovery | 29 exact omitted opportunities | All 29 reached the declared six-hour timeout; aggregate Rover results did not change | Closes censoring without rerunning already classified instances |
| Long Drone | Small long-instance side experiment | Retained in registry/provenance | Tests behavior outside the main 20-instance Drone distribution; not a primary claim |
| ENHSP leaf/estimator | Small estimator sensitivity | Retained in registry/provenance | Useful negative/sensitivity context; not a primary claim |

Held designs also remain registered: hard-run PW-30M validation, PW path
batching, resource-shape sensitivity, SAFE2, action-history ablation,
original-stop replication and the PUCT/estimator grid. None should be presented
as an empirical result until activated.
