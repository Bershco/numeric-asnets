# MCTS–policy divergence cause audit

## Question

Why does search depart from greedy-policy trajectories, and when does that
departure help or hurt? “MCTS follows visits rather than policy” describes the
mechanism but does not explain an outcome.

## Required outcome strata

1. policy succeeds, search fails;
2. both fail but their trajectories diverge;
3. policy fails, search succeeds;
4. both succeed, as a negative/control class.

For an exact checkpoint and instance, the primary unit is the first divergent
external decision. Roots from one trajectory are not independent samples.

## Evidence already established

- Counters selected VH-off cases: visit maxima tied, Q was effectively equal,
  action-index ordering selected a lower-prior action, and policy-prior
  tie-breaking recovered 3/3. This is causal selected-case evidence.
- Block Grouping PW trace: at two fully recorded roots, the policy action was
  favored by U/exploration while accumulated Q/exploitation favored the chosen
  non-policy action. Excessive exploration is not the demonstrated cause in
  those roots.
- Drone SAFE: known-terminal child handling explained and repaired 2/4
  selected failures.
- Context audit: the same physical state with different action-history inputs
  can change predictions, especially in Counters; contextual nodes exposed the
  mechanism but harmed coverage/memory when adopted broadly.

Entropy alone is descriptive, not causal. Low entropy can be confidently
wrong and high entropy can still have the correct argmax.

## Stage 0 result — frozen 17 September 2026

Stage 0 is complete locally. No cluster job was submitted.

The exact-provenance join recovered:

- **2,780/2,780 primary Stage-1 fixed-search instance pairs** across the five
  imperfect domains, both VH modes and ten seeds;
- **500 locally classified Stage-1 PW70 instance pairs**, each joined to the
  exact policy row by domain, VH mode, seed and instance;
- **138 first-divergence records**: 127 Counters strict-confirmation
  trajectories, three Counters Stage-2 causal-pilot roots, four Block Grouping
  fixed-search traces and four Block Grouping PW70 roots;
- **seven selected intervention summaries**: four Drone SAFE cases and the
  three VH-off Counters tie-break arms;
- SHA-256 checksums, resolved paths and row counts for all 14 source
  artifacts.

The fixed-search outcome universe is complete, but root-mechanism coverage is
not. Existing first-divergence evidence is concentrated in Block Grouping and
Counters, and only the four Block Grouping PW70 records have a locally cached
selected-versus-policy N/Q/U/prior decomposition. The corresponding remote PW
traces contain all children. The Counters selected pilot has exact tie counts,
visit entropy and policy/visit Jensen-Shannon divergence, but not per-child Q/U.
The strict Counters campaign records the divergence decision and actions but
not the root vector. These evidence levels are kept distinct in the CSV rather
than being silently treated as equivalent.

`both_fail_outcome_only` deliberately does **not** claim that the two failing
trajectories diverged. Those rows only establish the terminal outcome. A
compact trace must verify an actual first divergence before such a candidate
enters the required “both fail after divergence” stratum.

### Frozen Stage-0 artifacts

- `stage0_instance_outcomes.csv`: 3,280 normalized fixed/PW instance outcomes
  with checkpoint identity, seed, instance and original policy/search logs.
- `stage0_first_divergence_evidence.csv`: 138 records with the strongest local
  first-divergence fields available; absent root fields remain blank.
- `stage0_intervention_evidence.csv`: selected causal-treatment outcomes that
  cannot honestly be represented as complete root records.
- `stage0_coverage_missing_strata.csv`: all 24 domain × VH × fixed/PW cells,
  including zero-evidence MPrime/Drone-PW cells rather than omitting them.
- `stage0_missing_strata_manifest.csv`: 40 frozen candidate rows for every
  observed but unexplained stratum: 34 fixed-search candidates and six
  optional FO-Counters PW candidates. Candidates are grouped into jobs; they
  are not 40 proposed jobs.
- `stage0_source_artifacts.csv`: checksums and paths for the 14 input artifacts.
- `build_stage0.py`: deterministic reconstruction and validation script.

The broad fixed-search input currently exists in the shared checkout rather
than this worktree. Its exact path and hash are frozen in
`stage0_source_artifacts.csv`, while every normalized source row needed for the
audit is copied into `stage0_instance_outcomes.csv`. Rebuilding fails loudly if
an input, exact PW/policy join, or primary identity is missing or duplicated.

### What is still missing

The frozen matrix supports at most **ten primary fixed-search trace tasks**:
one domain/VH task for each of the five imperfect domains, with at most one
predeclared candidate instance per missing observed stratum (maximum four
instances in one task). It does not support forty separate submissions.

- Block Grouping/off and Counters/off reuse existing harmful-divergence traces;
  their tasks fill only the remaining outcome strata.
- Every other fixed domain/VH cell lacks root evidence for most or all observed
  strata and receives one grouped task.
- FO Counters PW has important positive coverage results but no root traces;
  its two VH tasks are optional after the fixed campaign.
- Existing Block Grouping PW roots already reject final-tie-breaking as the
  general explanation. Rover PW is approximately a parity result, so neither
  receives a new PW task at this stage.
- MPrime fixed/PW candidate selection waits for its exact per-instance outcome
  join. Stage 0 does not invent candidates from aggregate scores.
- Drone PW has no exact local per-instance join and is not a priority branch.

No Stage-2 task should be submitted until the compact recorder reproduces one
known Counters root and every selected candidate is checked against the final
current campaign state.

## First-divergence record

Record: normalized policy entropy; visit entropy; policy/visit Jensen-Shannon
divergence; selected action prior/rank; whether the policy action was expanded;
maximum-visit ties; top-two margin, winner share and leave-one-out prominence;
complete child N/Q/U/prior and `sign*Q+U`; Q-versus-U preference; safety,
goal-chase and duplicate-control overrides; branching width; search depth;
successor-generation and elapsed time; state/action-history digests; eventual
outcome/failure type; and whether trajectories later rejoin.

The compact root line itself is emitted at decision time and therefore cannot
contain future terminal outcome or a later-rejoin claim. Stage 2 must join the
terminal outcome/failure type from the exact completion ledger and frozen
candidate identity. Later rejoin remains a separate trajectory post-processing
question requiring aligned policy/search state-digest sequences; v1 does not
claim it. This distinction prevents blank future information from being
silently presented as a negative result.

The economical implementation is a compact first-divergence recorder with
optional sparse landmarks, not full debug output at every external action.

## Minimal workload

### Stage 0 — existing-evidence join (zero cluster jobs)

First join existing policy trajectories, fixed-search ledgers, PW ledgers and
the rich Counters/Block-Grouping root traces by exact checkpoint hash, seed and
instance.  Produce one row per first divergence and assign it to one of the
four outcome strata above.  This prevents a new campaign from repeating roots
that already have complete N/Q/U/prior evidence.

Completed on 17 September 2026. The deliverable is the frozen artifact set
listed above, not a scientific conclusion inferred from whichever traces
happened to be easy to locate.

### Stage 1 — compact recorder and preflight

Add one compact record at the first policy/search divergence plus optional
sparse time landmarks.  Unit-test entropy/Jensen–Shannon calculations, action
identity, expansion membership, N/Q/U/prior decomposition and override flags.
Smoke-test one already understood Counters root.  Estimated implementation and
preflight time: one development day; no primary test job is released unless
the recorded action and visits reproduce that known root.

#### Local implementation status — 18 September 2026

The compact recorder is implemented behind the evaluation-only flag
`--eval-mcts-first-divergence-record`. With the flag absent, the historical
selector path and historical completion-ledger signature are preserved. With
the flag present, each instance emits at most one
`[MCTS FIRST DIVERGENCE] {json}` line after the ordinary selector has already
chosen its action. The recorder is therefore an observer, not an alternative
action-selection implementation.

Schema `mcts-first-divergence-v1` records:

- checkpoint path, trainer/worker seed, evaluation index, exact PDDL paths,
  VH mode and all fixed/PW search settings;
- physical-state, action-history, applicability and successor-state digests;
- full action-space vectors for raw and applicable-masked network policy,
  MCTS visit distribution, edge visits, expansion membership, edge priors,
  Q, U and sign-correct `Q+U`; unexpanded PW actions are explicit `null`
  Q/U/prior entries rather than fabricated zero values;
- the actual external-selector path, including goal chase, terminal safety,
  duplicate control and visit-tie resolution. The final selector-input vector
  is retained separately from the original visit distribution;
- entropy/JS summaries, tie count, margin/share/prominence, policy rank and
  expansion membership, Q-versus-U argmax attribution, timing counters and
  search-depth histogram.

Focused local tests cover entropy/JS, full-vector alignment with unexpanded
actions, non-divergence suppression, goal-chase/tie provenance and a SAFE
transformation in which the final selector vector differs from the original
visit vector. The complete progressive-widening/selection test module passes
29/29 locally, including an exact expanded-root reconstruction from the
existing Counters job-21178321 trace fixture. Python syntax compilation passes
for the recorder, both
CLI layers, evaluation-spec plumbing, worker integration and manifest builder.

`build_stage1_manifest.py` freezes the Stage-0 candidates into
`stage1_grouped_tasks.csv` and `stage1_grouped_tasks.freeze.json`: ten primary
fixed tasks plus the two requested optional FO-Counters PW70/Kmin3 tasks,
40 exact candidate runs in total. Block Grouping and Counters reproduce the
narrow 5/20 comparator; Drone, FO Counters and Rover reproduce their original
normal fixed 20/70 arm. Both preserve estimator coefficient 0.5 and PUCT
exploration weight 0.1 from the source evaluations. A task is a sequential
group of up to four
exact checkpoint/seed/instance runs; candidates within a task do not
incorrectly share a checkpoint. Every row is marked `submitted=false` and is
gated on `known_counters_root_compute_smoke_passed`. No Slurm command was run
and no cluster state was modified.

The remaining pre-release work requires the compute-node environment and exact
remote checkpoint/log tree:

1. deploy the code to an isolated cluster worktree without changing the active
   experiment payload;
2. run one exact VH-on Counters 5/20, action-ID smoke using checkpoint
   `src20430427_e0000` and `fz_instance_51.pddl`;
3. require its first divergence to reproduce decision 0, selected action 101
   versus policy action 52, root visits 20 and total edge visits 19, while the
   JSON vectors pass schema/length/finite-value checks and match the frozen
   expanded-child fixture;
4. compare the emitted root against the existing counterfactual and visit
   audit, then freeze the deployed commit/checksum in the task ledger;
5. only after that gate passes, render exact per-candidate commands from the
   grouped manifest and request explicit approval for the 10 fixed plus two
   optional PW task submissions.

The local fixture smoke establishes record construction but cannot establish
that JPDDL, checkpoint loading and the cluster TensorFlow stack reproduce the
known root. Consequently Stage 1 is implemented and locally verified, but it
is not yet compute-node-smoke-complete and the grouped campaign remains
unsubmitted.

#### Live execution and exact reconciliation — 18 September 2026, 18:05 IDT

This section supersedes the pre-release status paragraph above. Commit
`a46e0a845dee83767c6bb43c2187cacc2b79e734` passed the compute-node gate in
Slurm job `21453647`: the recorder-disabled and recorder-enabled traces were
identical, and the known Counters root reproduced step 0, policy action 52,
selected action 101, 20 root visits and 19 edge visits. The exact recovery
array is job `21453648`; it contains only candidates without a preserved or
exactly reconciled terminal result.

The original 40 candidates currently comprise 37 terminal classifications and
three running candidates. Six hard timeouts were materialized from exact
`[EVAL INSTANCE] timeout` markers rather than rerun, one legacy FO-PW success
was normalized with its complete goal-chase selector input, and all previously
valid results were preserved. The remaining identities are Block Grouping
VH-on fixed candidates 2 and 3 and FO Counters VH-off fixed candidate 3. All
six optional FO-PW candidates are terminal.

Interim first-divergence evidence is descriptive, not yet a final cause:

- 28 of 37 terminal candidates have an observed policy/selected-action
  divergence; nine have none.
- Every observed divergence kept the policy action inside the expanded root.
  The current sample therefore gives no support to policy-action exclusion as
  the explanation, including under FO progressive widening.
- Twenty-one divergences end in root-visit argmax and seven in goal chase. Among
  the root decisions, 16 selected actions are signed-Q argmaxes, four are
  exact visit ties, one is neither Q- nor U-argmax, and none is uniquely
  U-aligned. This favors exploitation/value separation over an exploration-
  bonus account, but does not establish that following Q is beneficial: the
  mechanisms occur in both helpful and harmful historical strata.
- FO-PW contributes three Q-aligned root winners, two goal-chase overrides and
  one run with no observed divergence. The matched VH-off `instance_5` fixed
  and PW runs both first diverge at step 0 through the same goal-chase action,
  so PW is not the first-divergence cause of that paired search success.
- Exact visit ties are currently confined to three both-fail candidates and
  one both-success candidate. They do not cover the replicated harmful Drone
  policy-success/search-failure case, so a tie-break-only intervention is not
  presently broad enough to explain the important outcome asymmetry.

The exact snapshot, manifest hashes, recovery identities, resource requests
and cluster artifact locations are recorded in
`stage1_live_reconciliation_20260918.json`. Final mechanism counts and any RQ
change must wait for the three live candidates and a final outcome/trace join.

### Stage 2 — fill only genuinely missing strata

Missing strata then need at most one task per domain/VH cell, using the
declared search configuration and at most one frozen instance per stratum.

- Five current imperfect domains: at most 10 tasks.
- Add MPrime after its final endpoints freeze: at most 12 total.
- Per task: one worker, 2 CPU, 120 GiB, at most four exact instances, six-hour
  per-instance cap, approximately 26-hour allocation.
- Maximum six-domain concurrency: 24 CPU / 1.44 TiB.

If all tasks run concurrently, the scheduler hard bound is about 26 hours;
most tasks should finish earlier when the selected instance classifies before
six hours.  The manifest must record policy and search job/log provenance for
every selected instance.

Do not duplicate the campaign for PW. Add PW traces only where fixed-search
evidence cannot explain an important result: initially FO Counters and MPrime,
both VH modes, at most four additional tasks.

## 19 September interim mechanism join (superseded)

The former 39/40 and three-unknown snapshot was an artifact-discovery and
schema-reconciliation problem, not unfinished scientific inference. The final
40-case reconciliation below supersedes it. The interim artifacts under
`live_20260919/` remain historical provenance only.

### Stage 3 — causal intervention, gated

Only after Stage 2 identifies a repeated mechanism should a treatment be
tested: for example policy-prior tie-breaking for unresolved visit ties,
changing expansion/widening for policy actions that were never expanded, or a
PUCT/exploitation sensitivity check when Q rather than U repeatedly causes the
first harmful divergence.  Calibrate and freeze every threshold on validation
traces, then evaluate the unchanged rule on test.  Do not select a treatment
from test failures themselves.

Report by outcome stratum and domain with seed/domain-clustered bootstrap
intervals. Any new action-selection rule must be calibrated and frozen on
validation traces before test evaluation. The policy-failure/search-success
stratum is mandatory so a policy-preserving rule does not erase genuine search
wins.

The maximum initial new workload is therefore 10 fixed-search tasks for the
five current imperfect domains, or 12 after MPrime is available.  Optional PW
diagnostics add at most four tasks.  No such tasks are submitted until the
Stage-0 join produces the exact missing-strata manifest.

## 20 September final 40-case reconciliation

The apparent fortieth missing result and the three unknown no-divergence
outcomes were bookkeeping defects, now resolved without rerunning anything.

- Job `21473765_0` wrote to the newer
  `2026-09-19/mcts_policy_divergence_stage1_final_recovery` root, while the old
  audit inspected only the `2026-09-18/...stage1_recovery` root. Its result is
  a scientifically explicit hard timeout for FO Counters, VH-off, fixed 20/70,
  seed `1510771779`, checkpoint `src20401235_e0066`, `instance_8.pddl`.
- The three formerly unknown terminal joins use the version-1 result schema,
  where terminal status is stored in top-level `completion_record` rather than
  nested `outcome.classification`. They are all successes: Counters/VH-on
  fixed `fz_instance_10.pddl` (57 steps), FO Counters/VH-on PW
  `instance_2.pddl` (8 steps), and Rover/VH-off fixed `pfile1.pddl` (10 steps).
- The final join therefore contains 40/40 identities and zero unknown terminal
  outcomes: 27 successes, seven hard timeouts, five action-limit failures and
  one ordinary finished-unsolved result. Nine trajectories have no observed
  policy/search divergence: seven successes and two hard timeouts.

The selector taxonomy must distinguish raw root visits from transformations
applied before final argmax:

- eight divergences use known-goal chasing; all eight succeed in this selected
  sample;
- 20 use the unmodified root-visit argmax;
- three use root argmax only after duplicate-control transformed the visit
  vector. These are not evidence that raw visits or Q won. Duplicate-control
  produced two successes and one finished-unsolved result.

Among the 20 genuinely raw root-visit selections, the policy action was
expanded in every case. Sixteen selected actions are signed-Q argmaxes, none
is U argmax, eight are Q+U argmaxes and four have a maximum-visit tie. The
selected action has higher signed Q than the policy action in 17/20 roots and
equal Q in 3/20. Outcomes are ten successes, five action-limit failures and
five hard timeouts. By predeclared outcome stratum these are eight both-fail,
eight policy-failure/search-success, two both-success and two
policy-success/search-failure cases.

The two harmful raw-root cases are:

- Block Grouping/VH-on fixed, `instance_20_30_7_2.pddl`: selected visits
  2053 versus policy 1572, selected-minus-policy Q `+0.000703`, but
  selected-minus-policy U `-0.000760` and selected-minus-policy Q+U
  `-0.000057`; it reaches the action limit.
- FO Counters/VH-off fixed, `instance_8.pddl`: at step zero the selected action
  has 61 visits versus one for the policy action, Q difference `+0.000308`, U
  difference `-0.032490`, policy/visit Jensen-Shannon divergence `0.68596`,
  and winner visit share `0.884`; it reaches the six-hour timeout.

One additional harmful divergence in Drone is not a raw-root winner:
duplicate-control bans the 124-visit policy action, after which a 36-visit
action is selected and the run finishes unsolved. This should be tested with a
duplicate-control ablation rather than a tie-break or PUCT intervention.

Q is a backed-up MCTS return, not automatically a learned-value-head verdict.
VH-off uses the ENHSP-derived estimator (plus terminal outcomes) in the declared
leaf blend; VH-on blends learned value and ENHSP-derived estimation. The
current trace records only the blended Q. It therefore supports the descriptive
conclusion that accumulated backed-up exploitation evidence often overrides the
policy, but it cannot yet attribute a harmful Q difference to learned value,
ENHSP, terminal discovery or finite-search path dependence. It also cannot
compare divergent with non-divergent roots because version 1 records a root
vector only at first divergence.

The smallest causal-source follow-up is source-decomposed tracing for the two harmful
raw-root cases, the one duplicate-control case and descriptive helpful/control
roots. Record learned leaf value, transformed ENHSP value, blended value,
terminal source, per-action backed-up source contributions, raw visits before
selector transforms and final selector input. Use observation-only controls
at predeclared steps so non-divergent roots have comparable vectors.
This is a selected mechanism diagnostic, not an RQ population estimate.

The existing harmful divergences occurred after approximately 31.6 minutes
(Block Grouping), 7.7 minutes (Drone) and 16.8 seconds (FO Counters). A minimal
six-task design—those three exact cases plus one frozen matched control each—
can therefore request 2 CPU, 120 GiB and two hours per task, for at most
12 concurrent CPU and 720 GiB. It should stop immediately after materializing
the predeclared diagnostic root; it must not repeat the already-known full
six-hour terminal outcome. Code instrumentation and local validation are now
complete. The step-zero FO harmful task is the cluster compute-smoke gate for
the remaining five tasks. No recovery or causal job was submitted during this
reconciliation or preparation.

Canonical local outputs:

- `reconcile_result_roots.py`: version-aware, multi-root, deduplicating join;
- `stage1_final_40_reconciled.csv`: the 40 resolved rows with terminal,
  selector and root-vector evidence.

The reproducible cluster-side reconciliation command is:

```text
python3 reconcile_result_roots.py --format summary \
  /home/hersco/training_new_domains/2026-09-18/mcts_policy_divergence_stage1_v2 \
  /home/hersco/training_new_domains/2026-09-18/mcts_policy_divergence_stage1_recovery \
  /home/hersco/training_new_domains/2026-09-19/mcts_policy_divergence_stage1_final_recovery
```

This command is login-node parsing only and requests no Slurm resources.

## Source-decomposed causal follow-up (prepared, not submitted)

The minimal next diagnostic is frozen as six observation-only tasks: the two
harmful raw-root cases above, the harmful Drone duplicate-control case, and one
predeclared descriptive control for each. These are contrast groups, not
matched pairs or a harmful-versus-helpful causal comparison. The controls match
the most important available
axes: Block Grouping and FO
Counters match domain, search family and raw-root mechanism; Drone matches
domain, VH mode and fixed-search configuration and deliberately records an
agreement root. BG and FO cross VH mode, seed, checkpoint and instance; all
three controls differ in at least checkpoint/seed/instance. This limitation
must remain explicit, and each root should be interpreted individually.

The added instrumentation does not change scalar Q or action selection.  For
every backup it records the additive contributions from raw network value,
transformed ENHSP estimator value and terminal/rollout outcome.  At the frozen
external step it writes one complete root record and exits before executing the
selected action.  The result schema says both
`diagnostic_stopped_after_record=true` and `terminal_outcome_repeated=false`;
the generic evaluation completion ledger is disabled because it would label an
intentional early stop `finished_unsolved`. Only the dedicated validated
diagnostic result is written. The
historical terminal outcome remains joined only from the already completed
source evaluation.

Source statistics have scope `node_global_matches_child_q`.  This is
intentional: a transposed child node has one global Q across all of its parents,
so its accumulated source statistics use exactly the same scope.  Focused tests
cover blended, estimator-only and terminal backups, transposition/global-node
scope, agreement-root recording and diagnostic-stop result semantics.

Frozen artifacts:

- `source_decomposition_followup_manifest.json`: six exact roots and resources;
- `source_decomposition_followup_manifest.freeze.json`: SHA-256 lock against
  the manifest, source grouped manifest and final reconciliation;
- `../../scripts/run_mcts_source_decomposition_followup_20260920.py`: dry-run
  by default, exact join/record validation and fail-closed output handling;
- `../../scripts/mcts_source_decomposition_followup_20260920.sbatch`: reviewed
  Slurm payload with the global hard-node exclusion list.

Each task requests 2 CPUs, 120 GiB and two hours.  Running all six would request
at most 12 CPUs and 720 GiB; the gated release below peaks at 10 CPUs and
600 GiB after the smoke.  The longest historical target
root appeared after 31.6 minutes; the two-hour allocation leaves substantial
operational grace while still stopping immediately once the target record is
written.  A submission must first deploy a reviewed commit, create the output
directory, then run exactly:

```text
mkdir -p /home/hersco/training_new_domains/2026-09-20/mcts_source_decomposition_followup
smoke=$(sbatch --parsable --array=4 \
  --export=ALL,CODE_COMMIT=<reviewed-deployed-commit> \
  scripts/mcts_source_decomposition_followup_20260920.sbatch)
sbatch --parsable --dependency=afterok:$smoke --array=0-3,5 \
  --export=ALL,CODE_COMMIT=<reviewed-deployed-commit> \
  scripts/mcts_source_decomposition_followup_20260920.sbatch
```

As of preparation, no task has been submitted and no causal result exists.
This selected six-root diagnostic can attribute backed-up Q to its direct
sources; it is not a population estimate and does not alter any RQ table.
