# MCTS Inference Search Design and Experiment Plan

Status: implementation available on commit `fac11cb0`; not integrated into the
production checkout. The pilot uses an isolated sparse cluster checkout pinned
to aggregate commit `6701f4c1`, so existing queued jobs remain unaffected.

This document records the inference-search work developed for the numeric-ASNet
thesis, including the motivation, selected algorithms, rejected alternatives,
instrumentation, and planned experiments. It separates implemented behavior
from deferred ideas so later analysis does not accidentally treat a proposal as
an experimental result.

## 1. Motivation

The established inference configurations use fixed expansion widths:

- normal search: top 20 policy actions;
- narrow search: top 5 policy actions;
- automatically calculated simulations:

  `I = clip(10 + 3B, 10, 200)`

Thus, width 20 normally implies 70 simulations. Fixed width is costly in domains
where each MDPSim successor transition takes several milliseconds. It can spend
most of the budget constructing shallow alternatives rather than searching a
smaller number of promising trajectories deeply.

Inference also has a finite executable-action budget. If the evaluation allows
10,000 actions and 4,000 have already been executed, a goal more than 6,000 tree
edges from the current state cannot be executed before termination. A search
that ignores this remaining horizon may:

- explore unreachable depths;
- discover goals that cannot be executed in the remaining budget;
- backpropagate those goals as successful outcomes;
- distort root Q-values;
- deepen a long branch instead of finding a feasible shorter route.

Deep search can still create useful training targets, so the mismatch is most
directly relevant to inference.

## 2. Alternatives considered

The considered approaches were:

- policy-ordered progressive widening;
- stochastic action sampling from the policy;
- policy-mass or top-p admission;
- bound-based action elimination;
- sequential-halving or Gumbel-style root allocation;
- limited-discrepancy or policy-deviation search;
- beam-width control;
- hierarchical actions or options;
- novelty or transposition-based pruning;
- explicit finite-horizon or depth-limited search.

Policy-ordered progressive widening was selected as the first branching-control
method because it retains ordinary MCTS while adaptively limiting the number of
constructed successors.

## 3. Progressive widening

For tree node `s`, the permitted width is:

`K(s) = min(K_max, max(K_min, floor(c * N(s)^alpha)))`

Initial defaults:

- `K_min = 2`;
- `K_max = --mcts-expansion-size`, normally 20;
- `c = 0.6`;
- `alpha = 0.5`.

At 70 visits, these defaults admit approximately five root children. Deeper
nodes normally receive fewer visits and remain narrower. A second planned
variant uses `c = 1.0`, admitting approximately eight root children at 70
visits.

The implemented behavior is standard one-path, one-backup progressive widening:

1. Traverse the tree through selection.
2. Stop at the first node whose permitted width exceeds its existing child
   count.
3. Initially admit `K_min` actions.
4. On later widening events, admit exactly one additional action.
5. Admit actions in descending raw-policy order.
6. Break equal-policy ties deterministically by action ID.
7. Immediately evaluate and backpropagate the newly admitted child selected for
   that simulation.

The rejected alternative was collecting widening opportunities throughout the
complete selected path. That could generate several states in one simulation
and would no longer be conventional one-path, one-backup MCTS.

### Evaluation at admission

Fixed-width mode preserves its existing semantics:

- generate the fixed batch;
- evaluate and backpropagate the parent leaf;
- assign network outputs to children without immediately backing up each child.

Progressive-widening mode differs deliberately:

- generated children receive batched network predictions;
- the admitted child used by the current simulation receives a complete value
  evaluation, including the configured estimator blend;
- that value is immediately backpropagated.

This avoids leaving newly admitted children at an uninformative default Q until
a later simulation.

## 4. Memory and batching

No separate ordered-action list is retained per tree node. The implementation
reuses the policy vector, applicable-action mask, child action IDs, and existing
appendable child map. Existing visits, priors, and child pointers survive later
widening events.

Direct progressive-widening metadata should therefore be negligible, expected
to remain well below roughly one percent per node. Total memory should normally
decrease because fewer children are retained, although this must be measured:
transpositions and state representations can still dominate memory.

Expansion is only partially batched:

- MDPSim transitions remain sequential;
- parent conversion is reused;
- TensorFlow inference is batched.

Because successor generation appears to dominate runtime in expensive domains,
no complex path-batching mechanism was added. The new timing summaries will
show whether smaller neural batches introduce a material regression.

## 5. Remaining-horizon-aware inference

A separate opt-in inference mode uses:

`remaining_horizon = max_len - external_action_step`

When enabled:

- tree selection cannot exceed the remaining number of executable edges from
  the current root;
- cutoff nodes receive the ordinary network/estimator value;
- cutoff nodes are not expanded;
- known goals are chased only when their known tree distance is feasible;
- known but horizon-infeasible goals are logged separately.

The cutoff does not write a hard failure value. The same physical state may be
promising with 5,000 actions remaining and useless with ten. Since existing
transpositions share nodes by physical state, a harsh horizon penalty could
contaminate that state's Q-value in another context.

Progressive widening and horizon-aware inference must initially be evaluated as
separate factors. Combining them before measuring their individual effects
would obscure causal interpretation.

## 6. Deferred fully horizon-correct design

The theoretically complete search state is `(s, h)`, where `s` is the physical
state and `h` is the remaining executable horizon. Q-values and visit counts
would therefore be keyed by `(state_key, remaining_horizon)` rather than only by
`state_key`.

Possible implementations include horizon-specific statistics attached to a
physical-state node or distinct nodes for each `(s, h)` pair. Either approach
can multiply statistics and memory use across many horizons. It is therefore a
documented later correctness extension, not part of commit `fac11cb0` and not a
current experiment.

The implemented intermediate correction consists only of:

- remaining-depth cutoff;
- no expansion beyond the cutoff;
- feasible-only known-goal chasing;
- cutoff and infeasible-goal diagnostics;
- existing state-only transpositions.

### Action-history-aware transpositions (MCTS-SAFE-CONTEXT)

Numeric ASNets may include one cumulative count per grounded action in the
network input. The physical MCTS key omits that vector, so old behavior can
reuse priors, values, visits, and children obtained under a different action
history. SAFE-CONTEXT separates the identities:

- cycle identity remains the physical state;
- node/cache identity becomes `(physical_state_key, action_count_digest)`;
- the fixed 128-bit BLAKE2 digest covers **only** the named `action_count`
  column, never all auxiliary data;
- selection keeps an explicit physical-state path set, so contextual nodes do
  not weaken cycle blocking;
- diagnostics retain compact counters and at most 128 detailed witnesses per
  worker, including action-count distance and policy/value disagreement;
- no growing action-history list is stored per node.

The diagnostic pilot uses Drone, Rover, Counters, and Block Grouping; one seed
and VH mode; three representative instances; one worker; at most 100 external
actions; width 20; and 70 simulations. The separate MCTS-SAFE-2 proposal keys
statistics by `(state, remaining horizon)` and remains explicitly held because
combining both context dimensions immediately would confound effects and may
multiply memory in two dimensions.

### Binding-horizon efficacy experiment

The original Horizon pilot was non-binding: every job recorded zero cutoff
events. The replacement uses fresh aware and unaware arms from the same commit.
Drone receives a shared 750-action limit: the largest previously successful
matched Drone plan in the authoritative audit is 651 actions, leaving a
99-action margin while making the horizon relevant to long wandering failures.
The design is five matched seeds, both VH modes, two arms, width 20, 70
simulations, SAFE-1 enabled in both arms, and progressive widening disabled.

## 7. Defaults and removed option

Production callers already selected maximization because they passed
`inp.minimization`, whose default is false. The MCTS constructors themselves
misleadingly defaulted to minimization. Commit `fac11cb0` changes both relevant
constructor defaults to `minimization=False`.

The dormant `--mcts-smart-expansions` option was removed without a compatibility
alias.

## 8. Command-line interface

Progressive widening:

- `--mcts-progressive-widening`
- `--mcts-pw-min-width`
- `--mcts-pw-c`
- `--mcts-pw-alpha`
- existing `--mcts-expansion-size` supplies `K_max`.

Horizon-aware inference:

- `--eval-mcts-enforce-remaining-horizon`

The horizon option requires `--eval-with-mcts`. It does not affect training and
is unavailable for policy-only inference.

## 9. Diagnostics

New compact summaries cover:

- search calls and total simulations;
- retained and peak node counts;
- widening events and final node widths;
- root width before committed external actions;
- width at widening events and by search-depth band;
- selection-depth minimum, mean, median, p90, p95, and maximum;
- horizon-cutoff depths;
- policy ranks of admitted actions;
- raw-policy rank of selected external actions;
- feasible and horizon-infeasible known-goal decisions;
- selection, expansion, successor-generation, network-inference,
  estimator/evaluation, and backpropagation times.

Analysis must aggregate these separately for successful instances, ordinary
failures, per-instance timeouts, and scheduler interruptions.

## 10. Implementation provenance and tests

Commit: `fac11cb0` (`Add progressive widening and horizon-aware MCTS`).

Changed files:

- `post_training/monte_carlo_tree_search.py`
- `post_training/training_mcts.py`
- `post_training/action_selection_policy.py`
- `asnets/asnets/spawn_train_worker.py`
- `asnets/asnets/scripts/run_asnets.py`
- `asnets/asnets/scripts/run_experiment.py`
- `asnets/asnets/parllel_explore_spawn_grads.py`
- `asnets/tests/test_progressive_widening.py`

Completed smoke checks included compilation of all changed Python files, staged
diff validation, and ten focused unit/regression tests. The tests cover fixed
width behavior, appendable child-map preservation, maximization defaults,
initial and later widening, policy ordering, the width schedule, invalid
parameters, remaining-depth cutoff, no expansion beyond the horizon, and
horizon-aware known-goal chasing.

Result recorded by the implementation chat: `Ran 10 tests` / `OK`.

## 11. Completed widening-only experiment

The widening pilot domain is **Drone**. This was selected from complete Slurm-job
accounting, including terminal timeouts and OOM outcomes, rather than from policy
runtime or successful-instance runtime alone.

Preliminary replicated stage-1 completion records gave the following runtimes:

| Domain | Completed records | Median (s) | Mean (s) |
| --- | ---: | ---: | ---: |
| FO Counters | 135 | 84 | 355 |
| Rover | 137 | 96 | 747 |
| Block Grouping | 388 | 153 | 1,429 |
| Counters | 386 | 182 | 1,583 |
| Drone | 335 | 213 | 1,033 |

These records exclude hard-timeout instances, which are not persisted as
completed JSONL rows. They therefore cannot alone identify the cheapest complete
evaluation campaign. FO Counters has the fastest completed instances but several
jobs reach the three-day scheduler limit, while the Drone campaign completed.
The complete replicated stage-1 job accounting was:

| Domain | Completed | Timed out | OOM | Other failure | Still running | Mean terminal job time (h) | Median terminal job time (h) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Block Grouping | 20 | 0 | 0 | 0 | 0 | 5.02 | 8.19 |
| Drone | 20 | 0 | 0 | 0 | 0 | 11.44 | 12.13 |
| Rover | 5 | 0 | 15 | 0 | 0 | 28.60 | 30.04 |
| FO Counters | 5 | 9 | 5 | 1 | 0 | 52.36 | 60.27 |
| Counters | 0 | 0 | 1 | 0 | 19 | 60.91 | 60.91 |

The aggregate Block Grouping time is not a configuration-matched pilot baseline:
its ten sound VH-on jobs used width 5 and 20 simulations and averaged about
8.63 hours, while the much faster VH-off subset used the obsolete historical
width-5/three-simulation configuration. The corrected VH-off width-5/20 jobs are
still held. Drone is therefore the clean pilot choice: all twenty matched
width-20/70-simulation jobs completed, providing ten network seeds across both
value-head modes and allowing the fixed-top-20 arm to be reused without a search
configuration mismatch.

Compare four variants:

1. fixed top 20;
2. fixed top 5;
3. progressive `K_min=2, c=0.6, alpha=0.5`;
4. progressive `K_min=2, c=1.0, alpha=0.5`.

Hold constant:

- exactly 70 simulations;
- matched checkpoints and network seeds;
- PUCT coefficient;
- estimator configuration;
- external action-selection rule;
- time and action limits;
- horizon correction disabled.

For five network seeds and both value-head modes, the complete pilot contains
40 evaluations. Ten exactly matched fixed-top-20 evaluations may be reused,
leaving 30 new evaluations. A three-seed exploratory pilot needs 18 new jobs
after six reusable baselines; a one-seed pilot needs six new jobs after two
reusable baselines. One- and three-seed results must be labelled exploratory. A
promising variant should be expanded to at least five matched seeds.

Historical fixed-top-20 results may be reused only if checkpoint seed and every
other configuration match exactly.

The declared five pilot seeds are `1963100312`, `2011206605`, `1073581256`,
`1239739722`, and `1472491096`. Each value-head mode receives the same five
network seeds. The 30-job launch manifest is
`experiment_tracking/mcts_progressive_widening_pilot/manifest.csv`.

All 30 new evaluations completed and passed VAL. The immutable seed-level
results and aggregate statistics are stored in
`experiment_tracking/mcts_progressive_widening_pilot/results.csv` and
`experiment_tracking/mcts_progressive_widening_pilot/summary.csv`.

| Variant | n | Mean coverage | Matched top-20 | Paired change (95% CI) | Exact sign-flip p | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed top-5 | 10 | 8.70/20 | 8.60/20 | +0.10 [-0.31, +0.51] | 1.0000 | 11.24 h |
| PW c=0.6 | 10 | 6.70/20 | 8.60/20 | -1.90 [-3.35, -0.45] | 0.0195 | 8.12 h |
| PW c=1.0 | 10 | 6.20/20 | 8.60/20 | -2.40 [-3.92, -0.88] | 0.0039 | 5.03 h |

Progressive widening reduced runtime but failed the declared non-inferior
coverage criterion. Both progressive variants are significantly worse than
the matched fixed-top-20 baseline; this remains true under Holm correction
across the three displayed comparisons.

Weighted across all compact width summaries, the mean children count over all
retained nodes was 1.3269 for fixed top-5, 1.2516 for PW c=0.6, and 1.2603 for
PW c=1.0. This includes leaf nodes, so root branching is more interpretable:
4.7754, 5.0638, and 5.0212 respectively. PW retained 692,524 nodes for c=0.6
and 319,398 for c=1.0, versus 3,622,099 for fixed top-5. The runtime gain is
therefore consistent with a much smaller retained search graph, not with a
smaller final root width.

All 30 new jobs retain 70 simulations, PUCT 0.1, estimator coefficient 0.5,
the `hadd-astar` teacher, three workers, a six-hour per-instance limit, and a
three-day allocation. Remaining-horizon enforcement is disabled.

The completed eight-job SAFE Kmin=3 extension used the same fixed ordered
20-instance Drone test suite from `asnets/experiments_numeric/domain/drone.py`
for four matched seeds in each VH mode. Mean whole-job runtime was 3m58s for
policy-only inference, 11.76h for fixed top-20, and 9.99h for Kmin=3 PW.
Among successful instances with historical elapsed records, fixed top-20 had
median/mean/p90 runtimes of 130/395/893 seconds (n=84), while Kmin=3 had
94/147/318 seconds (n=76). This conditional runtime gain must be reported
beside the coverage loss: fixed top-20 solved 84 instances and Kmin=3 solved
76. Historical policy logs lack per-instance elapsed records, so eight exact
checkpoint timing recaptures were submitted as jobs 20720780--20720787; their
results are profiling evidence and do not replace the original policy scores.
All eight completed and reproduced the original scores. Whole policy jobs
averaged 238 seconds, although the policy evaluator still did not emit
per-instance elapsed records. Exactly two of 84 fixed-top-20 successes exceeded
30 minutes (42.94 and 102.11 minutes); zero of 76 Kmin=3 successes exceeded 30
minutes. Removing those two successes gives a 30-minute counterfactual of
policy 7.00, fixed top-20 10.25 and Kmin=3 9.50 solved instances per seed. The
underlying rows and plots are under `experiment_tracking/advisor_figures/`.

### Pilot profiling

Profiling is part of the pilot, but it uses the new compact per-instance MCTS
summaries rather than Python `cProfile`, which could perturb wall-clock
behavior. Aggregate coverage, VAL validity, runtime, retained/peak nodes,
widening events, width/depth distributions, successor-generation time, network
inference time, estimator time, and timeout/OOM incidence separately for
successes, ordinary failures, instance timeouts, and scheduler interruptions.
Use a separate cProfile follow-up only if these summaries cannot identify the
bottleneck.

## 12. Live horizon-only experiment

Compare:

- current horizon-unaware MCTS;
- identical MCTS with `--eval-mcts-enforce-remaining-horizon`.

Use the same checkpoints and seeds. Historical top-20 coverage can provide the
unaware baseline when all settings match. The intervening changes do not alter
fixed-width search when their new flags are disabled: progressive widening and
MCTS-SAFE are opt-in, the maximization default matches what existing callers
already passed, and the lifecycle repair changes timeout cleanup rather than
pre-timeout search decisions. Fresh unaware runs are therefore unnecessary for
coverage, but remain necessary for exact same-build runtime/profiling because
historical logs lack the new diagnostics and separate multiprocess runs are not
bitwise reproducible.

The primary success criterion is paired coverage non-inferior to the matched
horizon-unaware width-20/70 baseline. Secondary criteria are fewer per-instance
timeouts, lower runtime and retained-node work, feasible-only known-goal
chasing, and nonzero cutoff evidence showing that search was actually prevented
from exceeding the remaining executable horizon.

The first nonbinding ten-job Drone campaign remains provenance only. The fresh
binding campaign uses a 750-action limit and matched aware/unaware arms from
one commit. At the 2026-08-30 12:13 snapshot, 17/20 inference jobs were
terminal and three were running. Five newly terminal inference logs were
post-hoc VAL-confirmed by jobs 20723088--20723091 and 20723155 instead of
repeating inference.

The cutoff counter is working: aware job 20684990 recorded one actual cutoff
and scored 16/20 versus 15/20 for its matched unaware arm. All other terminal
aware jobs currently recorded zero. The five complete VH-off pairs tie exactly
at 8.8/20. Three complete VH-on pairs average 10.00 unaware versus 10.33 aware;
this is incomplete and cannot support a causal claim until both remaining
pairs terminate.

The apparent VH-on gain in the first nonbinding campaign is not a horizon
effect. Every run in that campaign recorded zero cutoffs and zero
horizon-infeasible known-goal decisions. The two differing seeds also changed
successful-instance membership and plan lengths: seed
1963100312 added `problem_8_1_2`; seed 2011206605 lost `problem_8_1_4` but added
`problem_4_2_5`, `problem_5_2_2`, and `problem_6_8_1`. Shared successful plans
were sometimes different too. Historical and new commands match except for the
horizon flag, but the new run uses the newer worker-lifecycle/code checkout;
separate TensorFlow/MDPSim worker executions are not bitwise reproducible. The
gain is therefore recorded as run-to-run/build variation. A causal horizon
claim requires fresh aware and unaware arms from one commit on a binding
horizon.

### SAFE-CONTEXT live matched experiment

SAFE-CONTEXT uses a separate `MCTSNode` for every `(physical_state,
action_history_digest)` registry key. Each node owns its own policy, value,
children, priors and visit statistics; the explicit selection-path set still
stores only physical-state keys for cycle prevention. No growing action history
is stored in the node key.

All twenty matched Drone physical/contextual jobs are running. The 2026-08-30
12:17 snapshot is stored in
`experiment_tracking/mcts_safe_context/live_progress_snapshot.csv`. Current
contextual node multipliers range from 1.045x to 35.886x, confirming both that
history aliasing occurs and that separating contexts can be expensive. Partial
success counts are not compared because paired jobs have classified different
numbers of instances; only terminal matched results will be used for coverage.

## 13. Deferred experiments

Do not run yet:

- MCTS-SAFE-2 horizon-indexed `(s, h)` statistics;
- network-first preliminary-Q ablation;
- estimator-at-admission ablation;
- large `c`/`alpha` sweep;
- MCTS-PW-PATHBATCH widening across multiple selected-path nodes in one
  simulation;
- combined progressive widening plus horizon correction before their separate
  effects are established.

## 14. Worker lifecycle is a separate change

The progressive-widening commit intentionally excludes the unfinished worker
lifecycle repair. That repair replaces shared result queues with per-worker
one-way pipes, kills timed-out worker process groups, reaps workers before
reusing their pool slot, closes communication handles, and raises a clear error
instead of attempting the impossible operation of restarting a stopped JPype
JVM inside the same Python process.

The lifecycle work must be tested and committed separately. Its working tree
also contains unrelated reproducibility/install changes, which must not be
accidentally included in the lifecycle commit.

## 15. Required integration sequence

1. Repair and rerun the isolated lifecycle test harness.
2. Split and commit the lifecycle repair independently.
3. Decide whether to cherry-pick or merge `fac11cb0` into `main`.
4. Pull the chosen production commits on the cluster only after that decision.
5. Select the pilot domain using measured MCTS instance runtimes.
6. Select one, three, or five matched seeds and label the scope accurately.
7. Submit the widening-only experiment with horizon correction disabled.
8. Submit the separate horizon-only comparison.
9. Keep the deferred designs and combined experiment on hold.

Progressive widening is now a completed negative coverage result with a clear
runtime/graph-size trade-off. Horizon-aware inference remains live and must not
be described as effective until it produces nonzero cutoff evidence on a
genuinely constrained workload.

### Progressive-widening sensitivity gate

Do not launch an undirected Cartesian grid. First add compact histograms for
node child count, visit-count bins, depth bands, and outcome class; raw per-node
dumps are explicitly excluded. Every new arm must use the same finalized
MCTS-SAFE behavior, so the pre-SAFE pilot cannot be mixed with post-SAFE arms.

The initial matched diagnostic changes one factor at a time:

1. `K_min=2, c=0.6, alpha=0.5, 70 simulations` (fresh SAFE baseline);
2. `K_min=3` only;
3. `alpha=0.65` only;
4. `140 simulations` only.

Use two seeds in each VH mode: sixteen jobs total. Do not raise `c` again in
this gate because `c=1.0` was already tested and lost more coverage. Promote
only a defensible arm to a five-seed confirmation. All sensitivity arms use
the frozen MCTS-SAFE behavior, so this gate is also the first SAFE+PW test.

The compact-v1 logger prints exact child-count histograms, visit-count bins,
selection-depth bins, and exact child widths by depth band. Each instance keeps
its own summary, allowing the evaluator result to partition distributions into
success, ordinary failure, timeout, and interruption without per-node dumps.

The two-seed gate was launched as jobs `20667974`--`20667989`. An earlier
preflight submission was cancelled within minutes because the SAFE diagnostic
printed complete root vectors on every external action. Bulk SAFE logging is
now event-only; complete root vectors are reserved for invariant failures. The
cancelled attempt IDs and their reason are preserved in
`experiment_tracking/mcts_progressive_widening_sensitivity/cancelled_verbose_submissions.tsv`
and are not experimental results.

## MCTS-SAFE-CONTEXT: action-history-aware transposition follow-up

MCTS-SAFE-1 is considered successful but incomplete: it repaired two of the
four targeted policy-only Drone failures. The other two no longer ended through
the same known-terminal selection, but still wandered to an unsolved outcome.

The next separate architectural question is contextual aliasing. The network input
contains a per-grounded-action application-count vector, while the current
transposition key contains only the physical planning state. Multiple histories
can therefore reuse one node even when the network would assign different
priors or values. The current key also intentionally omits the special
`total-time` fluent.

MCTS-SAFE-CONTEXT is staged rather than immediately changing node identity:

1. Instrument collisions where one physical-state key is observed with more
   than one action-count vector. Record collision frequency, action-count
   distance, and sampled policy/value disagreement.
2. Preserve physical-state identity for cycle and trajectory-duplicate safety.
3. Add a separate network-context identity—physical state plus action-count
   vector—for cached network predictions and MCTS statistics.
4. Compare MCTS-SAFE-1 against MCTS-SAFE-CONTEXT on matched Drone seeds.

This is not domain-specific and is not cheating: it prevents a
history-dependent network from being evaluated with another history's cached
statistics. It may improve cyclic domains, but reduces transposition sharing
and can increase runtime and memory. Instrumentation must therefore precede the
behavioral experiment.

The current implementation explains why this needs two identities. `MCTSNode`
stores one `CanonicalState`, its `as_network_input`, policy prediction, value,
visits and children. `state_key_to_node` is indexed only by the physical
`state_key`; when that key is revisited, the existing node and its cached
network input are reused even if the new `CanonicalState.aux_data` action-count
vector differs.

The failure is broader than reusing one cached value. In the physical-node
baseline, inference can overwrite the reused node's `act_dist` and predicted
value with the newly encountered action-history context while retaining the
already-created fixed-width child set, edge priors, visits and child
statistics from the first context. Fixed-width expansion then refuses to
expand the node again because it already has children. The result can be a
new policy distribution paired with stale top-k children and stale edge
priors. SAFE-CONTEXT prevents this entire mixed-context node state: each
context receives its own policy, value, children, priors, visits and child
statistics, while physical-key cycle checks remain shared.

The sound refactor would use:

- `physical_key = state.state_key` for ancestor-cycle and trajectory-duplicate
  checks;
- `context_key = (physical_key, digest(state.aux_data))` for MCTS nodes,
  network inputs, priors, values, visits and children;
- the existing physical-state estimator cache where the estimator itself is
  genuinely history-independent.

The context identifier must not be a growing Python list. Compute a canonical
128-bit BLAKE2 digest from the already-existing contiguous `float32`
`CanonicalState.aux_data` bytes. Each dictionary key then adds one fixed-size
digest rather than another action-count vector. In debug/sampled runs, retain a
small bounded digest-to-vector witness cache to assert that no observed digest
collision maps two different vectors to the same context. The main memory risk
is therefore not the 16-byte digest; it is creating multiple full `MCTSNode`
objects for one physical state when multiple histories are genuinely present.

Edges would point to context nodes. Cycle detection would compare only each
node's physical key along the current trajectory, so creating multiple context
nodes would not bypass duplicate safety. This costs transposition sharing and
memory, hence collision instrumentation comes before the behavioral refactor.

### Proposed MCTS-SAFE-CONTEXT implementation diff

1. `post_training/monte_carlo_tree_search.py`
   - add fixed `physical_key` and `context_digest` fields to `MCTSNode`;
   - rename the node registry to `context_key_to_node`;
   - add counters for physical-key collisions, distinct contexts per physical
     key, sampled action-count distance, and sampled policy/value disagreement.
2. `post_training/training_mcts.py`
   - compute the digest from `CanonicalState.aux_data` before node lookup;
   - key node reuse by `(state.state_key, context_digest)`;
   - keep estimator caching physical-state-only unless the estimator is later
     shown to consume auxiliary history;
   - keep trajectory cycle checks on `physical_key`, not `context_key`.
3. `post_training/action_selection_policy.py`
   - explicitly use physical identity for duplicate masks and SAFE fallbacks.
4. `asnets/asnets/scripts/run_asnets.py` and `run_experiment.py`
   - expose an instrumentation-only flag first; the behavioral contextual-node
     flag remains separate and opt-in.
5. Tests
   - same physical state plus same history reuses a node;
   - same physical state plus different history creates two context nodes;
   - those two nodes still trigger physical-cycle protection;
   - cached priors/values never cross contexts;
   - bounded witness sampling and aggregate counters do not grow per visit.

The instrumentation pilot is deliberately short: one seed and one VH mode in
Drone, Rover, Counters and Block Grouping; three representative instances per
domain; one worker; at most 100 external actions per instance; fixed width-20,
70-simulation MCTS. It records aggregate collision/context counts and samples
at most 128 disagreement witnesses per job. It is not a coverage experiment and
does not run complete test suites.

### SAFE-CONTEXT diagnostic outcome (2026-08-29)

The first cluster diagnostic invocation evaluated one representative instance
per domain rather than the intended three. All four inference calls completed,
but Slurm marked the wrappers failed afterward because the isolated checkout
did not contain `asnets/tools/validate_eval_log.py`. This is a post-hoc
validator-path defect, not an inference or SAFE-CONTEXT failure. The exact
inference evidence is authoritative in
`experiment_tracking/mcts_safe_context/diagnostic_results.csv`.

| Domain | Physical observations | Context mismatches | Node multiplier | Mean / max policy L1 disagreement | Inference |
|---|---:|---:|---:|---:|---:|
| Drone | 64 | 11 | 1.204 | 0.0036 / 0.0374 | 1/1 |
| Rover | 91 | 2 | 1.022 | 0.0620 / 0.0652 | 1/1 |
| Counters | 139 | 113 | 5.440 | 0.3456 / 1.0131 | 1/1 |
| Block Grouping | 140,000 | 56,239 | 1.902 | 0.0202 / 0.0358 | 0/1 at the diagnostic 100-action limit |

The pilot therefore establishes that action-history aliasing is real and can
materially change network predictions. Counters exhibits severe node
multiplication and prediction disagreement; Block Grouping also exhibits a
large structural effect. The full twenty-job Drone comparison remains held
until the wrapper is repaired and a corrected three-instance diagnostic,
including observed peak memory, passes the declared gate. Do not reinterpret
the four Slurm `FAILED` states as failed inference.

The two wrapper defects are repaired: the comma-bearing restriction is base64
encoded across Slurm's `--export` boundary, and validation uses the production
validator path. The stale held jobs 20684998--20685018 were cancelled without
running and archived in `stale_full_submissions_20260829.tsv`.

The corrected three-instance diagnostics 20688113--20688116 completed on
2026-08-30 with no tracking overflow. Exact per-instance evidence is in
`corrected_diagnostic_instance_results.csv`. Observed node multipliers were:

- Drone: 1.204 on the completed small instance; the two larger instances hit
  their four-hour diagnostic timeout before emitting a terminal summary;
- Rover: 1.022 on the successful instance and 3.379 on the 100-action failure;
- Counters: 5.440, 1.316 and 1.000 across the three representatives;
- Block Grouping: 1.902, 1.484 and 1.042.

Because all bounded jobs completed without tracking overflow, the operational
instrumentation gate passed. Twenty replacement matched Drone jobs
20688117--20688136 were released from user hold on 2026-08-30. The two timed-out
Drone diagnostics mean that hard-instance memory evidence is incomplete; any
full-run OOM is therefore retained as an experimental outcome rather than
silently retried with different resources.

The original one-instance runtimes were approximately 57 seconds (Drone), 79
seconds (Rover), 13 seconds (Counters) and 65 minutes (Block Grouping). Three
representatives are required because one instance cannot bound context
multiplication or peak memory; the corrected jobs retain a 12-hour limit.

### Counters Stage-2 narrow policy-loss audit

The six terminal VH-off Stage-2 pairs are joined instance by instance in
`experiment_tracking/mcts_counters_width_sensitivity/stage2_narrow_pair_audit.csv`.
The aggregate policy advantage comes entirely from two seeds:

- seed 1963100312: policy 59, narrow 49; nine policy successes became MCTS
  `unsolved` outcomes at exactly 10,000 actions and one reached the six-hour
  per-instance timeout;
- seed 1073581256: policy 53, narrow 51; two policy successes became MCTS
  `unsolved` outcomes at exactly 10,000 actions;
- the other four terminal seeds were equal or improved under narrow MCTS.

The decline is therefore search wandering on the largest instances, not
invalid output or a VAL discrepancy.

The compact trajectory audit in
`stage2_policy_only_trajectory_audit.csv` confirms the mechanism. Successful
policy plans for the twelve lost instances require only 1,378--2,760 actions.
Eleven MCTS failures persist a complete 10,000-action trajectory: they diverge
from the policy at the first action, contain roughly 3,286--4,059 decrement
actions, and contain 7,330--8,622 action occurrences above the successful
policy plan's action multiset. The remaining six-hour-timeout instance has a
2,607-action successful policy plan but no persisted partial MCTS trajectory
because its worker was killed before writing a completion record.

Both successful policy action lists and the eleven completed-unsolved MCTS
action lists are retained verbatim in their original logs/JSONL. This is
repeated increment/decrement wandering, not a terminal dead-end and therefore
not something MCTS-SAFE-1's terminal mask can repair by itself.

SAFE-CONTEXT is a plausible direct remedy only when the oscillation is caused
by reusing physical-state node statistics across distinct action-count network
contexts. The diagnostic evidence makes that mechanism especially plausible
in Counters: one representative had a 5.440 context-node multiplier and large
policy disagreement. It is not yet a causal result; the matched physical versus
contextual experiment must establish whether coverage improves.

MCTS-SAFE-2 could help indirectly when the same physical state is valued under
incompatible remaining action budgets, but horizon-indexed statistics do not
by themselves prevent increment/decrement cycles. A 10,000-action oscillation
can remain possible within one horizon context. Consequently neither
SAFE-CONTEXT nor SAFE-2 is claimed as a guaranteed cycle-prevention mechanism.

## MCTS-SAFE-2: fully horizon-indexed search

`MCTS-SAFE-2` is reserved for the fully horizon-correct design requested by the
experiment owner. Its search state is `(s, h)`, where `h` is remaining
executable actions. Q-values, visits, goal feasibility and child statistics are
therefore horizon-specific instead of sharing one physical-state node across
incompatible remaining budgets. The live MCTS-HORIZON experiment implements
only cutoff and feasible-goal chasing over state-only statistics; it is not
MCTS-SAFE-2. This design remains held because it can multiply node statistics
and memory substantially.

### Proposed MCTS-SAFE-2 implementation diff

1. Add `remaining_horizon` to `MCTSNode` and define exact node identity as
   `(physical_key, remaining_horizon)`; do not bucket horizons in the
   correctness implementation.
2. In selection/expansion, create a child with horizon `parent_horizon - 1`.
   Refuse expansion at zero and evaluate the cutoff without writing a failure
   penalty into any other horizon's statistics.
3. Store Q-values, visits, children, known-goal distance/feasibility and
   backpropagation statistics on the horizon-specific node.
4. Keep physical-key ancestor checks for cycle safety. The same physical state
   at another horizon remains a distinct statistics node but is still a
   physical repeat on the current trajectory.
5. Keep state-only estimator results shareable only if the estimator is truly
   horizon-independent; estimator blending happens separately in each
   horizon-specific MCTS node.
6. Add counters for physical states multiplied across horizons, peak
   horizon-specific node count, memory, cutoffs, and infeasible known goals.
7. Add regression tests for cross-horizon Q isolation, horizon decrement,
   zero-horizon expansion refusal, goal feasibility, cycle safety, and root
   reuse after an external action.

MCTS-SAFE-CONTEXT and MCTS-SAFE-2 should remain separate first. Their combined exact
identity would be `(physical_key, action_history_digest, remaining_horizon)` and
could multiply nodes along both dimensions; combining before measuring each
dimension would obscure both causality and memory cost.

### Binding-horizon counter audit (2026-08-30)

The cutoff counter is wired correctly. Selection returns the explicit
`horizon` stop reason at `selection_depth >= max_depth`, the iteration records
that depth, and focused tests assert both a nonzero cutoff count and refusal to
expand beyond the declared remaining depth. The first eight terminal jobs
genuinely logged zero cutoffs, but that was not a campaign-wide result.
Two still-running aware jobs have now exercised the mechanism:

- job 20684990: one cutoff at selection depth 59;
- job 20684994: 2,053 cutoffs over depths 11--41 (mean 25.47).

The experiment is therefore binding on at least two observed instances. Any
older statement that every run had zero cutoffs is superseded. Jobs
20684982--20684985 are also now post-hoc VAL-confirmed, so 12/20 inference
runs are terminal and validated while eight remain live.

## MCTS-PW-PATHBATCH: held multi-node widening

This held experiment implements the previously proposed nonstandard variant:
selection continues to a leaf while collecting every node on that path whose
permitted progressive width exceeds its current child count. After the path is
fixed, one new policy-ordered child is generated for each collected node and
their network inputs are evaluated as one batch. The simulation still performs
one declared leaf backup; additional admitted nodes receive an explicit,
separately defined initialization update rather than pretending that several
independent simulations occurred.

This can recover TensorFlow batching and widen several useful depths at once,
but it is not standard one-path/one-expansion MCTS and may spend multiple
successor generations per simulation. It therefore remains held until the
standard PW sensitivity and compact depth/width histograms are complete. A
future matched pilot must report generated states per simulation, network batch
size, successor-generation time, retained nodes, memory, runtime and coverage.
