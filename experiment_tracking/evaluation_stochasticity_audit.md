# Evaluation stochasticity and repeatability audit

This audit covers the policy and MCTS evaluation paths actually used by the
thesis campaigns. It separates deliberately stochastic code that is inactive
in current evaluation from active mechanisms that can make two nominally equal
runs diverge.

## What is deterministic in the declared evaluation configuration

- Both policy and MCTS experiment architectures set `ACTION_POLICY="argmax"`,
  epsilon 0 and no sampling temperature. External evaluation actions therefore
  use deterministic `numpy.argmax`.
- MCTS action admission is policy ordered with action-ID tie-breaking via
  `numpy.lexsort`. PUCT scans children in stable array order and uses a strict
  best-score comparison.
- Every worker derives a stable seed from the trainer seed and evaluation slot;
  Python, NumPy and TensorFlow RNGs are explicitly seeded.
- Inference calls the network with `training=False`; dropout/training-time
  behavior is inactive and weights are restored exactly.
- Problem metadata sorts propositions, fluents, comparisons and grounded
  actions, removing most hash/dictionary-order dependence.

Sampling implementations do exist in `action_selection_policy.py`, training
trajectory collection and legacy UCT code. They are not active in the current
argmax evaluation jobs. Their mere presence does not explain repeat-run drift.

## Active or plausible divergence sources

1. **TensorFlow/CPU numerical ordering.** Seeds are set, but the evaluation
   wrapper does not set `TF_DETERMINISTIC_OPS`, fixed TensorFlow intra/inter-op
   thread counts, or fixed OpenMP/MKL thread counts. CPU oneDNN/MKL reductions
   can differ slightly across processes/nodes. A tiny prior/value change near a
   PUCT tie can select a different branch and cascade into a completely
   different long trajectory.
2. **ENHSP estimator process.** Estimator coefficient .5 launches an external
   Java heuristic service. Its internal best-action tie ordering and numerical
   behavior are not seeded or logged strongly enough to prove repeatability.
   Because its output is blended into both action and value estimates, this is
   a serious candidate for the first divergence.
3. **Wall-clock boundaries.** Six-hour per-instance checks occur around external
   action/search work. Node speed and process scheduling can decide whether one
   more search/action completes. This explains differences close to 21,600s,
   but not trajectories that diverge early and finish far from the limit.
4. **Build/process differences.** Historical baselines and newer reruns span
   worker-lifecycle, SAFE, PW and horizon commits. Opt-in flags should isolate
   behavior, but a historical/fresh comparison is not a bitwise-replication
   claim. Fresh same-commit pairs are required for causality.
5. **State aliasing and rounding.** Physical keys round numeric fluents and omit
   designated special fluents. This is deterministic rather than random, but a
   tiny earlier action difference can cause distinct histories to reuse the
   same state statistics and amplify divergence.
6. **Multiprocessing order.** Forkserver workers have stable per-instance seeds,
   so launch order should not change an isolated result. It can still alter CPU
   contention and hence a result that is near a wall-clock boundary.

## Completed diagnostic: MCTS-DETERMINISM-AUDIT

Use one exact checkpoint and one known variable instance with one worker from
one commit/container.  The bounded first gate compares three exact repeats of:

1. current environment;
2. deterministic CPU environment (`TF_DETERMINISTIC_OPS=1`, fixed
   `PYTHONHASHSEED`, OpenMP/MKL/TF intra/inter-op threads all 1);
The estimator is not disabled in the first gate because changing its coefficient
would change the algorithm.  Its value is logged separately; an estimator-only
follow-up is justified only if the first differing record is the estimator.

At every root, record state digest, ordered applicable action IDs, network
prior/value checksum, estimator value/action, every child's N/Q/U/prior and the
selected action. The first differing record identifies the responsible layer:

- state/action order differs: environment or canonicalization;
- network tensor differs first: TensorFlow numerical execution;
- estimator differs first: Java/ENHSP;
- tensors agree but PUCT choice differs: ordering/statistics bug;
- everything agrees until the limit: timing-only censoring.

The opt-in logger is implemented behind `--action-debug`.  Container preflight
`20771356` passed; ordinary jobs `20771357--20771359` and deterministic-CPU
jobs `20771360--20771362` all completed.  Each used one worker, 2 CPUs and
20 GiB.  All six solved the same Drone instance in exactly 105 external
actions and finished in 137.51--151.03 seconds, so every arm is 1/1 at the
30-minute, two-hour and six-hour cutoffs.

The experiment localizes the observed variation more narrowly than the
original list of hypotheses:

- three ordinary repeats on `cs-cpu-07` were checksum-identical for all 105
  decisions;
- enabling deterministic TensorFlow/thread settings on the same
  `cs-cpu-07` node changed no checksum;
- two deterministic repeats on `ise-cpu-intl-07` were identical to each
  other;
- comparing `cs-cpu-07` with `ise-cpu-intl-07` changed the raw network-policy
  checksum at 78/105 decisions and child-statistics checksum at 45/105, while
  physical state, action history and selected action changed at 0/105.

The supported cause is therefore **hardware/node-type-dependent numerical
execution inside the network/low-level CPU math stack**, not random process
timing and not a different MCTS depth.  Different CPU instruction sets,
oneDNN/MKL kernels, vectorization or reduction order can produce slightly
different floating-point tensors.  Those differences propagated into child
statistics here, but did not cross an action-selection boundary.  The audit
does not yet prove that node type changes coverage; it proves the first layer
at which nominally identical runs can differ.  A causal follow-up should pin
ordinary/deterministic repeats to one node type and compare node types only on
an instance already known to select different actions.

See `mcts_determinism_audit/results.csv`, `checksum_summary.csv`,
`manifest.csv` and `submissions.tsv` for exact job and log provenance.

The selected node-pinned follow-up is the fresh same-commit
Drone VH-on seed `1239739722` Horizon pair: unaware job `20684991` scored 9/20
and aware job `20684992` scored 6/20, while the aware arm recorded zero horizon
cutoffs. Their per-instance membership was joined before submission. The aware
arm lost three instances and gained none; `problem_3_3_4` was selected because
the unaware arm solved it in 404.85 seconds while the aware arm ended unsolved
in 379.24 seconds. Preflight `20788665` passed. Three repeats pinned to
`cs-cpu-07` and three pinned to `ise-cpu-intl-07` are jobs
`20788666`--`20788671`; each uses one worker, two CPUs, 20 GiB and a two-hour
cap. They were eligible node-pending at the 20:35 snapshot. See
`mcts_determinism_audit/followup_candidates.csv` and
`followup_submissions.tsv`.

## Held architecture ablation: ACT-HISTORY-ABLATION

`USE_ACT_HISTORY_FEATURES=True` is inherited through
`experiments_numeric/architecture/actprop_2l_comparison.py`.  The feature is a
cumulative count per grounded action, not just total time.  Numeric ASNets does
not isolate its effect; the original ASNets work reported that action counters
helped direct policies, while their interaction with transposition-based search
remains untested.

Disabling it may remove a source of network-context aliasing, slightly reduce
the input and inference cost, and make physical-state transpositions internally
consistent.  It may also remove a useful repetition/cycle signal and reduce
policy quality.  Because input dimensionality and learned weights change, this
requires fresh Stage-1 training and downstream policy/MCTS evaluation; existing
weights cannot be reused.  The experiment is registered but held.  A defensible
first gate uses Drone and Counters plus one acyclic control before any full
campaign.

## Relevant implementation pointers

- `asnets/experiments_numeric/architecture/actprop_2l_comparison_mcts.py`:
  active argmax/epsilon/duplicate settings.
- `post_training/action_selection_policy.py`: deterministic argmax and inactive
  sampling alternatives.
- `asnets/asnets/spawn_train_worker.py`: stable worker seeding, inference and
  external MCTS action selection.
- `asnets/asnets/parllel_explore_spawn_grads.py`: stable evaluation slot IDs and
  forkserver worker lifecycle.
- `post_training/monte_carlo_tree_search.py`: deterministic PUCT/admission order.
- `post_training/enhspwrapper.py`: external heuristic service boundary.
- `asnets/asnets/utils/py_utils.py`: Python/NumPy/TensorFlow seed setup.
