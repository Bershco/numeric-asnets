# Endogenous 100-epoch KL-semantics pilot

## Question

Does replacing the historical dropout-current anchor KL with
deterministic-current KL improve complete Stage-2 training trajectories when
each arm generates its own replay, rather than consuming a frozen legacy
stream?

## Predeclared design

- Domains: Drone (cheap/stable) and FO Counters (clearest systematic
  PRIMARY-50 one-epoch harm/attenuation signal).
- Seeds: `534933607` and `923500475`, the first two canonical replication IDs
  in numeric order for each domain. They were not selected from KL outcomes.
- Networks: VH-off, validation-selected Stage-1 checkpoints, exact hashes in
  `manifest.csv`.
- Arms: historical dropout-current KL and deterministic-current KL.
- Both arms use the same current build and all common Stage-2 hyperparameters:
  anchor coefficient 3, learning rate 0.0003, 100 epochs, three workers,
  original training set, and the domain's frozen ENHSP teacher.
- Each matched pair passes the same canonical experiment `--random-seed` shown
  in `manifest.csv`. The special fixed-step RNG flag used by causal frozen-
  replay diagnostics is intentionally absent: this pilot compares the normal
  endogenous training algorithms, including their downstream replay feedback,
  rather than forcing an artificial per-update stochastic schedule.
- Replay is endogenous. No frozen-replay option is passed. Each arm generates
  and consumes its own states and targets throughout training.
- Policy coverage is evaluated at epochs `0,5,...,95,99`. Epoch 99 and the
  complete learning trajectory are predeclared outcomes; test results are not
  used to choose a checkpoint.
- MCTS is outside this pilot. It is warranted only if deterministic-current KL
  becomes a candidate replacement training method.

This is a two-seed method-development pilot, not a population-level efficacy
claim. A negative or inconsistent result stops expansion. A coherent benefit
justifies a ten-seed confirmation before replacing historical RQ results.

## Reuse audit

Historical MAIN-VAL legacy Stage-2 runs exist for all four source networks,
but they used the historical build. They are retained as provenance and may be
shown descriptively; they are not reused as matched pilot arms because doing so
would confound KL semantics with code/build changes. No existing
deterministic-current 100-epoch run starts from these exact four checkpoints.
Therefore all eight same-build training identities are new.

## Resources and dependency chain

- compute smoke: 2 CPU, 8 GiB, 10 minutes;
- training: eight tasks, each 6 CPU / 48 GiB / 18 hours;
- policy curves: 168 tasks, each 5 CPU / 20 GiB / 2 hours, released only after
  all training tasks succeed;
- finalizer: 1 CPU / 4 GiB / 30 minutes.

Known failing nodes are excluded. Slurm, rather than an artificial array cap,
controls how many tasks run concurrently.

## Outputs

Remote root:
`/home/hersco/training_new_domains/2026-09-18/endogenous_kl_100epoch_pilot`

The finalizer writes `learning_curve_results.csv`, `endpoint99_results.csv`,
and `domain_summary.csv`, retaining every source checkpoint, task identity,
training log and endpoint-evaluation log.

## Submission

The reviewed orchestration is commits `0930bb78` and `4a86e13a`. Scientific
training uses the frozen corrected-KL checkout at full commit
`c72c9080d2a314e6177b2e5df4079a02ef44e621`.

| Role | Job | Dependency |
|---|---:|---|
| compute smoke | 21453459 | none |
| eight training tasks | 21453460_[0-7] | afterok 21453459 |
| obsolete broad curve array | 21453461_[0-167] | cancelled; superseded without scientific output |
| obsolete broad finalizer | 21453462 | cancelled; superseded without scientific output |
| incremental watcher | 21455207 | active while training continues |
| first ready policy batch | 21455208 | 60 exact checkpoint identities submitted |

## Incremental curve and selected-search controller

The original curve array waits for all eight training tasks and its worker also
requires a terminal `training_complete.json`.  The replacement controller in
`scripts/endogenous_kl_incremental_controller_20260918.py` instead discovers
each stable checkpoint while training continues, appends an exact identity to
`policy_eval_manifest.csv`, freezes its weights SHA-256, and submits only a
missing curve evaluation.  It never rewrites an existing checkpoint identity.
The watcher runs every five minutes using one CPU and 2 GiB.

The watcher reuses every valid curve result already present.  Submission
ledgers prevent duplicate active tasks.  A scheduler-completed task without an
exact result is a fail-closed condition requiring targeted diagnosis rather
than broad automatic repetition.

After all eight trainings expose all 168 checkpoints, the same controller:

1. selects each arm's maximum held-out validation score from the per-epoch
   `[VALIDATION]` records, breaking ties by the
   earliest epoch;
2. freezes `selected_endpoints.csv`, including exact checkpoint hashes and
   the already-computed policy score;
3. ensures the eight selected policy evaluations are complete without waiting
   for non-selected curve points;
4. materializes 16 selected-search identities: eight fixed top-20/70 and eight
   PW70 (`Kmin=3`, `c=.6`, `alpha=.5`);
5. submits only missing search identities at six CPU, 120 GiB and 72 hours per
   identity, with six-hour per-instance limits and durable completion ledgers.

An `afterany` one-cycle controller is chained to each selected-search array.
It reconciles durable JSONL with exact 21,600-second timeout markers, retains
every classified instance, and reruns only an identity that still contains
unclassified instances; the evaluation ledger skips its durable successes.
When all 16 identities reach 20/20 classifications it writes
`selected_mcts_summary_6h.csv`.  Thirty-minute and two-hour recensoring remains
a downstream analysis of the immutable attempt logs; it is not fabricated by
this submission controller.

The remaining curve points continue concurrently.  When all 168 results are
available, the controller writes `learning_curve_results.csv`,
`endpoint99_results.csv`, and `domain_summary.csv`; selected MCTS is not held
behind those non-selected test-policy evaluations.

The obsolete array/finalizer were cancelled before the replacement was
released.  Watcher `21455207` is canonical; it discovered 60 stable checkpoint
identities on its first pass and submitted them as array `21455208`.  Later
passes append only newly stable checkpoints.  Validation-selected fixed 20/70
and PW70 jobs remain automatically gated on complete training validation
records plus the eight exact selected-policy results.

## 19 September interruption and continuation

All eight original training tasks ended `FAILED` after approximately 2h32 on
18 September.  This was not an OOM, timeout, or application exception: the
tasks ran on five distinct nodes, stopped within 77 seconds of one another,
and every log ends abruptly inside ordinary validation evaluation without a
traceback.  Other independent jobs also failed in that interval.  The evidence
therefore supports a synchronized operational interruption; its exact external
trigger is not present in the application or Slurm accounting logs.

The durable Stage-2 checkpoint frontiers were local epochs 25, 11, 28, 24,
67, 64, 66, and 56 for arms 0--7.  Recovery resumes each arm from its latest
complete weights, optimizer, trainer-state, validation-selection state, and
persisted original Stage-1 anchor.  It runs exactly `100 - (frontier + 1)`
additional epochs.  Replay memory and process RNG state are not checkpointed;
the continuation boundary and this limitation are retained as provenance.

Continuation snapshots live in a separate numbered `continuations/segment_*`
directory and never overwrite original logs or checkpoints.  The incremental
controller maps each segment's local snapshot number back to the canonical
Stage-2 epoch, preserves all valid policy evaluations, and submits only missing
curve identities.  `recovery_manifest.csv` freezes the eight exact source
checkpoint hashes before submission.

The recovery revision changes only orchestration, tests, and documentation;
the scientific implementation under `asnets/` remains the frozen `c72c9080`
build.  Existing policy results retain their exact original evaluator commit,
while new missing evaluations record the recovery-controller checkout.  They
are reused because the evaluator implementation is unchanged, not because
provenance was discarded.

The first recovery attempt was deployed from orchestration commit `77f2ec43`
at 11:40 IDT on 19 September:

| Role | Job | Tasks / resources | State at verification |
|---|---:|---|---|
| exact training continuations | `21463862_[0-7]` | 8 × 6 CPU / 48 GiB / 24h | cancelled after quota failures/holds; no continuation checkpoint produced |
| incremental curve/search watcher | `21463863` | 1 task / 2 GiB / 20h | cancelled after submitting retry array `21463871` |
| five missing policy identities | `21463871_[0-4]` | 5 × 5 CPU / 20 GiB / 4h | cancelled after quota failures/holds |

At release, 73 canonical curve checkpoints had been discovered and 68 already
had exact policy results.  The watcher correctly submitted only the five
missing existing identities.  The recovery then exposed a user-home quota
failure: Slurm could not create some task environments/stdout files, and the
tasks that did start stopped during initialization.  No task produced a new
continuation checkpoint.  The three job groups were cancelled to avoid
conflicting partial writers; all original checkpoints and the 68 exact policy
results remain reusable.

The dominant quota consumer is approximately 687 MiB of frozen first-update
batch payloads from the original causal audit.  These are scientific evidence
and were not deleted.  The repaired continuation deliberately does not request
a second first-update audit, accepts a new numbered segment and manifest, and
can restart from the unchanged frozen frontiers once sufficient persistent
quota is available.  Expected continuation time remains approximately 1--8
hours for seven arms and about 18--19 hours for the slow Drone arm; 24 hours is
the scheduler hard limit, not the expected duration.
