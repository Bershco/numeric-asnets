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
| 168 policy-curve tasks | 21453461_[0-167] | legacy broad gate; supersede before incremental release |
| statistical finalizer | 21453462 | legacy broad gate; supersede before incremental release |

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

Deployment must first cancel/supersede the old dependency-pending curve array
and finalizer so they cannot duplicate the incremental jobs.  The incremental
watcher can then be submitted with `after:<training-array-job-id>` (release
after the training array starts, not after it terminates).  No part of this
replacement chain was submitted during local implementation.
