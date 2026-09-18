# MPrime final validation-led Stage-2 search

The final Phase-B-A search campaign contains exactly forty identities:

- ten seeds x two value-head modes x fixed 20/70 MCTS;
- the same ten seeds x two value-head modes x PW70;
- the Phase-B-A-selected checkpoint for every lineage;
- the exact selected policy job/log, source training job, checkpoint path and
  checkpoint SHA-256 for every identity.

The obsolete pre-rescore array `21388436` was cancelled and dropped from all
result tables because it used the saturated training validator rather than the
final Phase-B-A selector. Its raw scheduler logs remain immutable operational
provenance only; they are not an experimental branch or “quarantined result.”

The corrected controller completed all 420 Phase-B-A checkpoint evaluations,
froze twenty endpoints, reused twelve exact-hash policy evaluations, and ran
only eight genuinely missing VH-on policy endpoints. All twenty selected policy
scores are now complete. Array `21429177_[0-39]` evaluates the exact selected
checkpoints under fixed 20/70 and PW70; all forty tasks were running at the
17 September 2026 11:51 IDT snapshot.
Each task requests six CPUs, 120 GiB and a 72-hour allocation, with three
rolling workers, a six-hour per-instance limit and a 10,000-action limit.
The fixed and PW arms share width 20, 70 simulations, PUCT 0.1 and estimator
mixture 0.5. PW additionally uses Kmin 3, c 0.6 and alpha 0.5.

These Stage-2 networks were trained with the historical **dropout-current KL**
semantics. Their exact training commands contain the constant anchor
coefficient but not `--policy-anchor-kl-deterministic-current`. Accordingly,
the MPrime RQ1/RQ3 extension is evidence for the historical implemented
Stage-2 pipeline, not evidence for a corrected deterministic-current pipeline.

`manifest.csv` is the scientific identity table. `submissions.tsv` maps every
identity to its Slurm array task. `source_ready_manifest_20260916_1024.csv` and
`selected_policy_scores_20260916_1024.txt` freeze the upstream policy evidence.
The local runner verifies the declared code revision and checkpoint hash before
evaluation and writes one durable completion ledger per identity.

At 15:33 IDT, 422/800 classifications were durable:

| Method | VH | Classified | 30m / 2h / 6h successes | Mean lower bound |
|---|---|---:|---:|---:|
| Fixed | off | 83/200 | 71 / 78 / 83 | >=7.1 / >=7.8 / >=8.3 |
| Fixed | on | 67/200 | 60 / 64 / 67 | >=6.0 / >=6.4 / >=6.7 |
| PW70 | off | 148/200 | 130 / 146 / 148 | >=13.0 / >=14.6 / >=14.8 |
| PW70 | on | 124/200 | 109 / 122 / 124 | >=10.9 / >=12.2 / >=12.4 |

These are live lower bounds, not final paired results; no confidence interval or
significance claim is made until every matched seed is terminal. Exact identity,
checkpoint, policy-job and log provenance is in
`phase_b_a_manifest_20260917.csv`; live counts and completion globs are in
`live_progress_20260917_1533.csv`.

At 16:32 IDT, 437/800 classifications were durable and 38/40 tasks remained
running (two were scheduler-complete):

| Method | VH | Classified | 30m / 2h / 6h successes | Mean lower bound |
|---|---|---:|---:|---:|
| Fixed | off | 90/200 | 77 / 85 / 90 | >=7.7 / >=8.5 / >=9.0 |
| Fixed | on | 75/200 | 67 / 71 / 75 | >=6.7 / >=7.1 / >=7.5 |
| PW70 | off | 148/200 | 130 / 146 / 148 | >=13.0 / >=14.6 / >=14.8 |
| PW70 | on | 124/200 | 109 / 122 / 124 | >=10.9 / >=12.2 / >=12.4 |

The array currently requests 228 CPU and 4,560 GiB.  The lower bounds still
cannot support paired confidence intervals or exact tests; no RQ2/RQ4 row is
updated until all matched identities terminate.  The machine-readable snapshot
is `live_progress_20260917_1632.csv`.

## Prepared terminal reconciliation and exact recovery

An `afterany` reconciliation/recovery chain is implemented locally but has
**not been submitted**. It preserves the original forty identities and all
their evidence. It does not rename, truncate or delete an original ledger,
attempt log or Slurm log.

The chain is intentionally instance-based rather than scheduler-state-based:

1. `reconcile_mprime_final_stage2_search_20260917.py` reads all forty durable
   completion JSONLs, every identity attempt log, and matching array Slurm
   logs.
2. It emits exactly 800 reconciliation rows and a recovery manifest containing
   only genuinely unclassified seed-instance identities.
3. `mprime_final_stage2_search_reconcile_controller_20260917.sbatch` refuses
   automatic recovery if two terminal sources disagree about an identity.
4. When the manifest is nonempty, the controller submits an unthrottled exact
   array. Every task retains the original checkpoint and fixed/PW
   hyperparameters, skips the other nineteen instances, uses one worker,
   2 CPUs, 120 GiB, the original six-hour per-instance cap and a seven-hour
   allocation.
5. A dependent `afterany` final reconciliation proves that all 800 identities
   are terminal. It does not loop or silently resubmit a second recovery wave;
   any surviving gap is emitted for review.

### Terminal evidence policy

- A completion-JSONL `success` or `finished_unsolved` record is terminal.
  `finished_unsolved` at 10,000 steps is labelled `action_limit`; an earlier
  completed dead end remains `finished_unsolved`.
- An exact `[EVAL INSTANCE] timeout` line is terminal only when its instance
  number/path match the frozen MPrime ordering and its limit is the declared
  21,600 seconds. Smoke or shortened timeouts are ignored.
- An exact instance-scoped `crashed` block with a memory-exhaustion marker is
  retained as an OOM recovery hint, not a scientific classification. That
  instance remains unclassified and is retried with 200 GiB.
- A job-level Slurm `OUT_OF_MEMORY`, cgroup kill, exit `-9`, or generic worker
  death cannot identify which of up to three active instances was complete.
  Those instances remain unclassified and are eligible for exact recovery.
- Conflicting terminal outcomes are never resolved by precedence. They stop
  the controller and require manual scientific review.

This distinction is essential: OOM is operational evidence, not a scientific
solved/unsolved classification. Generic scheduler OOM cannot even identify
which of the concurrently active instances was interrupted.

### Submission interface

After the primary array is terminal, deploy the scripts at the declared code
commit and submit the controller with the original array as an `afterany`
dependency:

```bash
CODE_COMMIT=$(git -C /home/hersco/bershco-nu-asnets/numeric-asnets-safe-context rev-parse HEAD)
sbatch --dependency=afterany:21429177 \
  --export=ALL,CODE_COMMIT="$CODE_COMMIT" \
  /home/hersco/bershco-nu-asnets/numeric-asnets-safe-context/scripts/mprime_final_stage2_search_reconcile_controller_20260917.sbatch
```

The default exact-recovery exclusion list contains only nodes already recorded
as incompatible by the current campaign family. The dynamic recovery array has
no artificial concurrency throttle; Slurm admits it under the account resource
limit. The controller writes versioned snapshots plus
`reconciliation_latest.csv`, `recovery_manifest_latest.csv`,
`conflicts_latest.csv`, `summary_latest.json` and `submissions.tsv` under
`$CAMPAIGN/reconciliation_20260917/`.

Implementation files:

- `scripts/reconcile_mprime_final_stage2_search_20260917.py`
- `scripts/run_mprime_final_stage2_search_exact_recovery_20260917.py`
- `scripts/mprime_final_stage2_search_exact_recovery_20260917.sbatch`
- `scripts/mprime_final_stage2_search_reconcile_controller_20260917.sbatch`
- optional exact-instance support in
  `scripts/run_mprime_final_stage2_search_20260916.py`

## Final reconciliation and RQ entry (18 September 2026)

The final reconciliation reached **800/800 scientifically terminal
seed-instance identities** with no conflicts or unclassified cases: 677
successes and 123 exact six-hour timeouts. The last 16 fixed-search identities
all reached their declared six-hour cap. Their recovery wrappers subsequently
reported `FAILED` because the historical evaluator does not append timeout
records to JSONL; this is an operational post-classification defect, not
missing scientific evidence.

Final means at 30 minutes / 2 hours / 6 hours are:

| VH | Policy | Fixed 20/70 | PW70 |
|---|---:|---:|---:|
| off | 16.5 | 14.2 / 15.8 / 16.6 | 16.0 / 17.7 / 17.9 |
| on | 16.7 | 12.9 / 14.6 / 16.0 | 15.2 / 17.0 / 17.2 |

All 21 VH-off and all 28 VH-on PW70 failures at six hours are exact timeouts;
none is an action-limit or unclassified outcome. PW70 exceeds fixed search in
both modes at every cutoff. The paired raw fixed-versus-PW exact tests are
`.0078/.0039/.0078` for VH-off and `.0039/.0039/.0547` for VH-on at
30m/2h/6h, respectively. Against policy, PW70 is descriptively positive by
two and six hours but is not raw-significant in either VH mode.

The primary fixed-search RQ2/RQ4 rows and the exploratory PW method comparison
are frozen in `final_method_comparisons_20260918.csv`; exact seed rows and
terminal failure causes are in `final_per_seed_results_20260918.csv`.
