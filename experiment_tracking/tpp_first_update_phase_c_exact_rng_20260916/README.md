# TPP Phase-C exact-RNG causal closure

## Purpose

The completed selected-pair Phase C prevented the catastrophic seed's
first-update collapse: its one-epoch endpoint was `20/20`, and the stable
control remained `20/20`.  However, rejected retries resampled dropout.  The
result therefore proved that the practical guard prevented this selected
collapse, but did not isolate rollback/learning-rate reduction from a lucky
new dropout realization.

This campaign closes only that causal gap.  It is not a new Stage-2 method,
population screen, or 100-epoch retraining campaign.

## Design

The experiment repeats exactly two one-epoch guarded treatments and two
dependent endpoint evaluations:

| Arm | Seed | Frozen start | Frozen replay | Historical first S2 |
|---|---:|---|---|---:|
| Catastrophic outlier | 1972442430 | Same Phase-C Stage-1 checkpoint | Same 60 batches | 10/20 |
| Stable control | 2082152039 | Same Phase-C Stage-1 checkpoint | Same 60 batches | 20/20 |

The coefficient (`3`), deterministic guard statistic, stable-control limits,
learning rate (`0.0003`), backtracking factor (`0.5`), two-retry maximum, and
endpoint probes remain frozen.  Both arms use the predeclared exact-RNG base
seed `314159`; each optimizer step derives a distinct deterministic seed.

Before a guarded step, the runtime snapshots Python, NumPy, TensorFlow global,
and every discoverable Keras dropout-generator state.  On rejection it restores
those RNG states together with the model and complete Adam state.  Crucially,
every attempt records a raw-gradient SHA-256, and the run aborts unless all
attempts for the same frozen batch have exactly the same hash.  This makes RNG
equivalence an observed invariant rather than an implementation assumption.
The compute smoke also forces one rejected TensorFlow/Keras dropout proposal,
restores all state, and requires the lower-learning-rate retry to have the
identical raw-gradient hash before either scientific arm can release.

## Scope and resources

- Scientific work: four tasks total—two one-epoch training arms and two
  endpoint evaluations.
- Preflight smoke: one small non-scientific job.
- Training maximum concurrency: 12 CPUs / 96 GiB for at most two hours.
- Endpoint maximum concurrency: 10 CPUs / 40 GiB for at most two hours.
- No new MCTS targets, Stage-1 training, 100-epoch training, or eight-seed
  population screen.
- Every job asserts that its checkout still resolves to the commit recorded by
  the submitter. The submission ledger is appended after each scheduler call,
  so a partial chain cannot be resubmitted into duplicate upstream jobs.

## Interpretation gate

- `20/20` bad and `20/20` control: exact-gradient rollback/backtracking
  prevents the selected collapse without harming the selected control.
- Bad below `20/20`: the former repair depended partly on dropout resampling
  or on the newly fixed stochastic realization.
- Control below `20/20`: the exact-gradient treatment is not a clean
  non-regression result.

Even a positive result remains selected-pair mechanism evidence; it does not
estimate how often the guard helps across seeds or domains.
The fixed step seeds deliberately define a fresh dropout sequence, so the new
endpoint is evidence about exact retry reproducibility under that frozen
realization—not a bit-for-bit replay of the earlier resampling-enabled run.

## Provenance

- Frozen protocol: `protocol.json`.
- Per-arm inputs and remote log routes: `manifest.csv`.
- Resource/dependency plan: `planned_jobs.csv`.
- RNG capture and gradient verification: `asnets/asnets/replay_rng.py` and
  `asnets/asnets/supervised.py`.
- Focused tests: `asnets/tests/test_replay_rng.py`.
- Slurm scripts:
  `scripts/tpp_first_update_phase_c_exact_rng_{smoke,train,endpoint}_20260916.sbatch`.
- Safe idempotent submitter:
  `scripts/submit_tpp_first_update_phase_c_exact_rng_20260916.py`.

## Status

Implementation and local tests are complete. Deployment must use a new
isolated checkout and a new remote output directory; the completed Phase-C
artifacts remain untouched. Submission is intentionally performed by the main
task after reviewing the exact commit and deployment bundle.
