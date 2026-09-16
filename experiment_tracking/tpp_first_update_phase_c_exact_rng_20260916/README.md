# TPP Phase-C exact-RNG causal closure

## Purpose

The completed selected-pair Phase C prevented the catastrophic seed's
first-update collapse: its one-epoch endpoint was `20/20`, and the stable
control remained `20/20`.  However, rejected retries resampled dropout.  The
result therefore proved that the practical guard prevented this selected
collapse, but did not isolate rollback/learning-rate reduction from a lucky
new dropout realization.

This campaign closes only that causal gap. It adds a matched inactive-guard
control under the identical fixed RNG schedule, so a good active-guard result
cannot be credited merely to a lucky replacement dropout stream. It is not a
new Stage-2 method, population screen, or 100-epoch retraining campaign.

## Design

The experiment runs four one-epoch treatments and four dependent endpoint
evaluations. Each seed is evaluated with an active guard and an inactive
instrumented guard. The inactive arm uses the same code and deterministic RNG
schedule, but frozen limits of `1e6` make rejection impossible in practice.

| Arm | Seed | Treatment | Frozen start | Frozen replay | Historical first S2 |
|---|---:|---|---|---|---:|
| Catastrophic outlier | 1972442430 | Active guard | Same Phase-C Stage-1 checkpoint | Same 60 batches | 10/20 |
| Stable control | 2082152039 | Active guard | Same Phase-C Stage-1 checkpoint | Same 60 batches | 20/20 |
| Catastrophic outlier | 1972442430 | Inactive guard | Same Phase-C Stage-1 checkpoint | Same 60 batches | 10/20 |
| Stable control | 2082152039 | Inactive guard | Same Phase-C Stage-1 checkpoint | Same 60 batches | 20/20 |

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

- Scientific work: eight tasks total—four one-epoch training arms and four
  endpoint evaluations.
- Preflight smoke: one small non-scientific job.
- Training maximum concurrency: 24 CPUs / 192 GiB for at most two hours.
- Endpoint maximum concurrency: 20 CPUs / 80 GiB for at most two hours.
- No new MCTS targets, Stage-1 training, 100-epoch training, or eight-seed
  population screen.
- Every job asserts that its checkout still resolves to the commit recorded by
  the submitter. The submission ledger is appended after each scheduler call,
  so a partial chain cannot be resubmitted into duplicate upstream jobs.

## Interpretation gate

- Active bad exceeds inactive bad, while active and inactive controls match:
  selected-pair causal evidence that rollback/backtracking prevents the
  collapse without harming the control.
- Active and inactive bad both score `20/20`: the fixed stochastic stream
  avoids the collapse, so no guard effect is established.
- Active bad scores below inactive bad: the treatment is harmful here.
- Active control scores below inactive control: the treatment fails the
  selected-control non-regression gate.

Even a positive result remains selected-pair mechanism evidence; it does not
estimate how often the guard helps across seeds or domains.
The fixed step seeds deliberately define a fresh dropout sequence, so this is
not a bit-for-bit replay of the earlier resampling-enabled run. The matched
inactive arm isolates the effect of activating rollback within that sequence.

## Provenance

- Frozen protocol: `protocol.json`.
- Per-arm inputs and remote log routes: `manifest.csv`.
- Resource/dependency plan: `planned_jobs.csv`.
- RNG capture and gradient verification: `asnets/asnets/replay_rng.py` and
  `asnets/asnets/supervised.py`.
- Focused tests: `asnets/tests/test_replay_rng.py`.
- Literature basis and broader-pilot decision:
  `../../docs/stage2_trust_region_guard_research_notes_20260916.md`.
- Slurm scripts:
  `scripts/tpp_first_update_phase_c_exact_rng_{smoke,train,endpoint}_20260916.sbatch`.
- Safe idempotent submitter:
  `scripts/submit_tpp_first_update_phase_c_exact_rng_20260916.py`.

## Completed result

Compute smoke `21394945`, training array `21394946_[0-3]` and endpoint array
`21394948_[0-3]` all completed. The fail-closed gradient-hash checks passed.

| Seed role | Active guard | Inactive guard | Active retries | Inactive retries |
|---|---:|---:|---:|---:|
| Catastrophic outlier `1972442430` | 20/20 | 20/20 | 8 | 0 |
| Stable control `2082152039` | 20/20 | 20/20 | 4 | 0 |

Canonical row-level evidence is in `results_final_20260916.csv`.

This matches the predeclared second interpretation gate: the fixed stochastic
stream itself avoided the collapse. The active guard did not outperform its
inactive exact-RNG match, so this experiment establishes **no causal guard
effect**. It also shows why the earlier selected-pair 20/20 repair cannot be
attributed uniquely to rollback or learning-rate backtracking.

The broader twelve-job, two-domain, three-arm pilot remains held. Submitting it
under the present gate would be scientifically unjustified; a redesigned test
must first expose more than one frozen stochastic stream or otherwise create a
setting in which active and inactive arms differ without selecting streams by
test outcome.
