# TPP KL-semantics multi-RNG susceptibility screen

This predeclared screen tests whether the selected TPP catastrophic seed is
reliably more susceptible to the historical dropout-current anchor KL than to
the corrected deterministic-current anchor KL.

The treatment changes only the forward pass used to compute the current side
of the anchor KL:

- `legacy_dropout_current`: the current policy in the KL term is evaluated
  with dropout active. The anchor is deterministic, so the KL gradient partly
  penalizes a stochastic dropout realization as if it were policy drift.
- `deterministic_current`: replay-policy learning still uses dropout, but the
  current policy used by the KL term is recomputed with dropout disabled. This
  makes the anchor compare two deterministic policies.

This is not a generic claim that stochastic training is wrong. It isolates a
specific semantic mismatch inside the anchor loss.

## Frozen design

- TPP/VH-off only.
- Catastrophic seed `1972442430` and stable control `2082152039`.
- The exact Stage-1 checkpoints and each seed's exact frozen 60-batch replay
  schedule from Phase A.
- One Stage-2 epoch, 60 optimizer steps, anchor coefficient 3, learning rate
  0.0003, no rollback or adaptive coefficient.
- Four predeclared optimizer-step RNG schedules: `314159`, `271828`,
  `1618033`, and `20260916`.
- Both KL semantics under every seed/RNG combination.

The already completed RNG-314159 identities are reused. The new array adds
only the remaining 12 training identities and their 12 policy endpoints.

Training array `21415031_[0-11]` completed all twelve new identities. Endpoint
array `21415032_[0-11]` began evaluation; tasks 0, 1 and 4 were held before
inference by Slurm user-environment retrieval. They were replaced exactly by
environment-independent array `21415252_[0,1,4]`. The original scientific
identities and outputs are unchanged; no completed endpoint is repeated.

The held task 4 subsequently completed and is reused.  Tasks 0 and 1 were
replaced without changing their scientific identities by exact endpoint jobs
`21428592_0` and `21428592_1` (scheduler display may expose their component
job IDs as `21428594` and `21428592`). Both completed on 17 September 2026.

## Current endpoint evidence

| RNG | Catastrophic legacy | Catastrophic deterministic | Stable legacy | Stable deterministic |
|---:|---:|---:|---:|---:|
| 314159 | 11/20 | 20/20 | 20/20 | 20/20 |
| 271828 | 10/20 | 20/20 | 20/20 | 20/20 |
| 1618033 | 20/20 | 20/20 | 20/20 | 20/20 |
| 20260916 | 19/20 | 20/20 | 20/20 | 20/20 |

Deterministic-current KL preserves 20/20 in all four catastrophic-seed
schedules and every stable control. Legacy KL produces 11, 10, 20 and 19/20
on the selected catastrophic seed (mean 15/20) while leaving the stable
control at 20/20. The predeclared screen is complete. Its defensible
conclusion is: **the catastrophic seed is stochastically susceptible to the
legacy dropout-current KL implementation, while deterministic-current KL
robustly protects it across the four tested optimizer RNG schedules.** This is
a selected-seed mechanism result, not a population estimate or replacement
for the historical 9/20 primary endpoint.

## Decision rule

Report the paired endpoint difference `deterministic - legacy` separately for
the catastrophic and stable seeds across the four RNG schedules. Do not claim
population-level causality from the selected catastrophic seed alone. The
result supports a broader implementation correction only if the deterministic
semantics repeatedly protects the catastrophic seed without harming the
stable control.

## Provenance

- Full 16-identity design: `manifest.csv`
- Newly submitted identities: `new_jobs_manifest.csv`
- Frozen protocol: `protocol.json`
- Slurm provenance: `submissions.tsv`
- Source selected-pair result:
  `../tpp_first_update_phase_d_legacy_exact_rng_20260916/results_final_20260916.csv`
