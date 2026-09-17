# Imperfect-domain KL-semantics first-epoch screen

## Question

Did the historical dropout-current anchor KL damage Stage-2 updates outside
TPP, and does the corrected deterministic-current KL change the immediate
policy result?

The historical 100 Stage-2 logs expose total loss/KL but not decomposed
policy-gradient norm, weighted anchor-gradient norm or policy/anchor cosine.
Those quantities cannot be reconstructed. This prospective screen adds the
missing instrumentation.

## Frozen design

- Five imperfect domains: Block Grouping, Drone, FO Counters, Rover, Counters.
- VH-off only, because the hypothesis concerns the anchor implementation and a
  compact first screen should not double scope.
- Two predeclared lineages per domain: one weak/declining historical lineage
  and one stable/median or improving control where available.
- Two treatments: exact historical dropout-current KL and corrected
  deterministic-current KL.
- One Stage-2 epoch (60 optimizer steps), anchor coefficient 3, LR 0.0003,
  fixed optimizer-step RNG, no rollback/adaptive controller.
- Policy inference on every new one-epoch endpoint.

One epoch is sufficient for the question asked here: it exposes the exact
policy and anchor gradients and tests immediate policy damage. It cannot prove
long-run superiority or justify replacing all 100-epoch Stage-2 networks.

These are deliberately selected mechanism-screen lineages, including the
historical seed-42/2026 tuning lineages; they do not estimate the prevalence of
the KL effect in the ten-seed domain populations.

## Three-way comparison

For each sampled lineage report:

1. historical true-legacy policy evidence already present in the learning
   curves (descriptive because it comes from the historical build/run);
2. prospective same-build dropout-current KL;
3. prospective same-build deterministic-current KL.

The intended causal comparison is (2) versus (3). Historical evidence is
context, not a same-build paired arm. Endpoint differences remain descriptive
unless the audit verifies replay membership, batch order, target tensors and
pre-treatment policy gradients across all sixty matched optimizer steps. Any
mismatch invalidates the causal interpretation and triggers a later frozen-
replay rerun rather than being silently accepted.

The historical comparator is always the first saved Stage-2 checkpoint
(epoch 0, meaning after the first epoch), not the historical
validation-selected checkpoint. The one Counters/534933607 epoch-0 policy
endpoint was not previously evaluated and is explicitly marked pending rather
than substituted with its epoch-8 selected score.

## Scope and resources

Twenty training tasks and twenty dependent endpoint tasks. Each training task
requests 6 CPU / 120 GiB / 12h; each endpoint requests 5 CPU / 20 GiB / 2h.
The maximum training allocation is 120 CPU / 2.4 TiB, below the declared 6 TiB
workload ceiling. No 100-epoch retraining or MCTS evaluation is part of this
screen.

The twelve-hour training value is a scheduler hard bound, not an expected
runtime.  All twenty one-epoch tasks completed in approximately three minutes
to two hours and three minutes.  The historical Counters/534933607 epoch-0
endpoint also completed at 4/59.

## Submission correction

The first array `21415079` used newer convenience architecture modules. Drone
and Rover correctly failed closed on checkpoint input-dimension mismatches;
the remaining tasks were cancelled before being used. Its dependent endpoint
array `21415080` was also cancelled. Those outputs are excluded.

The replacement uses the exact historical Stage-2 architecture module for
each domain and the exact historical policy-only evaluation module for each
endpoint. It writes to `v2_exact_legacy_architecture/`, leaving the failed
attempt intact as operational provenance. An explicit `--mcts-iterations 0`
means “use the architecture formula”; with twenty retained children this is
the historical normal budget of seventy simulations, not zero simulations.

## Interpretation gate

Expand only if deterministic-current KL materially changes gradient geometry
or one-epoch policy coverage in multiple sampled lineages without systematic
control harm. Otherwise retain the TPP finding as a selected-domain
implementation sensitivity rather than rerunning primary Stage 2.

## Results and causal limitation

All twenty one-epoch trainings and all twenty endpoint evaluations are now
complete.  The six held endpoint identities were replaced exactly by
`21428593_[2,3,8-11]`; no training was repeated.

| Domain / seed | Historical epoch 0 | Same-build legacy | Deterministic current | First-step weighted anchor-gradient norm, legacy / deterministic |
|---|---:|---:|---:|---:|
| Block Grouping / 42 | 13/20 | 14/20 | 15/20 | 0.309 / approximately 0 |
| Block Grouping / 2026 | 14/20 | 13/20 | 15/20 | 0.312831 / approximately 0 |
| Drone / 2026 | 4/20 | 5/20 | 5/20 | 2.282370 / approximately 0 |
| Drone / 42 | 4/20 | 6/20 | 4/20 | 3.883 / approximately 0 |
| FO Counters / 2026 | 10/20 | 6/20 | 6/20 | 1.299 / approximately 0 |
| FO Counters / 42 | 5/20 | 7/20 | 4/20 | 0.998257 / approximately 0 |
| Rover / 42 | 4/20 | 4/20 | 4/20 | 9.121 / approximately 0 |
| Rover / 2026 | 4/20 | 4/20 | 4/20 | 9.921906 / approximately 0 |
| Counters / 534933607 | 4/59 | 6/59 | 44/59 | 0.667 / approximately 0 |
| Counters / 2082152039 | 59/59 | 47/59 | 47/59 | 1.126834 / approximately 0 |

The first-step result is structurally consistent across the instrumented
screen: deterministic-current KL has an essentially zero anchor gradient when
the current and anchor weights are initially identical, while legacy dropout-
current KL can inject a large nonzero anchor gradient.  That proves the
historical semantics changed the optimizer update outside TPP.

However, the prospective treatment pairs did not reuse an identical frozen
replay schedule.  For example, the large Counters 6/59 versus 44/59 pair has
different worker ingestion/order and different first replay batches/targets.
It is therefore a striking descriptive signal, not a causal endpoint effect
attributable solely to KL semantics.  The proper next experiment is a compact
frozen-replay crossover: capture one 60-batch schedule per selected lineage,
replay the identical batches, targets, starting weights and optimizer RNG under
both KL semantics, and evaluate the two one-epoch endpoints.  No 100-epoch
training is justified by the present screen alone.

Historical total-gradient information cannot recover the missing anchor
gradient.  In vector form the applied gradient is the sum of policy, anchor
and other regularizer gradients; a total norm does not identify any component.
Subtraction would require identical weights, replay batches, targets, RNG and
all other gradient vectors, which the old stochastic runs do not provide.

Exact training/endpoint jobs and remote log paths for every score are in
`endpoint_results_20260917.csv`. Decomposed first-step anchor gradients,
policy/anchor cosine values, matched optimizer-RNG status and the failed replay-
identity gate are in `gradient_geometry_20260917.csv`.
