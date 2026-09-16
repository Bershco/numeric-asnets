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
