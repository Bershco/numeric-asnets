# Stage-2 rollback/trust-region guard: evidence and next test

Updated: 16 September 2026

## What the current guard does

For each Stage-2 Adam update, the active guard first computes the proposed
update and measures deterministic current-policy versus Stage-1-anchor KL. If
either the frozen mean-KL or p99-KL limit is exceeded, it rejects the update,
restores both network and Adam state, restores the same random-number state,
and recomputes the same raw gradient at half the learning rate. It allows at
most two retries. The anchor coefficient remains 3.

The lower learning rate is a per-update backtracking device. The configured
base learning rate is restored for the next optimizer step; it is not a
persistent global learning-rate decay.

## Completed exact-RNG causal closure

The first practical screen kept the catastrophic and stable TPP seeds at
20/20, but rejected retries resampled dropout. The exact-RNG closure therefore
uses a matched active versus inactive comparison for both seeds:

- identical Stage-1 checkpoint, frozen 60-batch replay schedule and code;
- identical deterministic per-step RNG schedule;
- active guard with stable-control-calibrated limits;
- inactive instrumented guard with limits of 1e6, which should never reject;
- four one-epoch training jobs and four dependent policy endpoints.

All four endpoints scored 20/20. The catastrophic arm used eight active-guard
retries but also scored 20/20 with the inactive guard and zero retries. The
stable control likewise scored 20/20 in both arms. Therefore the frozen RNG
stream avoided the collapse by itself and this matched experiment establishes
no causal guard effect. The earlier practical repair remains evidence that the
collapse was preventable, but not that rollback/LR backtracking caused the
repair.

## Related methods

The design is methodologically defensible, but no located paper uses the exact
combination of rejecting an Adam step, restoring network plus optimizer plus
RNG state, and retrying the same batch at a lower learning rate.

- [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html)
  formalizes limiting policy change with a KL trust region.
- [Truly Proximal Policy Optimization](https://proceedings.mlr.press/v115/wang20b.html)
  is the closest policy-optimization precedent: its rollback behavior is
  activated by a trust-region condition to restrict destabilizing changes.
- [Better Fine-Tuning by Reducing Representational Collapse](https://arxiv.org/abs/2008.03156)
  motivates trust-region-style regularization during fine-tuning to limit
  harmful representation drift.
- [Recall and Learn / RecAdam](https://arxiv.org/abs/2004.12651) uses an Adam
  variant with recall mechanisms to reduce forgetting during fine-tuning.

The literature therefore supports conservative/proximal fine-tuning and
rollback as principles. Our exact state-restoring Adam backtracking rule is a
new engineering treatment that must be validated empirically rather than
presented as a standard published recipe.

## Broader three-arm pilot after exact-RNG closure

If the active guard beats its inactive match without harming the control,
freeze the treatment and test two domains:

- TPP/off: the catastrophic seed and one predeclared stable seed;
- Drone/off: one predeclared weak seed and one median seed.

Each matched seed receives three fresh same-build arms:

1. unguarded, but with the same deterministic step-RNG schedule;
2. epoch-0-only guard: guard every update in Stage-2 epoch 0, then disable it;
3. all-updates guard: keep the guard active for all Stage-2 epochs.

This is 2 domains x 2 seeds x 3 arms = 12 training jobs. Policy curves and
endpoints are evaluated first. MCTS is submitted only for treatments that show
a meaningful policy result. TPP limits must not be copied to Drone: Drone
requires an independently frozen calibration from non-test control evidence.

### Epoch-0-only guard

Advantages:

- directly targets the observed TPP failure, which was already present after
  the first saved Stage-2 checkpoint;
- changes less of the later optimization trajectory;
- cheaper to justify and less likely to suppress useful later adaptation.

Risks:

- cannot prevent destructive updates after epoch 0;
- a protected first epoch can still place training on a different trajectory;
- success on the selected TPP failure may not generalize.

### All-updates guard

Advantages:

- protects against later as well as first-epoch instability;
- defines one consistent trust-region rule throughout refinement;
- may reduce rare catastrophic lineages beyond the selected TPP case.

Risks:

- changes the full Stage-2 optimizer trajectory;
- can reject useful exploration/adaptation and increase runtime through
  repeated proposals;
- adoption would require full-domain policy and MCTS reevaluation.

## Decision gate result

The inactive catastrophic arm remained 20/20, exactly as the negative gate
anticipated. The twelve-job broader pilot is therefore held and was not
submitted. Before cross-domain testing, redesign the causal screen around a
small predeclared panel of frozen stochastic streams or another non-test-based
way to produce both susceptible and stable update conditions.
