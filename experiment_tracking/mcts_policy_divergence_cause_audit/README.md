# MCTS–policy divergence cause audit

## Question

Why does search depart from greedy-policy trajectories, and when does that
departure help or hurt? “MCTS follows visits rather than policy” describes the
mechanism but does not explain an outcome.

## Required outcome strata

1. policy succeeds, search fails;
2. both fail but their trajectories diverge;
3. policy fails, search succeeds;
4. both succeed, as a negative/control class.

For an exact checkpoint and instance, the primary unit is the first divergent
external decision. Roots from one trajectory are not independent samples.

## Evidence already established

- Counters selected VH-off cases: visit maxima tied, Q was effectively equal,
  action-index ordering selected a lower-prior action, and policy-prior
  tie-breaking recovered 3/3. This is causal selected-case evidence.
- Block Grouping PW trace: at two fully recorded roots, the policy action was
  favored by U/exploration while accumulated Q/exploitation favored the chosen
  non-policy action. Excessive exploration is not the demonstrated cause in
  those roots.
- Drone SAFE: known-terminal child handling explained and repaired 2/4
  selected failures.
- Context audit: the same physical state with different action-history inputs
  can change predictions, especially in Counters; contextual nodes exposed the
  mechanism but harmed coverage/memory when adopted broadly.

Entropy alone is descriptive, not causal. Low entropy can be confidently
wrong and high entropy can still have the correct argmax.

## First-divergence record

Record: normalized policy entropy; visit entropy; policy/visit Jensen-Shannon
divergence; selected action prior/rank; whether the policy action was expanded;
maximum-visit ties; top-two margin, winner share and leave-one-out prominence;
complete child N/Q/U/prior and `sign*Q+U`; Q-versus-U preference; safety,
goal-chase and duplicate-control overrides; branching width; search depth;
successor-generation and elapsed time; state/action-history digests; eventual
outcome/failure type; and whether trajectories later rejoin.

The economical implementation is a compact first-divergence recorder with
optional sparse landmarks, not full debug output at every external action.

## Minimal workload

### Stage 0 — existing-evidence join (zero cluster jobs)

First join existing policy trajectories, fixed-search ledgers, PW ledgers and
the rich Counters/Block-Grouping root traces by exact checkpoint hash, seed and
instance.  Produce one row per first divergence and assign it to one of the
four outcome strata above.  This prevents a new campaign from repeating roots
that already have complete N/Q/U/prior evidence.

Expected local analysis time: roughly half a working day once every referenced
log is present locally.  The deliverable is a frozen missing-strata manifest,
not a scientific conclusion inferred from whichever traces happened to be
easy to locate.

### Stage 1 — compact recorder and preflight

Add one compact record at the first policy/search divergence plus optional
sparse time landmarks.  Unit-test entropy/Jensen–Shannon calculations, action
identity, expansion membership, N/Q/U/prior decomposition and override flags.
Smoke-test one already understood Counters root.  Estimated implementation and
preflight time: one development day; no primary test job is released unless
the recorded action and visits reproduce that known root.

### Stage 2 — fill only genuinely missing strata

Missing strata then need at most one task per domain/VH cell, using the
declared search configuration and at most one frozen instance per stratum.

- Five current imperfect domains: at most 10 tasks.
- Add MPrime after its final endpoints freeze: at most 12 total.
- Per task: one worker, 2 CPU, 120 GiB, at most four exact instances, six-hour
  per-instance cap, approximately 26-hour allocation.
- Maximum six-domain concurrency: 24 CPU / 1.44 TiB.

If all tasks run concurrently, the scheduler hard bound is about 26 hours;
most tasks should finish earlier when the selected instance classifies before
six hours.  The manifest must record policy and search job/log provenance for
every selected instance.

Do not duplicate the campaign for PW. Add PW traces only where fixed-search
evidence cannot explain an important result: initially FO Counters and MPrime,
both VH modes, at most four additional tasks.

### Stage 3 — causal intervention, gated

Only after Stage 2 identifies a repeated mechanism should a treatment be
tested: for example policy-prior tie-breaking for unresolved visit ties,
changing expansion/widening for policy actions that were never expanded, or a
PUCT/exploitation sensitivity check when Q rather than U repeatedly causes the
first harmful divergence.  Calibrate and freeze every threshold on validation
traces, then evaluate the unchanged rule on test.  Do not select a treatment
from test failures themselves.

Report by outcome stratum and domain with seed/domain-clustered bootstrap
intervals. Any new action-selection rule must be calibrated and frozen on
validation traces before test evaluation. The policy-failure/search-success
stratum is mandatory so a policy-preserving rule does not erase genuine search
wins.

The maximum initial new workload is therefore 10 fixed-search tasks for the
five current imperfect domains, or 12 after MPrime is available.  Optional PW
diagnostics add at most four tasks.  No such tasks are submitted until the
Stage-0 join produces the exact missing-strata manifest.
