# Counters visit-prominence follow-up — frozen design note

## What is established

The exact maximum-visit tie fallback to policy prior is causally supported on
three targeted VH-off cases: action-ID and Q tie-breaks solved 0/3, while
policy-prior solved 3/3. A two-visit lead is not established as sufficient;
it merely excludes reversal by moving one simulation and must not be presented
as a validated threshold.

## Candidate evidence score

For a unique visit winner, quantify how exceptional its count is relative to
the other eligible root actions with the variance-floored leave-one-out score

`z_LOO = (N_winner - mean(N_others)) /
sqrt(var(N_others) + max(mean(N_others), 1))`.

Record this together with the winner-versus-policy visit gap, winner visit
share, number of eligible actions, policy rank/prior, Q gap, and whether the
maximum is tied. The Poisson-like variance floor prevents near-zero sibling
variance from turning a one-visit difference into artificial certainty.

This is a standardized prominence heuristic, not an inferential z-test:
MCTS child visits are adaptive, dependent, and not exchangeable independent
samples. It therefore does not justify a p-value or an off-the-shelf standard
deviation threshold.

## Selection boundary

No threshold will be selected from the observed test failures. Candidate
thresholds must be calibrated and frozen on validation traces that include all
four outcome strata: policy+/MCTS-, policy-/MCTS+, both+, and both-. Selection
must balance preservation of demonstrated search gains against recovery of
policy successes. Only the frozen rule may then be evaluated on test.

Until that validation calibration exists, the only admissible behavioral
change is the exact-tie policy-prior fallback. The strict test campaign and its
dependent full-root trace diagnose transfer and mechanism; they do not tune a
near-tie threshold.

## Required trace

Once every strict identity is classified, rerun only the union of exact
policy-success/action-ID-failure instances with `--action-debug`. Preserve the
same checkpoints and narrow 5/20 configuration. Capture complete eligibility,
prior, visit, Q, U, selected action, goal-chasing/safety precedence, external
step, and elapsed time at the first divergence and subsequent roots.
