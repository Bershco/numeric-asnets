# PUCT exploitation/exploration feasibility audit

This is a read-only audit of existing evidence. It excludes Delivery, TPP and
Zenotravel, as requested, and does not tune a hyperparameter on test failures.

The implementation selects within the tree by
`sign * Q + U`, where `U = c * prior * sqrt(parent visits) / (1 + edge visits)`
and the declared experiments use `c = 0.1`. The external action is then chosen
from the final root visit counts, so a poor external action can arise from the
accumulated PUCT trajectory even when the final instantaneous U term is small.

## Finding

The hypothesis that exploration is too strong is plausible in the abstract,
but the available cross-domain evidence does not support it as a universal
failure mode. Block Grouping's quantified first divergences are mainly
Q/exploitation dominated: in both unique-winner roots where Q and U are fully
available, U favored the successful policy child while the accumulated Q
advantage favored the selected PW child. Counters demonstrates a different,
causal mechanism: equal Q and a tied maximum visit count allowed arbitrary
action-index ordering to discard a successful policy preference.

FO Counters, Rover and MPrime use the same PUCT coefficient without the same
failure signature. FO Counters and MPrime are positive PW results. Therefore a
broad coefficient sweep would mix domain mechanisms and would be difficult to
defend. If PUCT balance is studied further, the economical design is a tiny,
predeclared validation-trace sensitivity study in the domains whose completed
root traces actually show ambiguous Q/U evidence.

The row-level evidence and next-step gate are in
`mcts_puct_balance_feasibility_20260915.csv`.

## Visit-margin rule scope

The exact-tie policy-prior fallback is the only currently demonstrated rule.
A two-visit margin is not a reliability threshold; it only prevents one
simulation transfer from reversing the order. A standardized visit-prominence
score is more informative but becomes a separate model-selection problem.

A defensible Counters-only study would freeze an objective and 3-4 candidate
rules on validation traces, requiring roughly 30-40 validation treatment
evaluations, followed by 10 selected-rule test evaluations (20 if a same-build
baseline is required). Generalizing that calibration across all six imperfect
domains would exceed roughly 120 treatment evaluations and should be treated as
a separate thesis branch. The economical decision is to finish the exact-tie
confirmation first and open margin calibration only if substantial
policy-success losses remain.
