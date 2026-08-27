# Drone MCTS dead-end audit

## Scope

This audit explains the four `policy_only_success` Drone rows in
`experiment_tracking/policy_mcts_instance_audit.csv`. It uses the original
policy/MCTS logs and persisted MCTS completion JSONL files; no evaluation was
rerun.

All four MCTS evaluations used width 20, 70 simulations, PUCT 0.1, estimator
coefficient 0.5, argmax goal-chasing action selection, and trajectory-duplicate
penalty 0.0 (a hard ban where another non-banned action remains).

## Exact terminal outcomes

| Job | VH | Seed | Instance | Steps | Final state | Progress since last recharge |
|---|---|---:|---|---:|---|---|
| 20451722 | off | 1963100312 | problem_2_5_4.pddl | 135 | `(0,1,0)`, battery 0 | 23 actions, zero new visits |
| 20451722 | off | 1963100312 | problem_2_8_3.pddl | 100 | `(0,0,1)`, battery 0 | 27 actions, zero new visits |
| 20451725 | off | 534933607 | problem_5_2_2.pddl | 49 | `(0,1,0)`, battery 0 | 19 actions, zero new visits |
| 20430051 | on | 2011206605 | problem_6_8_1.pddl | 199 | `(1,0,0)`, battery 0 | 31 actions, zero new visits |

Drone movement and visit actions consume one battery unit. Recharge is possible
only at `(0,0,0)`. Therefore battery 0 at any other coordinate is a genuine
non-goal terminal dead end, not an action limit or timeout.

## Mechanism

Terminal non-goal nodes are evaluated as `worst_value()` (0 in the current
maximization convention) and that value is backed up. The committed evaluation
action is the **greedy argmax of root visit counts**. This is consistent with
AlphaZero-style evaluation; actions are not sampled during these evaluations.
Consequently, merely having one visit does not make a terminal action
selectable: it must have the largest visit count (or win an `argmax` tie).

The modern maximization evaluator should not produce negative values: the
value head is sigmoid-bounded, the ENHSP conversion is `exp(-coefficient*h)`,
their blend remains non-negative, and a terminal non-goal receives exactly
zero. A negative Q would therefore be an invariant violation. A terminal child
can still receive visits because its policy prior gives it positive PUCT
exploration utility while safe-state values may be close to zero. More
importantly, external duplicate masking is applied to the visit distribution:
it can zero a higher-visit safe duplicate and expose a lower-visit terminal
child as the post-mask argmax. The old logs do not contain the pre/post-mask
root vectors needed to distinguish these effects per action.

The stronger issue in these four trajectories is the interaction with
`PathDuplicatePenaltyMixin`. The Drone PDDL has no `total-time` fluent. More
generally, `env_state_key()` deliberately excludes `total-time` through
`SPECIAL_FUNCTIONS`, so elapsed time never distinguishes transpositions. After
wasting an entire battery charge without visiting a new location, recharging
at the origin recreates the same canonical state key already marked
`on_trajectory`: equal coordinates, battery, visited propositions, ordinary
numeric fluents and comparisons. The duplicate penalty can ban that recharge
action while leaving battery-consuming moves unmasked. With one battery unit
remaining, any selected move then creates a terminal dead end.

This is a state-transposition check, not a blanket claim that states reached at
different wall-clock/action times are duplicates. Two paths merge only when
their canonical propositions, non-special numeric fluents and comparisons
match after numeric rounding.

Across the checked numeric PDDL corpus, `total-time` appears in metric
expressions (or comments), not in preconditions, effects, or goals. Its
omission therefore does not merge states with different reachability for the
current coverage experiments. It would be unsafe for a genuinely
time-dependent domain or for an accumulated-cost result.

There is a separate contextual-alias risk: Numeric ASNets' action-history input
is a vector containing one cumulative application count for every grounded
action (`c_a`), not a scalar total-time/action-count feature. Two trajectories
can therefore have equal length and the same physical `env_state_key()` while
having different action-count vectors. The MCTS transposition dictionary can
then reuse the first node's cached policy/value for a different network input.
A future collision audit should compare network-input vectors whenever an
existing state key is reused.

In code this is Python-side auxiliary data, not a `BoundProp` or `BoundFlnt`.
`ActionCountDataGenerator` creates one float feature per grounded action and
increments the slot of the executed action. `CanonicalState.populate_aux_data`
flattens it into `_aux_data`; the network later reshapes it to
`[batch, num_actions, extra_dimension]` and routes the corresponding count to
each grounded action module. `CanonicalState.__eq__` can compare `_aux_data`,
but MCTS transpositions use `CanonicalState.state_key`, which is produced by
`env_state_key()` before auxiliary data is populated and contains no action
history.

## Required correction before a dedicated rerun

1. At external action selection, form a safe set from expanded children that
   are goals or are non-terminal. Hard-mask directly terminal non-goal children
   whenever that safe set is non-empty, then keep the existing greedy
   visit-count rule inside the safe set.
2. Apply trajectory-duplicate avoidance only after the terminal-safety mask.
   If duplicate avoidance would remove every safe action, restore safe duplicate
   actions before admitting a terminal non-goal action.
3. If every admitted child is terminal but unadmitted applicable actions remain
   (possible under fixed/progressive width), admit/evaluate additional actions
   before declaring that the root has no safe action.
4. Add an explicit diagnostic counter for terminal actions excluded, duplicate
   fallback events, and cases with no safe child.
5. For every external decision, log the raw visit vector, the vector after the
   terminal-safety mask, the vector after duplicate avoidance, the final eligible
   actions, the selected action, every child's terminal/goal/duplicate flags,
   and per-child `N`, `Q`, `U`, and prior. This preserves the pre-mask and every
   post-mask vector needed to explain a changed decision.
6. In the modern maximization evaluator, assert that every root-child Q-value is
   finite and lies in `[0, 1]` within numerical tolerance. Log the complete root
   vector before raising. Historical minimization/compatibility modes must check
   their own explicitly declared value convention instead.
7. Add regression tests for the Drone battery-one-at-origin state: recharge must
   be selected over a move that produces battery zero away from the origin.
8. Keep terminal nodes at the worst value; a negative penalty may be tested
   separately but is not a substitute for the external safety mask.

The minimal diagnostic rerun is the four known failing instances on their three
checkpoints. Because the selector can change any root where a terminal child
competes with a safe child, the confirmatory rerun must then cover all twenty
matched Drone Stage-1 checkpoints (ten seeds x two VH modes), not only those
four rows. Other domains need reruns only after an audit finds the same direct
terminal-selection condition.

The historical logs had action-level debug disabled, so the first earlier
policy/MCTS divergence cannot be reconstructed. The persisted complete failed
plans are sufficient to establish the final mechanism above.

The progressive-widening pilot is isolated from MCTS-SAFE. PW jobs execute from
`/home/hersco/bershco-nu-asnets/numeric-asnets-pw` at commit `6701f4c1`.
MCTS-SAFE executes from the separate
`/home/hersco/bershco-nu-asnets/numeric-asnets-mcts-safe` worktree. Neither the
running nor pending PW jobs can see the MCTS-SAFE changes, so the pilot does not
need to be relaunched merely because MCTS-SAFE was deployed.

## Provenance

- MCTS logs:
  - `/home/hersco/training_new_domains/2026-08-21/statistical_replication_stage1_mcts_eval/20451722_Ev_drone_drone_mcts_orig_novh_e.5_c.1_s1963100312_K0_SR10M_src20401207_e0157.txt`
  - `/home/hersco/training_new_domains/2026-08-21/statistical_replication_stage1_mcts_eval/20451725_Ev_drone_drone_mcts_orig_novh_e.5_c.1_s534933607_K0_SR10M_src20401213_e0069.txt`
  - `/home/hersco/training_new_domains/2026-08-21/statistical_replication_stage1_mcts_eval/20430051_Ev_drone_drone_mcts_orig_vh_e.5_c.1_s2011206605_K0_SR10M_src20401198_e0019.txt`
- Completion manifests are the corresponding job IDs under
  `/home/hersco/training_new_domains/2026-08-21/statistical_replication_stage1_mcts_eval/.resume_state/`.
