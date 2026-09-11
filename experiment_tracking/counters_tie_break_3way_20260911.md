# Counters MCTS root-visit tie-break comparison

This six-task matched diagnostic changes only how equal maximum root visit
counts are resolved after an otherwise identical narrow MCTS search.

| Mode | Rule after maximum visits | Final exact tie |
|---|---|---|
| `action_id` | Historical behavior | Lowest global action index |
| `q` | Best sign-correct child Q | Lowest global action index when Q differs by at most `1e-8` |
| `policy` | Largest root network prior | Lowest global action index |

Each mode runs both the VH-off failure checkpoint and matched VH-on checkpoint
on Counters instances 51, 55 and 59. Every other declared choice is frozen:
seed 1963100312, validation-led Stage-2 checkpoints, five retained children,
20 simulations, PUCT 0.1, estimator mixture 0.5, one worker, 10,000 external
actions and six hours per instance. Goal chasing and safety/eligibility masks
retain precedence over all three root tie-break rules.

The experiment uses six array tasks. Each requests 2 CPUs, 120 GiB and 24
hours; simultaneous maximum is 12 CPUs and 720 GiB. Slurm nice value 10000
keeps the diagnostic below ordinary-priority work. A compute-node smoke job
must succeed before the array is released.

The manifest is
`advisor_followup_20260910/counters_tie_break_3way_manifest.csv`. Each final
row will carry its job ID and output log for direct provenance.
