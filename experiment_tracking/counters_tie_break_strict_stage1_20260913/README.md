# Counters Stage-1 VH-off strict tie-break confirmation

This is the predeclared domain-wide confirmation of the targeted three-instance
causal result. It compares only two rules on the same current build:

- `action_id`: the historical lowest-global-action-index resolution of equal
  maximum root visit counts;
- `policy`: among equal maximum visits, use the largest network prior, then the
  stable action ID only for a remaining exact tie.

All ten validation-selected Stage-1 VH-off checkpoints and all 59 Counters test
instances are rerun under both rules. Every other setting is matched: narrow
search (five children, 20 simulations), PUCT 0.1, estimator mixture 0.5, three
workers, six-hour per-instance timeout, 10,000 external-action cap, six CPUs,
120 GiB and a 72-hour allocation. The array has low Slurm priority
(`nice=10000`). Q-only tie-breaking is excluded because it rescued 0/3 targeted
failures and is not a plausible default candidate.

The scientific scope is 20 full evaluation tasks (10 seeds x 2 rules). This
answers whether the policy tie-break improves *domain-wide* coverage, rather
than only the deliberately selected failures. It also makes the baseline and
candidate strictly comparable under the same build and scheduler-era setup.

`manifest.csv` carries the exact source checkpoint, Stage-1 training job,
policy-evaluation job, policy score and eventual remote output directory for
every task.

## Live status

At the 14 September 2026 00:21 IDT snapshot, four tasks were terminal and 16
were running. The two complete matched seed pairs were neutral: seed
`1073581256` scored 59/59 under both rules and seed `2011206605` scored 20/59
under both rules. Across all durable live ledgers, action-ID had classified
414/590 seed-instances with at least 254 successes; policy-prior had classified
409/590 with at least 249 successes. These whole-arm figures are conservative
live lower bounds, not a paired effect estimate. RQ2 remains frozen until all
ten pairs terminate. Exact current sources are in
`../advisor_followup_20260910/live_result_progress_20260914_0021.csv`.
