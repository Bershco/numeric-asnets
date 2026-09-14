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

At the 14 September 2026 15:02 IDT snapshot, six tasks were terminal and 14
were running. Three complete matched seed pairs were neutral: seed
`1073581256` scored 59/59 under both rules, seed `1239739722` scored 17/59
under both, and seed `2011206605` scored 20/59 under both. The matched
three-pair effect is exactly zero, but these three checkpoints contain zero
classified policy-success/action-ID-failure opportunities. Their neutrality
therefore does not test the proposed rescue mechanism. The informative
high-policy-coverage pairs remain live; unpaired lower bounds must not be used
as a treatment comparison. RQ2 remains frozen until all ten pairs terminate.

The complete interim, seed-level join is in `progress_20260914_1502.csv`. It
records the exact validation-selected checkpoint, source training and policy
logs, pure-policy score, classified count, search successes, classified
policy-success/action-ID-failure opportunities, gains over policy, and
unclassified policy-success instances for both rules. Current workload
provenance is refreshed separately in the advisor-followup workload ledger.
