# Current experiment status

Updated: 2026-09-12 18:53 IDT

## Live workload

| Experiment | Running | Ordinary pending | Dependency pending | CPU / RAM requested | Timing |
|---|---:|---:|---:|---:|---|
| FO Counters Stage-2 exact instance recovery | 1 | 0 | 0 | 2 CPU / 120 GiB | 1h12m elapsed; classification by ~23:42; 8h allocation ends ~01:41 |
| MPrime Phase-B-A anchor smoke | 0 | 1 | 0 | 3 CPU / 20 GiB | 2h hard bound; pending resources |
| MPrime Phase-B-A full rescore | 0 | 0 | 28 | 84 CPU / 560 GiB maximum | 24h/task hard bound; releases only after smoke succeeds |
| MPrime coefficient finalizer | 0 | 0 | 1 | 1 CPU / 2 GiB | 15m hard bound; releases only after all 28 tasks succeed |

MPrime jobs are `21222348` (smoke), `21222349[0-27]` (full rescore), and
`21222350` (analysis-only finalizer). The first smoke `21221744` failed before
evaluation because the isolated checkout lacked the frozen validator PDDLs;
its dependent array/finalizer were automatically cancelled. The 240 tracked
PDDL files were then deployed and counted before submitting the corrected
chain. The finalizer cannot submit Stage-2 training. FO job `21219947` runs only evaluator number 9
(`instance_10.pddl`) from source `20943885`; its current cell mean is
`≥6.1/20` and can rise only to `6.2/20`.

## Newly completed experiments

- **MPrime Phase C:** 347/347 candidate-checkpoint evaluations, 40/40 primary
  lineages. Phase-B replicate A is frozen as the final validator because its
  common-reference mean regret is marginally lower (1.625 versus 1.650 plans)
  and it was designed independently of test-suite structure.
- **Counters three-way tie-breaking:** policy-prior tie-breaking recovered all
  three preselected VH-off failures by two hours; action-index and Q
  tie-breaking recovered none. No rule recovered the VH-on controls.

Canonical RQ report:
`experiment_tracking/advisor_followup_20260910/rq_report_validation_led_20260912.md`.
Scheduler evidence: `experiment_tracking/cluster_workload_latest.csv`.
