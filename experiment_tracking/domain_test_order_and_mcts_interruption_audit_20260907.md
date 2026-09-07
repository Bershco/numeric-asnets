# Test-order and interrupted-MCTS opportunity audit — 7 September 2026

## Question and counting rule

This audit asks whether an OOM or Slurm allocation timeout ended an MCTS job
before some test instances received a genuine result.  An instance counts as
an **interruption opportunity** only when the complete original allocation log
contains neither:

- `[EVAL INSTANCE] completed ...`; nor
- `[EVAL INSTANCE] timeout ...` for the declared per-instance limit.

Ordinary unsolved trajectories and explicit six-hour instance timeouts are not
opportunities: the evaluator did run them to a declared terminal outcome.  A
crashed/started-only instance, or an instance never started before allocation
death, is an opportunity.  The existing fixed-budget score is never silently
rewritten; any selective retry must be reported as separate recovered-suite
evidence.

The reproducible builder is
`scripts/build_domain_order_interruption_report_20260907.py`.  It reads the
small completion ledgers and only the compact `[EVAL INSTANCE]` markers in the
original logs.  It writes one job-level row and one row per exact missing
instance, both with literal original-log and completion-ledger paths.

## Logical test-order classification

The user excluded PRESERVE-3, Counters and Block Grouping from this
classification.  Direct inspection of the remaining four domains' ordered
test PDDL files gives:

| Domain | Classification | Evidence | Audit consequence |
|---|---|---|---|
| Drone | Non-monotone | Structural size repeatedly falls: 48→18, 180→20 and 168→16 | Later unrun instances can be easier; audit interruptions |
| FO Counters | Strictly increasing by construction | `instance_n` has `n` counters, `n−1` chain goals and `max_int=2n` | Excluded from primary non-monotone cohort; retained as motivating supplement |
| Rover | Non-monotone with an increasing trend | Object count and goal count have adjacent reversals, including p10→p11 and p11→p12 | Later unrun instances can be easier; audit interruptions |
| MPrime | Non-monotone | Ordered object counts include 75→74→41→27 and 55→8 | Audit interruptions once MCTS exists; none exists yet |

The exact observed sequences are in `domain_test_order_20260907.csv`.

## Results

| Cohort | MCTS scope | Interrupted jobs with ≥1 opportunity | Completed/timed-out instance records | Interruption opportunities |
|---|---|---:|---:|---:|
| Primary | Rover Stage-1 validation-selected fixed 20/70 | 8 | 147 | 13 |
| Primary | Rover Stage-2 validation-led fixed 20/70 | 4 | 76 | 4 |
| Primary | Rover Stage-2 terminal-led fixed 20/70 | 9 | 168 | 12 |
| Primary | **Rover total** | **21** | **391** | **29** |
| Primary | Drone | 0 | — | 0 |
| Primary | MPrime | 0 (no MCTS campaign exists) | — | 0 |
| Supplement | FO Stage-1 fixed 20/70, job 20430072 | 1 | 19 | 1 (`instance_21.pddl`) |
| Supplement | FO PW70, job 21039205 | 1 | 19 | 1 (`instance_15.pddl`) |

Forty fixed-search Rover OOM allocations were inspected: 23 contained the 29
opportunities above; 17 had terminal completed/timeout records for every
instance despite their final OOM label.  Two earlier Rover PW70 allocations
also ended OOM but had all 20 terminal instance records, so they contribute
zero opportunities.

The curated Drone inventory contains 163 score-bearing or diagnostic MCTS
allocations.  None ended in Slurm `OUT_OF_MEMORY` or `TIMEOUT`.  Some wrapper
jobs ended `FAILED` only after inference because of the already documented
validator-path defect; their inference and post-hoc VAL evidence exist and are
not interruption opportunities.

The earlier description of FO PW70 job 21039205 as having only ten classified
instances was wrong.  The original log has 19 terminal records: seven
successes, three ordinary unsolved outcomes and nine explicit six-hour
timeouts.  Only instance 14 (`instance_15.pddl`) lacked a terminal record.

## What can be recovered later

The exact 29 Rover and two supplementary FO instance rows are frozen in
`mcts_interruption_opportunity_instances_20260907.csv`.  A selective retry can
therefore skip every already classified instance and run only those rows,
preferably with the separately proposed two-worker/160-GiB resource setting.

This retry is potentially valuable for Rover because its test order is not
monotone.  It is less promising for FO because both opportunities are later,
structurally harder instances in a strictly increasing suite.  Neither case
guarantees recovered successes; the audit identifies missing opportunity, not
counterfactual coverage.

## Durable files

- `domain_test_order_20260907.csv`: structural classification and evidence.
- `mcts_interruption_opportunity_summary_20260907.csv`: aggregate counts.
- `mcts_interruption_opportunity_jobs_20260907.csv`: exact job/state/config and
  original-log pointers.
- `mcts_interruption_opportunity_instances_20260907.csv`: exact selective-retry
  rows with original-log and completion-ledger pointers.
- `mcts_interruption_audit_inventory_20260907.csv`: the curated Drone job
  inventory used for the zero-interruption conclusion.

