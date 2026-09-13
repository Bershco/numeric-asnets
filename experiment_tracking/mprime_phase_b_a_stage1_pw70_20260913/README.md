# MPrime Stage-1 Phase-B-A PW70 screen

## Scientific question

Can progressive widening retain the useful coverage of MPrime's canonical
Stage-1 fixed 20/70 MCTS while opening substantially fewer children early in
search?

This is a four-task screen, not a confirmatory ten-seed result. The two seeds
`1963100312` and `2011206605` were predeclared because they are the same matched
two-seed screen identities used in the earlier cross-domain PW work. Both
VH-off and VH-on are evaluated, yielding exactly four domain/mode/seed cells.

## Frozen identity

Each row is derived directly from the corresponding canonical fixed-MCTS row
in `../mprime_phase_b_a_stage1_mcts_20260913/manifest_{off,on}.csv`. The
checkpoint, Phase-B-A selector, policy provenance, 70 simulations, PUCT 0.1,
estimator mixture 0.5, six-hour per-instance timeout, 10,000-action cap,
workers, CPUs, memory and 72-hour allocation are unchanged.

The only algorithmic change is progressive widening with:

- `Kmin = 3`;
- `c = 0.6`;
- `alpha = 0.5`.

Terminal-safe action selection remains disabled so the comparison matches the
canonical fixed-MCTS arm rather than silently combining PW with a second
safety intervention.

## Resources and outputs

- four array tasks;
- 6 CPUs and 120 GiB per task;
- maximum concurrent request: 24 CPUs and 480 GiB;
- three workers per task;
- six hours per instance and 72 hours per task;
- durable per-instance completion JSONL, immutable attempt logs, append-only
  attempt ledger and per-attempt VAL summary.

The full manifest is `manifest.csv`. Submission provenance is recorded in
`submissions.tsv`; the compute smoke is recorded in `smoke.tsv`.

## Status

Prepared locally; compute smoke and submission pending.
