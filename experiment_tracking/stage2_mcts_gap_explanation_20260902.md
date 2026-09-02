# Why the Stage-2 MCTS matrix has partial cells

The partial cells are not missing logs from a balanced ten-seed campaign. They
are the result of an August orchestration mistake: Stage-2 MCTS endpoints were
released manually as exploratory subsets while the user pending-job limit was
tight. The broad MCTS hold/cancellation then happened before a twenty-cell
branch-completeness matrix and balanced manifest enforced all ten seeds for
both Stage-2 lineages.

Later priorities completed selected cells—both Drone branches, terminal-led
Rover, validation-led Counters, and currently terminal-led FO Counters—but did
not retroactively fill the earlier exploratory validation-led subsets. This is
a bookkeeping/release-design defect, not a scientific reason to prefer one
lineage and not evidence that ten jobs once existed and were lost.

The canonical full historical scan is
`stage2_mcts_historical_log_audit_20260902.csv`; every one of the 200 intended
domain/VH/branch/seed identities has its policy log and either matching MCTS log
paths or an explicit absence. The aggregate matrix is
`stage2_mcts_branch_coverage_20260902.csv`, and the exact missing cells plus
runtime/cutoff evidence are in `stage2_mcts_gap_decisions_20260902.csv`.

At the 2 September audit, after counting the live terminal-led FO campaign but
before it finishes, the unresolved historical gaps are:

- Block Grouping: 12 validation-led jobs plus one terminal-led job = 13.
- FO Counters: five validation-led jobs = 5.
- Rover: nineteen validation-led jobs = 19.
- Counters: twenty terminal-led jobs = 20.
- Total: 57 jobs.

No missing gap should be submitted merely because it is absent. Block Grouping
loses meaningful coverage between two and six hours and therefore needs its
declared six-hour design. Rover's observed Stage-2 scores are identical at two
and six hours, making a hard two-hour completion campaign defensible. FO
Counters' Stage-1 analogue is unchanged after two hours, but its exact Stage-2
validation-led branch remains unmeasured; the live terminal-led campaign should
finish before freezing that cutoff. Counters terminal-led should use the already
declared narrow width-5/20 search.
