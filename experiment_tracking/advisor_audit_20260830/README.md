# Advisor audit package — 2026-08-30

This directory is the durable source for the advisor status presentation. It joins the authoritative experiment ledgers without copying raw logs.

- `rq_results.csv`: paired seed-level RQ estimands, exact sign-flip p-values, and the original five-domain Holm correction within each RQ. MPrime will be added only to a separately labelled six-domain extension after its Stage-2 confirmation.
- `cluster_workload.csv`: exact Slurm job/CPU/requested-memory snapshot grouped by experiment and state.
- `story_holes.csv`: threats, missing links, and defensible closure strategies.
- `improvement_priorities.csv`: at least one improvement for every registered experiment, scored for priority/relevance and cross-checked for overlap.
- `current_status.md` and `experiment_status.csv`: the human-readable and machine-readable live/held experiment snapshot, including estimates and dependencies.
- `numeric_asnets_advisor_status_20260830.pptx`: advisor-ready presentation (generated separately).

Raw provenance remains in `../experiment_results.csv`, `../policy_paired_seed_results.csv`, `../mcts_paired_seed_results.csv`, `../live_jobs.csv`, and experiment-specific subdirectories.
