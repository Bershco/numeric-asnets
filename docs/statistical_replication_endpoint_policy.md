# Statistical replication endpoint policy

This document fixes which checkpoints receive downstream evaluations in the
ten-seed replication campaigns.  It prevents controller refreshes from silently
changing the scientific comparison.

## Policy evaluation

- Stage 1: evaluate the every-five checkpoint curve and the terminal checkpoint.
- Stage 1: also evaluate the validation-selected checkpoint when it is not already
  one of the curve checkpoints.
- Stage 2: evaluate the every-five checkpoint curve and the terminal checkpoint.
- Stage 2: also evaluate the validation-selected checkpoint when it is not already
  one of the curve checkpoints.
- Report validation-selected and terminal results under explicit column names.
  A table containing both must not be titled "validation-led results".

## MCTS evaluation

- Stage 1: create held entries for both the validation-selected checkpoint and
  terminal checkpoint.  Deduplicate them when they are the same checkpoint.
- Stage 2: create a held entry for the validation-selected checkpoint only.
- Stage-2 terminal-checkpoint MCTS is intentionally excluded for now.  It may be
  added later as a separately declared endpoint comparison; it must not be
  introduced implicitly by a controller refresh.
- MCTS controllers remain held until explicitly released.  Policy controllers
  may continue to submit newly available policy work.

## Terminal job accounting

Scheduler timeouts and Slurm OOMs are terminal experimental outcomes, not missing
rows.  Status tables must therefore show separate counts for successful job
completion, scheduler timeout, OOM, other failure, still running, and total
accounted.  Coverage recovered from an interrupted log is marked with an asterisk
until every printed plan has been checked with VAL.  Unclassified instances count
as failures in the conservative score.

## Four previously perfect domains

Delivery, MPrime, TPP, and Zenotravel use the same endpoint rules above.  Their
Stage-1 policy and held-MCTS manifests can be populated as training finishes.
The two Stage-2 branches (validation-led and terminal-led) remain blocked until
their anchor/configuration selection is explicitly declared; the manifest builder
records this blocked state rather than guessing an anchor.

## Current manifest locations

- Five-domain campaign: `/home/hersco/training_new_domains/2026-08-22/statistical_replication_controller_option_a/`
- Four-domain campaign: `/home/hersco/training_new_domains/2026-08-23/completed_domains_stage1_10seed/`

The manifest builders are idempotent: repeated refreshes add only newly available
checkpoint work and remove no submitted result.
