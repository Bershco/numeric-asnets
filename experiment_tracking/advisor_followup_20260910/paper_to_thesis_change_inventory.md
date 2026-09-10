# What changed from Numeric ASNets to this thesis pipeline

The baseline remains a two-layer relational Numeric ASNet trained by supervised
imitation of ENHSP. The thesis adds two primary interventions: MCTS-guided
Stage-2 fine-tuning and inference-time MCTS, each crossed with value-head off/on.
The primary branch is validation-led and uses ten matched seeds.

## Adopted primary configuration

- Stage 1: supervised Numeric-ASNet imitation, learning rate 0.003.
- Stage 2: MCTS visit-distribution targets, learning rate 0.0003, 100 epochs,
  constant KL anchor selected without using test scores.
- Normal inference search: policy top-20 expansion, 70 simulations per external
  action, PUCT exploration 0.1, estimator mixture 0.5.
- Block Grouping and Counters primary fixed search: narrow top-5/20 simulations,
  because normal search exhibited prohibitive cost/memory; it is labelled
  separately and never pooled with normal 20/70.
- Evaluation: up to six hours per instance and 10,000 external actions, always
  reported again at deterministic 30-minute and two-hour success cutoffs.
- The primary MCTS tables preserve the exact historical implementation used by
  those jobs. MCTS-SAFE-1 is a separately labelled follow-up: it repaired two
  of four targeted Drone failures, but was not retroactively treated as though
  it had been present in the primary runs. Progressive widening is likewise a
  separately labelled method rather than a silent fixed-search replacement.

## Experimental changes that were not adopted

- Physical-state-plus-action-history contextual nodes reduced coverage and are
  not used in primary MCTS.
- Remaining-horizon enforcement almost never activated and did not help.
- Progressive widening is not universal: it is a strong FO Counters result,
  approximate Rover parity, and weaker in Drone/Counters under tested schedules.
- Adaptive target-KL is a focused TPP catastrophic-seed diagnostic, not the
  primary training method.
- Independent FO Counters/Rover generators were frozen and screened as an
  external-distribution robustness check.  They are not pooled with the paper
  test distribution.

The exact inventory, rationale, evidence and code/result provenance are in
`paper_to_thesis_change_inventory.csv`.
