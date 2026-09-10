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
- MCTS-SAFE-1 terminal masking is retained. Progressive widening remains a
  separately labelled method rather than silently replacing fixed search.

## Experimental changes that were not adopted

- Physical-state-plus-action-history contextual nodes reduced coverage and are
  not used in primary MCTS.
- Remaining-horizon enforcement almost never activated and did not help.
- Progressive widening is not universal: it is a strong FO Counters result,
  approximate Rover parity, and weaker in Drone/Counters under tested schedules.
- Adaptive target-KL is a focused TPP catastrophic-seed diagnostic, not the
  primary training method.

The exact inventory, rationale, evidence and code/result provenance are in
`paper_to_thesis_change_inventory.csv`.
