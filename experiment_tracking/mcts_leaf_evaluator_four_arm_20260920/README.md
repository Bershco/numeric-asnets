# Four-arm MCTS leaf-evaluator pilot

## Question

Which leaf signal gives the best coverage/runtime trade-off when every other
search choice is held fixed?

This is one 16-task factorial pilot, not an eight-task rollout comparison plus
a duplicated 16-task ablation:

- two domains: Drone and FO Counters;
- two predeclared Stage-1 VH-on seeds per domain: `1963100312` and
  `2011206605`;
- four leaf evaluators: bounded policy rollout, learned value only, ENHSP only,
  and the current `0.5` learned/ENHSP blend.

The seed pair is reused from the earlier cross-domain PW pilot. It was not
chosen from this experiment's test outcomes. The exact MAIN-VAL checkpoints,
source jobs, epochs and existing policy scores are frozen in `manifest.csv`.
Historical blend evaluations exist for these checkpoints, but they are not
substituted into the primary comparison: all four arms are intentionally
same-build runs because this pilot changes evaluator plumbing and needs a
causal comparison without a code-version confound. The historical blend
scores are a secondary consistency check, not four extra tasks or a
replacement for the four declared blend identities.

## Frozen common configuration

All arms use fixed expansion, top 20 children, 70 simulations per external
action, PUCT `0.1`, Action-ID root tie-breaking, the same 20 test instances,
three evaluation workers, a 10,000-action cap and a six-hour per-instance
limit. Progressive widening and terminal-safety overrides are intentionally
off. The only treatment is the leaf evaluator.

The three current-value arms already existed:

- learned-only: `--use-estimator 0`;
- ENHSP-only: `--use-estimator 1`;
- current blend: `--use-estimator 0.5`.

The rollout arm is an explicitly modernized reconstruction of the historical
rollout method family, not a bit-exact restoration. It is opt-in through
`--eval-mcts-leaf-evaluator policy_rollout --mcts-rollout-horizon 3`.
It restores the historical policy-rollout method family inside the current
tree/search implementation. It deliberately scores success/failure on the
current 0/1 scale. The historical `1000 / path_cost` reward would change Q's
scale relative to PUCT exploration and would therefore confound the leaf
evaluator comparison. The compatibility rollout also does not inject its
sampled trajectory into the explicit tree or goal-chasing path: it changes
only the scalar leaf signal, keeping the current tree mechanics fixed.

## Evidence and metrics

The pilot is exploratory (evidence class B). Report each arm at exact 30-minute,
two-hour and six-hour cutoffs, plus:

- coverage and paired per-seed changes against the current blend;
- instance and total runtime;
- MCTS `evaluation_seconds`, generated/peak nodes and search-call count;
- ordinary action-limit failures versus six-hour timeouts;
- rollout goal-hit rate (`rollout_goal_hits / rollout_evaluations`).

With two seeds, do not claim population significance or use this screen in the
primary RQ multiple-testing family. It can explain why a method was stopped or
justify a ten-seed expansion.

## Predeclared decision rule

An arm is **promising enough to expand** if, across the four matched
domain/seed cells, either:

1. its mean six-hour coverage is at least one plan above the blend and it is
   non-worse in at least three of four cells; or
2. its mean coverage is within 0.5 plans of the blend and its median runtime is
   at least 25% lower.

An arm is **stopped for futility** if the blend weakly dominates it in both
coverage and runtime in all four cells, or if it loses at least two plans on
average with no 25% runtime advantage. Any other pattern is inconclusive and
is not automatically expanded.

If an arm is expanded for a thesis-wide outcome claim, run that arm and the
blend on all ten matched seeds in both domains. The four pilot identities for
those two arms are reused, leaving 32 additional tasks rather than rerunning
40.

## Compatibility and release gate

Before scientific release:

1. run the local unit tests, including the rollout-mode regression;
2. deploy the code to an isolated checkout;
3. verify all four checkpoint paths and record their SHA-256 hashes;
4. run a four-task, one-instance compatibility smoke using the first checkpoint
   and all four arms;
5. require all four smokes to produce a terminal evaluation, a valid plan when
   successful, a completion ledger and an MCTS search summary naming the
   expected leaf evaluator;
6. only then submit the 16-task scientific array.

No cluster job has been submitted from this preparation.

The prepared release commands are deliberately two-stage (examples only;
they were not executed during preparation):

```bash
# The first four rows are one checkpoint crossed with all four arms.
sbatch --array=0-3 --cpus-per-task=2 --mem=20G --time=01:00:00 \
  --export=ALL,SMOKE=1 scripts/mcts_leaf_evaluator_four_arm.sbatch

python scripts/verify_leaf_evaluator_four_arm_manifest.py \
  experiment_tracking/mcts_leaf_evaluator_four_arm_20260920/manifest.csv \
  --smoke-done /home/hersco/training_new_domains/2026-09-20/mcts_leaf_evaluator_four_arm/smoke/done \
  --write-gate /home/hersco/training_new_domains/2026-09-20/mcts_leaf_evaluator_four_arm/smoke_gate.json

# The sbatch runner refuses science work if the gate file is absent.
sbatch --array=0-15 scripts/mcts_leaf_evaluator_four_arm.sbatch
```

## Resources and expected duration

Each scientific task requests 6 CPUs and 120 GiB for at most 72 hours. All 16
running simultaneously would request 96 CPUs and 1.875 TiB. The upper-bound
allocation is intentionally inherited from the matched fixed-search campaign:
`ceil(20 / 3) * 6h` plus setup/retry headroom. Drone should usually finish in
hours; FO Counters can occupy one to two days when many instances consume their
full allowance. The four one-instance smokes request 2 CPUs, 20 GiB and one
hour each.

## Provenance

- Historical transition: git commit `50835e09`.
- Checkpoint provenance: `experiment_tracking/experiment_results.csv` and the
  prior PW manifests.
- Method inventory: `docs/thesis_methods_configuration_inventory_20260916.md`.
- Runner: `scripts/mcts_leaf_evaluator_four_arm.sbatch`.
