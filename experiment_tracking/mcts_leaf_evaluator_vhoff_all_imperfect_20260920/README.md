# VH-off all-domain rollout versus current-leaf screen

## Why this extension exists

The existing four-arm Drone/FO decomposition was correctly restricted to
VH-on. Its learned-only arm requires an actually trained learned value head.
Under VH-off, the search code assigns nonterminal network leaf value zero; a
"learned-only" arm would therefore be a degenerate all-zero control rather than
a learned-value mechanism. ENHSP-only versus coefficient-0.5 would then also
mix signal source with a pure scale change. Repeating all four arms under
VH-off would not be the same interpretable decomposition.

That does not justify restricting the broader method-family conclusion to
VH-on. Rollout uses the policy output and is defined in both value-head modes,
and leaf choice can interact with whether a learned value is available. The
defensible VH-off question is therefore exactly two arms:

- `rollout`: bounded horizon-3 policy rollout on the current 0/1 scale;
- `current`: the frozen current VH-off value path with `--use-estimator 0.5`.

In the current implementation the VH-off network contribution is zero, so the
second arm is numerically coefficient-0.5 ENHSP plus terminal evidence. It is
called `current`, not `blend`, to avoid implying a learned value contribution
that does not exist. The estimand is still valid: replace the current deployed
VH-off leaf rule with policy rollout while holding the checkpoint, policy,
tree, search budget and code build fixed.

## Frozen design

The manifest contains 24 new, nonduplicative science identities:

- six imperfect domains: Block Grouping, Drone, FO Counters, Rover, Counters
  and MPrime;
- two predeclared Stage-1 VH-off seeds per domain: `1963100312` and
  `2011206605`;
- rollout and fresh same-build current-leaf control for each domain/seed cell.

No running VH-on job is changed or reused as a VH-off result. Older VH-off
current-leaf results are secondary consistency evidence only: they do not have
provably identical rollout-era build and diagnostic behavior. Fresh controls
avoid that code-version confound.

| Domain | Width | Simulations | Teacher |
| --- | ---: | ---: | --- |
| Block Grouping | 5 | 20 | `hadd-gbfs` |
| Counters | 5 | 20 | `hmrmax-astar` |
| Drone | 20 | 70 | `hadd-astar` |
| FO Counters | 20 | 70 | `hmrmax-astar` |
| Rover | 20 | 70 | `hmrp-ha-gbfs` |
| MPrime | 20 | 70 | `hmrp-ha-gbfs` |

All tasks use PUCT `0.1`, fixed-width MCTS, a 10,000-action cap, three workers,
a six-hour per-instance limit and code commit
`2b243a6e8f21723f99cfd84136770ad32d220ffa`. MPrime uses corrected IPC-scale
validation-selected checkpoints: jobs `20618716`/`20618717`, epochs 23/66.

This is an exploratory two-seed stop/drop characterization. It can establish
whether the dropped rollout direction behaves consistently enough to report
across both value-head modes, but it does not support a population-level
superiority claim or enter the primary RQ Holm family.

## Compatibility-only smoke

Use one first-seed rollout row per domain: manifest indices
`0,4,8,12,16,20`. Each smoke runs one test instance with at most 100 external
actions, one worker, 2 CPUs, 20 GiB and one hour. The verifier checks only:

- exact code/checkpoint identity;
- VH-off and rollout arguments;
- termination and completion ledger;
- expected search budget and leaf-evaluator marker;
- VAL validity when a plan exists.

Coverage, success and runtime never decide release. All 24 science identities
release if compatibility passes, regardless of smoke performance.

Prepared commands (documentation only; not executed):

```bash
python scripts/verify_mcts_leaf_evaluator_vhoff_all_imperfect.py \
  experiment_tracking/mcts_leaf_evaluator_vhoff_all_imperfect_20260920/manifest.csv

sbatch --array=0,4,8,12,16,20 --cpus-per-task=2 --mem=20G --time=01:00:00 \
  --exclude=ise-cpu-intl-[01,05,08-15,18,24-28] \
  --export=ALL,SMOKE=1 scripts/mcts_leaf_evaluator_vhoff_all_imperfect.sbatch

python scripts/verify_mcts_leaf_evaluator_vhoff_all_imperfect.py \
  experiment_tracking/mcts_leaf_evaluator_vhoff_all_imperfect_20260920/manifest.csv \
  --smoke-done /home/hersco/training_new_domains/2026-09-20/mcts_leaf_evaluator_vhoff_all_imperfect/smoke/done \
  --write-compatibility /home/hersco/training_new_domains/2026-09-20/mcts_leaf_evaluator_vhoff_all_imperfect/compatibility.json

sbatch --array=0-23 scripts/mcts_leaf_evaluator_vhoff_all_imperfect.sbatch
```

The idempotent release helper can instead be scheduled after successful
compatibility: `scripts/release_mcts_leaf_evaluator_vhoff_all_imperfect.sh`.

## Exact resources and remaining release steps

The six smokes request at most 12 CPUs and 120 GiB concurrently for one hour.
The 24 science tasks each request 6 CPUs and 120 GiB for at most 72 hours. A
fully concurrent launch therefore requests 144 CPUs and 2,880 GiB (2.8125
TiB). Scheduler concurrency can be capped without changing the design.

Before any submission:

1. review and commit this manifest, verifier, runner, release helper and tests;
2. deploy them to the isolated rollout checkout at the frozen code commit;
3. verify all 12 checkpoint directories and hashes;
4. run the six compatibility-only smokes under the established broad node
   exclusion `ise-cpu-intl-[01,05,08-15,18,24-28]`;
5. validate all six durable markers, then release all 24 tasks without a
   performance gate;
6. record immutable hashes and job IDs before analysis.

No VH-off smoke, gate or science job was submitted during this preparation.
