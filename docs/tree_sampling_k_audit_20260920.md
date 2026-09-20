# Stage-2 tree-sampling K audit

Last updated: 2026-09-20.

## Current status

The canonical Stage-2 configuration uses `tree_sampling_k = 0`, which disables
extra replay examples sampled from internal MCTS nodes. This is both the parser
default and the explicit value in the current primary submission pipelines.
It is a frozen pipeline choice, not a demonstrated optimum.

The implemented meanings are:

- `K = 0`: do not add internal tree nodes to replay;
- `K = -1`: add all eligible internal nodes;
- `K > 0`: sample at most `K` eligible internal nodes without replacement,
  weighted by squared visit count.

An internal node is eligible only after more than five visits. Its target is
the ordinary final MCTS policy/value target and its provenance is recorded as
`DataSource.TREE_SAMPLE`.

## Historical evidence

The cleanest surviving comparison contains 28 matched configuration cells from
Block Grouping, Drone, FO Counters and Rover. It compares historical `K=-1`
with `K=0`. These cells contain only two underlying stochastic seeds; value-head
and estimator variants are repeated conditions, not independent replications.

| Domain | Mean `K=-1` minus `K=0` | Win / tie / loss cells |
| --- | ---: | ---: |
| Block Grouping | +1.00 solved instances | 5 / 1 / 2 |
| Drone | 0.00 | 2 / 5 / 1 |
| FO Counters | +0.50 | 5 / 2 / 1 |
| Rover | 0.00 | 0 / 4 / 0 |

The interaction with value-head mode is material:

| Value-head mode | Cells | Mean change | Win / tie / loss cells |
| --- | ---: | ---: | ---: |
| Off | 12 | +1.25 | 7 / 5 / 0 |
| On | 16 | -0.19 | 5 / 7 / 4 |

Representative estimator-0.5 cells include Block Grouping `+3.5` with VH-off
but `-2.0` with VH-on, FO Counters `+1.0` with VH-off and `0.0` with VH-on,
Drone `0.0`/`+0.5`, and Rover `0.0`. Positive bounded values such as `K=10`
also appear in older Counters work, but no complete canonical matched outcome
table survives for `K=5`, `K=10` or `K=20`.

## Critical implementation caveat

The historical `K=-1` evidence predates commit `0c2570e9`, which changed tree
sample extraction from after every external action to once after the complete
episode. The old behavior could repeatedly add related states, overweight long
trajectories and alter replay volume much more strongly than the current code.
The table above is therefore development evidence, not a clean estimate of the
current implementation.

## Defensible thesis conclusion

Tree sampling was a genuine Stage-2 replay-augmentation direction. It was
sometimes promising, especially in VH-off Block Grouping and FO Counters, but
was mixed across value-head modes, evaluated with only two stochastic seeds,
and partly used an implementation later corrected. It was disabled in the
primary pipeline because it was insufficiently confirmed, **not** because it
was proven ineffective.

Tree sampling is related to HER only at the level of replay augmentation.
Tree sampling adds search-internal states with the original goal and ordinary
MCTS targets. HER relabels or reconstructs goal-directed supervision. They are
different training hypotheses and must be reported separately.

## Minimal current-code closure experiment

If the thesis needs a current-code outcome rather than historical development
evidence, use a predeclared matched `K=0` versus one nonzero setting. `K=10` is
the preferred bounded treatment; `K=-1` would instead be a direct historical
replication. Start with two contrasting imperfect domains, two seeds and both
value-head modes, while recording replay volume, unique sampled states, source
mixture, runtime and memory. A consistently negative frozen pilot can justify
stopping the direction as exploratory evidence. A broad effectiveness claim
would require ten matched seeds.

This experiment is documented but not submitted. It should not displace the
currently running corrected-KL, leaf-evaluator, value-head-quality or Counters
mechanism campaigns without an explicit prioritization decision.
