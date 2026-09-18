# Endogenous 100-epoch KL-semantics pilot

## Question

Does replacing the historical dropout-current anchor KL with
deterministic-current KL improve complete Stage-2 training trajectories when
each arm generates its own replay, rather than consuming a frozen legacy
stream?

## Predeclared design

- Domains: Drone (cheap/stable) and FO Counters (clearest systematic
  PRIMARY-50 one-epoch harm/attenuation signal).
- Seeds: `534933607` and `923500475`, the first two canonical replication IDs
  in numeric order for each domain. They were not selected from KL outcomes.
- Networks: VH-off, validation-selected Stage-1 checkpoints, exact hashes in
  `manifest.csv`.
- Arms: historical dropout-current KL and deterministic-current KL.
- Both arms use the same current build and all common Stage-2 hyperparameters:
  anchor coefficient 3, learning rate 0.0003, 100 epochs, three workers,
  original training set, and the domain's frozen ENHSP teacher.
- Each matched pair passes the same canonical experiment `--random-seed` shown
  in `manifest.csv`. The special fixed-step RNG flag used by causal frozen-
  replay diagnostics is intentionally absent: this pilot compares the normal
  endogenous training algorithms, including their downstream replay feedback,
  rather than forcing an artificial per-update stochastic schedule.
- Replay is endogenous. No frozen-replay option is passed. Each arm generates
  and consumes its own states and targets throughout training.
- Policy coverage is evaluated at epochs `0,5,...,95,99`. Epoch 99 and the
  complete learning trajectory are predeclared outcomes; test results are not
  used to choose a checkpoint.
- MCTS is outside this pilot. It is warranted only if deterministic-current KL
  becomes a candidate replacement training method.

This is a two-seed method-development pilot, not a population-level efficacy
claim. A negative or inconsistent result stops expansion. A coherent benefit
justifies a ten-seed confirmation before replacing historical RQ results.

## Reuse audit

Historical MAIN-VAL legacy Stage-2 runs exist for all four source networks,
but they used the historical build. They are retained as provenance and may be
shown descriptively; they are not reused as matched pilot arms because doing so
would confound KL semantics with code/build changes. No existing
deterministic-current 100-epoch run starts from these exact four checkpoints.
Therefore all eight same-build training identities are new.

## Resources and dependency chain

- compute smoke: 2 CPU, 8 GiB, 10 minutes;
- training: eight tasks, each 6 CPU / 48 GiB / 18 hours;
- policy curves: 168 tasks, each 5 CPU / 20 GiB / 2 hours, released only after
  all training tasks succeed;
- finalizer: 1 CPU / 4 GiB / 30 minutes.

Known failing nodes are excluded. Slurm, rather than an artificial array cap,
controls how many tasks run concurrently.

## Outputs

Remote root:
`/home/hersco/training_new_domains/2026-09-18/endogenous_kl_100epoch_pilot`

The finalizer writes `learning_curve_results.csv`, `endpoint99_results.csv`,
and `domain_summary.csv`, retaining every source checkpoint, task identity,
training log and endpoint-evaluation log.
