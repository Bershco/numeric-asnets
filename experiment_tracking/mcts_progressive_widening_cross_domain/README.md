# Cross-domain progressive-widening screen

This directory now contains two deliberately distinct experiments.  This
distinction was added after discovering that the first Block Grouping/Counters
cross-domain screen had inherited the narrow comparator's 20-simulation budget
rather than the normal 70-simulation PW budget.

1. `MCTS-PW-CROSS-DOMAIN` / `pw-kmin3-*`: the original **budget-matched PW20
   screen**.  It remains valid as a comparison against fixed narrow 5/20.
2. `MCTS-PW70-CROSS-DOMAIN` / `pw70-kmin3-*`: the separately submitted
   **standard-budget PW70 follow-up**.  It uses the same checkpoints and PW
   schedule but 70 simulations.  Its twelve jobs cover Block Grouping Stage 1,
   Counters Stage 1, and validation-led Counters Stage 2, both VH modes and two
   seeds.

The two arms must never be pooled or labelled interchangeably.  Final tables
must show policy, fixed 5/20, PW20, and PW70 in separate columns, plus fixed
20/70 wherever a matched result exists.

- Block Grouping Stage 1 and Counters Stage 1/Stage 2 use 20 simulations so
  Kmin=3 PW is matched to the scientifically relevant fixed-narrow 5/20 arm.
- FO Counters and Rover use 70 simulations so PW is matched to their available
  fixed-normal 20/70 arm.
- The completed Drone PW pilot and Kmin=3 extension use 70 simulations.

Therefore, a successful 20-simulation Counters row establishes that widening is
better than fixed width under the same small simulation budget. It does **not**
establish the result of PW with the standard 70-simulation budget. A promoted
Block Grouping/Counters cell must receive a separately labelled 70-simulation
confirmation if the thesis claim concerns standard-budget PW.

## Submitted PW70 follow-up

The twelve rows in `pw70_followup_manifest.csv` were submitted as jobs
`20755790`--`20755801`; exact job-to-row provenance is in
`pw70_followup_submissions.tsv`.  They are ordinary resource-pending jobs, not
held jobs.

The first terminal PW20 cell is Block Grouping Stage 1, VH-on.  For seeds
1963100312 and 2011206605, policy and fixed 5/20 both average 17/20, while PW20
averages 18/20.  Whole-job runtime averages about 8h38m for fixed 5/20 and
9h22m for PW20.  This is a coverage-positive candidate, but not yet a formal
promotion because it has not demonstrated an efficiency benefit; retained-node
evidence and the PW70 follow-up remain relevant.

## Promotion rule

A cell means one domain, training stage, and VH mode—not one favourable seed or
one test instance. The screen uses two matched seeds per cell. A cell is
promising when PW:

1. is no worse than policy on mean coverage;
2. is within five percentage points of its matched fixed-search mean;
3. reduces runtime, timeouts, OOMs, or retained nodes materially; and
4. does not introduce a systematic new policy-success regression.

To avoid choosing only a favourable VH result, if either VH cell for one
domain/stage is promoted, both VH modes are expanded to at least five matched
seeds. Fresh 30-minute confirmation remains a later, separately labelled arm.

## Terminal PW20 evidence at 31 August 2026 08:45 IDT

Seven of twenty PW20 jobs are terminal. Block Grouping Stage1/off is a complete
two-seed cell: policy averages 16.5/20, fixed 5/20 and PW20 both average
15.0/20, while whole-job runtime increases from about 7h39m to 12h26m. It is
unpromising: PW loses policy coverage, gains nothing over fixed narrow, and is
slower.

Block Grouping Stage1/on remains coverage-positive (policy and fixed 17.0/20,
PW20 18.0/20) but PW20 is slower (9h22m versus 8h38m). Neither fixed comparator
OOMed or ended early. Selection itself is a small fraction of runtime;
successor generation, evaluation, and longer searched trajectories dominate.
Under post-hoc cutoffs, BG/off fixed versus PW is 11.0 versus 11.5 at 30m and
13.5 versus 15.0 at 2h; BG/on is 11.5 versus 12.5 at 30m and 13.5 versus 17.0
at 2h. Thus PW is slower over the complete allocation but slightly better when
individual instances receive shorter budgets.

Counters Stage2/off has one strong screening seed: policy 59/59, fixed narrow
49/59 and PW20 59/59; whole-job runtime is 3h47m for PW versus 18h05m fixed.
At 30m the scores are PW52 versus fixed40, and at 2h PW59 versus fixed49. One
seed remains screening evidence, not a confidence-interval result. Exact rows
and log pointers are in `live_reconciliation_20260831.csv`.

## PW20 width and phase diagnostics

For Kmin=3, c=0.6 and alpha=0.5, the permitted width is 3 through visit 44,
4 for visits 45--69, 5 for 70--99, 6 for 100--136, 7 for 137--177 and 8 from
178.  A newly created root therefore has exactly three children after a
20-simulation call.  Across the four terminal Block Grouping PW20 jobs, the
actual weighted means are 2.015 children over all node observations and 5.860
children over root observations.  Roots can exceed three because an external
action may promote a node that already accumulated visits while deeper in the
retained tree.

Across completed instances in those jobs, the compact phase counters contain:

- successor generation: 93,800.6 seconds, 55.2% of completed wall time;
- evaluation/estimator: 55,166.4 seconds, 32.4%;
- network inference: 11,795.5 seconds, 6.9%;
- selection: 7,505.6 seconds, 4.4%;
- backpropagation: 559.3 seconds, 0.3%.

Expansion time overlaps successor-generation and network time and must not be
added to those percentages.  Thus “longer searched trajectory” means many
costly repeated root searches across the external-action trajectory; it does
not mean Python tree selection is the bottleneck.

For Block Grouping Stage1/off seed 1963100312, fixed 5/20 exhausted the
10,000-action budget on instances 19 and 20 in 4,817.67 and 15,772.01 seconds,
whereas PW20 hit the 21,600-second timeout on both.  Timeout workers are killed
before their final diagnostic summary, so exact phase totals for those two
instances do not exist.  The nearest completed PW instance (18) spent 2,831.29
of 5,464.02 seconds in successor generation and 2,493.73 seconds in evaluation,
but only 4.13 seconds in selection.  Historical fixed logs predate phase
logging; an exact fixed-versus-PW phase comparison would require one small
matched profiling rerun rather than inference from unavailable data.

The authoritative diagnostic aggregate is `pw20_block_grouping_diagnostics.csv`.
