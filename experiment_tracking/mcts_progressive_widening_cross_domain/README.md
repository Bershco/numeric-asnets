# Cross-domain progressive-widening screen

This directory now contains two deliberately distinct experiments.  This
distinction was added after discovering that the first Block Grouping/Counters
cross-domain screen had inherited the narrow comparator's 20-simulation budget
rather than the normal 70-simulation PW budget.

1. `MCTS-PW-CROSS-DOMAIN` / `pw-kmin3-*`: the original **mixed-budget
   screen**. Block Grouping and Counters use PW20 against fixed narrow 5/20;
   FO Counters and Rover use PW70 against fixed normal 20/70.
2. `MCTS-PW70-CROSS-DOMAIN` / `pw70-kmin3-*`: the separately submitted
   **standard-budget PW70 follow-up**.  It uses the same checkpoints and PW
   schedule but 70 simulations.  Its twelve jobs cover Block Grouping Stage 1,
   Counters Stage 1, and validation-led Counters Stage 2, both VH modes and two
   seeds.

The two arms must never be pooled or labelled interchangeably.  Final tables
must show policy, fixed 5/20, PW20, and PW70 in separate columns, plus fixed
20/70 wherever a matched result exists.

The label `PW20 cross-domain screen` is prohibited for the complete 20-row
screen: it is false for the eight FO Counters/Rover rows. Present it as
`MCTS-PW-MIXED-SCREEN`, while preserving the registered experiment ID for
provenance. The authoritative joined cutoff table is
`comparative_summary_20260901_1022.csv`; every row identifies the fixed search
as narrow 5/20 or normal 20/70 and keeps PW20 and PW70 separate.

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

## Two-hour PW20 appendix conclusion

Across the completed Block Grouping and Counters PW20 screening cells, the
two-hour and six-hour aggregate scores are identical except for Block Grouping
Stage1/on. That cell gains two additional solved instances across its two seeds
after two hours, moving its mean from 17/20 to 18/20. Every other completed
PW20 cell has the same mean at two and six hours.

This is an important efficiency appendix result: a two-hour post-hoc PW20 cap
would preserve nearly all observed PW20 coverage while bounding the long tail.
It is not yet a fair causal comparison against other search methods unless the
same two-hour cap is applied to fixed narrow, fixed normal and PW70. All main
tables therefore continue to report 30m/2h/6h together rather than promoting
the two-hour PW20 number alone.

Expansion recommendation after the two-seed screen:

- FO Counters qualifies most clearly for a five-seed PW70 confirmation because
  both VH cells match or exceed fixed normal coverage; add three matched seeds
  per VH rather than jumping directly to ten.
- Rover also qualifies on coverage, but its small gain and long/OOM-heavy jobs
  make it the second confirmation priority.
- Block Grouping and Counters should wait for the active PW70 correction before
  selecting confirmatory cells or a fresh hard-30-minute arm.

## Five-seed confirmation submitted 2 September 2026

The user approved expanding six cells: Counters validation-led Stage2,
FO Counters Stage1, and Rover Stage1, with both VH modes retained for each
domain/stage. The two original screening seeds remain part of each cell. Three
new matched seeds (`534933607`, `923500475`, and `1073581256`) were therefore
added per cell, producing 18 new jobs rather than restarting or duplicating the
two existing seeds.

The manifest is `pw70_confirmatory_expansion_manifest.csv` and exact submitted
job IDs are in `pw70_confirmatory_expansion_submissions.tsv` (jobs
`20838262`--`20838279`). All jobs use PW70: SAFE external selection, Kmin 3,
c 0.6, alpha 0.5, Kmax 20 and 70 simulations. They use Slurm `Nice=10000`,
verified to place them below the already pending ordinary work. This is an
ordinary resource-pending confirmation, not a scientific hold.

The expansion is defensible because it follows the predeclared screen-to-five
seed rule and keeps both VH modes even when only one mode looked strongest. It
does not establish a result until each cell has five matched terminal seeds.
PW20 and PW70 remain separate experiments and must never be pooled.

## Ten terminal jobs at 31 August 2026 18:05 IDT

Three additional PW70 jobs became terminal.  Every printed successful plan was
VAL-valid, with zero invalid plans:

- FO Counters Stage1/off, seed 2011206605: policy 4/20; matched fixed normal is
  9/10/10 at 30m/2h/6h; PW70 is 10/10/10.  The
  Slurm job OOMed after classifying 19/20 instances; the unclassified instance
  remains a conservative failure.
- FO Counters Stage1/on, seed 2011206605: policy 3/20; matched fixed normal is
  7/7/7; PW70 is 8/8/8.  The job classified all
  20 instances before its terminal OOM.
- Rover Stage1/off, seed 1963100312: policy 4/20; matched fixed normal is
  4/5/5; PW70 is 6/6/6.  The job completed normally.

FO Counters Stage1/off is now a complete two-seed screening cell: policy mean
3.5/20, fixed-normal mean 9.5/20 and PW70 mean 9.5/20.  It is coverage-positive
but not yet an efficiency success because one PW job still OOMed after 18h09m.
The paired VH-on job and the remaining Rover jobs are still live, so the domain
promotion rule has not yet been evaluated in full.

## Eleventh terminal job at 31 August 2026 20:13 IDT

Counters validation-led Stage2/on seed `2011206605` completed its budget-matched
PW20 run as job `20726021`. Policy scored 7/59, fixed narrow 5/20 scored 14/59,
and PW20 scored 12/59. All 59 instances were classified and all 12 printed
plans were VAL-valid. Every PW success completed within 482.39 seconds, so the
PW score is 12/59 under 30-minute, 2-hour, and 6-hour per-instance cutoffs.
PW20 therefore improves substantially over policy but does not match fixed
narrow on this seed. Whole-job elapsed time was 20h30m because the remaining
47 instances executed long unsuccessful 10,000-action trajectories; it was not
caused by late successful plans. This remains PW20 evidence. Its distinct PW70
counterpart is job `20755801`, currently resource-pending.
