# Thesis reproducibility artifacts

The stable advisor-facing Numeric ASNet thesis bundle is maintained separately:

- <https://github.com/Bershco/numeric-asnets-thesis-artifacts>

The development repository remains:

- <https://github.com/Bershco/numeric-asnets>

The artifact repository is the publication target for selected source revisions,
compact result tables, job/command manifests, plotting and statistical scripts,
original-log provenance, checksums, selected checkpoints and the Git-LFS-managed
Apptainer image. The current publication manifest is:

- `experiment_tracking/advisor_followup_20260910/reproducibility_publication_manifest.csv`

Primary results from 10 September 2026 onward use the validation-led branch.
Completed terminal-led results remain archived and traceable, but are not part of
the primary RQ analysis or future continuations.

The 10 September advisor follow-up additionally freezes an external-distribution
screen for FO Counters and Rover from SPL-BGU's PDDL generator commit
`b64e5d086117ebd5c1d53fbe9a9d93ae609a59fb`. The exact instance/checkpoint/task
manifest is `experiment_tracking/advisor_followup_20260910/yarin_external_screen_manifest.csv`.
The active screen uses policy inference only: two domains, two value-head modes,
and three fixed seeds (12 tasks). It does not add MCTS work.
