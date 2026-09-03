# TPP/off terminal-led Stage-2 continuation provenance

The affected lineage is seed `2082152039`, VH-off. This was not an ordinary
72-hour scheduler-timeout continuation.

1. Original job `20755752` reached Stage-2 snapshot 84 before the cluster
   outage/requeue event.
2. Slurm requeued the same batch script, but the wrapper restarted Stage 2 from
   its explicitly supplied Stage-1 source instead of auto-detecting the saved
   Stage-2 checkpoint. That restarted trajectory was not used as the scientific
   continuation.
3. Job `20834985` was then constructed as the true continuation from original
   snapshot 84. Its local snapshots 0--15 map to cumulative Stage-2 epochs
   85--100.
4. The restored validation state identifies original Stage-2 snapshot 75 as
   the validation-selected checkpoint. The final checkpoint is continuation
   snapshot 15, cumulative Stage-2 epoch 100.

The companion CSV stores both absolute snapshot paths and both training-log
paths. Policy materialization must combine the original snapshot directory at
offset 0 with the continuation directory at offset 85. No retraining is
required.
