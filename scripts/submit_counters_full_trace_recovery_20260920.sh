#!/usr/bin/env bash
set -euo pipefail

repo=/home/hersco/bershco-nu-asnets/numeric-asnets-tie-break
source_manifest="$repo/experiment_tracking/counters_tie_break_strict_stage1_20260913/full_trace_manifest.csv"
campaign=/home/hersco/training_new_domains/2026-09-20/counters_tie_break_full_trace_recovery
manifest="$campaign/recovery_manifest.csv"
summary="$campaign/reconciliation.json"
commit=$(git -C "$repo" rev-parse HEAD)

mkdir -p "$campaign/slurm"
python3 "$repo/scripts/build_counters_full_trace_recovery_20260920.py" \
  --source "$source_manifest" \
  --output "$manifest" \
  --summary "$summary" \
  --recovery-root "$campaign/results"
tasks=$(( $(wc -l < "$manifest") - 1 ))
(( tasks > 0 ))
last=$(( tasks - 1 ))
job=$(sbatch --parsable \
  --array="0-$last" \
  --export="ALL,CODE_COMMIT=$commit,TRACE_MANIFEST=$manifest,COUNTERS_REPO=$repo" \
  "$repo/scripts/counters_full_trace_recovery_20260920.sbatch")
printf 'job_id,tasks,code_commit,manifest,summary\n%s,%s,%s,%s,%s\n' \
  "$job" "$tasks" "$commit" "$manifest" "$summary" > "$campaign/submission.csv"
printf 'job=%s tasks=%s manifest=%s\n' "$job" "$tasks" "$manifest"
