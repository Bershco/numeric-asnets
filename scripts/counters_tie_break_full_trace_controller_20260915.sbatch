#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --nice=10000
#SBATCH --no-requeue
#SBATCH --job-name=COUNTERS_TRACE_CTL
#SBATCH --output=/home/hersco/training_new_domains/2026-09-15/counters_tie_break_full_trace/controller_%j.out

set -euo pipefail
: "${CODE_COMMIT:?CODE_COMMIT is required}"
repo=/home/hersco/bershco-nu-asnets/numeric-asnets-tie-break
tracking="$repo/experiment_tracking/counters_tie_break_strict_stage1_20260913"
base_manifest="$tracking/manifest.csv"
trace_manifest="$tracking/full_trace_manifest.csv"
trace_summary="$tracking/full_trace_source_summary.csv"
out=/home/hersco/training_new_domains/2026-09-15/counters_tie_break_full_trace
actual_commit=$(git -C "$repo" rev-parse HEAD)
[[ "$actual_commit" == "$CODE_COMMIT" ]]
mkdir -p "$out"
if [[ -e "$out/submissions.csv" ]]; then
  echo "refusing duplicate trace submission: $out/submissions.csv already exists" >&2
  exit 1
fi

sha256sum \
  "$repo/scripts/reconcile_counters_tie_break_terminal_records_20260915.py" \
  "$repo/scripts/reconcile_counters_tie_break_all_action_id_20260915.py" \
  "$repo/scripts/build_counters_tie_break_full_trace_manifest_20260915.py" \
  "$repo/scripts/run_counters_tie_break_full_trace_20260915.py" \
  "$repo/scripts/counters_tie_break_full_trace_20260915.sbatch" \
  "$repo/scripts/counters_tie_break_full_trace_controller_20260915.sbatch" \
  "$base_manifest" > "$out/deployed_inputs.sha256"

python "$repo/scripts/reconcile_counters_tie_break_all_action_id_20260915.py" \
  --manifest "$base_manifest" \
  > "$out/action_id_terminal_reconciliation.json"

python "$repo/scripts/build_counters_tie_break_full_trace_manifest_20260915.py" \
  --base-manifest "$base_manifest" \
  --output-manifest "$trace_manifest" \
  --output-summary "$trace_summary"

task_count=$(( $(wc -l < "$trace_manifest") - 1 ))
(( task_count > 0 ))
last=$(( task_count - 1 ))
trace_job=$(sbatch --parsable --array="0-${last}%12" \
  --export="ALL,CODE_COMMIT=$CODE_COMMIT" \
  "$repo/scripts/counters_tie_break_full_trace_20260915.sbatch")
printf 'controller_job_id,trace_array_job_id,trace_tasks,max_concurrency,code_commit,manifest,summary\n' \
  > "$out/submissions.csv"
printf '%s,%s,%s,12,%s,%s,%s\n' \
  "$SLURM_JOB_ID" "$trace_job" "$task_count" "$CODE_COMMIT" \
  "$trace_manifest" "$trace_summary" >> "$out/submissions.csv"
echo "[COUNTERS TRACE CONTROLLER] trace_job=$trace_job tasks=$task_count concurrency=12"
