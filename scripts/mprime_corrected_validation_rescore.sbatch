#!/usr/bin/env bash
#SBATCH --job-name=MPRIME_VAL_V1_RESCORE
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-26/mprime_validation_ipc_scale_v1/rescore_%A_%a.log

set -euo pipefail

repo=/home/hersco/bershco-nu-asnets/numeric-asnets
work=/home/hersco/training_new_domains/2026-08-26/mprime_validation_ipc_scale_v1
manifest="$work/lineage_manifest.tsv"
validation_tool="$repo/asnets/tools/validate_eval_log_with_summary.py"
line_number=$((SLURM_ARRAY_TASK_ID + 2))
row=$(sed -n "${line_number}p" "$manifest")
IFS=$'\t' read -r manifest_id seed value_head source_job snapshots_dir source_log validation_version status <<< "$row"

if [[ -z "$manifest_id" || "$status" != "ready-after-planner-sanity" ]]; then
  echo "Invalid manifest row at array index $SLURM_ARRAY_TASK_ID: $row" >&2
  exit 2
fi

out="$work/rescore/$manifest_id"
mkdir -p "$out"

mapfile -t checkpoints < <(
  find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d -name 'snapshot_*' \
    | sort -V
)
if [[ ${#checkpoints[@]} -eq 0 ]]; then
  echo "No snapshots found: $snapshots_dir" >&2
  exit 3
fi

for checkpoint in "${checkpoints[@]}"; do
  base=$(basename "$checkpoint")
  epoch=$(sed -E 's/^snapshot_([0-9]+)_.*/\1/' <<< "$base")
  log="$out/epoch_${epoch}.log"
  summary="$out/epoch_${epoch}_val.csv"
  done_marker="$out/epoch_${epoch}.done"
  if [[ -s "$done_marker" ]]; then
    echo "[SKIP] existing VAL summary epoch=$epoch"
    continue
  fi

  # A previous campaign version completed the evaluator but then failed because
  # its production validator did not support --summary-csv.  Reuse those logs:
  # VAL is deterministic and there is no reason to repeat policy inference.
  if grep -q '\[EVAL FINAL\]' "$log" 2>/dev/null; then
    echo "[RECOVER] validating existing completed evaluation epoch=$epoch"
    apptainer exec \
      --bind /home/hersco:/home/hersco \
      --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
      --pwd "$repo/asnets" \
      /home/hersco/Docker/image.sif \
      /bin/bash -lc "
        source ../venv-asnets/bin/activate
        python '$validation_tool' \
          --log '$log' \
          --domain mprime_validation_ipc_scale_v1 \
          --validator /home/hersco/tools/VAL/build/bin/Validate \
          --summary-csv '$summary'
      "
    printf 'checkpoint=%s\n' "$checkpoint" > "$done_marker"
    continue
  fi
  vh_flag=()
  if [[ "$value_head" == "off" ]]; then
    vh_flag=(--disable-value-head)
  fi
  apptainer exec \
    --bind /home/hersco:/home/hersco \
    --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
    --pwd "$repo/asnets" \
    /home/hersco/Docker/image.sif \
    /bin/bash -lc "
      source ../venv-asnets/bin/activate
      set -o pipefail
      ./run_experiment \
        experiments_numeric.architecture_2.mprime \
        experiments_numeric.domain.mprime_validation_ipc_scale_v1 \
        --resume-from '$checkpoint' \
        --num-workers 10 \
        --jpddl-max-heap 4g \
        --random-seed '$seed' \
        --worker-logs \
        ${vh_flag[*]} \
        2>&1 | tee '$log'
      python '$validation_tool' \
        --log '$log' \
        --domain mprime_validation_ipc_scale_v1 \
        --validator /home/hersco/tools/VAL/build/bin/Validate \
        --summary-csv '$summary'
    "
  printf 'checkpoint=%s\n' "$checkpoint" > "$done_marker"
done

echo "[RESCORE COMPLETE] manifest_id=$manifest_id checkpoints=${#checkpoints[@]}"
