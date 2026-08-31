#!/usr/bin/env bash
#SBATCH --job-name=MPRIME_ANCHOR_VAL_V1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/hersco/training_new_domains/2026-08-31/mprime_anchor_corrected_validation/rescore_%A_%a.log

set -euo pipefail

repo=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
production=/home/hersco/bershco-nu-asnets/numeric-asnets
work=/home/hersco/training_new_domains/2026-08-31/mprime_anchor_corrected_validation
manifest="$work/anchor_selection_invalid.csv"
validation_tool="$repo/asnets/tools/validate_eval_log_with_summary.py"
mkdir -p "$work/rescore"

row=$(
  /home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/python - \
    "$manifest" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv
import sys
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
row = rows[int(sys.argv[2])]
print("\t".join(row[key] for key in (
    "manifest_id", "job_id", "value_head", "seed", "anchor", "source_log"
)))
PY
)
IFS=$'\t' read -r manifest_id source_job value_head seed anchor source_log <<< "$row"
[[ -s "$source_log" ]] || { echo "missing training log: $source_log" >&2; exit 2; }

snapshots_dir=$(grep -m1 '^Snapshot directory: ' "$source_log" | sed 's/^Snapshot directory: //')
[[ -d "$snapshots_dir" ]] || { echo "missing snapshots: $snapshots_dir" >&2; exit 3; }

out="$work/rescore/$manifest_id"
mkdir -p "$out"
mapfile -t all_checkpoints < <(
  find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d -name 'snapshot_*' | sort -V
)
[[ ${#all_checkpoints[@]} -gt 0 ]] || { echo "no checkpoints: $snapshots_dir" >&2; exit 3; }
last_checkpoint=${all_checkpoints[-1]}
checkpoints=()
for checkpoint in "${all_checkpoints[@]}"; do
  epoch=$(basename "$checkpoint" | sed -E 's/^snapshot_([0-9]+)_.*/\1/')
  if (( epoch % 5 == 0 )) || [[ "$checkpoint" == "$last_checkpoint" ]]; then
    checkpoints+=("$checkpoint")
  fi
done

if [[ -n "${MAX_CHECKPOINTS:-}" ]]; then
  checkpoints=("${checkpoints[@]:0:$MAX_CHECKPOINTS}")
fi

vh_flag=()
if [[ "$value_head" == "off" ]]; then vh_flag=(--disable-value-head); fi

for checkpoint in "${checkpoints[@]}"; do
  base=$(basename "$checkpoint")
  epoch=$(sed -E 's/^snapshot_([0-9]+)_.*/\1/' <<< "$base")
  log="$out/epoch_${epoch}.log"
  summary="$out/epoch_${epoch}_val.csv"
  done_marker="$out/epoch_${epoch}.done"
  if [[ -s "$done_marker" ]]; then
    echo "[SKIP] $manifest_id epoch=$epoch"
    continue
  fi

  apptainer exec \
    --bind /home/hersco:/home/hersco \
    --bind /home/hersco/apptainer_fake_passwd:/etc/passwd \
    --pwd "$repo/asnets" \
    /home/hersco/Docker/image.sif \
    /bin/bash -lc "
      source '$production/venv-asnets/bin/activate'
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
  printf 'checkpoint=%s\nanchor=%s\nsource_job=%s\n' \
    "$checkpoint" "$anchor" "$source_job" > "$done_marker"
done

echo "[ANCHOR RESCORE COMPLETE] manifest_id=$manifest_id points=${#checkpoints[@]}"
