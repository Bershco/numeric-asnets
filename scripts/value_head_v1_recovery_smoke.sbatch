#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/hersco/training_new_domains/2026-09-18/value_head_v1/recovery_smoke_%j.txt

set -euo pipefail
REPO=/home/hersco/bershco-nu-asnets/numeric-asnets-v1
PRODUCTION_REPO=/home/hersco/bershco-nu-asnets/numeric-asnets
ROOT=/home/hersco/training_new_domains/2026-09-18/value_head_v1
OUT="$ROOT/recovery_smoke/$SLURM_JOB_ID"
mkdir -p "$OUT"

bash -n \
  "$REPO/scripts/value_head_v1_state_capture.sbatch" \
  "$REPO/scripts/value_head_v1_manifest_rebuild.sbatch" \
  "$REPO/scripts/submit_value_head_v1_exact_recovery_20260919.sh"

apptainer exec --bind /home/hersco:/home/hersco /home/hersco/Docker/image.sif /bin/bash -lc "
  set -euo pipefail
  source '$PRODUCTION_REPO/venv-asnets/bin/activate'
  cd '$REPO'
  export PYTHONPATH='$REPO/asnets':'$REPO':\${PYTHONPATH:-}
  python tests/test_value_head_audit.py
  python -c \"import runpy; n=runpy.run_path('asnets/tests/test_value_head_v1_manifest.py'); n['test_balanced_selection_skips_missing_instances_and_duplicate_states'](); n['test_balanced_selection_respects_existing_cross_source_hashes'](); print('V1_MANIFEST_UNIT_OK')\"
  python scripts/materialize_value_head_v1_state_manifest.py \
    --domain drone --seed 534933607 \
    --capture-root '$ROOT/captures/drone_534933607' \
    --output '$OUT/vhv1-drone-534933607-smoke.jsonl'
  python scripts/materialize_value_head_v1_state_manifest.py \
    --domain drone --seed 923500475 \
    --capture-root '$ROOT/captures/drone_923500475' \
    --output '$OUT/vhv1-drone-923500475-smoke.jsonl'
"
printf 'V1_RECOVERY_SMOKE_OK|%s\n' "$OUT"
