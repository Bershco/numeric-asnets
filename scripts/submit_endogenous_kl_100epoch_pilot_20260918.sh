#!/bin/bash
set -euo pipefail
: "${CHECKOUT:?}"; : "${CODE_COMMIT:?}"
ROOT=/home/hersco/training_new_domains/2026-09-18/endogenous_kl_100epoch_pilot
test -f "$ROOT/manifest.csv"; mkdir -p "$ROOT/outputs"
common="ALL,CHECKOUT=$CHECKOUT,CODE_COMMIT=$CODE_COMMIT,CAMPAIGN_ROOT=$ROOT"
smoke=$(sbatch --parsable --export="$common" "$CHECKOUT/scripts/endogenous_kl_100epoch_smoke_20260918.sbatch" | cut -d';' -f1)
train=$(sbatch --parsable --dependency=afterok:"$smoke" --export="$common" "$CHECKOUT/scripts/endogenous_kl_100epoch_train_20260918.sbatch" | cut -d';' -f1)
policy=$(sbatch --parsable --dependency=afterok:"$train" --export="$common" "$CHECKOUT/scripts/endogenous_kl_100epoch_policy_curve_20260918.sbatch" | cut -d';' -f1)
final=$(sbatch --parsable --dependency=afterok:"$policy" --export="$common" "$CHECKOUT/scripts/endogenous_kl_100epoch_finalize_20260918.sbatch" | cut -d';' -f1)
printf 'smoke=%s\ntraining=%s\npolicy=%s\nfinalizer=%s\ncommit=%s\ncheckout=%s\n' "$smoke" "$train" "$policy" "$final" "$CODE_COMMIT" "$CHECKOUT" | tee "$ROOT/submission_ids.txt"
