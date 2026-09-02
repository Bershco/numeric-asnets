#!/usr/bin/env bash
set -euo pipefail

checkout=/home/hersco/bershco-nu-asnets/numeric-asnets-safe-context
ledger="$checkout/experiment_tracking/mprime_validation_ipc_scale_v1/stage2_submissions_corrected.tsv"
campaign="$checkout/experiment_tracking/mprime_validation_ipc_scale_v1"
controller="$checkout/scripts/stage2_policy_refresh.sbatch"

bash -n "$controller"
/home/hersco/bershco-nu-asnets/numeric-asnets/venv-asnets/bin/python \
  "$checkout/experiment_tracking/materialize_stage2_policy_from_ledger.py" \
  --help | grep -q -- --value-head

# This is the obsolete validation-branch-wide controller.  It has not run.
scancel 20832933

off_job=$(sbatch --parsable \
  --job-name=MPRIME_CORR_VAL_OFF_POLICY_CTRL \
  --dependency=afterany:20832890:20832891:20832892:20832893:20832894:20832895:20688832:20832896:20688839:20832897 \
  --export=ALL,TRAINING_LEDGER="$ledger",READY_MANIFEST="$campaign/validation_led_stage2_policy_ready_off.csv",POLICY_LEDGER="$campaign/validation_led_stage2_policy_submissions_off.tsv",ROLE_PREFIX=mprime_validation_led_stage2,SUFFIX_PREFIX=MPEXT6VP_OFF,OUTPUT_PREFIX=mprime_validation_led_stage2_policy,DOMAIN_FILTER=mprime,BRANCH_FILTER=validation_led,VALUE_HEAD_FILTER=off,EXCLUDE_REUSED=0 \
  "$controller")

on_job=$(sbatch --parsable \
  --job-name=MPRIME_CORR_VAL_ON_POLICY_CTRL \
  --dependency=afterany:20832898:20832899:20832900:20832901:20832902:20832903:20688846:20832904:20688853:20832906 \
  --export=ALL,TRAINING_LEDGER="$ledger",READY_MANIFEST="$campaign/validation_led_stage2_policy_ready_on.csv",POLICY_LEDGER="$campaign/validation_led_stage2_policy_submissions_on.tsv",ROLE_PREFIX=mprime_validation_led_stage2,SUFFIX_PREFIX=MPEXT6VP_ON,OUTPUT_PREFIX=mprime_validation_led_stage2_policy,DOMAIN_FILTER=mprime,BRANCH_FILTER=validation_led,VALUE_HEAD_FILTER=on,EXCLUDE_REUSED=0 \
  "$controller")

printf 'off=%s on=%s\n' "$off_job" "$on_job"
squeue -h -j "$off_job,$on_job" -o '%i|%T|%j|%R'
