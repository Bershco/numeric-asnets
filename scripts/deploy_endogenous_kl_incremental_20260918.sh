#!/bin/bash
set -euo pipefail

SOURCE=/home/hersco/bershco-nu-asnets/numeric-asnets-v1
EVAL=/home/hersco/bershco-nu-asnets/numeric-asnets-endo-kl-eval
CAMPAIGN=/home/hersco/training_new_domains/2026-09-18/endogenous_kl_100epoch_pilot
COMMIT=99716d2a

# These two jobs are the obsolete all-training-complete gate.  Leaving them
# queued would duplicate the incremental curve tasks and finalizer.
scancel 21453461 21453462

git -C "$SOURCE" fetch origin codex/mcts-safe-context
if test -e "$EVAL/.git"; then
  test -z "$(git -C "$EVAL" status --porcelain)"
  git -C "$EVAL" fetch origin codex/mcts-safe-context
  git -C "$EVAL" checkout --detach "$COMMIT"
else
  git -C "$SOURCE" worktree add --detach "$EVAL" "$COMMIT"
fi
test "$(git -C "$EVAL" rev-parse --short=8 HEAD)" = "$COMMIT"

OP_SOURCE=/home/hersco/bershco-nu-asnets/numeric-asnets/asnets/asnets/ops/_asnet_ops_impl.so
OP_TARGET="$EVAL/asnets/asnets/ops/_asnet_ops_impl.so"
test -e "$OP_SOURCE"
if ! test -e "$OP_TARGET"; then
  ln -s "$OP_SOURCE" "$OP_TARGET"
fi
test -e "$OP_TARGET"

full_commit=$(git -C "$EVAL" rev-parse HEAD)
raw=$(sbatch --parsable --dependency=after:21453460 \
  --export="ALL,CHECKOUT=$EVAL,CODE_COMMIT=$full_commit,CAMPAIGN_ROOT=$CAMPAIGN,SCRIPT_ROOT=$EVAL/scripts" \
  "$EVAL/scripts/endogenous_kl_incremental_controller_20260918.sbatch")
job=${raw%%;*}
printf 'incremental_controller\t%s\t%s\n' "$job" "$full_commit" \
  > "$CAMPAIGN/incremental_controller_job.tsv"
printf 'ENDO_KL_INCREMENTAL|job=%s|commit=%s\n' "$job" "$full_commit"
