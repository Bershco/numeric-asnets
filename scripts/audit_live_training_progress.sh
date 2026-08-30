#!/usr/bin/env bash
set -euo pipefail

for job_id in "$@"; do
    stdout_path=$(scontrol show job -o "$job_id" | sed -n 's/.* StdOut=\([^ ]*\).*/\1/p')
    elapsed=$(squeue -h -j "$job_id" -o %M)
    if [[ ! -f "$stdout_path" ]]; then
        printf '%s|%s|missing_log=%s\n' "$job_id" "$elapsed" "$stdout_path"
        continue
    fi
    epoch=$({ grep -aE 'epoch:[^[:cntrl:]]*[0-9]+/[0-9]+' "$stdout_path" || true; } \
        | tail -n 1 | sed -nE 's/.*[[:space:]]([0-9]+)\/([0-9]+).*/\1\/\2/p')
    validation=$({ grep -aE '(validation|VALIDATION).*(coverage|success)' "$stdout_path" || true; } \
        | tail -n 1 | tr '\r\n' ' ' | cut -c1-220)
    printf '%s|%s|epoch=%s|validation=%s|%s\n' \
        "$job_id" "$elapsed" "${epoch:-unknown}" "${validation:-none}" "$stdout_path"
done
