#!/usr/bin/env bash
set -euo pipefail

for job_id in "$@"; do
    stdout_path=$(scontrol show job -o "$job_id" | sed -n 's/.* StdOut=\([^ ]*\).*/\1/p')
    elapsed=$(squeue -h -j "$job_id" -o %M)
    if [[ ! -f "$stdout_path" ]]; then
        printf '%s|%s|missing_log=%s\n' "$job_id" "$elapsed" "$stdout_path"
        continue
    fi
    classified=$({ grep -E '\[EVAL INSTANCE\] (completed|skip completed)' "$stdout_path" || true; } \
        | sed -n 's/.*number=\([0-9]*\).*/\1/p' | sort -nu | wc -l)
    successes=$({ grep -E '\[EVAL INSTANCE\] (completed|skip completed)' "$stdout_path" || true; } \
        | { grep -E 'success=(1\.0|True)' || true; } \
        | sed -n 's/.*number=\([0-9]*\).*/\1/p' | sort -nu | wc -l)
    started=$({ grep -E '\[EVAL INSTANCE\] started' "$stdout_path" || true; } \
        | sed -n 's/.*number=\([0-9]*\).*/\1/p' | sort -nu | wc -l)
    timeouts=$({ grep -E '\[EVAL INSTANCE\] timeout' "$stdout_path" || true; } \
        | sed -n 's/.*number=\([0-9]*\).*/\1/p' | sort -nu | wc -l)
    cutoffs=$({ grep -o 'horizon_cutoffs=(count=[0-9]*' "$stdout_path" || true; } \
        | sed 's/.*=//' | awk '{total += $1} END {print total + 0}')
    printf '%s|%s|classified=%s|success=%s|started=%s|timeouts=%s|cutoffs=%s|%s\n' \
        "$job_id" "$elapsed" "$classified" "$successes" "$started" "$timeouts" "$cutoffs" "$stdout_path"
done
