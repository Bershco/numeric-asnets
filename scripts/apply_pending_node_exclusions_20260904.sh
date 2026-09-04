#!/usr/bin/env bash
set -euo pipefail

excluded="ise-cpu128-03,ise-cpu128-04,ise-cpu-intl-13"
updated=0
while IFS='|' read -r job_id job_name; do
  case "$job_name" in
    *P4TPPS2P*|*SR10M*)
      scontrol update "JobId=$job_id" "ExcNodeList=$excluded"
      shown=$(scontrol show job -o "$job_id")
      case "$shown" in
        *"ExcNodeList=$excluded"*) ;;
        *) echo "verification failed for $job_id" >&2; exit 1 ;;
      esac
      echo "updated $job_id $job_name"
      updated=$((updated + 1))
      ;;
  esac
done < <(squeue -u hersco -h -t PENDING -o '%i|%j')
echo "updated_pending_jobs=$updated"
