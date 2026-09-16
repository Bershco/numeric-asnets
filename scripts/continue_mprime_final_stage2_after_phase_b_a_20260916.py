#!/usr/bin/env python3
"""Build/reuse/submit final MPrime searches after Phase-B-A selection."""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


RATE = re.compile(r"Inference success rate:\s*([0-9.]+)")


def read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write(path: Path, rows: list[dict[str, str]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=delimiter, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def policy_evidence(paths: list[Path], training_job: str, epoch: int) -> dict[str, str] | None:
    suffix = f"_src{training_job}_e{epoch:04d}.txt"
    candidates = []
    for root in paths:
        for path in root.glob(f"*{suffix}"):
            content = path.read_text(errors="replace")
            matches = RATE.findall(content)
            if matches and "Traceback (most recent call last)" not in content:
                candidates.append((path, float(matches[-1])))
    if not candidates:
        return None
    candidates.sort(key=lambda item: int(item[0].name.split("_", 1)[0]))
    path, rate = candidates[-1]
    return {
        "source_policy_job_id": path.name.split("_", 1)[0],
        "source_policy_log": str(path),
        "selected_test_policy_score": str(round(rate * 20)),
    }


def reusable(old_root: Path, template: dict[str, str], selected_hash: str) -> tuple[bool, str]:
    if template["checkpoint_sha256"] != selected_hash:
        return False, "checkpoint_hash_differs"
    leaf = old_root / template["search_method"] / template["value_head"] / template["seed"]
    completion = leaf / "completion" / f"{template['manifest_id']}.jsonl"
    attempts = leaf / "attempts.tsv"
    if not completion.is_file() or sum(1 for line in completion.open(errors="ignore") if line.strip()) != 20:
        return False, "not_20_durable_classifications"
    if not attempts.is_file():
        return False, "missing_attempt_ledger"
    valid = any(
        row["manifest_id"] == template["manifest_id"]
        and row["evaluation_returncode"] == "0"
        and row["validation_returncode"] == "0"
        for row in read(attempts, "\t")
    )
    return (valid, "exact_hash_config_complete" if valid else "no_valid_terminal_attempt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--old-output", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, action="append", required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runner-source", type=Path, required=True)
    parser.add_argument("--sbatch", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--policy-submit-script", type=Path,
        default=Path("/home/hersco/bershco-nu-asnets/numeric-asnets-mprime-policy-live-20260915/experiment_tracking/submit_stage2_policy.py"),
    )
    parser.add_argument(
        "--policy-submitter", type=Path,
        default=Path("/home/hersco/training_new_domains/2026-09-15/mprime_final_stage2_policy/submit_training_mprime_policy_wrapper_20260916.sh"),
    )
    args = parser.parse_args()

    selected, old = read(args.selected), read(args.old_manifest)
    if len(selected) != 20 or len(old) != 40:
        raise RuntimeError("expected twenty selected endpoints and forty templates")
    templates = {(row["value_head"], row["seed"], row["search_method"]): row for row in old}
    rows, audit, missing_policy = [], [], []
    for endpoint in sorted(selected, key=lambda row: (row["value_head"], int(row["seed"]))):
        epoch = int(endpoint["snapshot_epoch"])
        policy = policy_evidence(args.policy_root, endpoint["source_training_job_id"], epoch)
        if policy is None:
            missing_policy.append(endpoint)
            continue
        for method in ("fixed", "pw70"):
            template = dict(templates[(endpoint["value_head"], endpoint["seed"], method)])
            template.update({
                "array_index": str(len(rows)),
                "manifest_id": f"mprime-final-s2-{endpoint['value_head']}-{endpoint['seed']}-e{epoch:04d}-{method}20x70",
                "selected_epoch": str(epoch),
                "selected_validation_score": endpoint["selected_validation_score"],
                **policy,
                "source_training_job_id": endpoint["source_training_job_id"],
                "source_training_log": endpoint["training_log"],
                "checkpoint": endpoint["source_checkpoint_ref"],
                "checkpoint_sha256": endpoint["source_checkpoint_sha256"],
                "ready_manifest": str(args.selected),
                "ready_manifest_sha256": "phase_b_a_selected_endpoints",
                "selected_policy_source": policy["source_policy_log"],
                "selected_policy_source_sha256": "log_identity_in_manifest",
                "status": "ready_not_submitted",
            })
            reuse, reason = reusable(args.old_output, templates[(endpoint["value_head"], endpoint["seed"], method)], endpoint["source_checkpoint_sha256"])
            audit.append({
                "array_index": template["array_index"], "manifest_id": template["manifest_id"],
                "reusable": int(reuse), "reason": reason,
                "old_manifest_id": templates[(endpoint["value_head"], endpoint["seed"], method)]["manifest_id"],
                "checkpoint_sha256": endpoint["source_checkpoint_sha256"],
            })
            if reuse:
                template["status"] = "reused_exact_hash_config"
            rows.append(template)
    if missing_policy:
        recovery_manifest = args.campaign / "selected_policy_recovery_manifest.csv"
        recovery_ledger = args.campaign / "selected_policy_recovery_submissions.tsv"
        recovery_prefix = "mprime_final_stage2_selected_policy_recovery"
        recovery_root = (
            Path("/home/hersco/training_new_domains")
            / datetime.now().astimezone().date().isoformat()
            / f"{recovery_prefix}_mprime"
        )
        recovery_rows = []
        for endpoint in missing_policy:
            epoch = int(endpoint["snapshot_epoch"])
            recovery_rows.append({
                "manifest_id": f"mprime-final-s2-{endpoint['value_head']}-{endpoint['seed']}-policy-e{epoch:04d}-pba-selected-recovery",
                "task_type": "policy_eval", "domain": "mprime",
                "value_head": endpoint["value_head"], "seed": endpoint["seed"],
                "stage": "stage2", "status": "ready", "teacher": "hmrp-ha-gbfs",
                "source_checkpoint_ref": endpoint["source_checkpoint_ref"],
                "source_checkpoint_sha256": endpoint["source_checkpoint_sha256"],
                "source_training_job_id": endpoint["source_training_job_id"],
                "snapshot_epoch": str(epoch),
                "analysis_roles": "mprime_final_validation_stage2_phase_b_a_selected_policy",
                "training_state": "COMPLETED", "training_log": endpoint["training_log"],
            })
        args.campaign.mkdir(parents=True, exist_ok=True)
        write(recovery_manifest, recovery_rows)
        if recovery_ledger.exists():
            names = ", ".join(f"{r['value_head']}/{r['seed']}/e{r['snapshot_epoch']}" for r in recovery_rows)
            raise RuntimeError("selected policy recovery completed but evidence is still absent: " + names)
        environment = dict(**__import__("os").environ)
        environment["STAGE2_POLICY_SUBMITTER"] = str(args.policy_submitter)
        subprocess.run([
            sys.executable, str(args.policy_submit_script),
            "--manifest", str(recovery_manifest), "--ledger", str(recovery_ledger),
            "--suffix-prefix", "MPFINALV_S2SEL", "--output-prefix", recovery_prefix,
            "--queue-cap", "1999", "--max-per-cycle", str(len(recovery_rows)),
            "--max-active", str(len(recovery_rows)), "--one-cycle",
        ], env=environment, check=True)
        submitted = read(recovery_ledger, "\t")
        job_ids = [row["slurm_job_id"] for row in submitted]
        if len(job_ids) != len(recovery_rows):
            raise RuntimeError("selected-policy recovery ledger is incomplete")
        continuation = subprocess.run([
            "sbatch", "--parsable", "--dependency=afterany:" + ":".join(job_ids),
            "--job-name=MPRIME_S2_DOWNSTREAM_R", "--cpus-per-task=1", "--mem=2G",
            "--time=00:30:00", f"--output={args.campaign}/downstream_recovery_%j.log",
            "--wrap=" + shlex.join([
                sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                "--policy-root", str(recovery_root),
            ]),
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        print(f"selected_policy_recovery={len(job_ids)} continuation={continuation.stdout.strip()}")
        return
    if len(rows) != 40:
        raise RuntimeError("failed to build forty final search identities")
    args.campaign.mkdir(parents=True, exist_ok=True)
    (args.campaign / "slurm").mkdir(parents=True, exist_ok=True)
    write(args.campaign / "manifest.csv", rows)
    write(args.campaign / "reuse_audit.csv", audit)
    shutil.copy2(args.runner_source, args.campaign / args.runner_source.name)
    missing = [row["array_index"] for row in rows if row["status"] == "ready_not_submitted"]
    submitted = []
    if missing:
        result = subprocess.run([
            "sbatch", "--parsable", "--array=" + ",".join(missing),
            f"--output={args.campaign}/slurm/%A_%a.out",
            f"--export=ALL,CAMPAIGN={args.campaign},MANIFEST={args.campaign / 'manifest.csv'},CODE_COMMIT={args.code_commit}",
            str(args.sbatch),
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        job = result.stdout.strip().splitlines()[-1].split(";", 1)[0]
        now = datetime.now(timezone.utc).isoformat()
        submitted = [{"array_index": index, "slurm_job_id": f"{job}_{index}", "submitted_at": now} for index in missing]
        write(args.campaign / "submissions.tsv", submitted, "\t")
        print(f"submitted={job} tasks={len(missing)} reused={40-len(missing)}")
    else:
        print("submitted=none tasks=0 reused=40")


if __name__ == "__main__":
    main()
