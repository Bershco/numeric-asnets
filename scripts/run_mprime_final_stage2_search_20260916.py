#!/usr/bin/env python3
"""Run one exact fixed/PW70 search on a final MPrime Stage-2 endpoint."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path


def load_row(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 40:
        raise ValueError(f"manifest must contain exactly 40 rows, found {len(rows)}")
    matches = [row for row in rows if row["array_index"] == str(index)]
    if len(matches) != 1:
        raise ValueError(f"array index {index}: expected one row, found {len(matches)}")
    row = matches[0]
    expected = {
        "experiment_id": "MPRIME-FINAL-S2-SEARCH",
        "branch": "validation_led", "stage": "stage2",
        "checkpoint_selection": "phase_b_replicate_a",
        "domain_module": "experiments_numeric.domain.mprime",
        "architecture_module": "experiments_numeric.architecture_2.mprime_mcts",
        "teacher": "hmrp-ha-gbfs", "width": "20", "iterations": "70",
        "puct": "0.1", "estimator": "0.5", "workers": "3", "cpus": "6",
        "memory": "120G", "walltime": "3-00:00:00",
        "instance_timeout_seconds": "21600", "max_external_actions": "10000",
        "evaluation_scheduling": "rolling", "expected_test_instances": "20",
        "status": "ready_not_submitted",
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"{row.get('manifest_id')}: {field}={row.get(field)!r}, expected {value!r}")
    if row["search_method"] not in {"fixed", "pw70"}:
        raise ValueError("invalid search method")
    if row["value_head"] not in {"off", "on"}:
        raise ValueError("invalid value-head mode")
    epoch = int(row["selected_epoch"])
    if epoch not in set(range(0, 100, 5)) | {99}:
        raise ValueError(f"invalid selected epoch: {epoch}")
    if not row.get("selected_validation_score"):
        raise ValueError("missing Phase-B-A validation score")
    return row


def append_attempt(path: Path, fields: dict[str, object]) -> None:
    columns = [
        "manifest_id", "slurm_job_id", "restart_count", "started_at", "ended_at",
        "evaluation_returncode", "validation_returncode", "log_path",
        "completion_record_path", "val_summary_path",
    ]
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(fields)
        stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--only-instance-number", type=int,
        help=(
            "Run exactly one stable ordered test-set position (1-20). "
            "Used only by exact recovery jobs; the primary campaign omits it."
        ),
    )
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()

    if args.only_instance_number is not None and not 1 <= args.only_instance_number <= 20:
        parser.error("--only-instance-number must be between 1 and 20")

    row = load_row(args.manifest, args.index)
    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != args.code_commit:
        raise RuntimeError(f"checkout commit {actual_commit} != declared {args.code_commit}")
    if not Path(row["checkpoint"]).is_dir():
        raise RuntimeError(f"checkpoint missing: {row['checkpoint']}")
    weights = Path(row["checkpoint"]) / "weights.joblib"
    if not weights.is_file():
        raise RuntimeError(f"checkpoint weights missing: {weights}")
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != row["checkpoint_sha256"]:
        raise RuntimeError(
            f"checkpoint hash {digest.hexdigest()} != declared {row['checkpoint_sha256']}"
        )
    for required in (args.container, args.validator, args.production / "venv-asnets/bin/activate"):
        if not required.exists():
            raise RuntimeError(f"required runtime artifact missing: {required}")

    output = args.output_root / row["search_method"] / row["value_head"] / row["seed"]
    completion_dir, attempts_dir, validation_dir = (
        output / "completion", output / "attempts", output / "validation"
    )
    for directory in (completion_dir, attempts_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    identity = row["manifest_id"]
    completion = completion_dir / f"{identity}.jsonl"
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")
    attempt = f"{job_id}_r{restart}"
    log_path = attempts_dir / f"{attempt}.txt"
    val_summary = validation_dir / f"{attempt}.val.csv"

    effective_workers = "1" if args.only_instance_number is not None else row["workers"]
    experiment_args = [
        "./run_experiment", row["architecture_module"], row["domain_module"],
        "--resume-from", row["checkpoint"], "--eval-with-mcts",
        "--mcts-expansion-size", row["width"], "--mcts-iterations", row["iterations"],
        "--mcts-exploration-weight", row["puct"], "--use-estimator", row["estimator"],
        "--eval-scheduling", row["evaluation_scheduling"],
        "--eval-completion-file", str(completion),
        "--eval-instance-timeout", row["instance_timeout_seconds"],
        "--eval-max-actions", row["max_external_actions"],
        "--num-workers", effective_workers, "--jpddl-max-heap", "4g", "--worker-logs",
        "--random-seed", row["seed"],
    ]
    if args.only_instance_number is not None:
        skipped = [
            str(number) for number in range(1, 21)
            if number != args.only_instance_number
        ]
        experiment_args += ["--skip-instance-numbers", ",".join(skipped)]
    if row["search_method"] == "pw70":
        experiment_args += [
            "--mcts-progressive-widening", "--mcts-pw-min-width", row["pw_min_width"],
            "--mcts-pw-c", row["pw_c"], "--mcts-pw-alpha", row["pw_alpha"],
        ]
    if row["value_head"] == "off":
        experiment_args.append("--disable-value-head")
    inner = (
        "set -euo pipefail; "
        f"cd {shlex.quote(str(args.checkout / 'asnets'))}; "
        f"source {shlex.quote(str(args.production / 'venv-asnets/bin/activate'))}; "
        f"export PYTHONPATH={shlex.quote(str(args.checkout))}:{shlex.quote(str(args.checkout / 'asnets'))}:${{PYTHONPATH:-}}; "
        f"export ENHSP_CONFIG_OVERRIDE={shlex.quote(row['teacher'])}; "
        + shlex.join(experiment_args)
    )
    command = [
        "apptainer", "exec", "--bind", "/home/hersco:/home/hersco",
        "--bind", "/home/hersco/apptainer_fake_passwd:/etc/passwd",
        str(args.container), "/bin/bash", "-lc", inner,
    ]
    if args.print_command:
        print(shlex.join(command)); return 0

    print(
        f"[MPRIME FINAL S2 SEARCH] identity={identity} method={row['search_method']} "
        f"vh={row['value_head']} seed={row['seed']} epoch={row['selected_epoch']}"
    )
    if args.only_instance_number is not None:
        print(
            "[MPRIME FINAL S2 SEARCH] exact-recovery "
            f"instance_number={args.only_instance_number} workers=1"
        )
    print("[MPRIME FINAL S2 SEARCH] width=20 simulations=70 puct=.1 estimator=.5")
    print(f"[MPRIME FINAL S2 SEARCH] checkpoint={row['checkpoint']} commit={actual_commit}")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
        evaluation_rc = process.wait()
    validation_rc = subprocess.run([
        sys.executable, str(args.checkout / "asnets/tools/validate_eval_log_with_summary.py"),
        "--log", str(log_path), "--domain", "mprime", "--validator", str(args.validator),
        "--allow-incomplete", "--summary-csv", str(val_summary),
    ]).returncode
    if validation_rc == 0 and not val_summary.exists():
        with val_summary.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream, lineterminator="\n").writerow(
                ["instance", "candidate_plans", "selected_steps", "val_valid", "reason"]
            )
    append_attempt(output / "attempts.tsv", {
        "manifest_id": identity, "slurm_job_id": job_id, "restart_count": restart,
        "started_at": started, "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_returncode": evaluation_rc, "validation_returncode": validation_rc,
        "log_path": log_path, "completion_record_path": completion,
        "val_summary_path": val_summary,
    })
    return validation_rc or evaluation_rc


if __name__ == "__main__":
    raise SystemExit(main())
