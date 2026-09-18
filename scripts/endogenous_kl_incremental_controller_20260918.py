#!/usr/bin/env python3
"""Incrementally release endogenous-KL policy curves and selected MCTS.

The controller is deliberately fail-closed:

* a checkpoint identity is append-only and includes its weights SHA-256;
* an existing identity whose provenance changes aborts the controller;
* a valid result, an active Slurm attempt, or a successful attempt is never
  submitted again;
* validation-best selection occurs only after all 168 curve results exist;
* fixed20/70 and PW70 are materialized from the frozen selected endpoints.

Without ``--submit`` this is a read-only/dry-run planner.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


EPOCHS = (0, *range(5, 100, 5), 99)
ACTIVE_STATES = {
    "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
    "RESIZING", "REQUEUED", "REQUEUE_FED", "SIGNALING", "STAGE_OUT",
}
POLICY_FIELDS = (
    "identity", "curve_index", "arm_index", "domain", "seed", "semantics",
    "epoch", "validation_score", "total", "evaluation_module", "teacher",
    "mcts_module",
    "checkpoint", "checkpoint_sha256", "source_checkpoint",
    "source_checkpoint_sha256", "source_training_job_id", "code_commit",
    "discovered_at",
)
SUBMISSION_FIELDS = (
    "kind", "identity", "array_job_id", "array_task_id", "slurm_job_id",
    "checkpoint_sha256", "submitted_at",
)
SELECTED_FIELDS = (
    "arm_index", "domain", "seed", "semantics", "selected_epoch",
    "selected_validation_score", "checkpoint", "checkpoint_sha256",
    "policy_score", "policy_result", "selection_rule", "frozen_at",
)
MCTS_FIELDS = (
    "array_index", "identity", "arm_index", "domain", "seed", "semantics",
    "method", "checkpoint", "checkpoint_sha256", "selected_epoch",
    "selected_validation_score", "policy_score", "evaluation_module", "teacher",
    "width", "iterations", "puct", "estimator", "pw_min_width", "pw_c",
    "pw_alpha", "workers", "instance_timeout_seconds", "max_external_actions",
    "expected_instances", "code_commit",
)


def read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def atomic_write_csv(
    path: Path, rows: list[dict[str, object]], fields: tuple[str, ...], *, delimiter: str = ","
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...], *, delimiter: str = "\t") -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base_manifest(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    if len(rows) != 8:
        raise RuntimeError(f"expected 8 pilot arms, found {len(rows)}")
    if sorted(int(row["arm_index"]) for row in rows) != list(range(8)):
        raise RuntimeError("arm_index must be exactly 0..7")
    identities = {(row["domain"], row["seed"], row["semantics"]) for row in rows}
    if len(identities) != 8:
        raise RuntimeError("pilot arm identities are not unique")
    for row in rows:
        if row["semantics"] not in {"legacy_dropout_current", "deterministic_current"}:
            raise RuntimeError(f"invalid semantics: {row['semantics']}")
        weights = Path(row["source_checkpoint"]) / "weights.joblib"
        if weights.exists() and sha256(weights) != row["source_checkpoint_sha256"]:
            raise RuntimeError(f"source checkpoint hash mismatch: {weights}")
    return rows


def snapshot_root(campaign: Path, arm: dict[str, str]) -> Path | None:
    output = campaign / "outputs" / (
        f"arm_{arm['arm_index']}_{arm['domain']}_{arm['seed']}_{arm['semantics']}"
    )
    root_file = output / "snapshot_root.txt"
    if root_file.exists():
        value = root_file.read_text(encoding="utf-8").strip()
        return Path(value) if value else None
    log = output / "training.stdout"
    if not log.exists():
        return None
    matches = re.findall(r"^Snapshot directory:\s*(.+?)\s*$", log.read_text(errors="replace"), re.MULTILINE)
    return Path(matches[-1]) if matches else None


def discover_policy_rows(
    campaign: Path,
    arms: list[dict[str, str]],
    existing: list[dict[str, str]],
    *,
    code_commit: str,
    min_age_seconds: int,
    now: dt.datetime,
) -> list[dict[str, object]]:
    by_identity = {row["identity"]: row for row in existing}
    discovered = list(existing)
    for arm in arms:
        root = snapshot_root(campaign, arm)
        if root is None or not root.is_dir():
            continue
        training_log = campaign / "outputs" / (
            f"arm_{arm['arm_index']}_{arm['domain']}_{arm['seed']}_{arm['semantics']}"
        ) / "training.stdout"
        if not training_log.exists():
            continue
        validation_rates = [
            float(value) for value in re.findall(
                r"\[VALIDATION\] Current network validation success rate:\s*([0-9.]+)",
                training_log.read_text(errors="replace"),
            )
        ]
        for position, epoch in enumerate(EPOCHS):
            matches = sorted(root.glob(f"snapshot_{epoch}_*"))
            if not matches:
                continue
            if len(matches) != 1:
                raise RuntimeError(f"arm {arm['arm_index']} epoch {epoch}: {len(matches)} snapshots")
            checkpoint = matches[0]
            weights = checkpoint / "weights.joblib"
            if not weights.is_file():
                continue
            if time.time() - weights.stat().st_mtime < min_age_seconds:
                continue
            # Stage-2 snapshot suffixes contain training/replay coverage, not
            # validation coverage.  VALIDATE_EVERY is frozen to one in this
            # training build, so the Nth validation record is epoch N.
            if epoch >= len(validation_rates):
                continue
            validation_score = validation_rates[epoch]
            curve_index = int(arm["arm_index"]) * len(EPOCHS) + position
            identity = f"arm{int(arm['arm_index']):02d}-epoch{epoch:04d}"
            row: dict[str, object] = {
                "identity": identity,
                "curve_index": curve_index,
                "arm_index": int(arm["arm_index"]),
                "domain": arm["domain"],
                "seed": arm["seed"],
                "semantics": arm["semantics"],
                "epoch": epoch,
                "validation_score": f"{validation_score:.12g}",
                "total": arm["total"],
                "evaluation_module": arm["evaluation_module"],
                "mcts_module": arm["training_module"],
                "teacher": arm["teacher"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(weights),
                "source_checkpoint": arm["source_checkpoint"],
                "source_checkpoint_sha256": arm["source_checkpoint_sha256"],
                "source_training_job_id": arm["source_training_job_id"],
                "code_commit": code_commit,
                "discovered_at": now.isoformat(),
            }
            old = by_identity.get(identity)
            if old is not None:
                immutable = set(POLICY_FIELDS) - {"discovered_at"}
                differences = [field for field in immutable if str(old[field]) != str(row[field])]
                if differences:
                    raise RuntimeError(f"checkpoint identity mutated: {identity}: {differences}")
                continue
            by_identity[identity] = {key: str(value) for key, value in row.items()}
            discovered.append(row)
    return sorted(discovered, key=lambda row: int(row["curve_index"]))


def valid_policy_result(campaign: Path, row: dict[str, str]) -> bool:
    output = campaign / "outputs" / (
        f"arm_{row['arm_index']}_{row['domain']}_{row['seed']}_{row['semantics']}"
    ) / f"policy_epoch_{int(row['epoch']):04d}" / "result.json"
    if not output.exists():
        return False
    try:
        result = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required = {
        "identity": row["identity"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "domain": row["domain"],
        "seed": int(row["seed"]),
        "semantics": row["semantics"],
        "epoch": int(row["epoch"]),
        "total": int(row["total"]),
    }
    return all(result.get(key) == value for key, value in required.items()) and int(result.get("seen", -1)) == int(row["total"])


def slurm_state(job_id: str) -> str:
    queue = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    states = [line.strip().upper() for line in queue.stdout.splitlines() if line.strip()]
    if states:
        return states[-1]
    accounting = subprocess.run(
        ["sacct", "-n", "-X", "-j", job_id, "--format=State"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    states = [line.strip().split()[0].split("+", 1)[0].upper() for line in accounting.stdout.splitlines() if line.strip()]
    return states[-1] if states else "UNKNOWN"


def identities_blocked_by_ledger(ledger: list[dict[str, str]]) -> set[str]:
    blocked: set[str] = set()
    for identity in {row["identity"] for row in ledger}:
        attempts = [row for row in ledger if row["identity"] == identity]
        state = slurm_state(attempts[-1]["slurm_job_id"])
        if state in ACTIVE_STATES:
            blocked.add(identity)
    return blocked


def completed_without_result(
    ledger: list[dict[str, str]], valid_identities: set[str]
) -> list[str]:
    """Return successful Slurm attempts whose scientific result is absent.

    Automatically repeating such a task risks duplicating a late filesystem
    write or concealing a wrapper bug.  The watcher therefore stops and asks
    for exact recovery instead of broadly resubmitting it.
    """
    broken = []
    for identity in {row["identity"] for row in ledger} - valid_identities:
        attempts = [row for row in ledger if row["identity"] == identity]
        if slurm_state(attempts[-1]["slurm_job_id"]) == "COMPLETED":
            broken.append(identity)
    return sorted(broken)


def submit_array(
    *,
    indices: list[int],
    script: Path,
    export: dict[str, str],
    dependency: str | None = None,
) -> str:
    command = ["sbatch", "--parsable", "--array=" + ",".join(map(str, indices))]
    if dependency:
        command.append("--dependency=" + dependency)
    command += ["--export=" + ",".join(["ALL", *(f"{key}={value}" for key, value in export.items())]), str(script)]
    output = subprocess.check_output(command, text=True).strip()
    return output.split(";", 1)[0]


def collect_policy_results(campaign: Path, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    results = []
    for row in rows:
        if not valid_policy_result(campaign, row):
            continue
        path = campaign / "outputs" / (
            f"arm_{row['arm_index']}_{row['domain']}_{row['seed']}_{row['semantics']}"
        ) / f"policy_epoch_{int(row['epoch']):04d}" / "result.json"
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def freeze_selected_and_mcts(
    campaign: Path,
    policy_rows: list[dict[str, str]],
    results: list[dict[str, object]],
    *,
    code_commit: str,
    now: dt.datetime,
) -> list[dict[str, object]]:
    if len(policy_rows) != 168:
        raise RuntimeError("selection requires exactly 168 frozen checkpoint rows")
    result_by_identity = {str(row["identity"]): row for row in results}
    selected: list[dict[str, object]] = []
    for arm_index in range(8):
        candidates = [row for row in policy_rows if int(row["arm_index"]) == arm_index]
        if len(candidates) != 21:
            raise RuntimeError(f"arm {arm_index}: expected 21 policy candidates")
        winner = min(candidates, key=lambda row: (-float(row["validation_score"]), int(row["epoch"])))
        policy = result_by_identity.get(winner["identity"])
        selected.append({
            "arm_index": arm_index,
            "domain": winner["domain"],
            "seed": winner["seed"],
            "semantics": winner["semantics"],
            "selected_epoch": winner["epoch"],
            "selected_validation_score": winner["validation_score"],
            "checkpoint": winner["checkpoint"],
            "checkpoint_sha256": winner["checkpoint_sha256"],
            "policy_score": "" if policy is None else policy["score"],
            "policy_result": "" if policy is None else policy["evaluation_log"],
            "selection_rule": "maximum_training_validation_score_then_earliest_epoch",
            "frozen_at": now.isoformat(),
        })
    selected_path = campaign / "selected_endpoints.csv"
    old_selected = read_csv(selected_path)
    if old_selected:
        for old, new in zip(old_selected, selected, strict=True):
            for field in set(SELECTED_FIELDS) - {"frozen_at", "policy_score", "policy_result"}:
                if str(old[field]) != str(new[field]):
                    raise RuntimeError(f"selected endpoint changed: arm {new['arm_index']} field {field}")
            if old["policy_score"] and str(old["policy_score"]) != str(new["policy_score"]):
                raise RuntimeError(f"selected endpoint policy score changed: arm {new['arm_index']}")
        if any(not row["policy_score"] for row in old_selected) and all(row["policy_score"] for row in selected):
            atomic_write_csv(selected_path, selected, SELECTED_FIELDS)
    else:
        atomic_write_csv(selected_path, selected, SELECTED_FIELDS)

    selected_epoch = {int(row["arm_index"]): int(row["selected_epoch"]) for row in selected}
    base = {
        int(row["arm_index"]): row for row in policy_rows
        if int(row["epoch"]) == selected_epoch[int(row["arm_index"])]
    }
    if any(not endpoint["policy_score"] for endpoint in selected):
        return []
    mcts: list[dict[str, object]] = []
    for endpoint in selected:
        source = base[int(endpoint["arm_index"])]
        for method in ("fixed20x70", "pw70"):
            mcts.append({
                "array_index": len(mcts),
                "identity": f"arm{int(endpoint['arm_index']):02d}-{method}",
                **{key: endpoint[key] for key in (
                    "arm_index", "domain", "seed", "semantics", "checkpoint",
                    "checkpoint_sha256", "selected_epoch", "selected_validation_score", "policy_score",
                )},
                "method": method,
                "evaluation_module": source["mcts_module"],
                "teacher": source["teacher"],
                "width": 20,
                "iterations": 70,
                "puct": 0.1,
                "estimator": 0.5,
                "pw_min_width": 3 if method == "pw70" else "",
                "pw_c": 0.6 if method == "pw70" else "",
                "pw_alpha": 0.5 if method == "pw70" else "",
                "workers": 3,
                "instance_timeout_seconds": 21600,
                "max_external_actions": 10000,
                "expected_instances": 20,
                "code_commit": code_commit,
            })
    mcts_path = campaign / "selected_mcts_manifest.csv"
    old_mcts = read_csv(mcts_path)
    if old_mcts:
        for old, new in zip(old_mcts, mcts, strict=True):
            for field in MCTS_FIELDS:
                if str(old[field]) != str(new[field]):
                    raise RuntimeError(f"MCTS identity changed: {new['identity']} field {field}")
    else:
        atomic_write_csv(mcts_path, mcts, MCTS_FIELDS)
    return mcts


def write_curve_aggregates(campaign: Path, results: list[dict[str, object]]) -> None:
    if len(results) != 168:
        return
    aggregate = sorted(
        results,
        key=lambda row: (row["domain"], int(row["seed"]), row["semantics"], int(row["epoch"])),
    )
    result_fields = (
        "identity", "arm_index", "domain", "seed", "semantics", "epoch",
        "validation_score", "score", "seen", "total", "source_checkpoint",
        "source_checkpoint_sha256", "checkpoint", "checkpoint_sha256",
        "code_commit", "evaluation_log",
    )
    atomic_write_csv(campaign / "learning_curve_results.csv", aggregate, result_fields)
    endpoints = [row for row in aggregate if int(row["epoch"]) == 99]
    atomic_write_csv(campaign / "endpoint99_results.csv", endpoints, result_fields)
    summary: list[dict[str, object]] = []
    for domain in sorted({str(row["domain"]) for row in endpoints}):
        subset = [row for row in endpoints if row["domain"] == domain]
        means = {
            semantics: sum(int(row["score"]) for row in subset if row["semantics"] == semantics) / 2
            for semantics in ("legacy_dropout_current", "deterministic_current")
        }
        summary.append({
            "domain": domain, "n_pairs": 2,
            "legacy_mean": means["legacy_dropout_current"],
            "deterministic_mean": means["deterministic_current"],
            "det_minus_legacy": means["deterministic_current"] - means["legacy_dropout_current"],
            "interpretation": "two-seed endogenous pilot; no population claim",
        })
    atomic_write_csv(
        campaign / "domain_summary.csv", summary,
        ("domain", "n_pairs", "legacy_mean", "deterministic_mean", "det_minus_legacy", "interpretation"),
    )


def valid_mcts_output(campaign: Path, row: dict[str, object]) -> bool:
    return len(mcts_terminal_records(campaign, row)) == int(row["expected_instances"])


def mcts_terminal_records(campaign: Path, row: dict[str, object]) -> dict[int, str]:
    """Join durable JSONL with exact declared-timeout attempt evidence."""
    completion = (
        campaign / "selected_mcts" / str(row["method"])
        / f"arm_{row['arm_index']}" / "completion.jsonl"
    )
    terminal: dict[int, str] = {}
    if completion.exists():
        for line in completion.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            number = int(record["instance_number"])
            classification = "success" if bool(record.get("hit_goal")) else (
                "action_limit" if int(record.get("steps", -1)) >= int(row["max_external_actions"])
                else "finished_unsolved"
            )
            old = terminal.setdefault(number, classification)
            if old != classification:
                raise RuntimeError(f"conflicting MCTS evidence for {row['identity']} instance {number}")
    output = campaign / "selected_mcts" / str(row["method"]) / f"arm_{row['arm_index']}"
    timeout_pattern = re.compile(
        r"\[EVAL INSTANCE\] timeout number=(\d+) path=\S+ limit=([0-9.]+)s"
    )
    for attempt in output.glob("attempt_*.txt"):
        for number_text, limit_text in timeout_pattern.findall(attempt.read_text(errors="replace")):
            if abs(float(limit_text) - float(row["instance_timeout_seconds"])) > 0.5:
                continue
            number = int(number_text)
            old = terminal.setdefault(number, "six_hour_timeout")
            if old != "six_hour_timeout":
                raise RuntimeError(f"conflicting timeout evidence for {row['identity']} instance {number}")
    return terminal


def write_mcts_summary(campaign: Path, rows: list[dict[str, object]]) -> None:
    if not rows or not all(valid_mcts_output(campaign, row) for row in rows):
        return
    output = []
    for row in rows:
        terminal = mcts_terminal_records(campaign, row)
        output.append({
            "identity": row["identity"], "arm_index": row["arm_index"],
            "domain": row["domain"], "seed": row["seed"],
            "semantics": row["semantics"], "method": row["method"],
            "policy_score": row["policy_score"],
            "successes_6h": sum(value == "success" for value in terminal.values()),
            "timeouts_6h": sum(value == "six_hour_timeout" for value in terminal.values()),
            "action_limits_6h": sum(value == "action_limit" for value in terminal.values()),
            "other_unsolved_6h": sum(value == "finished_unsolved" for value in terminal.values()),
            "classified": len(terminal), "checkpoint_sha256": row["checkpoint_sha256"],
        })
    atomic_write_csv(
        campaign / "selected_mcts_summary_6h.csv", output,
        (
            "identity", "arm_index", "domain", "seed", "semantics", "method",
            "policy_score", "successes_6h", "timeouts_6h", "action_limits_6h",
            "other_unsolved_6h", "classified", "checkpoint_sha256",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--policy-sbatch", type=Path, required=True)
    parser.add_argument("--mcts-sbatch", type=Path, required=True)
    parser.add_argument("--controller-sbatch", type=Path)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--training-array-job-id")
    parser.add_argument("--snapshot-min-age-seconds", type=int, default=120)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    args.base_manifest = args.base_manifest or args.campaign / "manifest.csv"
    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != args.code_commit:
        raise RuntimeError(f"checkout {actual_commit} != declared {args.code_commit}")
    arms = load_base_manifest(args.base_manifest)
    now = dt.datetime.now(dt.timezone.utc)
    policy_manifest = args.campaign / "policy_eval_manifest.csv"
    policy_rows = discover_policy_rows(
        args.campaign, arms, read_csv(policy_manifest), code_commit=args.code_commit,
        min_age_seconds=args.snapshot_min_age_seconds, now=now,
    )
    atomic_write_csv(policy_manifest, policy_rows, POLICY_FIELDS)
    policy_ledger_path = args.campaign / "policy_eval_submissions.tsv"
    policy_ledger = read_csv(policy_ledger_path, delimiter="\t")
    valid_policy_identities = {
        row["identity"] for row in policy_rows if valid_policy_result(args.campaign, row)
    }
    blocked = identities_blocked_by_ledger(policy_ledger) if policy_ledger else set()
    ready = [
        row for row in policy_rows
        if not valid_policy_result(args.campaign, row) and row["identity"] not in blocked
    ]
    print(json.dumps({
        "discovered": len(policy_rows),
        "policy_complete": sum(valid_policy_result(args.campaign, row) for row in policy_rows),
        "policy_ready_to_submit": len(ready),
    }, sort_keys=True))
    results = collect_policy_results(args.campaign, policy_rows)
    training_done = all(
        (args.campaign / "outputs" / f"arm_{row['arm_index']}_{row['domain']}_{row['seed']}_{row['semantics']}" / "training_complete.json").exists()
        for row in arms
    )
    selected_identities: set[str] = set()
    if training_done and len(policy_rows) == 168:
        for arm_index in range(8):
            candidates = [row for row in policy_rows if int(row["arm_index"]) == arm_index]
            winner = min(candidates, key=lambda row: (-float(row["validation_score"]), int(row["epoch"])))
            selected_identities.add(winner["identity"])
    if ready and args.submit:
        # Selected endpoints receive their own array submission first.  This
        # lets their MCTS start without waiting behind non-selected curves.
        groups = [
            [row for row in ready if row["identity"] in selected_identities],
            [row for row in ready if row["identity"] not in selected_identities],
        ]
        for group in groups:
            if not group:
                continue
            indices = [int(row["curve_index"]) for row in group]
            job = submit_array(
                indices=indices, script=args.policy_sbatch,
                export={
                    "CHECKOUT": str(args.checkout), "CODE_COMMIT": args.code_commit,
                    "CAMPAIGN_ROOT": str(args.campaign), "POLICY_MANIFEST": str(policy_manifest),
                },
            )
            append_csv(policy_ledger_path, [{
                "kind": "policy_selected" if row["identity"] in selected_identities else "policy_curve",
                "identity": row["identity"], "array_job_id": job,
                "array_task_id": row["curve_index"], "slurm_job_id": f"{job}_{row['curve_index']}",
                "checkpoint_sha256": row["checkpoint_sha256"], "submitted_at": now.isoformat(),
            } for row in group], SUBMISSION_FIELDS)

    write_curve_aggregates(args.campaign, results)
    if not (training_done and len(policy_rows) == 168):
        return 0
    mcts_rows = freeze_selected_and_mcts(
        args.campaign, policy_rows, results, code_commit=args.code_commit, now=now
    )
    if not mcts_rows:
        print(json.dumps({"selected_endpoints": 8, "selected_policy_results": sum(
            valid_policy_result(args.campaign, row)
            for row in policy_rows
            if any(
                int(selected["arm_index"]) == int(row["arm_index"])
                and int(selected["selected_epoch"]) == int(row["epoch"])
                for selected in read_csv(args.campaign / "selected_endpoints.csv")
            )
        )}, sort_keys=True))
        return 0
    write_mcts_summary(args.campaign, mcts_rows)
    mcts_ledger_path = args.campaign / "selected_mcts_submissions.tsv"
    mcts_ledger = read_csv(mcts_ledger_path, delimiter="\t")
    valid_mcts_identities = {
        str(row["identity"]) for row in mcts_rows if valid_mcts_output(args.campaign, row)
    }
    mcts_blocked = identities_blocked_by_ledger(mcts_ledger) if mcts_ledger else set()
    missing_mcts = [
        row for row in mcts_rows
        if row["identity"] not in mcts_blocked and row["identity"] not in valid_mcts_identities
    ]
    print(json.dumps({"selected_endpoints": 8, "mcts_ready_to_submit": len(missing_mcts)}, sort_keys=True))
    if missing_mcts and args.submit:
        indices = [int(row["array_index"]) for row in missing_mcts]
        job = submit_array(
            indices=indices, script=args.mcts_sbatch,
            export={
                "CHECKOUT": str(args.checkout), "CODE_COMMIT": args.code_commit,
                "CAMPAIGN_ROOT": str(args.campaign),
                "MCTS_MANIFEST": str(args.campaign / "selected_mcts_manifest.csv"),
            },
        )
        append_csv(mcts_ledger_path, [{
            "kind": "mcts", "identity": row["identity"], "array_job_id": job,
            "array_task_id": row["array_index"], "slurm_job_id": f"{job}_{row['array_index']}",
            "checkpoint_sha256": row["checkpoint_sha256"], "submitted_at": now.isoformat(),
        } for row in missing_mcts], SUBMISSION_FIELDS)
        if args.controller_sbatch is not None:
            command = [
                "sbatch", "--parsable", f"--dependency=afterany:{job}",
                "--export=" + ",".join([
                    "ALL", f"CHECKOUT={args.checkout}", f"CODE_COMMIT={args.code_commit}",
                    f"CAMPAIGN_ROOT={args.campaign}",
                    f"SCRIPT_ROOT={args.controller_sbatch.parent}", "ONE_CYCLE=1",
                ]), str(args.controller_sbatch),
            ]
            followup = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
            append_csv(
                args.campaign / "controller_followups.tsv",
                [{
                    "kind": "mcts_followup", "identity": "all", "array_job_id": followup,
                    "array_task_id": "", "slurm_job_id": followup,
                    "checkpoint_sha256": "manifest_frozen", "submitted_at": now.isoformat(),
                }], SUBMISSION_FIELDS,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
