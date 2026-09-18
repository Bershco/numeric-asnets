#!/usr/bin/env python3
"""Reconcile Stage-1 artifacts and freeze only genuinely unresolved runs.

Read-only is the default.  ``--write`` materializes result JSON for exact
hard-timeout or already-completed evidence and writes a filtered recovery
manifest; it never deletes or overwrites an existing result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_mcts_policy_divergence_stage1_20260918 import (
    EXPECTED_MANIFEST_SHA256,
    RESTRICTED_EVALUATOR_NUMBER,
    atomic_json,
    divergence_records,
    load_frozen_tasks,
    normalize_record,
    sha256_file,
    terminal_outcome,
)


RESULT_SCHEMAS = {
    "mcts-divergence-stage1-candidate-result-v1",
    "mcts-divergence-stage1-candidate-result-v2",
}


def candidate_stem(candidate_index: int, candidate: dict[str, Any]) -> str:
    return (
        f"{candidate_index:02d}_{candidate['checkpoint_identity']}_"
        f"s{candidate['seed']}_{Path(candidate['instance']).stem}"
    )


def validate_existing_result(
    result: dict[str, Any], *, candidate: dict[str, Any],
) -> None:
    if result.get("schema_version") not in RESULT_SCHEMAS:
        raise RuntimeError("unsupported candidate-result schema")
    for key in ("checkpoint_identity", "instance"):
        if result.get(key) != candidate[key]:
            raise RuntimeError(f"candidate result {key} mismatch")
    divergence = result.get("first_divergence")
    if divergence is not None:
        normalize_record(
            divergence, instance_name=candidate["instance"], allow_legacy=True)
    completion = result.get("completion_record")
    outcome = result.get("outcome")
    if completion is None and not (
            isinstance(outcome, dict)
            and outcome.get("classification") == "hard_timeout"):
        raise RuntimeError("candidate result has no terminal outcome evidence")


def reconcile(
    *, repo: Path, manifest: Path, freeze: Path, artifact_root: Path,
    write: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = load_frozen_tasks(manifest, freeze)
    statuses = []
    recovery_tasks = []
    preserved = materialized = timeout_count = legacy_normalized = 0
    for task_index, row in enumerate(rows):
        unresolved = []
        reasons = []
        candidates = json.loads(row["candidate_runs_json"])
        task_dir = artifact_root / row["task_id"]
        for candidate_index, candidate in enumerate(candidates):
            stem = candidate_stem(candidate_index, candidate)
            log = task_dir / f"{stem}.txt"
            completion = task_dir / f"{stem}.completed.jsonl"
            result_path = task_dir / f"{stem}.result.json"
            base_status = {
                "source_task_index": task_index,
                "source_task_id": row["task_id"],
                "candidate_index": candidate_index,
                "checkpoint_identity": candidate["checkpoint_identity"],
                "seed": int(candidate["seed"]),
                "instance": candidate["instance"],
                "log": str(log),
                "completion": str(completion),
                "result": str(result_path),
            }
            if result_path.exists():
                try:
                    validate_existing_result(
                        json.loads(result_path.read_text(encoding="utf-8")),
                        candidate=candidate,
                    )
                except Exception as error:
                    status = {**base_status, "status": "invalid_result",
                              "reason": str(error)}
                    unresolved.append(candidate)
                    reasons.append(status)
                else:
                    preserved += 1
                    status = {**base_status, "status": "preserved_valid_result"}
                statuses.append(status)
                continue
            if not log.exists():
                status = {**base_status, "status": "unresolved_not_run",
                          "reason": "no candidate log"}
                statuses.append(status)
                unresolved.append(candidate)
                reasons.append(status)
                continue
            try:
                records = divergence_records(log)
                if len(records) > 1:
                    raise RuntimeError("multiple first-divergence records")
                normalized = None
                if records:
                    old_schema = records[0].get("schema_version")
                    normalized = normalize_record(
                        records[0], instance_name=candidate["instance"],
                        allow_legacy=True)
                    if old_schema != normalized.get("schema_version"):
                        legacy_normalized += 1
                outcome = terminal_outcome(
                    log=log,
                    completion=completion,
                    instance_name=candidate["instance"],
                    evaluation_index=RESTRICTED_EVALUATOR_NUMBER,
                    max_actions=10000,
                )
                if outcome is None:
                    raise RuntimeError("no explicit terminal outcome evidence")
            except Exception as error:
                status = {**base_status, "status": "unresolved_invalid",
                          "reason": str(error)}
                statuses.append(status)
                unresolved.append(candidate)
                reasons.append(status)
                continue
            if outcome["classification"] == "hard_timeout":
                timeout_count += 1
            payload = {
                "schema_version": "mcts-divergence-stage1-candidate-result-v2",
                "source": "post_run_exact_reconciliation",
                "source_task_index": task_index,
                "task_id": row["task_id"],
                "candidate_index": candidate_index,
                "checkpoint_identity": candidate["checkpoint_identity"],
                "instance": candidate["instance"],
                "missing_stratum": candidate["missing_stratum"],
                "first_divergence_observed": normalized is not None,
                "first_divergence": normalized,
                "outcome": outcome,
                "completion_record": outcome["completion_record"],
                "log": str(log),
                "completion": str(completion),
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "reconciled_without_rerun": True,
            }
            if write:
                atomic_json(result_path, payload)
            materialized += 1
            statuses.append({
                **base_status,
                "status": "materialized_terminal_result" if write
                          else "reconcilable_terminal_result",
                "classification": outcome["classification"],
                "legacy_record_normalized": bool(
                    normalized and normalized.get("normalized_from_schema")),
            })
        if unresolved:
            recovery_tasks.append({
                "recovery_index": len(recovery_tasks),
                "source_task_index": task_index,
                "source_task_id": row["task_id"],
                "release_class": row["release_class"],
                "candidate_runs": unresolved,
                "candidate_indices": [
                    int(reason["candidate_index"]) for reason in reasons],
                "candidate_count": len(unresolved),
                "reasons": reasons,
            })
    recovery_manifest = {
        "schema_version": "mcts-divergence-stage1-recovery-v1",
        "source_manifest": str(manifest),
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "artifact_root": str(artifact_root),
        "task_count": len(recovery_tasks),
        "candidate_count": sum(task["candidate_count"] for task in recovery_tasks),
        "tasks": recovery_tasks,
    }
    report = {
        "schema_version": "mcts-divergence-stage1-reconciliation-v1",
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "artifact_root": str(artifact_root),
        "preserved_valid_results": preserved,
        "materialized_terminal_results": materialized,
        "explicit_hard_timeouts": timeout_count,
        "legacy_records_normalized": legacy_normalized,
        "recovery_tasks": len(recovery_tasks),
        "recovery_candidates": recovery_manifest["candidate_count"],
        "candidates": statuses,
    }
    return report, recovery_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--recovery-manifest", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write and (args.report is None or args.recovery_manifest is None):
        parser.error("--write requires --report and --recovery-manifest")
    report, recovery = reconcile(
        repo=args.repo,
        manifest=args.manifest,
        freeze=args.freeze,
        artifact_root=args.artifact_root,
        write=args.write,
    )
    if args.write:
        atomic_json(args.report, report)
        atomic_json(args.recovery_manifest, recovery)
        freeze_path = args.recovery_manifest.with_suffix(".freeze.json")
        atomic_json(freeze_path, {
            "schema_version": "mcts-divergence-stage1-recovery-freeze-v1",
            "recovery_manifest": str(args.recovery_manifest),
            "recovery_manifest_sha256": sha256_file(args.recovery_manifest),
            "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "submitted": False,
        })
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
