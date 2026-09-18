#!/usr/bin/env python3
"""Run one frozen Stage-1 divergence task or its known-root compute smoke.

This payload is intentionally opt-in: the recorder flag is added only by this
runner, and execution requires an explicit ``--execute``.  The default mode
only renders commands and performs immutable provenance checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


AUDIT_RELATIVE = Path("experiment_tracking/mcts_policy_divergence_cause_audit")
EXPECTED_MANIFEST_SHA256 = (
    "56204d595a70d4b92eabf88970612d36c69506af864c4b56506136f524744659"
)
EXPECTED_TASK_COUNTS = {"primary_fixed": 10, "optional_fo_pw": 2}
RECORDER_MARKER = "[MCTS FIRST DIVERGENCE] "
DETERMINISM_MARKER = "[MCTS DETERMINISM] "
RECORDER_SCHEMA = "mcts-first-divergence-v2"
LEGACY_RECORDER_SCHEMA = "mcts-first-divergence-v1"
GATE_SCHEMA = "mcts-divergence-stage1-compute-smoke-v1"
# Every campaign command restricts the evaluation to exactly one TEST_RUNS
# entry.  The rolling evaluator numbers that sole scheduled worker from one;
# this is deliberately distinct from the zero-based TEST_RUNS index supplied
# to --restrict-test-probs.
RESTRICTED_EVALUATOR_NUMBER = 1
IDENTITY_RE = re.compile(r"^src(?P<job>\d+)_e(?P<epoch>\d+)$")
POLICY_JOB_RE = re.compile(r"^(?P<job>\d+)_")
SNAPSHOT_RE = re.compile(r"(?:^|/)snapshot_(?P<epoch>\d+)(?:_|$)")
HARD_TIMEOUT_RE = re.compile(
    r"\[EVAL INSTANCE\] timeout number=(?P<number>\d+) "
    r"path=(?P<path>\S+) limit=(?P<limit>[0-9.]+)s"
)
COMPLETED_RE = re.compile(
    r"\[EVAL INSTANCE\] completed number=(?P<number>\d+) "
    r"path=(?P<path>\S+) status=(?P<status>success|unsolved) "
    r"elapsed=(?P<elapsed>[0-9.]+)s success=(?P<success>True|False|1\.0|0\.0) "
    r"steps=(?P<steps>-?\d+)"
)
CRASH_RE = re.compile(r"\[EVAL INSTANCE\] (?:crashed|died) number=(?P<number>\d+)")

DOMAIN_MODULES = {
    domain: (
        f"experiments_numeric.architecture_2.{domain}_mcts",
        f"experiments_numeric.domain.{domain}",
    )
    for domain in ("block_grouping", "counters", "drone", "fo_counters", "rover")
}

SMOKE = {
    "checkpoint_identity": "src20430427_e0000",
    "evaluation_job": "20465252",
    "domain": "counters",
    "value_head": "on",
    "seed": 1963100312,
    "instance": "fz_instance_51.pddl",
    # The historical log 21178321 first diverges at step 0.  Step 1 is also a
    # policy/search mismatch, but it is not the first one and is therefore not
    # the correct invariant for a first-divergence recorder.
    "step": 0,
    "root_visits": 20,
    "total_edge_visits": 19,
    "policy_action": 52,
    "selected_action": 101,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_frozen_tasks(manifest: Path, freeze: Path) -> list[dict[str, str]]:
    actual_hash = sha256_file(manifest)
    if actual_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"grouped manifest hash mismatch: {actual_hash} != "
            f"{EXPECTED_MANIFEST_SHA256}"
        )
    frozen = json.loads(freeze.read_text(encoding="utf-8"))
    if frozen.get("grouped_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("freeze metadata does not name the compiled-in manifest hash")
    if frozen.get("submitted") is not False:
        raise RuntimeError("frozen manifest unexpectedly claims submission")
    rows = read_csv(manifest)
    counts: dict[str, int] = {}
    candidate_total = 0
    for row in rows:
        counts[row["release_class"]] = counts.get(row["release_class"], 0) + 1
        candidates = json.loads(row["candidate_runs_json"])
        if len(candidates) != int(row["candidate_count"]):
            raise RuntimeError(f"{row['task_id']}: candidate count mismatch")
        candidate_total += len(candidates)
        if row["submitted"] != "false":
            raise RuntimeError(f"{row['task_id']}: task is not frozen as unsubmitted")
        if row["recorder_flag"] != "--eval-mcts-first-divergence-record":
            raise RuntimeError(f"{row['task_id']}: recorder flag mismatch")
        if row["release_gate"] != "known_counters_root_compute_smoke_passed":
            raise RuntimeError(f"{row['task_id']}: release gate mismatch")
    if counts != EXPECTED_TASK_COUNTS or candidate_total != 40:
        raise RuntimeError(
            f"unexpected frozen task inventory: counts={counts}, candidates={candidate_total}"
        )
    return rows


def load_recovery_tasks(
    recovery_manifest: Path,
    recovery_freeze: Path,
    source_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    metadata = json.loads(recovery_freeze.read_text(encoding="utf-8"))
    actual_hash = sha256_file(recovery_manifest)
    if metadata.get("recovery_manifest_sha256") != actual_hash:
        raise RuntimeError("recovery manifest hash mismatch")
    if metadata.get("source_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("recovery freeze source-manifest hash mismatch")
    if metadata.get("submitted") is not False:
        raise RuntimeError("recovery manifest unexpectedly claims submission")
    payload = json.loads(recovery_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "mcts-divergence-stage1-recovery-v1":
        raise RuntimeError("unsupported recovery manifest schema")
    if payload.get("source_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("recovery manifest source hash mismatch")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or payload.get("task_count") != len(tasks):
        raise RuntimeError("recovery task count mismatch")
    seen = set()
    candidate_total = 0
    for recovery_index, task in enumerate(tasks):
        if task.get("recovery_index") != recovery_index:
            raise RuntimeError("recovery indices must be contiguous")
        source_index = int(task["source_task_index"])
        if not 0 <= source_index < len(source_rows):
            raise RuntimeError("recovery source task index out of range")
        source_candidates = json.loads(
            source_rows[source_index]["candidate_runs_json"])
        indices = task.get("candidate_indices")
        candidates = task.get("candidate_runs")
        if not isinstance(indices, list) or not isinstance(candidates, list):
            raise RuntimeError("recovery candidates must be explicit lists")
        if len(indices) != len(candidates) or len(indices) != task["candidate_count"]:
            raise RuntimeError("recovery candidate count mismatch")
        for candidate_index, candidate in zip(indices, candidates):
            key = (source_index, int(candidate_index))
            if key in seen:
                raise RuntimeError("duplicate recovery candidate identity")
            seen.add(key)
            if not 0 <= int(candidate_index) < len(source_candidates):
                raise RuntimeError("recovery candidate index out of range")
            if candidate != source_candidates[int(candidate_index)]:
                raise RuntimeError("recovery candidate differs from frozen source")
        candidate_total += len(indices)
    if payload.get("candidate_count") != candidate_total:
        raise RuntimeError("recovery manifest total candidate count mismatch")
    return tasks


def _identity(identity: str) -> tuple[str, int]:
    match = IDENTITY_RE.fullmatch(identity)
    if not match:
        raise RuntimeError(f"invalid checkpoint identity: {identity}")
    return match.group("job"), int(match.group("epoch"))


def resolve_checkpoint(
    registry_rows: Iterable[dict[str, str]],
    identity: str,
    *,
    policy_log: str | None = None,
    evaluation_job: str | None = None,
) -> tuple[str, dict[str, str]]:
    source_job, epoch = _identity(identity)
    if policy_log is not None:
        match = POLICY_JOB_RE.match(Path(policy_log).name)
        if not match:
            raise RuntimeError(f"cannot recover policy job from {policy_log}")
        evaluation_job = match.group("job")
    matches = []
    for row in registry_rows:
        if row.get("source_training_job_id") != source_job:
            continue
        if evaluation_job is not None and row.get("job_id") != evaluation_job:
            continue
        checkpoint = row.get("checkpoint", "").replace("\\", "/")
        epoch_match = SNAPSHOT_RE.search(checkpoint)
        if not epoch_match or int(epoch_match.group("epoch")) != epoch:
            continue
        if policy_log is not None:
            recorded_log = row.get("source_evaluation_log", "").replace("\\", "/")
            if recorded_log != policy_log.replace("\\", "/"):
                continue
        matches.append((checkpoint, row))
    if len(matches) != 1:
        raise RuntimeError(
            f"{identity}: expected one exact registry checkpoint for evaluation "
            f"job {evaluation_job}, found {len(matches)}"
        )
    return matches[0]


def instance_index(repo: Path, domain: str, instance_name: str) -> int:
    module_root = str(repo / "asnets")
    if module_root not in sys.path:
        sys.path.insert(0, module_root)
    _arch, problem_module = DOMAIN_MODULES[domain]
    problem = importlib.import_module(problem_module)
    matches = []
    for index, (pddls, _name) in enumerate(problem.TEST_RUNS):
        basenames = [Path(path).name for path in pddls]
        if instance_name in basenames:
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(
            f"{domain}/{instance_name}: expected one TEST_RUNS index, found {matches}"
        )
    return matches[0]


def build_command(
    *,
    repo: Path,
    domain: str,
    value_head: str,
    seed: int,
    checkpoint: str,
    instance_name: str,
    search_config: dict[str, Any],
    tie_break: str,
    terminal_safe: bool,
    completion: Path,
    instance_timeout: int,
    max_actions: int = 10000,
    action_debug: bool = False,
) -> list[str]:
    if value_head not in {"off", "on"}:
        raise RuntimeError(f"invalid value-head setting: {value_head}")
    arch_module, problem_module = DOMAIN_MODULES[domain]
    command = [
        "./run_experiment",
        arch_module,
        problem_module,
        "--resume-from", checkpoint,
        "--eval-with-mcts",
        "--eval-mcts-first-divergence-record",
        "--eval-mcts-root-visit-tie-break", tie_break,
        "--mcts-expansion-size", str(int(search_config["mcts_expansion_k"])),
        "--mcts-iterations", str(int(search_config["mcts_iterations"])),
        "--mcts-exploration-weight", str(search_config["mcts_exploration_weight"]),
        "--use-estimator", str(search_config["estimator_coeff"]),
        "--restrict-test-probs", str(instance_index(repo, domain, instance_name)),
        "--serial-test",
        "--eval-scheduling", "rolling",
        "--eval-completion-file", str(completion),
        "--eval-instance-timeout", str(instance_timeout),
        "--eval-max-actions", str(max_actions),
        "--num-workers", "1",
        "--jpddl-max-heap", "4g",
        "--worker-logs",
        "--random-seed", str(seed),
    ]
    if action_debug:
        command.append("--action-debug")
    if terminal_safe:
        command.append("--eval-mcts-terminal-safe-action-selection")
    if bool(search_config.get("progressive_widening")):
        command.extend([
            "--mcts-progressive-widening",
            "--mcts-pw-min-width", str(int(search_config["pw_min_width"])),
            "--mcts-pw-c", str(search_config["pw_c"]),
            "--mcts-pw-alpha", str(search_config["pw_alpha"]),
        ])
    if value_head == "off":
        command.append("--disable-value-head")
    return command


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def divergence_records(log: Path) -> list[dict[str, Any]]:
    records = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = line.find(RECORDER_MARKER)
        if marker < 0:
            continue
        records.append(json.loads(
            line[marker + len(RECORDER_MARKER):],
            parse_constant=_reject_json_constant,
        ))
    return records


def determinism_records(log: Path) -> list[dict[str, Any]]:
    records = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = line.find(DETERMINISM_MARKER)
        if marker < 0:
            continue
        record = json.loads(
            line[marker + len(DETERMINISM_MARKER):],
            parse_constant=_reject_json_constant,
        )
        record.pop("elapsed_seconds", None)
        records.append(record)
    return records


def normalize_record(
    record: dict[str, Any], *, instance_name: str, allow_legacy: bool = False,
) -> dict[str, Any]:
    """Validate one record and add explicit v2 selection evidence if needed."""
    schema = record.get("schema_version")
    if schema != RECORDER_SCHEMA and not (
            allow_legacy and schema == LEGACY_RECORDER_SCHEMA):
        raise RuntimeError(f"unexpected divergence record schema: {schema}")
    if Path(record.get("instance", "")).name != instance_name:
        raise RuntimeError("divergence record instance mismatch")
    root = record.get("root", {})
    act_dim = root.get("act_dim")
    if not isinstance(act_dim, int) or act_dim <= 0:
        raise RuntimeError("divergence record has invalid action dimension")
    vector_fields = (
        "action_ids", "action_names", "applicable", "expanded",
        "raw_network_policy", "masked_network_policy", "visit_distribution",
        "edge_visit_counts", "edge_priors", "q_values", "u_values",
        "signed_q_plus_u", "child_terminal", "child_goal",
        "child_state_digests",
    )
    for field in vector_fields:
        if not isinstance(root.get(field), list) or len(root[field]) != act_dim:
            raise RuntimeError(f"divergence record root.{field} length mismatch")
    override_path = record.get("override_path")
    if not isinstance(override_path, list) or not override_path:
        raise RuntimeError("divergence record is missing the selector override path")
    selection = record.get("selection")
    if isinstance(selection, dict):
        selector = selection.get("selector_input_distribution")
        source = selection.get("selector_distribution_source")
        final_stage = selection.get("final_stage")
        if int(selection.get("selected_action", -1)) != int(
                record["selected_action"]):
            raise RuntimeError("selection evidence action mismatch")
    else:
        selector = source = final_stage = None
    if not isinstance(selector, list) or len(selector) != act_dim:
        selected = int(record["selected_action"])
        chosen_index = None
        for index in range(len(override_path) - 1, -1, -1):
            stage = override_path[index]
            if int(stage.get("selected_action", -1)) != selected:
                continue
            candidate = stage.get("selector_input_distribution")
            if isinstance(candidate, list) and len(candidate) == act_dim:
                selector = candidate
                source = "override_path"
                final_stage = stage.get("stage")
                chosen_index = index
                break
            if (allow_legacy and stage.get("stage") == "goal_chase"
                    and stage.get("applied") is True):
                selector = root["visit_distribution"]
                source = "visit_distribution_legacy_goal_chase_input"
                final_stage = "goal_chase"
                chosen_index = index
                break
        if not isinstance(selector, list) or len(selector) != act_dim:
            raise RuntimeError(
                "terminating selector-input distribution is missing or truncated")
        record = json.loads(json.dumps(record))
        record["selection"] = {
            "final_stage": final_stage,
            "override_path_index": chosen_index,
            "selected_action": selected,
            "selector_input_distribution": selector,
            "selector_distribution_source": source,
        }
        if schema == LEGACY_RECORDER_SCHEMA:
            record["normalized_from_schema"] = LEGACY_RECORDER_SCHEMA
            record["schema_version"] = RECORDER_SCHEMA
    if any(not isinstance(value, (int, float))
           or not math.isfinite(float(value)) or float(value) < 0.0
           for value in selector):
        raise RuntimeError("terminating selector-input distribution is invalid")
    if final_stage is None or not source:
        raise RuntimeError("terminating selector provenance is incomplete")
    return record


def validate_record(record: dict[str, Any], *, instance_name: str) -> None:
    normalize_record(record, instance_name=instance_name, allow_legacy=False)


def terminal_outcome(
    *,
    log: Path,
    completion: Path,
    instance_name: str,
    evaluation_index: int,
    max_actions: int,
) -> dict[str, Any] | None:
    """Return a defensible terminal classification, never infer from silence."""
    if completion.exists():
        lines = [line for line in completion.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 1:
            raise RuntimeError(
                f"expected one completion record, found {len(lines)}: {completion}")
        record = json.loads(lines[0], parse_constant=_reject_json_constant)
        if int(record.get("instance_number", -1)) != evaluation_index:
            raise RuntimeError("completion record evaluator identity mismatch")
        if Path(record.get("instance_path", "")).name != instance_name:
            raise RuntimeError("completion record instance path mismatch")
        status = record.get("status")
        if status not in {"success", "finished_unsolved"}:
            raise RuntimeError(f"unsupported completion status: {status}")
        if status == "success":
            classification = "success"
        elif int(record.get("steps", -1)) >= max_actions:
            classification = "action_limit"
        else:
            classification = "finished_unsolved"
        return {
            "classification": classification,
            "evidence": "completion_jsonl",
            "completion_record": record,
        }
    text = log.read_text(encoding="utf-8", errors="replace")
    if any(int(match.group("number")) == evaluation_index
           for match in CRASH_RE.finditer(text)):
        return None
    completed = [match for match in COMPLETED_RE.finditer(text)
                 if int(match.group("number")) == evaluation_index
                 and Path(match.group("path")).name == instance_name]
    if len(completed) == 1:
        marker = completed[0]
        steps = int(marker.group("steps"))
        success = marker.group("success") in {"True", "1.0"}
        if marker.group("status") == "success" and success:
            classification = "success"
        elif marker.group("status") == "unsolved" and not success:
            classification = (
                "action_limit" if steps >= max_actions else "finished_unsolved")
        else:
            return None
        return {
            "classification": classification,
            "evidence": "eval_instance_completed_log_marker",
            "completion_record": None,
            "instance_number": evaluation_index,
            "instance_path": marker.group("path"),
            "elapsed_seconds": float(marker.group("elapsed")),
            "steps": steps,
            "marker": marker.group(0),
        }
    matches = [match for match in HARD_TIMEOUT_RE.finditer(text)
               if int(match.group("number")) == evaluation_index
               and Path(match.group("path")).name == instance_name]
    if len(matches) != 1:
        return None
    marker = matches[0]
    return {
        "classification": "hard_timeout",
        "evidence": "eval_instance_timeout_log_marker",
        "completion_record": None,
        "instance_number": evaluation_index,
        "instance_path": marker.group("path"),
        "limit_seconds": float(marker.group("limit")),
        "marker": marker.group(0),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_gate(path: Path, *, code_commit: str) -> dict[str, Any]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": GATE_SCHEMA,
        "status": "passed",
        "code_commit": code_commit,
        "grouped_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "checkpoint_identity": SMOKE["checkpoint_identity"],
        "instance": SMOKE["instance"],
        "step": SMOKE["step"],
        "policy_action": SMOKE["policy_action"],
        "selected_action": SMOKE["selected_action"],
        "root_visits": SMOKE["root_visits"],
        "total_edge_visits": SMOKE["total_edge_visits"],
        "disabled_mode_invariance": True,
    }
    mismatches = {key: (gate.get(key), value) for key, value in expected.items()
                  if gate.get(key) != value}
    if mismatches:
        raise RuntimeError(f"compute-smoke gate mismatch: {mismatches}")
    return gate


def _run(command: list[str], *, cwd: Path, log: Path, header: dict[str, Any]) -> None:
    if log.exists():
        raise RuntimeError(f"refusing to overwrite existing log: {log}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("[MCTS DIVERGENCE STAGE1 RUN] " + json.dumps(
            header, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        result = subprocess.run(
            command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def compute_smoke(args: argparse.Namespace) -> None:
    registry_rows = read_csv(args.registry)
    checkpoint, registry_row = resolve_checkpoint(
        registry_rows,
        SMOKE["checkpoint_identity"],
        evaluation_job=SMOKE["evaluation_job"],
    )
    smoke_dir = args.output_root / "compute_smoke"
    baseline_log = smoke_dir / "known_counters_root_recorder_disabled.txt"
    baseline_completion = (
        smoke_dir / "known_counters_root_recorder_disabled.completed.jsonl")
    log = smoke_dir / "known_counters_root_recorder_enabled.txt"
    completion = smoke_dir / "known_counters_root_recorder_enabled.completed.jsonl"
    gate = smoke_dir / "gate.json"
    command = build_command(
        repo=args.repo,
        domain=SMOKE["domain"],
        value_head=SMOKE["value_head"],
        seed=SMOKE["seed"],
        checkpoint=checkpoint,
        instance_name=SMOKE["instance"],
        search_config={
            "mcts_expansion_k": 5,
            "mcts_iterations": 20,
            "mcts_exploration_weight": 0.1,
            "estimator_coeff": 0.5,
            "progressive_widening": False,
        },
        tie_break="action_id",
        terminal_safe=False,
        completion=completion,
        instance_timeout=1200,
        max_actions=2,
        action_debug=True,
    )
    if not args.execute:
        baseline_command = [
            item for item in command
            if item != "--eval-mcts-first-divergence-record"
        ]
        baseline_command[baseline_command.index(str(completion))] = str(
            baseline_completion)
        print(json.dumps({
            "mode": "dry-run",
            "checkpoint": checkpoint,
            "recorder_disabled_command": baseline_command,
            "recorder_enabled_command": command,
        }, indent=2))
        return
    if gate.exists() or completion.exists() or baseline_completion.exists():
        raise RuntimeError("refusing to overwrite an existing smoke gate/completion")
    if not Path(checkpoint).is_dir():
        raise RuntimeError(f"missing compute-smoke checkpoint: {checkpoint}")
    baseline_command = [
        item for item in command
        if item != "--eval-mcts-first-divergence-record"
    ]
    baseline_command[baseline_command.index(str(completion))] = str(
        baseline_completion)
    _run(baseline_command, cwd=args.repo / "asnets", log=baseline_log, header={
        "role": "known-root-compute-smoke-recorder-disabled",
        "code_commit": args.code_commit,
        "checkpoint_identity": SMOKE["checkpoint_identity"],
        "checkpoint": checkpoint,
        "registry_job": registry_row["job_id"],
        "registry_sha256": sha256_file(args.registry),
    })
    if divergence_records(baseline_log):
        raise RuntimeError("disabled-mode smoke unexpectedly emitted a recorder event")
    _run(command, cwd=args.repo / "asnets", log=log, header={
        "role": "known-root-compute-smoke",
        "code_commit": args.code_commit,
        "checkpoint_identity": SMOKE["checkpoint_identity"],
        "checkpoint": checkpoint,
        "registry_job": registry_row["job_id"],
        "registry_sha256": sha256_file(args.registry),
    })
    records = divergence_records(log)
    if len(records) != 1:
        raise RuntimeError(f"compute smoke emitted {len(records)} divergence records")
    record = records[0]
    validate_record(record, instance_name=SMOKE["instance"])
    observed = {
        "step": record["step"],
        "policy_action": record["policy_action"],
        "selected_action": record["selected_action"],
        "root_visits": record["root"]["root_visits"],
        "total_edge_visits": record["root"]["total_edge_visits"],
    }
    expected = {key: SMOKE[key] for key in observed}
    if observed != expected:
        raise RuntimeError(f"known-root smoke mismatch: {observed} != {expected}")
    baseline_determinism = determinism_records(baseline_log)
    enabled_determinism = determinism_records(log)
    if len(baseline_determinism) != 2 or baseline_determinism != enabled_determinism:
        raise RuntimeError(
            "recorder observer changed the two-step deterministic root/action trace: "
            f"disabled={len(baseline_determinism)} enabled={len(enabled_determinism)}"
        )
    atomic_json(gate, {
        "schema_version": GATE_SCHEMA,
        "status": "passed",
        "code_commit": args.code_commit,
        "grouped_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "checkpoint_identity": SMOKE["checkpoint_identity"],
        "instance": SMOKE["instance"],
        **observed,
        "disabled_mode_invariance": True,
        "determinism_records_compared": len(enabled_determinism),
        "recorder_disabled_log": str(baseline_log),
        "recorder_disabled_completion": str(baseline_completion),
        "log": str(log),
        "completion": str(completion),
        "registry_sha256": sha256_file(args.registry),
    })
    print(f"compute smoke passed; gate={gate}")


def run_task(args: argparse.Namespace) -> None:
    rows = load_frozen_tasks(args.manifest, args.freeze)
    if not 0 <= args.task_index < len(rows):
        raise RuntimeError(f"task index out of range: {args.task_index}")
    row = rows[args.task_index]
    if args.execute:
        validate_gate(args.smoke_gate, code_commit=args.code_commit)
    registry_rows = read_csv(args.registry)
    source_candidates = json.loads(row["candidate_runs_json"])
    if args.candidate_indices:
        selected_indices = [int(value) for value in args.candidate_indices.split(",")]
        if len(selected_indices) != len(set(selected_indices)):
            raise RuntimeError("candidate indices must be unique")
        if any(index < 0 or index >= len(source_candidates)
               for index in selected_indices):
            raise RuntimeError("candidate index out of range")
    else:
        selected_indices = list(range(len(source_candidates)))
    candidates = [(index, source_candidates[index]) for index in selected_indices]
    rendered = []
    for candidate_index, candidate in candidates:
        checkpoint, registry_row = resolve_checkpoint(
            registry_rows,
            candidate["checkpoint_identity"],
            policy_log=candidate["policy_log"],
        )
        stem = (
            f"{candidate_index:02d}_{candidate['checkpoint_identity']}_"
            f"s{candidate['seed']}_{Path(candidate['instance']).stem}"
        )
        task_dir = args.output_root / row["task_id"]
        log = task_dir / f"{stem}.txt"
        completion = task_dir / f"{stem}.completed.jsonl"
        result_path = task_dir / f"{stem}.result.json"
        command = build_command(
            repo=args.repo,
            domain=row["domain"],
            value_head=row["value_head"],
            seed=int(candidate["seed"]),
            checkpoint=checkpoint,
            instance_name=candidate["instance"],
            search_config=json.loads(row["search_config"]),
            tie_break=row["tie_break"],
            terminal_safe=row["terminal_safe"].lower() == "true",
            completion=completion,
            instance_timeout=int(row["per_instance_timeout_seconds"]),
        )
        rendered.append({
            "candidate_index": candidate_index,
            "checkpoint": checkpoint,
            "instance_index": instance_index(
                args.repo, row["domain"], candidate["instance"]),
            "command": command,
        })
        if not args.execute:
            continue
        if not Path(checkpoint).is_dir():
            raise RuntimeError(f"missing candidate checkpoint: {checkpoint}")
        if completion.exists() or result_path.exists():
            raise RuntimeError(f"refusing to overwrite candidate artifacts: {stem}")
        _run(command, cwd=args.repo / "asnets", log=log, header={
            "task_index": args.task_index,
            "task_id": row["task_id"],
            "release_class": row["release_class"],
            "candidate_index": candidate_index,
            "missing_stratum": candidate["missing_stratum"],
            "checkpoint_identity": candidate["checkpoint_identity"],
            "checkpoint": checkpoint,
            "seed": int(candidate["seed"]),
            "instance": candidate["instance"],
            "policy_log": candidate["policy_log"],
            "search_log": candidate["search_log"],
            "registry_job": registry_row["job_id"],
            "code_commit": args.code_commit,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        })
        records = divergence_records(log)
        if len(records) > 1:
            raise RuntimeError(f"{stem}: emitted multiple first-divergence records")
        if records:
            records[0] = normalize_record(
                records[0], instance_name=candidate["instance"])
        outcome = terminal_outcome(
            log=log,
            completion=completion,
            instance_name=candidate["instance"],
            evaluation_index=RESTRICTED_EVALUATOR_NUMBER,
            max_actions=10000,
        )
        if outcome is None:
            raise RuntimeError(
                f"{stem}: no explicit success, action-limit, finished-unsolved, "
                "or hard-timeout evidence")
        atomic_json(result_path, {
            "schema_version": "mcts-divergence-stage1-candidate-result-v2",
            "task_index": args.task_index,
            "task_id": row["task_id"],
            "candidate_index": candidate_index,
            "checkpoint_identity": candidate["checkpoint_identity"],
            "checkpoint": checkpoint,
            "instance": candidate["instance"],
            "missing_stratum": candidate["missing_stratum"],
            "first_divergence_observed": bool(records),
            "first_divergence": records[0] if records else None,
            "outcome": outcome,
            "completion_record": outcome["completion_record"],
            "log": str(log),
            "completion": str(completion),
            "code_commit": args.code_commit,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "registry_sha256": sha256_file(args.registry),
        })
    if not args.execute:
        print(json.dumps({
            "mode": "dry-run",
            "task_index": args.task_index,
            "task_id": row["task_id"],
            "release_class": row["release_class"],
            "runs": rendered,
        }, indent=2))


def run_recovery(args: argparse.Namespace) -> None:
    source_rows = load_frozen_tasks(args.manifest, args.freeze)
    recovery_tasks = load_recovery_tasks(
        args.recovery_manifest, args.recovery_freeze, source_rows)
    if not 0 <= args.recovery_index < len(recovery_tasks):
        raise RuntimeError("recovery index out of range")
    recovery = recovery_tasks[args.recovery_index]
    args.task_index = int(recovery["source_task_index"])
    args.candidate_indices = ",".join(
        str(value) for value in recovery["candidate_indices"])
    run_task(args)


def common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--execute", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("compute-smoke")
    common_parser(smoke)
    smoke.set_defaults(func=compute_smoke)
    task = subparsers.add_parser("run-task")
    common_parser(task)
    task.add_argument("--manifest", type=Path, required=True)
    task.add_argument("--freeze", type=Path, required=True)
    task.add_argument("--task-index", type=int, required=True)
    task.add_argument(
        "--candidate-indices",
        help="optional comma-separated exact source-candidate indices")
    task.add_argument("--smoke-gate", type=Path, required=True)
    task.set_defaults(func=run_task)
    recovery = subparsers.add_parser("run-recovery")
    common_parser(recovery)
    recovery.add_argument("--manifest", type=Path, required=True)
    recovery.add_argument("--freeze", type=Path, required=True)
    recovery.add_argument("--recovery-manifest", type=Path, required=True)
    recovery.add_argument("--recovery-freeze", type=Path, required=True)
    recovery.add_argument("--recovery-index", type=int, required=True)
    recovery.add_argument("--smoke-gate", type=Path, required=True)
    recovery.set_defaults(func=run_recovery)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
