#!/usr/bin/env python3
"""Run one fail-closed V1 checkpoint audit task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "asnets"))

from asnets.models import PropNetwork  # noqa: E402
from asnets.prob_dom_meta import DomainType  # noqa: E402
from asnets.state_reprs import CanonicalState, sample_next_state  # noqa: E402
from asnets.supervised import PlannerExtensions  # noqa: E402
from asnets.value_head_audit import (  # noqa: E402
    DeterministicContinuationProvider,
    ENHSPRawProvider,
    ENHSPSearchValueProvider,
    ReplayTargetProvider,
    SuccessorValueRow,
    apply_label_provider,
    restore_canonical_state,
)
from post_training.enhspwrapper import ENHSPEstimator  # noqa: E402


ENHSP_CONFIG = {
    "drone": "hadd-astar",
    "fo_counters": "hmrmax-astar",
    "rover": "hmrp-ha-gbfs",
    "mprime": "hmrp-ha-gbfs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_preflight(records: list[dict[str, object]]) -> list[dict[str, object]]:
    quotas = {"common_planner": 2, "common_random_legal": 2,
              "stage1_on_policy": 1, "stage2_on_policy": 1}
    chosen: list[dict[str, object]] = []
    for source, quota in quotas.items():
        subset = sorted(
            (row for row in records if row["state_source"] == source),
            key=lambda row: (str(row["instance_identity"]), str(row["state_sha256"])),
        )
        if len(subset) < quota:
            raise RuntimeError(f"preflight stratum {source} has {len(subset)} states")
        chosen.extend(subset[:quota])
    return chosen


class AlarmTimeout:
    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        def handler(_signum, _frame):
            raise TimeoutError(f"operation exceeded {self.seconds}s")
        self.previous = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.seconds)

    def __exit__(self, *_args):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous)


def stable_policy_action(network, state) -> int | None:
    output = network(state.to_network_input()[None], training=False)
    policy = output[0] if isinstance(output, (tuple, list)) else output
    policy = np.asarray(policy).reshape(-1)
    mask = state.get_applicable_action_mask()
    valid = np.flatnonzero(mask)
    if not len(valid):
        return None
    masked = np.full(policy.shape, -np.inf, dtype=np.float64)
    masked[valid] = policy[valid]
    return int(np.argmax(masked))


def continuation_result(network, state, planner_exts, *, max_steps: int,
                        timeout_seconds: int) -> tuple[str, float | None]:
    if state.is_goal:
        return "valid", 0.0
    if state.is_terminal:
        return "unsolved", None
    started = time.monotonic()
    current = state
    for steps in range(1, max_steps + 1):
        if time.monotonic() - started > timeout_seconds:
            return "timeout", None
        action = stable_policy_action(network, current)
        if action is None:
            return "unsolved", None
        current, _ = sample_next_state(
            current, action, planner_exts, ignore_disabled=False
        )
        if current.is_goal:
            return "valid", float(steps)
        if current.is_terminal:
            return "unsolved", None
    return "timeout", None


def load_replay(path: Path | None, expected_sha: str | None):
    if path is None:
        return {}, "missing://no-frozen-replay-cache"
    if not expected_sha or sha256(path) != expected_sha:
        raise RuntimeError("replay cache hash mismatch or missing expected hash")
    values = {}
    for row in read_jsonl(path):
        key = (str(row["state_id"]), int(row["action_id"]), int(row["successor_index"]))
        values[key] = ("valid", float(row["label_value"]))
    return values, str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--replay-cache", type=Path)
    parser.add_argument("--replay-cache-sha256")
    parser.add_argument("--enhsp-timeout", type=int, default=300)
    parser.add_argument("--continuation-timeout", type=int, default=900)
    parser.add_argument("--continuation-max-steps", type=int, default=10000)
    args = parser.parse_args()

    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.candidates.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not 0 <= args.task_index < len(rows):
        raise RuntimeError("task index outside frozen candidate manifest")
    task = rows[args.task_index]
    checkpoint = Path(task["checkpoint_path"])
    weights_path = checkpoint / "weights.joblib"
    manifest = Path(task["state_manifest_path"])
    if sha256(weights_path) != task["checkpoint_sha256"]:
        raise RuntimeError("checkpoint payload hash mismatch")
    if sha256(manifest) != task["state_manifest_sha256"]:
        raise RuntimeError("paired state manifest hash mismatch")

    transform_path = ROOT / "experiment_tracking/value_head_quality_audit/v1_enhsp_search_transform.json"
    label_sources_path = ROOT / "experiment_tracking/value_head_quality_audit/v1_label_sources.csv"
    with label_sources_path.open(newline="", encoding="utf-8") as stream:
        transform_row = next(
            row for row in csv.DictReader(stream)
            if row["label_source"] == "enhsp_search_v"
        )
    if sha256(transform_path) != transform_row["transform_config_sha256"]:
        raise RuntimeError("ENHSP search transform hash mismatch")
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    if (transform["formula"] != "exp(-coefficient * h)"
            or float(transform["coefficient"]) != 1.0
            or bool(transform["minimization"])):
        raise RuntimeError("unexpected ENHSP search transform")

    records = read_jsonl(manifest)
    records = select_preflight(records) if args.mode == "preflight" else records
    if len(records) != (6 if args.mode == "preflight" else 60):
        raise RuntimeError("unexpected selected state count")
    replay_results, replay_log = load_replay(
        args.replay_cache, args.replay_cache_sha256
    )
    wm = joblib.load(weights_path)
    if not getattr(wm, "value_head_enabled", False):
        raise RuntimeError("V1 task requires a VH-on checkpoint")
    CanonicalState.network_input_config(use_fluents=True, use_comparisons=True)

    output_path = args.output_dir / "successor_labels.jsonl"
    label_counts: dict[str, dict[str, int]] = {}
    total_successors = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            instance_path = str(record["instance_path"])
            domain_path = str(Path(instance_path).parents[1] / "domain.pddl")
            planner_exts = PlannerExtensions(
                [domain_path, instance_path], DomainType.NUMERIC,
                dg_use_act_history=True,
            )
            network = PropNetwork(weight_manager=wm, problem_meta=planner_exts.problem_meta)
            state = restore_canonical_state(record, planner_exts)
            state_id = str(record["state_sha256"])
            successor_objects = {}
            pending = []
            mdpsim_state = state.to_mdpsim(planner_exts)
            for action_id in np.flatnonzero(state.get_applicable_action_mask()):
                successor, _ = sample_next_state(
                    state, int(action_id), planner_exts,
                    mdpsim_state=mdpsim_state, ignore_disabled=False,
                )
                pending.append((int(action_id), 0, successor))
            if not pending:
                continue
            observations = np.asarray([item[2].to_network_input() for item in pending], dtype=np.float32)
            network_output = network(observations, training=False)
            if not isinstance(network_output, (tuple, list)) or len(network_output) != 2:
                raise RuntimeError("checkpoint did not produce value-head output")
            raw_values = np.asarray(network_output[1]).reshape(-1)
            value_rows = []
            for (action_id, successor_index, successor), raw_value in zip(pending, raw_values):
                row = SuccessorValueRow(
                    state_id=state_id, action_id=action_id,
                    successor_index=successor_index,
                    transition_probability=1.0,
                    raw_network_value=float(raw_value),
                    successor_terminal=bool(successor.is_terminal),
                    successor_goal=bool(successor.is_goal),
                )
                value_rows.append(row)
                successor_objects[(state_id, action_id, successor_index)] = successor
            total_successors += len(value_rows)

            enhsp = ENHSPEstimator(planner_exts, enhsp_config=ENHSP_CONFIG[task["domain"]])
            raw_results = {}
            continuation_results = {}
            try:
                for key, successor in successor_objects.items():
                    if successor.is_goal:
                        raw_results[key] = ("valid", 0.0)
                    elif successor.is_terminal:
                        raw_results[key] = ("unsolved", None)
                    else:
                        try:
                            with AlarmTimeout(args.enhsp_timeout):
                                h, _ = enhsp.get_cstate_h_and_pi(successor)
                            raw_results[key] = (
                                ("valid", float(h)) if np.isfinite(h)
                                else ("unsolved", None)
                            )
                        except TimeoutError:
                            raw_results[key] = ("timeout", None)
                            enhsp.close()
                            enhsp = ENHSPEstimator(
                                planner_exts, enhsp_config=ENHSP_CONFIG[task["domain"]]
                            )
                        except Exception:
                            raw_results[key] = ("error", None)
                    try:
                        continuation_results[key] = continuation_result(
                            network, successor, planner_exts,
                            max_steps=args.continuation_max_steps,
                            timeout_seconds=args.continuation_timeout,
                        )
                    except Exception:
                        continuation_results[key] = ("error", None)
            finally:
                enhsp.close()

            providers = (
                ReplayTargetProvider(replay_results, label_log_path=replay_log),
                DeterministicContinuationProvider(
                    continuation_results, label_log_path=str(output_path)
                ),
                ENHSPRawProvider(raw_results, label_log_path=str(output_path)),
                ENHSPSearchValueProvider(
                    ENHSPRawProvider(raw_results, label_log_path=str(output_path)),
                    coefficient=1.0, minimization=False,
                ),
            )
            for provider in providers:
                for labelled in apply_label_provider(value_rows, provider):
                    source = str(labelled["label_source"])
                    status = str(labelled["label_status"])
                    label_counts.setdefault(source, {}).setdefault(status, 0)
                    label_counts[source][status] += 1
                    payload = {
                        "task_id": task["task_id"], "domain": task["domain"],
                        "seed": int(task["seed"]), "stage": task["stage"],
                        "checkpoint_role": task["checkpoint_role"],
                        "checkpoint_sha256": task["checkpoint_sha256"],
                        "state_manifest_sha256": task["state_manifest_sha256"],
                        "trajectory_id": f"{record['state_source']}:{record['instance_identity']}",
                        "state_source": record["state_source"],
                        "instance": record["instance_identity"],
                        "step": int(record["step"]), **labelled,
                    }
                    output.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")

    elapsed = time.monotonic() - started
    max_rss_kib = max(
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
    )
    summary = {
        "schema": "value-head-audit-task-summary-v1",
        "task_id": task["task_id"], "mode": args.mode,
        "states": len(records), "successors": total_successors,
        "elapsed_seconds": elapsed, "max_rss_kib": max_rss_kib,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "label_counts": label_counts,
        "output_path": str(output_path), "output_sha256": sha256(output_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("V1_TASK_COMPLETE|" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
