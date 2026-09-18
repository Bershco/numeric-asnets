"""Pure manifest validation for the value-head V1 audit."""

from __future__ import annotations

import re
from typing import Mapping, Sequence


def validate_task_manifest_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    domains: Sequence[str],
    seeds: Sequence[str],
) -> list[str]:
    """Return submission blockers for a frozen V1 task manifest."""

    errors: list[str] = []
    expected = {
        (domain, str(seed), stage)
        for domain in domains
        for seed in seeds
        for stage in ("stage1", "stage2")
    }
    actual: set[tuple[str, str, str]] = set()
    sha_re = re.compile(r"^[0-9a-f]{64}$")
    required_labels = {
        "replay_target", "deterministic_continuation", "enhsp_raw_h",
        "enhsp_search_v",
    }
    required_sources = {
        "common_planner", "common_random_legal",
        "stage1_on_policy", "stage2_on_policy",
    }

    state_manifest_by_lineage: dict[tuple[str, str], tuple[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        identity = (row.get("domain", ""), row.get("seed", ""), row.get("stage", ""))
        if identity in actual:
            errors.append(f"row {index}: duplicate identity {identity}")
        actual.add(identity)
        if not row.get("task_id"):
            errors.append(f"row {index}: missing task_id")
        if not row.get("checkpoint_path", "").startswith("/"):
            errors.append(f"row {index}: checkpoint_path is not an absolute cluster path")
        for field in ("checkpoint_sha256", "state_manifest_sha256", "source_manifest_sha256"):
            if not sha_re.fullmatch(row.get(field, "")):
                errors.append(f"row {index}: {field} is missing or not sha256")
        state_path = row.get("state_manifest_path", "")
        if not state_path:
            errors.append(f"row {index}: missing state_manifest_path")
        lineage = identity[:2]
        state_identity = (state_path, row.get("state_manifest_sha256", ""))
        previous = state_manifest_by_lineage.setdefault(lineage, state_identity)
        if previous != state_identity:
            errors.append(f"row {index}: Stage1/Stage2 do not share one frozen state manifest for {lineage}")
        sources = set(filter(None, row.get("state_sources", "").split("+")))
        if sources != required_sources:
            errors.append(f"row {index}: unexpected state_sources {sorted(sources)}")
        labels = set(filter(None, row.get("label_sources", "").split("+")))
        if labels != required_labels:
            errors.append(f"row {index}: unexpected label_sources {sorted(labels)}")
        try:
            if any(int(row.get(field, "0")) <= 0 for field in ("cpus", "memory_gib", "time_limit_hours")):
                raise ValueError
        except ValueError:
            errors.append(f"row {index}: invalid resource request")

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"missing task identities: {missing}")
    if extra:
        errors.append(f"unexpected task identities: {extra}")
    return errors


def validate_state_mixture_rows(
    rows: Sequence[Mapping[str, str]], *, domains: Sequence[str]
) -> list[str]:
    """Check the predeclared 60-state, four-source lineage mixture."""

    errors: list[str] = []
    expected_quotas = {
        "common_planner": 20,
        "common_random_legal": 20,
        "stage1_on_policy": 10,
        "stage2_on_policy": 10,
    }
    for domain in domains:
        subset = [row for row in rows if row.get("domain") == domain]
        sources = [row.get("state_source", "") for row in subset]
        if set(sources) != set(expected_quotas) or len(sources) != len(expected_quotas):
            errors.append(f"{domain}: state-source factorial is not {sorted(expected_quotas)}")
            continue
        total = 0
        for row in subset:
            source = row["state_source"]
            try:
                quota = int(row.get("quota_per_lineage_manifest", ""))
            except ValueError:
                errors.append(f"{domain}/{source}: invalid quota")
                continue
            total += quota
            if quota != expected_quotas[source]:
                errors.append(f"{domain}/{source}: expected quota {expected_quotas[source]}, got {quota}")
            if not row.get("sampling_rule") or not row.get("validation_instances"):
                errors.append(f"{domain}/{source}: sampling rule/instances are not frozen")
        if total != 60:
            errors.append(f"{domain}: expected 60 states, got {total}")
    extras = sorted({row.get("domain", "") for row in rows} - set(domains))
    if extras:
        errors.append(f"unexpected mixture domains: {extras}")
    return errors


def validate_label_source_rows(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Check that independent label families have explicit failure semantics."""

    errors: list[str] = []
    expected = {
        "replay_target", "deterministic_continuation", "enhsp_raw_h",
        "enhsp_search_v",
    }
    names = [row.get("label_source", "") for row in rows]
    if set(names) != expected or len(names) != len(expected):
        errors.append(f"label-source factorial is not {sorted(expected)}")
    for row in rows:
        source = row.get("label_source", "")
        if row.get("orientation") not in {"higher_is_better", "lower_is_better"}:
            errors.append(f"{source}: orientation is not explicit")
        if not row.get("cache_identity") or not row.get("do_not_do"):
            errors.append(f"{source}: cache identity/prohibited fallback is not frozen")
        statuses = set(filter(None, row.get("valid_statuses", "").split(";")))
        if "valid" not in statuses:
            errors.append(f"{source}: valid status is missing")
        if source != "replay_target" and not {"timeout", "unsolved", "error"}.issubset(statuses):
            errors.append(f"{source}: timeout/unsolved/error statuses are incomplete")
        comparable = row.get("scale_comparable")
        expected_comparable = (
            "true" if source in {"replay_target", "enhsp_search_v"} else "false"
        )
        if source in expected and comparable != expected_comparable:
            errors.append(
                f"{source}: scale_comparable must be {expected_comparable} for V1"
            )
        if source == "enhsp_search_v":
            transform_path = row.get("transform_config_path", "")
            transform_sha = row.get("transform_config_sha256", "")
            if not transform_path or not re.fullmatch(r"[0-9a-f]{64}", transform_sha):
                errors.append(
                    "enhsp_search_v: exact transform config path/hash is not frozen"
                )
    return errors
