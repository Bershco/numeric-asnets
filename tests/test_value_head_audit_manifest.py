from __future__ import annotations

import unittest

from asnets.value_head_audit_manifest import (
    validate_label_source_rows,
    validate_state_mixture_rows,
    validate_task_manifest_rows,
)


class ValueHeadAuditManifestTest(unittest.TestCase):
    def row(self, *, stage: str) -> dict[str, str]:
        return {
            "task_id": f"vhv1-drone-1-{stage}",
            "domain": "drone", "seed": "1", "stage": stage,
            "checkpoint_path": f"/cluster/{stage}",
            "checkpoint_sha256": "a" * 64,
            "state_manifest_path": "states/paired.jsonl",
            "state_manifest_sha256": "b" * 64,
            "source_manifest_sha256": "c" * 64,
            "state_sources": "common_planner+common_random_legal+stage1_on_policy+stage2_on_policy",
            "label_sources": "replay_target+deterministic_continuation+enhsp_raw_h",
            "cpus": "4", "memory_gib": "48", "time_limit_hours": "8",
        }

    def test_accepts_complete_paired_lineage(self):
        rows = [self.row(stage="stage1"), self.row(stage="stage2")]
        self.assertEqual(validate_task_manifest_rows(rows, domains=["drone"], seeds=["1"]), [])

    def test_rejects_different_stage_state_manifests(self):
        rows = [self.row(stage="stage1"), self.row(stage="stage2")]
        rows[1]["state_manifest_path"] = "states/stage2.jsonl"
        errors = validate_task_manifest_rows(rows, domains=["drone"], seeds=["1"])
        self.assertIn("do not share", "\n".join(errors))

    def test_rejects_missing_hash(self):
        rows = [self.row(stage="stage1"), self.row(stage="stage2")]
        rows[1]["checkpoint_sha256"] = ""
        errors = validate_task_manifest_rows(rows, domains=["drone"], seeds=["1"])
        self.assertIn("checkpoint_sha256", "\n".join(errors))

    def test_accepts_frozen_state_mixture(self):
        rows = []
        for source, quota in (
            ("common_planner", 20), ("common_random_legal", 20),
            ("stage1_on_policy", 10), ("stage2_on_policy", 10),
        ):
            rows.append({
                "domain": "drone", "state_source": source,
                "quota_per_lineage_manifest": str(quota),
                "sampling_rule": "frozen", "validation_instances": "frozen",
            })
        self.assertEqual(validate_state_mixture_rows(rows, domains=["drone"]), [])

    def test_label_sources_require_failure_semantics(self):
        rows = [
            {"label_source": "replay_target", "orientation": "higher_is_better",
             "scale_comparable": "true", "cache_identity": "key",
             "do_not_do": "no fallback", "valid_statuses": "valid;missing"},
            {"label_source": "deterministic_continuation", "orientation": "lower_is_better",
             "scale_comparable": "false", "cache_identity": "key",
             "do_not_do": "no fallback", "valid_statuses": "valid;timeout;unsolved;error"},
            {"label_source": "enhsp_raw_h", "orientation": "lower_is_better",
             "scale_comparable": "false", "cache_identity": "key",
             "do_not_do": "no fallback", "valid_statuses": "valid;timeout;unsolved;error"},
        ]
        self.assertEqual(validate_label_source_rows(rows), [])

    def test_rejects_raw_cost_as_scale_comparable(self):
        rows = [
            {"label_source": "replay_target", "orientation": "higher_is_better",
             "scale_comparable": "true", "cache_identity": "key",
             "do_not_do": "no fallback", "valid_statuses": "valid;missing"},
            {"label_source": "deterministic_continuation", "orientation": "lower_is_better",
             "scale_comparable": "true", "cache_identity": "key",
             "do_not_do": "no fallback", "valid_statuses": "valid;timeout;unsolved;error"},
            {"label_source": "enhsp_raw_h", "orientation": "lower_is_better",
             "scale_comparable": "false", "cache_identity": "key",
             "do_not_do": "no fallback", "valid_statuses": "valid;timeout;unsolved;error"},
        ]
        self.assertIn("scale_comparable", "\n".join(validate_label_source_rows(rows)))


if __name__ == "__main__":
    unittest.main()
