from __future__ import annotations

import unittest

import numpy as np

from asnets.value_head_audit import (
    CallableLabelProvider,
    ENHSPSearchValueProvider,
    LabelResult,
    MappingLabelProvider,
    apply_label_provider,
    canonical_state_record,
    enhsp_search_value,
    evaluate_successor_values,
    rows_as_dicts,
    validate_canonical_state_record,
    validate_task_manifest_rows,
)


class _State:
    def __init__(self, obs, *, mask=None, terminal=False, goal=False):
        self._obs = np.asarray(obs, dtype=np.float32)
        self._mask = None if mask is None else np.asarray(mask, dtype=bool)
        self.is_terminal = terminal
        self.is_goal = goal

    def get_applicable_action_mask(self):
        return self._mask

    def to_network_input(self):
        return self._obs


class _ValueNetwork:
    def __call__(self, observations, *, training):
        assert training is False
        values = np.sum(observations, axis=1, keepdims=True)
        policy = np.zeros((len(observations), 3), dtype=np.float32)
        return policy, values


class _SerializableState:
    def __init__(self):
        class _Named:
            def __init__(self, name):
                self.unique_ident = name

        self.aux_data = np.asarray([0.0, 1.0], dtype=np.float32)
        self._aux_data_interp = ["is_enabled", "action_count"]
        self.is_terminal = False
        self.is_goal = False
        self.props_true = ((_Named("at rover waypoint0"), True),)
        self.flnt_values = (
            (_Named("fuel rover"), 3.5),
            (_Named("total-cost"), 7.0),
        )

    def to_tup_state(self):
        return (("at rover waypoint0",), (("fuel rover", 3.5),))

    def to_network_input(self):
        return np.asarray([1.0, 0.0, 1.0, 3.5], dtype=np.float32)


class ValueHeadAuditTest(unittest.TestCase):
    def test_canonical_state_hash_excludes_capture_metadata_and_detects_mutation(self):
        state = _SerializableState()
        first = canonical_state_record(state, instance_name="p0", step=1)
        second = canonical_state_record(state, instance_name="p1", step=9)
        self.assertEqual(first["fluents"][-1], ["total-cost", 7.0])
        self.assertEqual(first["state_sha256"], second["state_sha256"])
        validate_canonical_state_record(first)
        first["aux_data"][0] = 1.0
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_canonical_state_record(first)

    def test_batches_all_applicable_successors_and_preserves_identity(self):
        root = _State([0], mask=[True, False, True])
        successors = {
            0: [(1.0, _State([1, 2], terminal=False, goal=False))],
            2: [
                (0.25, _State([3, 4], terminal=True, goal=False)),
                (0.75, _State([5, 6], terminal=True, goal=True)),
            ],
        }
        rows = evaluate_successor_values(
            state=root,
            state_id="state-7",
            network=_ValueNetwork(),
            successor_fn=lambda _state, action: successors[action],
        )
        self.assertEqual([row.action_id for row in rows], [0, 2, 2])
        self.assertEqual([row.successor_index for row in rows], [0, 0, 1])
        self.assertEqual([row.raw_network_value for row in rows], [3.0, 7.0, 11.0])
        self.assertAlmostEqual(sum(row.transition_probability for row in rows[1:]), 1.0)
        self.assertTrue(rows[-1].successor_goal)
        self.assertEqual(rows_as_dicts(rows)[0]["state_id"], "state-7")

    def test_rejects_invalid_transition_distribution(self):
        root = _State([0], mask=[True])
        with self.assertRaisesRegex(ValueError, "probabilities sum"):
            evaluate_successor_values(
                state=root,
                state_id="bad",
                network=_ValueNetwork(),
                successor_fn=lambda _state, _action: [(0.5, _State([1]))],
            )

    def test_requires_value_head_output(self):
        root = _State([0], mask=[True])
        with self.assertRaisesRegex(ValueError, "VH-on"):
            evaluate_successor_values(
                state=root,
                state_id="novh",
                network=lambda observations, training: np.zeros((len(observations), 1)),
                successor_fn=lambda _state, _action: [(1.0, _State([1]))],
            )

    def test_mapping_provider_keeps_missing_separate_from_numeric_values(self):
        rows = evaluate_successor_values(
            state=_State([0], mask=[True, True]),
            state_id="s",
            network=_ValueNetwork(),
            successor_fn=lambda _state, action: [(1.0, _State([action, 1]))],
        )
        provider = MappingLabelProvider(
            label_source="replay_target",
            values={("s", 0, 0): 0.75},
            higher_is_better=True,
            scale_comparable=True,
            label_log_path="cache/replay.csv",
        )
        labelled = apply_label_provider(rows, provider)
        self.assertEqual(labelled[0]["label_status"], "valid")
        self.assertEqual(labelled[0]["label_value"], 0.75)
        self.assertEqual(labelled[1]["label_status"], "missing")
        self.assertIsNone(labelled[1]["label_value"])

    def test_timeout_cannot_be_encoded_as_numeric_label(self):
        rows = evaluate_successor_values(
            state=_State([0], mask=[True]),
            state_id="s",
            network=_ValueNetwork(),
            successor_fn=lambda _state, _action: [(1.0, _State([1]))],
        )
        provider = CallableLabelProvider(
            lambda _row: LabelResult(
                label_source="deterministic_continuation",
                label_status="timeout",
                label_value=10_000.0,
                label_higher_is_better=False,
                label_scale_comparable=False,
                label_log_path="logs/continuation.txt",
            )
        )
        with self.assertRaisesRegex(ValueError, "must not carry"):
            apply_label_provider(rows, provider)

    def test_valid_label_requires_finite_value_and_log(self):
        rows = evaluate_successor_values(
            state=_State([0], mask=[True]),
            state_id="s",
            network=_ValueNetwork(),
            successor_fn=lambda _state, _action: [(1.0, _State([1]))],
        )
        provider = CallableLabelProvider(
            lambda _row: LabelResult(
                label_source="enhsp_raw_h",
                label_status="valid",
                label_value=float("nan"),
                label_higher_is_better=False,
                label_scale_comparable=False,
                label_log_path="logs/enhsp.txt",
            )
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            apply_label_provider(rows, provider)

    def test_manifest_preflight_requires_hashes_and_exact_factorial(self):
        row = {
            "task_id": "vhv1-drone-1-stage1",
            "domain": "drone",
            "seed": "1",
            "stage": "stage1",
            "checkpoint_path": "/cluster/checkpoint",
            "checkpoint_sha256": "a" * 64,
            "state_manifest_path": "states.jsonl",
            "state_manifest_sha256": "b" * 64,
            "source_manifest_sha256": "c" * 64,
            "state_sources": "common_planner+common_random_legal+stage1_on_policy+stage2_on_policy",
            "label_sources": "replay_target+deterministic_continuation+enhsp_raw_h+enhsp_search_v",
            "cpus": "4",
            "memory_gib": "48",
            "time_limit_hours": "8",
        }
        errors = validate_task_manifest_rows([row], domains=["drone"], seeds=["1"])
        self.assertIn("missing task identities", "\n".join(errors))
        stage2 = dict(row, task_id="vhv1-drone-1-stage2", stage="stage2")
        self.assertEqual(
            validate_task_manifest_rows([row, stage2], domains=["drone"], seeds=["1"]),
            [],
        )
        bad = dict(stage2, checkpoint_sha256="")
        self.assertIn("checkpoint_sha256", "\n".join(
            validate_task_manifest_rows([row, bad], domains=["drone"], seeds=["1"])
        ))

    def test_search_enhsp_transform_matches_deployed_formula(self):
        self.assertAlmostEqual(enhsp_search_value(0.0), 1.0)
        self.assertAlmostEqual(enhsp_search_value(2.0), np.exp(-2.0))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            enhsp_search_value(-1.0)

    def test_search_enhsp_provider_transforms_only_valid_raw_h(self):
        rows = evaluate_successor_values(
            state=_State([0], mask=[True, True]),
            state_id="s",
            network=_ValueNetwork(),
            successor_fn=lambda _state, action: [(1.0, _State([action, 1]))],
        )
        raw = MappingLabelProvider(
            label_source="enhsp_raw_h",
            values={("s", 0, 0): 2.0},
            higher_is_better=False,
            scale_comparable=False,
            label_log_path="cache/enhsp.csv",
        )
        labelled = apply_label_provider(rows, ENHSPSearchValueProvider(raw))
        self.assertAlmostEqual(labelled[0]["label_value"], np.exp(-2.0))
        self.assertTrue(labelled[0]["label_scale_comparable"])
        self.assertEqual(labelled[1]["label_status"], "missing")
        self.assertIsNone(labelled[1]["label_value"])


if __name__ == "__main__":
    unittest.main()
