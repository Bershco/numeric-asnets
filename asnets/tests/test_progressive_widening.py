import importlib
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def _install_lightweight_import_stubs():
    """Keep these MCTS unit tests independent of JPDDL/TensorFlow installs."""
    module_names = (
        "rpyc",
        "asnets.state_reprs",
        "asnets.spawn_context",
        "asnets.utils.pddl_utils",
        "post_training.enhspwrapper",
        "tensorflow",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    if "rpyc" not in sys.modules:
        rpyc = types.ModuleType("rpyc")
        rpyc.BaseNetref = type("BaseNetref", (), {})
        sys.modules["rpyc"] = rpyc

    state_reprs = types.ModuleType("asnets.state_reprs")
    state_reprs.CanonicalState = type("CanonicalState", (), {})
    sys.modules["asnets.state_reprs"] = state_reprs

    spawn_context = types.ModuleType("asnets.spawn_context")
    spawn_context.LocalExploreContext = type("LocalExploreContext", (), {})
    sys.modules["asnets.spawn_context"] = spawn_context

    pddl_utils = types.ModuleType("asnets.utils.pddl_utils")
    pddl_utils.replace_init_state = lambda *args: None
    pddl_utils.hlist_to_sexprs = lambda *args: ""
    sys.modules["asnets.utils.pddl_utils"] = pddl_utils

    enhspwrapper = types.ModuleType("post_training.enhspwrapper")
    enhspwrapper.EstimatorMode = types.SimpleNamespace(
        V_ONLY="v_only", BOTH="both")
    sys.modules["post_training.enhspwrapper"] = enhspwrapper

    tensorflow = types.ModuleType("tensorflow")
    tensorflow.stack = np.stack
    tensorflow.squeeze = np.squeeze
    tensorflow.float32 = np.float32
    sys.modules["tensorflow"] = tensorflow

    return saved_modules


_saved_modules = _install_lightweight_import_stubs()

search = importlib.import_module("post_training.monte_carlo_tree_search")
training = importlib.import_module("post_training.training_mcts")
policies = importlib.import_module("post_training.action_selection_policy")
recorder = importlib.import_module("asnets.mcts_first_divergence")

# Keep the lightweight imports local to this test module.  A larger test run
# may subsequently need to import these modules against the real dependencies.
for _module_name, _saved_module in _saved_modules.items():
    if _saved_module is None:
        sys.modules.pop(_module_name, None)
    else:
        sys.modules[_module_name] = _saved_module
for _module_name in (
        "post_training.monte_carlo_tree_search",
        "post_training.training_mcts",
        "post_training.action_selection_policy",
        "asnets.mcts_first_divergence"):
    sys.modules.pop(_module_name, None)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


class FakeState:
    def __init__(self, key, depth=0, goal=False, terminal=False,
                 action_counts=None, unrelated_aux=0.0):
        self.state_key = key.encode("ascii")
        self.depth = depth
        self.is_goal = goal
        self.is_terminal = terminal
        self.action_counts = np.asarray(
            [0.0, 0.0, 0.0, 0.0]
            if action_counts is None else action_counts,
            dtype=np.float32,
        )
        self.unrelated_aux = unrelated_aux

    def get_aux_data_dimension(self, dim_name):
        if dim_name != "action_count":
            raise KeyError(dim_name)
        return self.action_counts

    def to_network_input(self):
        return np.asarray([self.depth], dtype=np.float32)

    def get_applicable_action_mask(self):
        return np.ones(4, dtype=bool)

    def to_tup_state(self):
        return (self.depth,)


class FakeNetwork:
    value_head_enabled = True
    value_network_enabled = True

    def __call__(self, inputs, training=False):
        batch = np.asarray(inputs)
        batch_size = 1 if batch.ndim == 1 else batch.shape[0]
        policies_out = np.tile(
            np.asarray([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32),
            (batch_size, 1),
        )
        values_out = np.full((batch_size, 1), 0.5, dtype=np.float32)
        return FakeTensor(policies_out), FakeTensor(values_out)


class FakeContext:
    estimator = None
    estimator_h_to_v_coeff = 1.0

    def get_act_dim(self):
        return 4

    def env_simulate_batch_steps(self, state, actions):
        results = []
        for action in actions:
            child = FakeState(
                f"{state.depth + 1}:{int(action)}",
                depth=state.depth + 1,
            )
            results.append((
                int(action), child, 1.0, False, False,
                child.to_network_input(),
                child.get_applicable_action_mask(),
            ))
        return results


def make_node(state, policy=(0.4, 0.3, 0.2, 0.1)):
    node = search.wrapInMCTSNode(state, cost_until_now=state.depth)
    node.act_dist = np.asarray(policy, dtype=np.float32)
    node.pred_value = 0.25
    return node


class FixedChildMapTests(unittest.TestCase):
    def test_append_preserves_existing_edge_statistics(self):
        first = make_node(FakeState("first"))
        second = make_node(FakeState("second"))
        children = search.FixedChildMap([2], [first], [0.4])
        children.increment_visit(2)

        children.append(1, second, 0.3)

        self.assertEqual(list(children.keys()), [1, 2])
        self.assertEqual(children[1], second)
        self.assertEqual(children[2], first)
        self.assertEqual(children.visits.tolist(), [0, 1])
        self.assertEqual(children.priors.dtype, np.float32)


class ContextIdentityTests(unittest.TestCase):
    def test_digest_uses_action_count_only(self):
        first = FakeState(
            "same", action_counts=[1, 2, 3, 4], unrelated_aux=10)
        second = FakeState(
            "same", action_counts=[1, 2, 3, 4], unrelated_aux=999)
        changed = FakeState(
            "same", action_counts=[1, 2, 4, 4], unrelated_aux=10)

        self.assertEqual(
            search.action_history_digest(first),
            search.action_history_digest(second),
        )
        self.assertNotEqual(
            search.action_history_digest(first),
            search.action_history_digest(changed),
        )

    def test_contextual_registry_separates_histories(self):
        mcts = training.TrainingMCTS(
            FakeNetwork(), FakeContext(), contextual_nodes=True)
        first = FakeState("same", action_counts=[1, 0, 0, 0])
        second = FakeState("same", action_counts=[0, 1, 0, 0])
        self.assertNotEqual(
            mcts.node_registry_key(first), mcts.node_registry_key(second))
        self.assertEqual(
            mcts.node_registry_key(first)[0],
            mcts.node_registry_key(second)[0],
        )

    def test_contextual_cycle_mask_uses_physical_state(self):
        mcts = training.TrainingMCTS(
            FakeNetwork(), FakeContext(), contextual_nodes=True)
        parent = make_node(FakeState("same", action_counts=[1, 0, 0, 0]))
        child = make_node(FakeState("same", action_counts=[2, 0, 0, 0]))
        parent.children = search.FixedChildMap([0], [child], [1.0])
        mcts._select_counter = 1
        action, selected = mcts._puct_select_argmax(
            parent, forbidden_physical_keys={parent.state_key})
        self.assertIsNone(action)
        self.assertIsNone(selected)


class ProgressiveWideningTests(unittest.TestCase):
    def setUp(self):
        self.root = make_node(FakeState("root"))
        self.mcts = training.TrainingMCTS(
            network=FakeNetwork(),
            ctx=FakeContext(),
            iterations=1,
            expansion_k=20,
            progressive_widening=True,
            pw_min_width=2,
            pw_c=0.6,
            pw_alpha=0.5,
        )
        self.mcts.curr_tree_root = self.root
        self.mcts.original_tree_root = self.root
        self.mcts.state_key_to_node[self.root.state_key] = self.root

    def test_initial_expansion_adds_two_and_evaluates_generated_child(self):
        self.mcts.run_search()

        self.assertEqual(len(self.root.children), 2)
        self.assertEqual(list(self.root.children.keys()), [0, 1])
        self.assertEqual(self.root.visit_count, 1)
        evaluated_children = [
            child for child in self.root.children.values()
            if child.visit_count == 1
        ]
        self.assertEqual(len(evaluated_children), 1)
        self.assertAlmostEqual(evaluated_children[0].Q_value, 0.5)

    def test_widening_adds_one_policy_ordered_child(self):
        self.mcts.run_search()
        self.root.visit_count = 25

        self.mcts.mcts_iteration_value_based(self.root)

        self.assertEqual(len(self.root.children), 3)
        self.assertEqual(list(self.root.children.keys()), [0, 1, 2])

    def test_schedule_starts_at_two_and_reaches_five_at_seventy(self):
        expected = {1: 2, 10: 2, 25: 3, 50: 4, 70: 5}
        for visits, width in expected.items():
            self.root.visit_count = visits
            self.assertEqual(self.mcts._progressive_width(self.root), width)

    def test_horizon_prevents_expansion_beyond_remaining_depth(self):
        self.mcts.iterations = 4

        self.mcts.run_search(remaining_horizon=1)

        self.assertEqual(len(self.root.children), 2)
        self.assertTrue(all(
            child.children is None
            for child in self.root.children.values()
        ))
        self.assertGreater(
            sum(self.mcts.horizon_cutoff_depth_hist.values()), 0)

    def test_two_step_remaining_horizon_records_grandchild_cutoff(self):
        """A search at external step max_len-2 must count depth-2 visits."""
        self.mcts.iterations = 20

        self.mcts.run_search(remaining_horizon=2)

        self.assertGreater(
            self.mcts.horizon_cutoff_depth_hist[2], 0,
            "a non-terminal grandchild reached at the two-step boundary must "
            "be recorded as a horizon cutoff",
        )
        self.assertEqual(
            sum(
                count for depth, count
                in self.mcts.horizon_cutoff_depth_hist.items()
                if depth > 2
            ),
            0,
        )

    def test_invalid_widening_parameters_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "pw_min_width"):
            training.TrainingMCTS(
                FakeNetwork(), FakeContext(), expansion_k=2,
                pw_min_width=3)

    def test_compact_distribution_helpers(self):
        counts = training.Counter({0: 5, 2: 3, 5: 1})
        self.assertEqual(
            self.mcts._compact_hist(counts), "0=5,2=3,5=1")
        self.assertEqual(self.mcts._visit_bucket(0), "0")
        self.assertEqual(self.mcts._visit_bucket(4), "3-4")
        self.assertEqual(self.mcts._visit_bucket(65), "65-128")
        self.assertEqual(self.mcts._visit_bucket(129), "129+")


class FixedExpansionRegressionTests(unittest.TestCase):
    def test_fixed_expansion_keeps_parent_leaf_evaluation_semantics(self):
        root = make_node(FakeState("fixed-root"))
        mcts = training.TrainingMCTS(
            network=FakeNetwork(),
            ctx=FakeContext(),
            iterations=1,
            expansion_k=2,
            progressive_widening=False,
        )
        mcts.curr_tree_root = root
        mcts.original_tree_root = root
        mcts.state_key_to_node[root.state_key] = root

        mcts.run_search()

        self.assertEqual(len(root.children), 2)
        self.assertEqual(root.visit_count, 1)
        self.assertTrue(all(
            child.visit_count == 0
            for child in root.children.values()
        ))


class HorizonTests(unittest.TestCase):
    def test_selection_stops_at_remaining_edge_depth(self):
        root = make_node(FakeState("root"))
        child = make_node(FakeState("child", depth=1))
        grandchild = make_node(FakeState("grandchild", depth=2))
        root.children = search.FixedChildMap([0], [child], [1.0])
        child.children = search.FixedChildMap([0], [grandchild], [1.0])
        mcts = search.MCTS()

        path, reason = mcts._select(root, max_depth=1)

        self.assertEqual(path, [root, child])
        self.assertEqual(reason, "horizon")

    def test_default_is_maximization(self):
        self.assertEqual(search.MCTS().sign, 1)

    def test_goal_chase_respects_remaining_horizon(self):
        root = make_node(FakeState("root"))
        goal_child = make_node(FakeState("goal", goal=True))
        other = make_node(FakeState("other"))
        root.children = search.FixedChildMap(
            [0, 1], [goal_child, other], [0.4, 0.6])
        root.known_distance_to_goal = 5
        root.best_goal_child = goal_child
        mcts = types.SimpleNamespace(curr_tree_root=root)
        policy = policies.build_action_policy(
            "argmax", distance_threshold=np.inf)

        self.assertEqual(
            policy.select_action(
                mcts, np.asarray([0.1, 0.9]), remaining_horizon=5),
            0,
        )
        self.assertEqual(
            policy.select_action(
                mcts, np.asarray([0.1, 0.9]), remaining_horizon=4),
            1,
        )


class RootVisitTieBreakTests(unittest.TestCase):
    @staticmethod
    def _root():
        root = make_node(FakeState("root"))
        children = [
            make_node(FakeState("a")),
            make_node(FakeState("b")),
            make_node(FakeState("c")),
        ]
        children[0].Q_value = 0.1
        children[1].Q_value = 0.7
        children[2].Q_value = 0.4
        root.children = search.FixedChildMap(
            [0, 1, 2], children, [0.2, 0.3, 0.5])
        root.act_dist = np.asarray([0.2, 0.6, 0.2])
        return root

    @staticmethod
    def _mcts(root, minimization=False):
        return types.SimpleNamespace(
            curr_tree_root=root,
            minimization=minimization,
            sign=-1 if minimization else 1,
        )

    def test_historical_mode_uses_lowest_action_id(self):
        root = self._root()
        policy = policies.build_action_policy(
            "argmax", root_visit_tie_break="action_id")
        self.assertEqual(
            policy.select_action(self._mcts(root), np.asarray([0.5, 0.5, 0.])),
            0,
        )

    def test_q_mode_uses_sign_correct_q(self):
        root = self._root()
        policy = policies.build_action_policy(
            "argmax", root_visit_tie_break="q")
        pi = np.asarray([0.5, 0.5, 0.])
        self.assertEqual(policy.select_action(self._mcts(root), pi), 1)
        self.assertEqual(
            policy.select_action(self._mcts(root, minimization=True), pi),
            0,
        )

    def test_policy_mode_uses_root_network_prior(self):
        root = self._root()
        policy = policies.build_action_policy(
            "argmax", root_visit_tie_break="policy")
        self.assertEqual(
            policy.select_action(self._mcts(root), np.asarray([0.5, 0.5, 0.])),
            1,
        )

    def test_goal_chase_retains_precedence(self):
        root = self._root()
        root.known_distance_to_goal = 1
        root.best_goal_child = root.children[2]
        policy = policies.build_action_policy(
            "argmax", distance_threshold=np.inf,
            root_visit_tie_break="policy")
        self.assertEqual(
            policy.select_action(self._mcts(root), np.asarray([0.5, 0.5, 0.])),
            2,
        )

    def test_opt_in_trace_records_goal_and_policy_tie_break_path(self):
        root = self._root()
        policy = policies.build_action_policy(
            "argmax",
            distance_threshold=np.inf,
            root_visit_tie_break="policy",
            selection_trace_enabled=True,
        )
        policy.begin_selection_trace()

        selected = policy.select_action(
            self._mcts(root), np.asarray([0.5, 0.5, 0.]))
        trace = policy.consume_selection_trace()

        self.assertEqual(selected, 1)
        self.assertEqual(
            [entry["stage"] for entry in trace],
            ["goal_chase", "root_visit_argmax"],
        )
        self.assertFalse(trace[0]["applied"])
        self.assertTrue(trace[1]["tie_break_applied"])
        self.assertEqual(trace[1]["selected_action"], 1)


class FirstDivergenceRecorderTests(unittest.TestCase):
    @staticmethod
    def _mcts(root):
        return types.SimpleNamespace(
            curr_tree_root=root,
            sign=1,
            exploration_weight=1.0,
            selection_seconds=0.1,
            successor_generation_seconds=0.2,
            network_inference_seconds=0.3,
            evaluation_seconds=0.4,
            backpropagation_seconds=0.5,
            selection_depth_hist={0: 2, 1: 1},
        )

    def test_entropy_and_js_are_normalized_and_symmetric(self):
        self.assertAlmostEqual(recorder.entropy([2.0, 2.0]), np.log(2.0))
        left_right = recorder.jensen_shannon_divergence([1, 0], [0, 1])
        right_left = recorder.jensen_shannon_divergence([0, 1], [1, 0])
        self.assertAlmostEqual(left_right, np.log(2.0))
        self.assertAlmostEqual(left_right, right_left)

    def test_complete_vectors_keep_unexpanded_actions_explicit(self):
        root = make_node(FakeState("record-root"), policy=(0.6, 0.1, 0.3, 0.0))
        first = make_node(FakeState("child-zero"))
        third = make_node(FakeState("child-two"))
        first.Q_value = 0.2
        third.Q_value = 0.8
        root.children = search.FixedChildMap(
            [0, 2], [first, third], [0.6, 0.3])
        root.children.visits[:] = [4, 6]
        root.visit_count = 10
        root.state.acts_enabled = tuple(
            (types.SimpleNamespace(unique_ident=f"action-{action}"), True)
            for action in range(4)
        )

        result = recorder.build_first_divergence_record(
            self._mcts(root),
            visit_policy=[0.4, 0.0, 0.6, 0.0],
            selected_action=2,
            step=3,
            instance_name="fixture.pddl",
            elapsed_seconds=1.5,
            override_path=[{
                "stage": "root_visit_argmax", "selected_action": 2}],
            provenance={"checkpoint_path": "fixture.ckpt", "seed": 7},
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["policy_action"], 0)
        self.assertEqual(result["selected_action"], 2)
        self.assertEqual(
            result["root"]["action_names"],
            ["action-0", "action-1", "action-2", "action-3"],
        )
        self.assertEqual(
            result["root"]["expanded"], [True, False, True, False])
        self.assertEqual(
            result["root"]["edge_visit_counts"], [4, 0, 6, 0])
        self.assertEqual(
            result["root"]["q_values"], [0.2, None, 0.8, None])
        self.assertAlmostEqual(result["root"]["edge_priors"][0], 0.6)
        self.assertIsNone(result["root"]["edge_priors"][1])
        self.assertAlmostEqual(result["root"]["edge_priors"][2], 0.3)
        self.assertIsNone(result["root"]["edge_priors"][3])
        self.assertEqual(result["summary"]["max_visit_tie_count"], 1)
        self.assertEqual(result["summary"]["signed_q_argmax"], 2)
        self.assertEqual(
            result["selection_depth_histogram"], {"0": 2, "1": 1})

    def test_matching_policy_and_search_emits_nothing(self):
        root = make_node(FakeState("same-root"))
        child = make_node(FakeState("same-child"))
        root.children = search.FixedChildMap([0], [child], [0.4])
        root.children.visits[:] = [1]
        root.visit_count = 1

        result = recorder.build_first_divergence_record(
            self._mcts(root),
            visit_policy=[1.0, 0.0, 0.0, 0.0],
            selected_action=0,
            step=0,
            instance_name="fixture.pddl",
            elapsed_seconds=None,
            override_path=[],
            provenance={},
        )
        self.assertIsNone(result)

    def test_existing_counters_trace_fixture_reproduces_root(self):
        fixture_path = (
            Path(__file__).resolve().parents[2]
            / "experiment_tracking"
            / "mcts_policy_divergence_cause_audit"
            / "fixtures"
            / "counters_known_divergence_src20430427_e0000_instance51_step1.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        act_dim = 1 + max(row["action"] for row in fixture["children"])
        raw_policy = np.zeros(act_dim, dtype=np.float32)
        root = search.wrapInMCTSNode(
            FakeState("known-counters-root"), cost_until_now=0)
        root.applicable_action_mask = np.ones(act_dim, dtype=bool)
        actions, children, priors = [], [], []
        for row in fixture["children"]:
            action = row["action"]
            child = make_node(FakeState(f"known-child-{action}"))
            child.Q_value = row["Q"]
            raw_policy[action] = row["prior"]
            actions.append(action)
            children.append(child)
            priors.append(row["prior"])
        root.act_dist = raw_policy
        root.children = search.FixedChildMap(actions, children, priors)
        root.children.visits[:] = [
            row["N"] for row in fixture["children"]]
        root.visit_count = fixture["root_visits"]
        visit_policy = np.zeros(act_dim, dtype=np.float64)
        for row in fixture["children"]:
            visit_policy[row["action"]] = row["N"]

        fixture_mcts = self._mcts(root)
        fixture_mcts.exploration_weight = fixture["exploration_weight"]
        result = recorder.build_first_divergence_record(
            fixture_mcts,
            visit_policy=visit_policy,
            selected_action=fixture["selected_action"],
            step=fixture["step"],
            instance_name=fixture["instance"],
            elapsed_seconds=None,
            override_path=[],
            provenance={
                "checkpoint_identity": fixture["checkpoint_identity"],
                "source_job": fixture["source_job"],
            },
        )

        self.assertEqual(result["policy_action"], 50)
        self.assertEqual(result["selected_action"], 52)
        self.assertEqual(result["root"]["root_visits"], 25)
        self.assertEqual(result["root"]["total_edge_visits"], 24)
        for row in fixture["children"]:
            self.assertAlmostEqual(
                result["root"]["u_values"][row["action"]], row["U"])


class TerminalSafeActionSelectionTests(unittest.TestCase):
    @staticmethod
    def _mcts(root):
        return types.SimpleNamespace(
            curr_tree_root=root,
            minimization=False,
            exploration_weight=0.1,
        )

    def test_known_terminal_is_masked_when_safe_child_exists(self):
        root = make_node(FakeState("safe-root"))
        terminal = make_node(FakeState("terminal", terminal=True))
        safe = make_node(FakeState("safe"))
        root.children = search.FixedChildMap(
            [0, 1], [terminal, safe], [0.9, 0.1])
        policy = policies.build_action_policy(
            "argmax", duplicate_penalty=0.0, terminal_safe=True)

        selected = policy.select_action(
            self._mcts(root), np.asarray([0.9, 0.1]))

        self.assertEqual(selected, 1)
        self.assertEqual(policy.terminal_actions_excluded, 1)

    def test_trace_retains_original_and_transformed_selector_inputs(self):
        root = make_node(FakeState("safe-trace-root"))
        terminal = make_node(FakeState("terminal", terminal=True))
        safe = make_node(FakeState("safe"))
        root.children = search.FixedChildMap(
            [0, 1], [terminal, safe], [0.9, 0.1])
        policy = policies.build_action_policy(
            "argmax", duplicate_penalty=0.0, terminal_safe=True,
            selection_trace_enabled=True)
        policy.begin_selection_trace()

        selected = policy.select_action(
            self._mcts(root), np.asarray([0.9, 0.1]))
        trace = policy.consume_selection_trace()

        self.assertEqual(selected, 1)
        self.assertEqual(trace[0]["stage"], "terminal_safe")
        self.assertEqual(trace[0]["post_terminal_policy"], [0.0, 1.0])
        self.assertEqual(trace[1]["stage"], "root_visit_argmax")
        self.assertEqual(
            trace[1]["selector_input_distribution"], [0.0, 1.0])

    def test_safe_duplicate_is_restored_before_terminal_fallback(self):
        root = make_node(FakeState("duplicate-root"))
        terminal = make_node(FakeState("terminal", terminal=True))
        safe_duplicate = make_node(FakeState("safe-duplicate"))
        safe_duplicate.on_trajectory = True
        root.children = search.FixedChildMap(
            [0, 1], [terminal, safe_duplicate], [0.9, 0.1])
        policy = policies.build_action_policy(
            "argmax", duplicate_penalty=0.0, terminal_safe=True)

        selected = policy.select_action(
            self._mcts(root), np.asarray([0.9, 0.1]))

        self.assertEqual(selected, 1)
        self.assertEqual(policy.safe_duplicate_fallbacks, 1)

    def test_invalid_modern_q_value_fails_with_invariant(self):
        root = make_node(FakeState("invalid-q-root"))
        child = make_node(FakeState("child"))
        child.Q_value = -0.25
        root.children = search.FixedChildMap([0], [child], [1.0])
        policy = policies.build_action_policy(
            "argmax", duplicate_penalty=0.0, terminal_safe=True)

        with self.assertRaisesRegex(ValueError, "value convention"):
            policy.select_action(self._mcts(root), np.asarray([1.0]))

    def test_all_terminal_root_forces_safe_action_admission(self):
        root = make_node(FakeState("force-root"))
        terminal = make_node(FakeState("terminal", terminal=True))
        root.children = search.FixedChildMap([0], [terminal], [0.4])
        mcts = training.TrainingMCTS(
            network=FakeNetwork(),
            ctx=FakeContext(),
            iterations=1,
            expansion_k=1,
            pw_min_width=1,
            progressive_widening=False,
        )
        mcts.curr_tree_root = root
        mcts.original_tree_root = root
        mcts.state_key_to_node[root.state_key] = root

        self.assertTrue(mcts.ensure_safe_root_child())
        self.assertGreater(len(root.children), 1)
        self.assertTrue(any(
            not child.terminal_state for child in root.children.values()))


if __name__ == "__main__":
    unittest.main()
