import importlib
import sys
import types
import unittest

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
        "post_training.action_selection_policy"):
    sys.modules.pop(_module_name, None)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


class FakeState:
    def __init__(self, key, depth=0, goal=False, terminal=False):
        self.state_key = key.encode("ascii")
        self.depth = depth
        self.is_goal = goal
        self.is_terminal = terminal

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
