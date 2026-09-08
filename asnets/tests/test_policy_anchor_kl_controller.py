import unittest

from asnets.policy_anchor import PolicyAnchorKLController


class PolicyAnchorKLControllerTests(unittest.TestCase):
    def test_constant_mode_never_changes_coefficient(self):
        controller = PolicyAnchorKLController("constant", 3.0)
        record = controller.observe(0.5)
        self.assertEqual(record["action"], "constant")
        self.assertEqual(controller.coefficient, 3.0)
        self.assertEqual(controller.adjustments, 0)

    def test_adaptive_mode_increases_holds_and_decreases(self):
        controller = PolicyAnchorKLController(
            "adaptive_target", 3.0, target=0.05)
        self.assertEqual(controller.observe(0.08)["action"], "increase")
        self.assertEqual(controller.coefficient, 6.0)
        self.assertEqual(controller.observe(0.05)["action"], "hold")
        self.assertEqual(controller.coefficient, 6.0)
        self.assertEqual(controller.observe(0.03)["action"], "decrease")
        self.assertEqual(controller.coefficient, 3.0)
        self.assertEqual(controller.adjustments, 2)

    def test_bounds_are_enforced(self):
        controller = PolicyAnchorKLController(
            "adaptive_target", 4.0, target=0.05,
            min_coefficient=2.0, max_coefficient=8.0)
        controller.observe(1.0)
        controller.observe(1.0)
        self.assertEqual(controller.coefficient, 8.0)
        controller.observe(0.0)
        controller.observe(0.0)
        controller.observe(0.0)
        self.assertEqual(controller.coefficient, 2.0)

    def test_state_round_trip_and_configuration_guard(self):
        source = PolicyAnchorKLController(
            "adaptive_target", 3.0, target=0.05)
        source.observe(0.08)
        state = source.to_dict()

        resumed = PolicyAnchorKLController(
            "adaptive_target", 3.0, target=0.05)
        self.assertTrue(resumed.restore(state))
        self.assertEqual(resumed.to_dict(), state)

        incompatible = PolicyAnchorKLController(
            "adaptive_target", 3.0, target=0.04)
        with self.assertRaisesRegex(ValueError, "continuation mismatch"):
            incompatible.restore(state)

    def test_adaptive_mode_requires_target_and_positive_coefficient(self):
        with self.assertRaises(ValueError):
            PolicyAnchorKLController("adaptive_target", 3.0)
        with self.assertRaises(ValueError):
            PolicyAnchorKLController("adaptive_target", 0.0, target=0.05)


if __name__ == "__main__":
    unittest.main()
