import unittest

from asnets.policy_anchor_trust_region import PolicyAnchorTrustRegion


class PolicyAnchorTrustRegionTests(unittest.TestCase):
    def test_rejects_when_either_frozen_limit_is_exceeded(self):
        guard = PolicyAnchorTrustRegion(
            0.2, 1.5, max_retries=2, learning_rate_factor=0.5)
        self.assertFalse(guard.decide([0.1, 0.2]).excessive)
        self.assertTrue(guard.decide([0.21, 0.21]).excessive)
        self.assertTrue(guard.decide([0.0] * 98 + [2.0, 2.0]).excessive)

    def test_threshold_equality_is_admitted(self):
        guard = PolicyAnchorTrustRegion(0.2, 0.2)
        decision = guard.decide([0.2, 0.2])
        self.assertAlmostEqual(decision.mean_kl, 0.2)
        self.assertAlmostEqual(decision.p99_kl, 0.2)
        self.assertFalse(decision.excessive)

    def test_configuration_validation_and_serialization(self):
        with self.assertRaises(ValueError):
            PolicyAnchorTrustRegion(0.0, 1.0)
        with self.assertRaises(ValueError):
            PolicyAnchorTrustRegion(1.0, 1.0, max_retries=-1)
        with self.assertRaises(ValueError):
            PolicyAnchorTrustRegion(1.0, 1.0, learning_rate_factor=1.0)
        guard = PolicyAnchorTrustRegion(
            0.2123709661, 1.5138580917, 2, 0.5)
        self.assertEqual(guard.to_dict(), {
            "mean_kl_limit": 0.2123709661,
            "p99_kl_limit": 1.5138580917,
            "max_retries": 2,
            "learning_rate_factor": 0.5,
        })

    def test_learning_rate_backtracking_schedule(self):
        guard = PolicyAnchorTrustRegion(0.2, 1.5, 2, 0.5)
        self.assertAlmostEqual(guard.retry_learning_rate(0.0003, 1), 0.00015)
        self.assertAlmostEqual(guard.retry_learning_rate(0.0003, 2), 0.000075)
        with self.assertRaises(ValueError):
            guard.retry_learning_rate(0.0003, 3)


if __name__ == "__main__":
    unittest.main()
