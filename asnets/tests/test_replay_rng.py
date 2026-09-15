import random
import unittest

import numpy as np

from asnets.replay_rng import (
    ReplayRNGSnapshot,
    gradient_sha256,
    replay_step_seed,
    tensorflow_rng_variables,
)


class _Variable:
    def __init__(self, value):
        self.value = np.asarray(value).copy()

    def numpy(self):
        return self.value

    def assign(self, value):
        self.value = np.asarray(value).copy()


class _Generator:
    def __init__(self, value):
        self.state = _Variable(value)


class _Random:
    def __init__(self, generator):
        self.generator = generator
        self.seeds = []

    def set_seed(self, seed):
        self.seeds.append(seed)

    def get_global_generator(self):
        return self.generator


class _TF:
    def __init__(self, generator):
        self.random = _Random(generator)


class _Layer:
    def __init__(self, generator):
        self._random_generator = type("Wrapper", (), {"_generator": generator})()
        self.layers = []


class ReplayRNGTests(unittest.TestCase):
    def test_snapshot_restores_all_rng_sources(self):
        global_generator = _Generator([1, 2])
        layer_generator = _Generator([3, 4])
        tf_module = _TF(global_generator)
        layer = _Layer(layer_generator)
        random.seed(11)
        np.random.seed(12)
        snapshot = ReplayRNGSnapshot.capture(tf_module, [layer], 123)
        expected_python = random.random()
        expected_numpy = np.random.random()
        global_generator.state.assign([9, 9])
        layer_generator.state.assign([8, 8])
        snapshot.restore(tf_module)
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)
        np.testing.assert_array_equal(global_generator.state.numpy(), [1, 2])
        np.testing.assert_array_equal(layer_generator.state.numpy(), [3, 4])
        self.assertEqual(tf_module.random.seeds, [123, 123])
        self.assertEqual(snapshot.tensorflow_generator_count, 2)

    def test_generator_discovery_deduplicates_state(self):
        shared = _Generator([1])
        tf_module = _TF(shared)
        variables = tensorflow_rng_variables(tf_module, [_Layer(shared)])
        self.assertEqual(len(variables), 1)

    def test_step_seed_is_stable_and_step_specific(self):
        self.assertEqual(replay_step_seed(314159, 0), 314159)
        self.assertEqual(replay_step_seed(314159, 1), 1314162)
        self.assertNotEqual(
            replay_step_seed(314159, 1), replay_step_seed(314159, 2))
        with self.assertRaises(ValueError):
            replay_step_seed(-1, 0)

    def test_gradient_digest_is_order_and_value_sensitive(self):
        left = gradient_sha256([
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([3.0], dtype=np.float32),
        ])
        self.assertEqual(left, gradient_sha256([
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([3.0], dtype=np.float32),
        ]))
        self.assertNotEqual(left, gradient_sha256([
            np.asarray([2.0, 1.0], dtype=np.float32),
            np.asarray([3.0], dtype=np.float32),
        ]))


if __name__ == "__main__":
    unittest.main()
