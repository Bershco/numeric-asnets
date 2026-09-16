#!/usr/bin/env python3
"""Compute-node smoke for fixed per-step replay/dropout RNG."""

import tensorflow as tf

from asnets.replay_rng import ReplayRNGSnapshot, gradient_sha256


def one_gradient():
    tf.keras.backend.clear_session()
    tf.random.set_seed(17)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="elu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2),
    ])
    inputs = tf.reshape(tf.range(48, dtype=tf.float32), (8, 6)) / 10.0
    targets = tf.ones((8, 2), dtype=tf.float32)
    model(inputs, training=False)
    for index, variable in enumerate(model.trainable_variables):
        variable.assign(tf.ones_like(variable) * (index + 1) / 100.0)
    model(inputs, training=True)
    snapshot = ReplayRNGSnapshot.capture(tf, [model], 314159)
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=True)
        loss = tf.reduce_mean(tf.square(prediction - targets))
    gradients = tape.gradient(loss, model.trainable_variables)
    return float(loss.numpy()), gradient_sha256(gradients), snapshot.tensorflow_generator_count


def main():
    left = one_gradient()
    right = one_gradient()
    if left[:2] != right[:2]:
        raise SystemExit(
            "Fixed-step RNG smoke failed: matched fresh models did not "
            "reproduce the same stochastic gradient")
    print(
        "[TPP-FIXED-STEP-RNG-SMOKE] matched gradient="
        f"{left[1]}; keras_rng_states={left[2]}")


if __name__ == "__main__":
    main()
