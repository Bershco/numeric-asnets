#!/usr/bin/env python3
"""Real TensorFlow/Keras smoke for exact stochastic-gradient replay."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from asnets.replay_rng import ReplayRNGSnapshot, gradient_sha256


def optimizer_variables(optimizer):
    variables = optimizer.variables
    return list(variables() if callable(variables) else variables)


def snapshot_training_state(model, optimizer):
    return (
        [variable.numpy().copy() for variable in model.trainable_variables],
        [variable.numpy().copy() for variable in optimizer_variables(optimizer)],
    )


def restore_training_state(model, optimizer, snapshot):
    model_values, optimizer_values = snapshot
    for variable, value in zip(model.trainable_variables, model_values):
        variable.assign(value)
    for variable, value in zip(optimizer_variables(optimizer), optimizer_values):
        variable.assign(value)


def proposal(model, optimizer, inputs, targets):
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=True)
        loss = tf.reduce_mean(tf.square(prediction - targets))
    gradients = tape.gradient(loss, model.trainable_variables)
    digest = gradient_sha256(gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return float(loss.numpy()), digest


def main():
    tf.random.set_seed(17)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="elu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2),
    ])
    inputs = tf.reshape(tf.range(48, dtype=tf.float32), (8, 6)) / 10.0
    targets = tf.ones((8, 2), dtype=tf.float32)
    model(inputs, training=False)
    # Materialize any lazily-created Keras dropout generator before capture.
    model(inputs, training=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    # Materialize Adam slots, then return to an unchanged scientific state.
    optimizer.apply_gradients([
        (tf.zeros_like(variable), variable)
        for variable in model.trainable_variables
    ])
    optimizer.iterations.assign(0)
    state = snapshot_training_state(model, optimizer)
    rng = ReplayRNGSnapshot.capture(tf, [model], 314159)

    first_loss, first_digest = proposal(
        model, optimizer, inputs, targets)
    # Force this proposal to be treated as rejected, then restore every state
    # and retry at the lower learning rate used by the scientific treatment.
    restore_training_state(model, optimizer, state)
    rng.restore(tf)
    optimizer.learning_rate.assign(0.00015)
    retry_loss, retry_digest = proposal(
        model, optimizer, inputs, targets)

    if first_digest != retry_digest or first_loss != retry_loss:
        raise SystemExit(
            "Exact-RNG smoke failed: rejected proposal and retry did not "
            "reproduce the same stochastic gradient")
    print(
        "[TPP-EXACT-RNG-SMOKE] forced rejection reproduced identical "
        f"gradient={first_digest}; keras_rng_states="
        f"{rng.tensorflow_generator_count}")


if __name__ == "__main__":
    main()
