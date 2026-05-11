import tensorflow as tf
from asnets.supervised import ManualLoss, SupervisedObjective

# Dummy model producing policy + value from params
class MiniNet(tf.keras.Model):
    def __init__(self, act_dim):
        super().__init__()
        self.dense = tf.keras.layers.Dense(act_dim, activation="softmax")
        self.value = tf.keras.layers.Dense(1, activation="tanh")

    def call(self, x):
        return self.dense(x), self.value(x)

net = MiniNet(act_dim=4)
loss_fn = ManualLoss([], net.trainable_variables, None, 0, 0, 0,
                     SupervisedObjective.MCTS_VISIT_DIST)

x = tf.random.normal((3, 5))
pi_targets = tf.nn.softmax(tf.random.normal((3, 4)), axis=-1)
z_targets = tf.random.uniform((3, 1), -1, 1)

with tf.GradientTape() as tape:
    policy_pred, value_pred = net(x)
    loss = loss_fn([policy_pred], [pi_targets],
                   target_values=[z_targets], pred_values=[value_pred])
grads = tape.gradient(loss, net.trainable_variables)

print("Loss:", float(loss))
print("Any None grads?", any(g is None for g in grads))
