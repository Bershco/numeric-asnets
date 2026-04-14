# tf_cpu_preload.py

import os

# Hide GPU before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 0 = all logs, 1 = filter INFO, 2 = filter INFO/WARNING, 3 = filter INFO/WARNING/ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

import tensorflow as tf

# Optional but safe: force CPU visibility
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass