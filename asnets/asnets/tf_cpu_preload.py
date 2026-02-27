# tf_cpu_preload.py

import os

# Hide GPU before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# Optional but safe: force CPU visibility
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass