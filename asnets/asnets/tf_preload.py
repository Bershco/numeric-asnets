# tf_preload.py
import os
# 0 = all logs, 1 = filter INFO, 2 = filter INFO/WARNING, 3 = filter INFO/WARNING/ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

import tensorflow as tf