from .actprop_2l_comparison import *

EXPLORATION_ALGORITHM = 'mcts'
ROLLOUTS = 2
EXPLORATION_LEARNING_RATIO = 10
MAX_OPT_EPOCHS = 100
TIME_LIMIT_SECONDS = int(60 * 60 * 8)
# TIME_LIMIT_SECONDS = int(60 * 90)
# USE_NUMERIC_LANDMARKS = False
#TODO: turn off both of the following after single instance overfit test
DROPOUT = 0.0
# TRAINING_LIMIT_TURNS = 5
L2_REG = 0.0
L1_REG = 0.0
USE_NUMERIC_LANDMARKS = False
SUPERVISED_LEARNING_RATE = 0.003  # EXPERIMENTAL

#### Model settings ####
NUM_LAYERS = 3
HIDDEN_SIZE = 32
SKIP = True
