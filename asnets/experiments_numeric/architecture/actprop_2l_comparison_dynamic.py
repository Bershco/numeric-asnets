from .actprop_2l_comparison import *

EXPLORATION_ALGORITHM = 'dynamic'
ROLLOUTS = 2
# TIME_LIMIT_SECONDS = int(60 * 60 * 8)

#### Action Policy Settings ####
ACTION_POLICY = "sample"
ACTION_POLICY_EPSILON = 0.05
ACTION_POLICY_TEMPERATURE = None
ACTION_POLICY_DECAY_RATE = None