"""Reduced optimizer-step count for the compute-only first-update smoke."""

from .tpp_mcts import *

OPT_BATCH_PER_EPOCH = 1
SAVE_EVERY_N_EPOCHS = 1

