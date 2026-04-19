# asnets/explorer_spawn_grads.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from asnets.parllel_explore_spawn_grads import run_epoch_spawn_grads, run_epoch_spawn_eval
from asnets.spawn_train_worker import WorkerOutput
from asnets.utils.generator_utils import ProgressionLevel


@dataclass
class ParallelMCTSExplorerGrads:
    specs: list[Any]
    dropout: float
    debug: bool
    policy_only: bool
    log: bool

    # loss cfg
    mse_coeff: float
    l2_reg_coeff: float
    l1_reg_coeff: float
    l1_l2_reg_coeff: float

    PROFILE_DIR: Optional[str] = None
    curr_epoch: int = 0
    max_workers: Optional[int] = None
    bootstrap_timeout: Optional[int] = 300
    rolling_worker_times = deque(maxlen=10)
    timeout_multiplier: Optional[int] = 3

    progression_level: ProgressionLevel = ProgressionLevel.LEVEL1

    #corruption testing settings
    corrupt_pi: Optional[str] = None
    corrupt_z: Optional[str] = None

    def explore(self, weights_np: dict, limit_workers=None,) -> list[WorkerOutput]:
        self.curr_epoch += 1
        max_rolling_worker_times = max(self.rolling_worker_times) if len(self.rolling_worker_times) > 0 else 0
        timeout= max(self.bootstrap_timeout, self.timeout_multiplier * max_rolling_worker_times)
        return run_epoch_spawn_grads(
            specs=self.specs,
            curr_epoch=self.curr_epoch-1, # so the first is 0
            weights_np=weights_np,
            dropout=self.dropout,
            debug=self.debug,
            policy_only=self.policy_only,
            log=self.log,
            PROFILE_DIR=self.PROFILE_DIR,
            corrupt_pi=self.corrupt_pi,
            corrupt_z=self.corrupt_z,
            mse_coeff=self.mse_coeff,
            l2_reg_coeff=self.l2_reg_coeff,
            l1_reg_coeff=self.l1_reg_coeff,
            l1_l2_reg_coeff=self.l1_l2_reg_coeff,
            max_workers=self.max_workers if limit_workers is None else min(limit_workers,self.max_workers),
            epoch_timeout=timeout,
        )

    def num_slots(self):
        return len(self.specs)

    def estimator_decay_end_epoch(self):
        return self.specs[0].estimator_decay_epochs if self.specs[0].estimator_decay else 0

    def advance_progression_level(self):
        if self.progression_level == ProgressionLevel.LEVEL5:
            return
        print(f"Starting to advance progression level from {self.progression_level} to {self.progression_level.next()}")
        self.progression_level = self.progression_level.next()
        self.set_specs_according_to_progression_level()
        print(f"Current progression level is {self.progression_level}, specs were given the following difficulties:")
        diff_list_from_specs = [str(self.specs[i].difficulty) for i in range(len(self.specs))]
        print(",".join(diff_list_from_specs))

    def set_specs_according_to_progression_level(self):
        diff_seq = self.progression_level.generate_difficulty_sequence(len(self.specs))
        assert len(diff_seq) == len(self.specs)
        for i, diff in enumerate(diff_seq):
            self.specs[i].difficulty = diff


@dataclass
class ParallelMCTSExplorerEval:
    specs: list[Any]
    max_workers: Optional[int] = None

    def evaluate(self, weights_np):
        outs = run_epoch_spawn_eval(
            specs=self.specs,
            weights_np=weights_np,
            max_workers=self.max_workers,
        )
        solved = [o.hit_goal for o in outs]
        success_rate = float(np.mean(solved))
        return success_rate, outs
