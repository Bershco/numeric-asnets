# asnets/explorer_spawn_grads.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import time
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
    bootstrap_timeout_s: Optional[int] = 300
    rolling_epoch_times = deque(maxlen=10)
    timeout_multiplier: Optional[int] = 3
    max_epoch_timeout_s = 3600

    progression_level: ProgressionLevel = ProgressionLevel.LEVEL1

    #corruption testing settings
    corrupt_pi: Optional[str] = None
    corrupt_z: Optional[str] = None

    def explore(
            self,
            weights_np: dict,
            limit_workers: Optional[int] = None,
    ) -> list[WorkerOutput]:
        """
        Run one exploration epoch and update rolling timing statistics.

        Args:
            weights_np: Current network weights to send to workers.
            limit_workers: Optional upper bound on the number of workers to use.

        Returns:
            A list of successful worker outputs for this epoch.
        """
        self.curr_epoch += 1
        epoch_timeout = self._compute_epoch_timeout()
        effective_max_workers = (
            self.max_workers
            if limit_workers is None
            else min(limit_workers, self.max_workers)
        )
        start_time = time()
        outputs = run_epoch_spawn_grads(
            specs=self.specs,
            curr_epoch=self.curr_epoch - 1,  # first epoch is 0
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
            max_workers=effective_max_workers,
            epoch_timeout=epoch_timeout,
        )
        self.rolling_epoch_times.append(time() - start_time)
        return outputs

    def _compute_epoch_timeout(self) -> float:
        """
        Compute the epoch timeout from rolling worker times.

        The timeout is:
            max(bootstrap_timeout, timeout_multiplier * rolling_max_worker_time)

        and is capped at one hour. A message is printed if the cap is reached.

        Returns:
            Timeout in seconds.
        """
        rolling_max = max(self.rolling_epoch_times) if self.rolling_epoch_times else 0.0

        timeout = max(
            float(self.bootstrap_timeout_s),
            float(self.timeout_multiplier) * float(rolling_max),
        )

        if timeout >= self.max_epoch_timeout_s:
            print(
                f"[TRAINER] Timeout capped at 1 hour "
                f"({self.max_epoch_timeout_s}s) | raw_timeout={timeout:.1f}s"
            )
            return float(self.max_epoch_timeout_s)

        return float(timeout)

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
