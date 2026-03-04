# asnets/explorer_spawn_grads.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from asnets.parllel_explore_spawn_grads import run_epoch_spawn_grads
from asnets.spawn_train_worker import WorkerOutput


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

    max_workers: Optional[int] = None

    #corruption testing settings
    corrupt_pi: Optional[str] = None
    corrupt_z: Optional[str] = None


    def explore(self, weights_np: dict, limit_workers=None) -> list[WorkerOutput]:
        return run_epoch_spawn_grads(
            specs=self.specs,
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
        )

    def num_slots(self):
        return len(self.specs)
