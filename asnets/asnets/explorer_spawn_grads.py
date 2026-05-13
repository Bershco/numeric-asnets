# asnets/explorer_spawn_grads.py
from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
from time import time
from typing import Any, Optional, Callable

import numpy as np

from asnets.parllel_explore_spawn_grads import run_epoch_spawn_grads, run_epoch_spawn_eval, SpawnExploreSpec
from asnets.spawn_train_worker import WorkerOutput, EvalWorkerInput, EvalWorkerOutput
from asnets.utils.generator_utils import ProgressionLevel


@dataclass
class ParallelMCTSExplorerGrads:
    specs: list[Any]
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
    max_curr_est_coeff: float = 1.0

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
        epoch_timeout = self._compute_epoch_timeout()
        effective_max_workers = (
            self.max_workers
            if limit_workers is None
            else min(limit_workers, self.max_workers)
        )
        start_time = time()
        outputs = run_epoch_spawn_grads(
            specs=self.specs,
            curr_epoch=self.curr_epoch,
            weights_np=weights_np,
            log=self.log,
            PROFILE_DIR=self.PROFILE_DIR,
            corrupt_pi=self.corrupt_pi,
            corrupt_z=self.corrupt_z,
            mse_coeff=self.mse_coeff,
            max_estimator_coeff=self.max_curr_est_coeff,
            l2_reg_coeff=self.l2_reg_coeff,
            l1_reg_coeff=self.l1_reg_coeff,
            l1_l2_reg_coeff=self.l1_l2_reg_coeff,
            max_workers=effective_max_workers,
            epoch_timeout=epoch_timeout,
        )
        self.curr_epoch += 1
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
        return self.specs[0].estimator_decay_epochs if self.specs[0].use_estimator else 0

    def advance_progression_level(self):
        if self.progression_level == ProgressionLevel.LEVEL5:
            return False
        print(f"Starting to advance progression level from {self.progression_level} to {self.progression_level.next()}")
        self.progression_level = self.progression_level.next()
        self.set_specs_according_to_progression_level()
        print(f"Current progression level is {self.progression_level}, specs were given the following difficulties:")
        diff_list_from_specs = [str(self.specs[i].difficulty) for i in range(len(self.specs))]
        print(",".join(diff_list_from_specs))
        return True

    def set_specs_according_to_progression_level(self):
        diff_seq = self.progression_level.generate_difficulty_sequence(len(self.specs))
        assert len(diff_seq) == len(self.specs)
        for i, diff in enumerate(diff_seq):
            self.specs[i] = self.specs[i].change_diff_to(diff)

    def decay_estimator_coefficient(self):
        self.max_curr_est_coeff *= 0.7

@dataclass
class ParallelEvaluator:
    specs: list[SpawnExploreSpec]
    worker_fn: Callable[[EvalWorkerInput],EvalWorkerOutput]
    max_workers: Optional[int] = None

    def evaluate(self, weights_np):
        print(f"[EVAL] worker_fn={self.worker_fn.__name__}")

        outs = run_epoch_spawn_eval(
            specs=self.specs,
            weights_np=weights_np,
            max_workers=self.max_workers,
            worker_fn=self.worker_fn,
        )

        # ------------------------------------------------------------
        # Aggregate metrics per difficulty
        # ------------------------------------------------------------

        metrics = defaultdict(lambda: {
            "hits": [],
            "steps_success": [],
            "steps_fail": [],
        })

        successful_plans = []

        for spec, out in zip(self.specs, outs):
            diff_metrics = metrics[spec.difficulty]

            diff_metrics["hits"].append(out.hit_goal)

            if out.hit_goal:
                diff_metrics["steps_success"].append(out.steps)

                successful_plans.append({
                    "instance": Path(spec.pddls[1]).name,
                    "plan": out.plan,
                    "steps": out.steps,
                    "difficulty": spec.difficulty.name,
                })

            else:
                diff_metrics["steps_fail"].append(out.steps)

        # ------------------------------------------------------------
        # Compute + print metrics
        # ------------------------------------------------------------

        success_rates = {}

        for diff, diff_metrics in metrics.items():
            hits = diff_metrics["hits"]

            success_rate = float(np.mean(hits)) if hits else 0.0
            success_rates[diff] = success_rate

            success_steps = diff_metrics["steps_success"]
            fail_steps = diff_metrics["steps_fail"]

            avg_success_len = (
                float(np.mean(success_steps))
                if success_steps else float("nan")
            )

            avg_fail_len = (
                float(np.mean(fail_steps))
                if fail_steps else float("nan")
            )

            print(
                f"[EVAL] {diff.name:<10} | "
                f"success={success_rate:.3f} | "
                f"instances={len(hits):>3} | "
                f"avg_len_success={avg_success_len:.2f} | "
                f"avg_len_fail={avg_fail_len:.2f}"
            )

        # ------------------------------------------------------------
        # Successful plans logging
        # ------------------------------------------------------------

        print("\n[EVAL] SUCCESSFUL PLANS")

        if successful_plans:
            for entry in successful_plans:
                print(
                    f"[EVAL][PLAN] "
                    f"{entry['difficulty']:<10} | "
                    f"{entry['instance']} | "
                    f"steps={entry['steps']} | "
                    f"plan={entry['plan']}"
                )
        else:
            print("[EVAL][PLAN] No successful plans.")

        # ------------------------------------------------------------
        # Overall metrics
        # ------------------------------------------------------------

        all_hits = [out.hit_goal for out in outs]
        overall_success = float(np.mean(all_hits)) if all_hits else 0.0

        print(f"\n[EVAL] OVERALL success={overall_success:.3f}")

        return success_rates, overall_success, outs

