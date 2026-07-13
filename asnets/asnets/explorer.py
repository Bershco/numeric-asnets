"""Exploration algorithms"""
from abc import ABC, abstractmethod
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed, Future

import numpy as np
from typing import List, Optional, Tuple, Any, Iterable
from collections import Counter, deque
import multiprocessing as mp

from asnets.spawn_train_worker import run_multiple_trajectory_collection, ProblemInitData, collect_problem_dims_worker, \
    PolicyDrivenWorkerInput, run_worker_opt_profiled, WorkerInput

LOGGER = logging.getLogger(__name__)


class SingleProblem(object):
    """Wrapper to store all information relevant to training on a single
    problem."""

    def __init__(self, spec):
        self.name = None  # this changes later when explorer is created
        self.spec = spec
        self.replay = WeightedReplayBuffer()
        self.sampled_states_replay = WeightedReplayBuffer()
        self.problem_meta = None
        self.dom_meta = None
        self.obs_dim = None
        self.act_dim = None
        self.ssipp_dead_end_value = None
        self.network = None

    def flatten_obs_qvs(self, rich_targets):
        if self.network.value_head_enabled:
            cstates, rich_qvs, values = zip(*rich_targets)
        else:
            cstates, rich_qvs = zip(*rich_targets)
        obs_tensor = np.stack([s.to_network_input() for s in cstates], axis=0, )
        qv_lists = []
        for qv_pairs in rich_qvs:
            qv_dict = dict(qv_pairs)
            qv_list = [qv_dict[ba] for ba in self.problem_meta.bound_acts_ordered]
            qv_lists.append(qv_list)
        qv_tensor = np.array(qv_lists, dtype=float)
        if self.network.value_head_enabled:
            value_tensor = np.array(values, dtype=float)
            return obs_tensor, qv_tensor, value_tensor
        return obs_tensor, qv_tensor

    def weighted_dataset(self, replay=None):
        replay = self.replay if replay is None else replay
        rich_targets, counts = replay.get_full_dataset()
        assert len(rich_targets) > 0, f"Empty replay {replay}"
        counts = np.asarray(counts, dtype="float32")
        flattened = self.flatten_obs_qvs(rich_targets)
        if self.network.value_head_enabled:
            obs_tensor, qv_tensor, value_tensor = flattened
            return obs_tensor, qv_tensor, value_tensor, counts
        else:
            obs_tensor, qv_tensor = flattened
            return obs_tensor, qv_tensor, counts


class WeightedReplayBuffer:
    """Replay buffer for previously-encountered states. The 'weighted' in the
    name comes from the fact that it's really a multiset that lets you sample
    states weighted by multiplicity."""

    def __init__(self):
        """Initialize the replay buffer."""
        self.counter = Counter()
        self.added_items = deque()

    def update(self, new_elems: Iterable[Any]) -> None:
        """Add new elements to the replay buffer.

        Args:
            new_elems (Iterable[Any]): New elements to add to the replay
            buffer.
        """
        item_counter = Counter(new_elems)
        self.counter.update(item_counter)
        self.added_items.append(item_counter)

    def __len__(self) -> int:
        """Get the number of unique elements in the replay buffer.

        Returns:
            int: Number of unique elements in the replay buffer.
        """
        return len(self.counter)

    def get_full_dataset(self) -> Tuple[List[Any], List[int]]:
        """Get the full dataset stored in the replay buffer.

        Returns:
            Tuple[List[Any], List[int]]: List of elements in the replay buffer
            and list of their counts.
        """
        rich_dataset = list(self.counter)
        counts = [self.counter[item] for item in rich_dataset]
        return rich_dataset, counts

    def remove_oldest(self) -> bool:
        """Remove the oldest insertion batch from the replay buffer."""
        if not self.added_items:
            return False

        item_counter = self.added_items.popleft()
        self.counter.subtract(item_counter)
        self.counter += Counter()  # remove zero and negative counts
        return True


class Explorer(ABC):
    def __init__(self, problems: List[SingleProblem],
                 max_replay_size: int = None):
        self.problems = problems
        self.spec_to_problem = {problem.spec: problem for problem in self.problems}
        self.max_replay_size = max_replay_size
        self.curr_weights_np = None
        # The following are the same across all specs
        self.enhsp_config = self.problems[0].spec.enhsp_config
        self.only_one_good_action = self.problems[0].spec.only_one_good_action
        self.use_teacher_envelope = self.problems[0].spec.use_teacher_envelope
        self.cached_dims = {problem.name: (problem.obs_dim, problem.act_dim) for problem in self.problems}

    def get_cached_shapes_per_problem(self):
        if not all(problem.obs_dim is not None and problem.act_dim for problem in self.problems):
            print(f"[EXPLORER_DIM_CACHE] dimension caching problem")
        return self.cached_dims

    def _collect_trajectories(self,
                              num_per_problem: int,
                              epoch_num: int,
                              dynamic: bool,
                              progress: bool = True) -> None:
        """Collects trajectories for each problem."""
        assert self.curr_weights_np is not None
        specs_and_all_problems_all_trajectories = run_parallel_multiple_traj_collection(
            specs=[problem.spec for problem in self.problems], epoch_num=epoch_num, weights_np=self.curr_weights_np,
            num_traj=num_per_problem, dynamic=dynamic,
            min_new_pairs=self.min_new_pairs if hasattr(self, "min_new_pairs") else None,
            max_new_pairs=self.max_new_pairs if hasattr(self, "max_new_pairs") else None,
            recent_learning_time=self.recent_learning_time if hasattr(self, "recent_learning_time") else 0,
            expl_learn_ratio=self.expl_learn_ratio if hasattr(self, "expl_learn_ratio") else None)
        assert len(specs_and_all_problems_all_trajectories) == len(self.problems)
        for spec, all_trajectories_single_problem in specs_and_all_problems_all_trajectories:
            problem = self.spec_to_problem[spec]
            expert_traj, policy_traj_hit_goal_list = all_trajectories_single_problem
            if expert_traj:
                problem.replay.update(expert_traj)
            if policy_traj_hit_goal_list:
                hit_goal_list = [hit_goal for _, hit_goal in policy_traj_hit_goal_list]
                self.hit_goal[problem].extend(hit_goal_list)

    def _trim_replays(self) -> None:
        """Trims replays for each problem if needed."""
        if self.max_replay_size is None:
            return
        while True:
            replay_size = sum(len(problem.replay) for problem in self.problems)

            if replay_size <= self.max_replay_size:
                break

            for problem in self.problems:
                LOGGER.info(f'[{problem.name}] trimming replay buffer')
                problem.replay.remove_oldest()

    def extend_replay(self, weights_np, epoch_num) -> List[Tuple[SingleProblem, float]]:
        self.hit_goal = {problem: [] for problem in self.problems}
        self.traj_sizes = {problem: 0 for problem in self.problems}
        self.curr_weights_np = weights_np
        self.explore(epoch_num)
        self._trim_replays()
        return [(problem,
                 sum(self.hit_goal[problem]) / len(self.hit_goal[problem]) if len(self.hit_goal[problem]) > 0 else 0)
                for problem in self.problems]

    def update_learning_time(self, learning_time: float) -> None:
        """Updates the learning time."""
        pass

    @abstractmethod
    def explore(self, epoch_num):
        pass


class StaticExplorer(Explorer):
    """The static exploration algorithm from the original ASNets."""

    def __init__(self, problems, trajs_per_problem: int, max_replay_size: int):
        super().__init__(problems, max_replay_size)
        self.trajs_per_problem = trajs_per_problem

    def explore(self, epoch_num) -> None:
        self._collect_trajectories(self.trajs_per_problem, epoch_num=epoch_num, dynamic=False)


class DynamicExplorer(Explorer):
    """The dynamic exploration algorithm."""

    def __init__(self,
                 problems,
                 init_trajs_per_problem: int,
                 min_new_pairs: int,
                 max_new_pairs: int,
                 expl_learn_ratio: float,
                 max_replay_size: int):
        super().__init__(problems, max_replay_size)
        self.init_trajs_per_problem = init_trajs_per_problem
        self.min_new_pairs = min_new_pairs
        self.max_new_pairs = max_new_pairs
        self.expl_learn_ratio = expl_learn_ratio
        self.recent_learning_times = []
        self.recent_learning_time = 0
        # Might not make the best sense to have the explorer manage this, but
        # this is the easiest way to do it. Also different exploration
        # algorithm manage the buffer differently
        self.first_explore = True

    def _is_first_explore(self) -> bool:
        if self.first_explore:
            self.first_explore = False
            return True
        return False

    def _sample_problem(self) -> Optional[SingleProblem]:
        """Samples a problem to explore from."""
        total_traj_size = sum(self.traj_sizes.values())
        if total_traj_size == 0:
            return None

        return random.choices(
            list(self.traj_sizes.keys()),
            list(self.traj_sizes.values()),
            k=1)[0]

    def update_learning_time(self, learning_time: float) -> None:
        # learning_time = time for the learning itself, not the whole training epoch, i.e. network prediction + applying gradients
        """Updates the learning time."""
        self.recent_learning_times.append(learning_time)
        if len(self.recent_learning_times) > 10:
            self.recent_learning_times.pop(0)
        self.recent_learning_time = sum(self.recent_learning_times) / \
                                    len(self.recent_learning_times)

    def explore(self, epoch_num) -> None:
        if self._is_first_explore():
            LOGGER.info('First exploration phase, collecting less trajectories'
                        ' and terminating exploration as soon as all problems'
                        ' have at least one new pair.')
            self._collect_trajectories(1, epoch_num=epoch_num, progress=True, dynamic=True)
        else:
            self._collect_trajectories(self.init_trajs_per_problem, epoch_num=epoch_num, progress=True, dynamic=True)


def run_parallel_multiple_traj_collection(specs, epoch_num, weights_np, num_traj, dynamic: bool, min_new_pairs,
                                          max_new_pairs, recent_learning_time, expl_learn_ratio, max_workers=None):
    ctx = mp.get_context("forkserver")
    with ProcessPoolExecutor(
            max_workers=max_workers or len(specs),
            mp_context=ctx,
    ) as ex:
        spec_map = {spec.slot_id: spec for spec in specs}
        fut_to_idx: dict[Future[Any], int] = {}
        for spec in specs:
            inp = PolicyDrivenWorkerInput(spec=spec, weights_np=weights_np, epoch=epoch_num, log=False,
                                          log_weights=False, PROFILE_DIR=None, num_trajectories=num_traj,
                                          dynamic=dynamic, min_new_pairs=min_new_pairs, max_new_pairs=max_new_pairs,
                                          recent_learning_time=recent_learning_time, expl_learn_ratio=expl_learn_ratio)
            fut: Future[Any] = ex.submit(run_worker_opt_profiled, inp, run_multiple_trajectory_collection)
            fut_to_idx[fut] = spec.slot_id
        outs: list[Optional[Any]] = [None] * len(specs)
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            outs[idx] = (spec_map[idx], fut.result())
    return outs


def run_parallel_problem_init_data_collection(
        specs: list[Any],
        max_workers: int | None = None,
        PROFILE_DIR = None,
) -> list[ProblemInitData]:
    ctx = mp.get_context("forkserver")

    max_workers = max_workers or min(len(specs), mp.cpu_count())
    outs: list[ProblemInitData | None] = [None] * len(specs)

    with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
    ) as ex:
        fut_to_slot = {
            ex.submit(run_worker_opt_profiled, WorkerInput(spec=spec, weights_np={}, epoch=0, log=False, log_weights=False, PROFILE_DIR=PROFILE_DIR), collect_problem_dims_worker): spec.slot_id
            for spec in specs
        }

        for fut in as_completed(fut_to_slot):
            slot_id = fut_to_slot[fut]
            dims = fut.result()
            assert dims.slot_id == slot_id
            outs[slot_id] = dims

    assert all(x is not None for x in outs)
    return outs
