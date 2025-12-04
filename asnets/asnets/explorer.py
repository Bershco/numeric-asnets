"""Exploration algorithms"""
from abc import ABC, abstractmethod
import concurrent
import logging
import random
from time import time
import tqdm.auto as tqdm
from typing import List, Optional, Tuple

from collections import Counter
import gc

from asnets.multiprob import to_local

LOGGER = logging.getLogger(__name__)

class Explorer(ABC):
    def __init__(self, problems: List['SingleProblem'],
                 max_replay_size: int=None):
        self.problems = problems
        self.max_replay_size = max_replay_size
    
    def _collect_trajectories(self,
                              num_per_problem: int,
                              progress: bool = True) -> None:
        """Collects trajectories for each problem."""
        def inner(problem):
            hit_goal = []
            for _ in range(num_per_problem):
                network_weights_to_send = problem.network.get_weights()
                hit_goal.append(
                    problem.problem_service.collect_trajectory(network_weights_to_send))
            
            return problem, hit_goal
        
        def update_result(result):
            problem, hit_goal = result
            self.hit_goal[problem].extend(hit_goal)

        # Concurrency doesn't seem to help too much, especially on single-core
        # systems
        CONCURREENT = False
        if CONCURREENT:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(inner, problem)
                           for problem in self.problems]
            
                if progress:
                    futures = tqdm.tqdm(concurrent.futures.as_completed(futures),
                                        total=len(futures),
                                        desc='trajectories (concurrent)')
                
                for future in futures:
                    update_result(future.result())

        else:
            tr = tqdm.tqdm(self.problems, desc='trajectories') if progress else \
                self.problems
            for problem in tr:
                update_result(inner(problem))
            if hasattr(tr,'close'):
                tr.close()
        
        for problem in self.problems:
            self.traj_sizes[problem] = \
                problem.problem_service.get_num_traj_states()
    

    def _trim_replays(self) -> None:
        """Trims replays for each problem if needed."""
        if self.max_replay_size is None:
            return
        while True:
            replay_size = sum(
                to_local(problem.problem_service.get_replay_size())
                for problem in self.problems)
            
            if replay_size <= self.max_replay_size:
                break

            for problem in self.problems:
                problem.problem_service.trim_replay()

    
    def extend_replay(self) -> List[Tuple['SingleProblem', float]]:
        self.hit_goal = {problem: [] for problem in self.problems}
        self.traj_sizes = {problem: 0 for problem in self.problems}
        self.explore()
        for problem in self.problems:
            problem.problem_service.finish_explore()
        self._trim_replays()
        return [
            (problem, sum(self.hit_goal[problem]) / len(self.hit_goal[problem]) if len(self.hit_goal[problem])>0 else 0)
            for problem in self.problems]

    def update_learning_time(self, learning_time: float) -> None:
        """Updates the learning time."""
        pass

    @abstractmethod
    def explore(self):
        pass


class StaticExplorer(Explorer):
    """The static exploration algorithm from the original ASNets."""
    def __init__(self, problems, trajs_per_problem: int):
        super().__init__(problems)
        self.trajs_per_problem = trajs_per_problem

    def explore(self) -> None:
        self._collect_trajectories(self.trajs_per_problem)
        for problem in tqdm.tqdm(self.problems, desc='static explore'):
            # problem.problem_service.explore_from_trajectories(problem.network)
            problem.problem_service.explore_from_trajectories(problem.network.get_weights())


class DynamicExplorer(Explorer):
    """The dynamic exploration algorithm."""
    def __init__(self,
                 problems,
                 init_trajs_per_problem: int,
                 min_new_pairs: int,
                 max_new_pairs: int,
                 expl_learn_ratio: float,
                 max_replay_size: int,
                 debug_memory: bool = False):
        super().__init__(problems)
        self.init_trajs_per_problem = init_trajs_per_problem
        self.min_new_pairs = min_new_pairs
        self.max_new_pairs = max_new_pairs
        self.expl_learn_ratio = expl_learn_ratio
        self.recent_learning_times = []
        # Might not make the best sense to have the explorer manage this, but
        # this is the easiest way to do it. Also different exploration
        # algorithm manage the buffer differently
        self.max_replay_size = max_replay_size
        self.debug_memory = debug_memory
    
    def _is_first_explore(self) -> bool:
        return len(self.recent_learning_times) == 0
    
    def _terminate(self, start_time: float, t: tqdm.tqdm) -> bool:
        """Whether to terminate the exploration phase."""
        new_pairs = [to_local(problem.problem_service.get_num_new_pairs())
                     for problem in self.problems]
        total_new_pairs = sum(new_pairs)

        if self._is_first_explore():
            t.update(total_new_pairs - t.n)
            return total_new_pairs >= self.min_new_pairs and \
                all(n > 0 for n in new_pairs)

        # Terminating when there seems to be no progress
        if total_new_pairs == t.n:
            if time() - self.last_progress_time > 10:
                LOGGER.warning(
                    'No progress in exploration phase for 10s, aborting')
                return True
        else:
            self.last_progress_time = time()
            t.update(total_new_pairs - t.n)
        
        # hard termination when we take too long
        if time() - start_time > 3 * self.expl_learn_ratio * self.recent_learning_time:
            print('[DYNAMIC_EXPLORE_TERMINATED] Cause: hard termination for taking too long')
            return True
        if total_new_pairs >= self.max_new_pairs:
            print('[DYNAMIC_EXPLORE_TERMINATED] Cause: total_new_pairs >= max_new_pairs')
            return True
        if total_new_pairs >= self.min_new_pairs:
            if time() - start_time >= self.expl_learn_ratio * self.recent_learning_time:
                print('[DYNAMIC_EXPLORE_TERMINATED] Cause: time() - start_time >= self.expl_learn_ratio * self.recent_learning_time ')
                return True
        if t.n > t.total:
            print('[DYNAMIC_EXPLORE_TERMINATED] Cause: t.n > t.total')
            return True
        return False
    
    def _sample_problem(self) -> Optional['SingleProblem']:
        """Samples a problem to explore from."""
        total_traj_size = sum(self.traj_sizes.values())
        if total_traj_size == 0:
            return None
    
        return random.choices(
            list(self.traj_sizes.keys()),
            list(self.traj_sizes.values()),
            k=1)[0]
        
    def update_learning_time(self, learning_time: float) -> None:
        """Updates the learning time."""
        self.recent_learning_times.append(learning_time)
        if len(self.recent_learning_times) > 10:
            self.recent_learning_times.pop(0)
        self.recent_learning_time = sum(self.recent_learning_times) / \
            len(self.recent_learning_times)
    
    def explore(self) -> None:
        start_time = time()
        if self._is_first_explore():
            LOGGER.info('First exploration phase, collecting less trajectories'
                        ' and terminating exploration as soon as all problems'
                        ' have at least one new pair.')
            self._collect_trajectories(1, progress=True)
        else:
            self._collect_trajectories(self.init_trajs_per_problem,
                                       progress=True)
        
        t = tqdm.tqdm(desc='dynamic explore', total=self.max_new_pairs)
        self.last_progress_time = time()
        while not self._terminate(start_time, t):
            problem = self._sample_problem()

            if problem is None:
                self._collect_trajectories(1, progress=False)
                continue
        
            self.traj_sizes[problem] -= 1
            # problem.problem_service.explore_from_random_state(problem.network)
            problem.problem_service.explore_from_random_state(problem.network.get_weights())
        t.close()

        if self.debug_memory:
            objs = gc.get_objects()
            types = Counter(type(o).__name__ for o in objs)
            top = types.most_common(10)
            print("[MEM] top object types:", top)
            print("[MEM] total objects:", len(objs))



class MCTSExplorer(DynamicExplorer):

    def __init__(self,
                 problems,
                 init_trajs_per_problem: int,
                 min_new_pairs: int,
                 max_new_pairs: int,
                 expl_learn_ratio: float,
                 max_replay_size: int,
                 debug_memory: bool = False):
        super().__init__(problems, init_trajs_per_problem, min_new_pairs, max_new_pairs, expl_learn_ratio, max_replay_size, debug_memory)
        self.exploration_count_by_problem: dict['SingleProblem', int] = {prob: 0 for prob in problems}
        assert len(self.exploration_count_by_problem) == len(self.problems)
        self.solved_count_by_problem: dict['SingleProblem', int] = {prob: 0 for prob in problems}
        self.solved_threshold = 0.1
        self.init_state_h_by_problem: dict['SingleProblem', float] = {prob: prob.problem_service.get_state_h(prob.problem_service.env_reset()) for prob in problems}
        # self.planner_plan_length_by_probem: dict['SingleProblem, int] = {prob: TODO: <here goes a method to retrieve planner plan length from initial state per problem> for prob in problems}

    def compute_weight(self, problem: 'SingleProblem'):
        if self.exploration_count_by_problem[problem] == 0:
            return 1.0  # unexplored = high weight

        success_rate = self.solved_count_by_problem[problem] / self.exploration_count_by_problem[problem]
        w = 1 - success_rate
        return max(w, 0.05)  # keep minimal weight to prevent forgetting

    def _sample_problem(self) -> Optional['SingleProblem']:
        weights = [self.compute_weight(prob) for prob in self.problems]
        return random.choices(self.problems, weights=weights, k=1)[0]

    def explore(self) -> None:
        start_time = time()
        t = tqdm.tqdm(desc='MCTS explore', total=self.max_new_pairs)
        self.last_progress_time = time() #not sure what this is used for, trying to keep the status quo
        while not self._terminate(start_time, t):
            problem = self._sample_problem()
            self.hit_goal[problem].append(problem.problem_service.explore_from_init_state(problem.network.get_weights()))
        t.close()
