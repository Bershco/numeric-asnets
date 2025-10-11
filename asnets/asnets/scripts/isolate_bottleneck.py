#!/usr/bin/env python3
import argparse, time, numpy as np, tensorflow as tf
import logging

import rpyc

from post_training.enhspwrapper import ENHSPEstimator

rpyc.core.protocol.DEFAULT_CONFIG.update({
    "allow_pickle": True,
    "sync_request_timeout": 1800,  # optional but nice to have for long calls
})

# Your code paths
from asnets.prob_dom_meta import DomainType
from asnets.state_reprs import get_init_cstate, sample_next_state, compute_action_dim
from asnets.supervised import PlannerExtensions, ProblemServiceConfig
from asnets.multiprob import ProblemServer, to_local

# Import your evaluator & node wrappers
from run_asnets import MonteCarloPolicyEvaluator, move_to_next_state  # uses your _expand/find_children
from post_training.monte_carlo_tree_search import wrapInMCTSNode, FixedChildMap

from asnets import state_reprs

def uniform_policy(act_dim: int):
    def _pi(in_obs_batch, training=False):
        return tf.nn.softmax(tf.zeros((1, act_dim), dtype=tf.float32)), 1
    return _pi

class LocalService:
    """In-process service exposing the minimal API your evaluator calls."""
    def __init__(self, p):
        self.p = p
        self.estimator = ENHSPEstimator(planner_exts=self.p, enhsp_config="hadd-gbfs")
    def env_reset(self):
        return get_init_cstate(self.p)
    def env_simulate_step(self, cstate_from, action_num: int):
        return sample_next_state(cstate_from, int(action_num), self.p)
    def get_state_h(self, cstate):
        # value-based path (if used)
        # return float(hash(cstate) & 1023)
        return self.estimator.get_cstate_h(cstate)
    def get_act_dim(self):
        return compute_action_dim(self.p)

def _maybe_to_local_pair(x):
    """Robustly obtain (state, cost) from local or RPyC result."""
    res = to_local(x)
    if isinstance(res, tuple) and len(res) == 2:
        s, c = res
        try:
            c = float(c)
        except Exception:
            c = float(to_local(c))
        return s, c
    # Fallback if service returns only a state
    return res, 0.0

def bench(mode: str, pddl_domain: str, pddl_problem: str, problem_name: str,
          domain_type: str, iterations: int, decisions: int, k: int,
          reroot_per_decision: bool):
    dtype = DomainType.NUMERIC if domain_type == "numeric" else DomainType.PROBABILISTIC
    pddl_files = [pddl_domain, pddl_problem]

    if mode == "local":
        p = PlannerExtensions(pddl_files, problem_name, dtype,
                              dg_ssipp_heuristic_name=None,
                              dg_use_lm_cuts=False,
                              dg_use_numeric_landmarks=False,
                              dg_use_contributions=False,
                              dg_use_act_history=False)
        service = LocalService(p)
    elif mode == "rpyc":
        cfg = ProblemServiceConfig(pddl_files, problem_name, dtype, teacher_planner="enhsp")
        srv = ProblemServer(cfg)
        srv.service.initialise()
        srv.service.initialise_estimator("hadd-gbfs")
        service = srv.service
    else:
        raise SystemExit("mode must be local or rpyc")

    act_dim = service.get_act_dim()
    policy = uniform_policy(act_dim)

    # Wire your evaluator
    mcts = MonteCarloPolicyEvaluator(policy=policy,
                                     problem_service=service,
                                     iterations=iterations,
                                     horizon=0,
                                     num_cstates_to_generate_per_expansion=k,
                                     use_value_based=True,
                                     debug_memory=False)

    # Turn on your internal timing buckets
    mcts.debug_time_mcts_iterations = True
    mcts.start_times = []; mcts.after_selection_times = []
    mcts.after_expansion_times = []; mcts.after_eval_times = []; mcts.end_times = []

    # Seed root (state + tree)
    # curr_cstate = to_local(service.env_reset()) TODO: check that it's okay that the first cstate is a netref
    curr_cstate = service.env_reset()
    total_cost = 0.0
    mcts.curr_tree_root = wrapInMCTSNode(curr_cstate, cost_until_now=total_cost, previous_action=None)

    # Run N decisions; each decision performs `iterations` loops and then chooses an action
    t0 = time.perf_counter()
    for _ in range(decisions):
        # Important: pass the CURRENT state & cost (matches inference usage)
        action = int(mcts.get_action_from_cstate(curr_cstate, total_cost))

        if reroot_per_decision:
            # Try to re-root to the existing child node for `action` if present
            curr_cstate, step_cost = move_to_next_state(problem_service=service, policy_evaluator=mcts, action=action, cost=total_cost, current_code=False)
            # new_root = None
            # try:
            #     root = mcts.curr_tree_root
            #     # if root in mcts.children and mcts.children[root] is not None:
            #     if root.children is not None:
            #         # root_children = mcts.children[root]
            #         root_children = root.children
            #         next_node = None
            #         if isinstance(root_children, FixedChildMap) and action in root_children:
            #             next_node = root_children[action]
            #         elif hasattr(root_children, "get"):
            #             try:
            #                 next_node = root_children.get(action)
            #             except Exception:
            #                 next_node = None
            #         # Some containers expose .__getitem__
            #         if next_node is None and hasattr(root_children, "__getitem__"):
            #             try:
            #                 next_node = root_children[action]
            #             except Exception:
            #                 next_node = None
            #         if next_node is not None:
            #             new_root = next_node
            # except Exception:
            #     new_root = None
            #
            # if new_root is not None:
            #     # Reuse subtree & state if available
            #     mcts.curr_tree_root = new_root
            #     if hasattr(new_root, "state"):
            #         curr_cstate = new_root.state
            #     # keep total_cost as-is; some trees track cost on node already
            # else:
            #     # Fall back to simulating the chosen action and rebuilding root at the next state
            #     next_state, step_cost = _maybe_to_local_pair(service.env_simulate_step(curr_cstate, action))
            #     total_cost += step_cost
            #     curr_cstate = next_state
            #     mcts.curr_tree_root = wrapInMCTSNode(curr_cstate, cost_until_now=total_cost, previous_action=action)


    wall = time.perf_counter() - t0

    # Collate stage times from your arrays
    sel = sum(b - a for a, b in zip(mcts.start_times, mcts.after_selection_times))
    expd = sum(b - a for a, b in zip(mcts.after_selection_times, mcts.after_expansion_times))
    evalv = sum(b - a for a, b in zip(mcts.after_expansion_times, mcts.after_eval_times))
    back = sum(b - a for a, b in zip(mcts.after_eval_times, mcts.end_times))
    total_iters = len(mcts.end_times)

    print(f"\n=== {mode.upper()} ===")
    print(f"decisions={decisions}, iterations_per_decision={iterations}, k={k}")
    print(f"total_iterations={total_iters}, wall={wall:.3f}s  ->  {1e3*wall/max(1,total_iters):.3f} ms/iter")
    print(f" selection={sel:.3f}s  expansion={expd:.3f}s  evaluation={evalv:.3f}s  backprop={back:.3f}s")
    print(f" per-iter: sel={1e3*sel/max(1,total_iters):.3f} ms  expd={1e3*expd/max(1,total_iters):.3f} ms  "
          f"eval={1e3*evalv/max(1,total_iters):.3f} ms  back={1e3*back/max(1,total_iters):.3f} ms")

    if mode == "rpyc":
        try:
            srv.stop()
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--problem-name", required=True)
    ap.add_argument("--domain-type", choices=["numeric","probabilistic"], required=True)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--decisions", type=int, default=20)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--run-grid-test", action="store_true")
    ap.add_argument("--run-local", action="store_true")
    ap.add_argument("--run-rpyc", action="store_true")
    ap.add_argument("--reroot-per-decision", action="store_true",
                    help="After each decision, re-root the tree to the chosen child (simulate if necessary) and continue from the next state.")

    args = ap.parse_args()
    logger = logging.getLogger(__name__)

    run_modes = []
    if args.run_local:
        run_modes.append("local")
    if args.run_rpyc:
        run_modes.append("rpyc")

    if args.run_grid_test:
        logger.info("Running grid test")
        for k_value in (1,3,5,10):
            for iterations_num in (3,5,10):
                for action_decisions_num in (5,10,50,100):
                    for mode in run_modes:
                        logger.info(
                            f"Running {mode} mode for {action_decisions_num} action decisions, "
                            f"with {iterations_num} mcts iterations, and {k_value} nodes generated per partial expansion."
                        )
                        bench(mode, args.domain, args.problem, args.problem_name, args.domain_type,
                              iterations_num, decisions=action_decisions_num, k=k_value,
                              reroot_per_decision=args.reroot_per_decision)
    else:
        for mode in run_modes:
            bench(mode, args.domain, args.problem, args.problem_name, args.domain_type,
                  args.iterations, args.decisions, args.k,
                  reroot_per_decision=args.reroot_per_decision)

if __name__ == "__main__":
    main()
