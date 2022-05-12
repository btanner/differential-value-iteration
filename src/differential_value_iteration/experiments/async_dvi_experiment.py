"""Runs control algorithms a few times to generate timing in a few problems."""

import functools
import time
from typing import Callable
from typing import Sequence

import numpy as np
from absl import logging

from differential_value_iteration.algorithms import algorithm
from differential_value_iteration.algorithms import dvi
from differential_value_iteration.algorithms import mdvi
from differential_value_iteration.algorithms import random
from differential_value_iteration.algorithms import rvi
from differential_value_iteration.environments import garet
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import mm1_queue
from differential_value_iteration.environments import structure


def run(
    environment: structure.MarkovDecisionProcess,
    alg: algorithm.Control,
    num_iters: int,
    convergence_tolerance: float,
    synchronized: bool,
    eval_all_states: bool,
    measure_iters: Sequence[int],
):
    """Runs one algorithm on one environment and return results.
    Params:
      environment: Markov Decision Processes to run.
      alg: The algorithm to use.
      num_iters: Total number of iterations for benchmark.
      convergence_tolerance: Criteria for convergence.
      synchronized: Run algorithms in synchronized or asynchronous mode.
      eval_all_states: Empirically test return from all states (or just S0).
      measure_iters: Which iterations to evaluate policy on. Final policy is
        always evaluate.
    """
    log_changes_every = 100
    inner_loop_range = 1 if synchronized else environment.num_states
    changes = [0.0]  # Dummy starting value.
    last_policy = None
    policy_switches = 0
    total_time = 0
    all_late_policies = set()
    policy_measurements = {}
    logged_value_changes = {}
    converged = False
    diverged = False
    for i in range(num_iters):
        if i in measure_iters:
            mean_returns, std_returns = estimate_policy_average_reward(
                alg.greedy_policy(), environment, eval_all_states
            )
            policy_measurements[i] = (mean_returns, std_returns, alg.greedy_policy())

        change_summary = 0.0
        for _ in range(inner_loop_range):
            start_time = time.time()
            changes = alg.update()
            end_time = time.time()
            total_time += end_time - start_time
            # Mean instead of sum so tolerance scales with num_states.
            change_summary += np.mean(np.abs(changes))
        # Basically divide by num_states if running async.
        change_summary /= inner_loop_range
        if i % log_changes_every == 0:
            logged_value_changes[i] = change_summary
        # Track all policy changes after 10_000 iterations.
        if i == 10_000:
            last_policy = str(alg.greedy_policy())
            all_late_policies.add(last_policy)
        if i > 10_000:
            this_policy = str(alg.greedy_policy())
            if this_policy != last_policy:
                policy_switches += 1
                last_policy = this_policy
                all_late_policies.add(last_policy)

        if alg.diverged():
            diverged = True
            converged = False
            break

        if alg.converged(convergence_tolerance) and i > 1:
            converged = True
            break
    converged_string = "YES\t" if converged else "NO\t"

    mean_returns, std_returns = estimate_policy_average_reward(
        alg.greedy_policy(), environment, eval_all_states
    )
    this_policy = str(alg.greedy_policy())
    policy_measurements[i] = (mean_returns, std_returns, this_policy)

    if diverged:
        converged_string = "DIVERGED"
    logging.info(
        f"Summary: Average Time:{1000. * total_time / i:.3f} ms\tConverged:{converged_string}\t{i} iters\tMean final Change:{np.mean(np.abs(changes)):.5f}"
    )
    if not converged:
        logging.info(
            f"After iter 10000, policy switched:{policy_switches} times and saw these policies:{all_late_policies}"
        )
    return {
        "average_time_ms": 1000.0 * total_time / i,
        "last_iteration": i,
        "converged": converged,
        "diverged": diverged,
        "late_policy_switches": policy_switches,
        "all_late_policies": all_late_policies,
        "last_policy": this_policy,
        "policy_measurements": policy_measurements,
        "value_changes": logged_value_changes,
    }


def sample_return(policy, environment, start_state, length):
    stochastic = True if policy.ndim == 2 else False
    state = start_state
    total_return = 0.0

    for _ in range(length):
        if stochastic:
            action = np.random.choice(a=environment.num_actions, p=policy[:, state])
        else:
            action = policy[state]
        total_return += environment.rewards[action, state]
        state = np.random.choice(
            a=environment.num_states, p=environment.transitions[action, state]
        )
    return total_return / length


def estimate_policy_average_reward(policy, environment, all_states: bool):
    length = 1000
    num_reps = 10
    starting_states = np.arange(environment.num_states) if all_states else [0]

    means = np.zeros(len(starting_states), dtype=np.float64)
    stdevs = np.zeros(len(starting_states), dtype=np.float64)

    for starting_state in starting_states:
        returns = []
        for _ in range(num_reps):
            returns.append(sample_return(policy, environment, starting_state, length))
        means[starting_state] = np.mean(returns)
        stdevs[starting_state] = np.std(returns)
    return means, stdevs
