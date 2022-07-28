"""Implements a discrete-time M/M/1 Queue environment.

Converted from continuous time and simplified to fixed set of states as per:
https://www.aaai.org/Papers/AAAI/1996/AAAI96-130.pdf

This implementation is a bit messy. We intended to clean it up, but did not.


State Representation
---------------------
In paper, state has 2 components: (jobs_in_queue: int, job_available: bool)

We have packed these into a single integer:
  state = jobs_in_queue*2 + job_available.

Invalid Actions
---------------------
In the paper, the algorithm does not have the option to admit a job when none
is available. Since our MDP transition and rewards are stored as matrices, we
we modified the reward/transition function such that the ADMIT action when no
job has arrived. The transitions are applied as if CONTINUE was chose, but an
extra penalty is added to the reward. Essentially, it's always worse to ADMIT
than to CONTINUE when no jobs have arrived.

Finite Implementation
---------------------
In the paper, the algorithm can (in theory) store an unbounded number of jobs.
Our implementation has a limit on the number of jobs that can be stored so that
we can represent transition and reward dynamics with a fixed-size matrix.

This limit should be set such that no reasonable algorithm would admit a new job
when the limit of jobs has been reached.

If the algorithm chooses to ADMIT a new job when the maximum is reached, there
is no reward for admitting the job. Further, there is no probability that an
existing job will be completed on the transition.

Essentially, ADMIT at capacity gives same reward as CONTINUE, but expected next
 state is worse than CONTINUE.
"""
import functools
import itertools
from typing import Callable

import numpy as np

from differential_value_iteration.environments import structure

COST_FN = Callable[[int], float]
CONTINUE = 0
ADMIT = 1


def linear_cost_fn(cost_constant: float, jobs_waiting: int) -> float:
  if cost_constant <= 0.:
    raise ValueError(f'cost_constant must be positive, got: {cost_constant}')
  if jobs_waiting < 0:
    raise ValueError(f'jobs_waiting must be non-negative, got: {jobs_waiting}')
  return jobs_waiting * cost_constant

def global_state_to_paper_state(global_state: int):
  jobs_in_queue = global_state // 2
  new_job = True if global_state % 2 == 1 else False
  return jobs_in_queue, new_job


def to_global_state(jobs_in_queue: int, new_job: bool):
  return jobs_in_queue * 2 + new_job


def create(arrival_rate: float, service_rate: float, admit_reward: float,
    cost_fn: COST_FN, max_stored_jobs: int,
    dtype: np.dtype) -> structure.MarkovDecisionProcess:
  """Creates a new MM1 Queue MDP.

  Args:
    arrival_rate: Rate of new jobs arriving. Should be positive.
    service_rate: Rate that accepted jobs are processed. Should be positive.
    admit_reward: Reward for accepting a job for processing.
    cost_fn: Function that returns cost (expressed as a positive number) for
      holding some number of jobs.
    max_stored_jobs: Limit on number of stored jobs transitions and rewards can
      be stored as fixed-size matrices.
    dtype: NumPy dtype of MDP, should be a float type, probably np.float64.

  Returns: An MDP.
  """
  arrive_prob = arrival_rate / (arrival_rate + service_rate)
  complete_prob = service_rate / (arrival_rate + service_rate)
  joint_rate = service_rate + arrival_rate
  num_states = max_stored_jobs * 2
  num_actions = 2

  transitions = np.zeros((num_actions, num_states, num_states), dtype=dtype)
  rewards = np.zeros((num_actions, num_states), dtype=dtype)

  transition_possibilities = itertools.product(
      range(max_stored_jobs),
      [False, True],
      [ADMIT, CONTINUE])

  for num_queued, new_job, action in transition_possibilities:
    s = to_global_state(jobs_in_queue=num_queued,
                        new_job=new_job)

    # Base Case.
    if num_queued == 0:
      if not new_job:
        no_new_job_next = s
        new_job_next = to_global_state(jobs_in_queue=num_queued,
                                       new_job=True)
        transitions[action, s, no_new_job_next] = complete_prob
        transitions[action, s, new_job_next] = arrive_prob
        if action == CONTINUE:
          rewards[action, s] = 0.
        elif action == ADMIT:
          rewards[action, s] = -1.
      elif new_job:
        if action == CONTINUE:
          no_new_job_next = to_global_state(jobs_in_queue=num_queued,
                                            new_job=False)
          new_job_next = s
          transitions[action, s, no_new_job_next] = complete_prob
          transitions[action, s, new_job_next] = arrive_prob
          rewards[action, s] = 0.
        elif action == ADMIT:
          no_new_job_next = to_global_state(jobs_in_queue=num_queued,
                                            new_job=False)
          new_job_next = to_global_state(jobs_in_queue=num_queued + 1,
                                         new_job=True)
          transitions[action, s, no_new_job_next] = complete_prob
          transitions[action, s, new_job_next] = arrive_prob
          rewards[action, s] = (admit_reward - cost_fn(
            jobs_waiting=num_queued + 1)) * joint_rate
    # General Case.
    elif num_queued > 0 and (num_queued < max_stored_jobs - 1):
      if not new_job:
        no_new_job_next = to_global_state(jobs_in_queue=num_queued - 1,
                                          new_job=False)
        new_job_next = to_global_state(jobs_in_queue=num_queued,
                                       new_job=True)
        transitions[action, s, no_new_job_next] = complete_prob
        transitions[action, s, new_job_next] = arrive_prob
        if action == CONTINUE:
          rewards[action, s] = (-cost_fn(jobs_waiting=num_queued)) * joint_rate
        elif action == ADMIT:
          # Small penalty for invalid action.
          rewards[action, s] = (-cost_fn(
            jobs_waiting=num_queued + 1)) * joint_rate

      elif new_job:
        if action == CONTINUE:
          no_new_job_next = to_global_state(jobs_in_queue=num_queued - 1,
                                            new_job=False)
          new_job_next = to_global_state(jobs_in_queue=num_queued,
                                         new_job=True)
          transitions[action, s, no_new_job_next] = complete_prob
          transitions[action, s, new_job_next] = arrive_prob
          rewards[action, s] = (-cost_fn(jobs_waiting=num_queued)) * joint_rate
        elif action == ADMIT:
          no_new_job_next = to_global_state(jobs_in_queue=num_queued,
                                            new_job=False)
          new_job_next = to_global_state(jobs_in_queue=num_queued + 1,
                                         new_job=True)
          transitions[action, s, no_new_job_next] = complete_prob
          transitions[action, s, new_job_next] = arrive_prob
          rewards[action, s] = (admit_reward - cost_fn(
            jobs_waiting=num_queued + 1)) * joint_rate

    # In our finite model, we cannot add more jobs in this state.
    elif num_queued == max_stored_jobs - 1:
      # Same as general case.
      if not new_job:
        new_job_next = to_global_state(jobs_in_queue=num_queued,
                                       new_job=True)
        no_new_job_next = to_global_state(jobs_in_queue=num_queued - 1,
                                          new_job=False)
        transitions[action, s, new_job_next] = arrive_prob
        transitions[action, s, no_new_job_next] = complete_prob
        if action == CONTINUE:
          rewards[action, s] = (-cost_fn(jobs_waiting=num_queued)) * joint_rate
        elif action == ADMIT:
          # Small penalty for invalid action.
          rewards[action, s] = (-cost_fn(
            jobs_waiting=num_queued + 1)) * joint_rate
      elif new_job:
        # Same as general case.
        if action == CONTINUE:
          new_job_next = to_global_state(jobs_in_queue=num_queued,
                                         new_job=True)
          no_new_job_next = to_global_state(jobs_in_queue=num_queued - 1,
                                            new_job=False)
          transitions[action, s, new_job_next] = arrive_prob
          transitions[action, s, no_new_job_next] = complete_prob
          rewards[action, s] = -cost_fn(jobs_waiting=num_queued) * joint_rate
        elif action == ADMIT:
          # Stuck here, cannot add another job to the queue.
          new_job_next = to_global_state(jobs_in_queue=num_queued,
                                         new_job=True)
          no_new_job_next = to_global_state(jobs_in_queue=num_queued,
                                            new_job=False)
          transitions[action, s, new_job_next] = arrive_prob
          transitions[action, s, no_new_job_next] = complete_prob
          # Same as passing b/c could not admit job.
          rewards[action, s] = -cost_fn(jobs_waiting=num_queued) * joint_rate
  name = f'MM1 {arrival_rate}:{service_rate}:{admit_reward}:{max_stored_jobs}:{dtype}:{cost_fn.func.__name__}'
  return structure.MarkovDecisionProcess(transitions=transitions,
                                         rewards=rewards,
                                         name=name)


MM1_QUEUE_1 = functools.partial(create,
                                arrival_rate=1.,
                                service_rate=1.,
                                admit_reward=10.,
                                cost_fn=functools.partial(linear_cost_fn,
                                                          cost_constant=1.),
                                max_stored_jobs=20)

MM1_QUEUE_2 = functools.partial(create,
                                arrival_rate=1.5,
                                service_rate=1.,
                                admit_reward=4.,
                                cost_fn=functools.partial(linear_cost_fn,
                                                          cost_constant=1.),
                                max_stored_jobs=20)

MM1_QUEUE_3 = functools.partial(create,
                                arrival_rate=1.,
                                service_rate=1.5,
                                admit_reward=4.,
                                cost_fn=functools.partial(linear_cost_fn,
                                                          cost_constant=1.),
                                max_stored_jobs=20)
