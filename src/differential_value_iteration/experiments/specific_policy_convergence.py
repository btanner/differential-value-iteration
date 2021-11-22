"""Runs control algorithms a few times to generate timing in a few problems."""

import functools
from typing import Callable
from typing import Sequence

import numpy as np
from absl import app
from absl import flags

from differential_value_iteration.algorithms import algorithm
from differential_value_iteration.algorithms import dvi
from differential_value_iteration.algorithms import mdvi
from differential_value_iteration.algorithms import rvi
from differential_value_iteration.environments import garet
from differential_value_iteration.environments import mm1_queue
from differential_value_iteration.environments import structure

FLAGS = flags.FLAGS
_NUM_ITERS = flags.DEFINE_integer('num_iters', 1000,
                                  'Number of iterations per algorithm')
_SYNCHRONIZED = flags.DEFINE_bool('synchronized', True,
                                  'Run algorithms in synchronized mode')
_32bit = flags.DEFINE_bool('32bit', False,
                           'Use 32 bit precision (default is 64 bit)')

_CONVERGENCE_TOLERANCE = flags.DEFINE_float('convergence_tolerance', 1e-5,
                                            'Tolerance for convergence')

_DVI = flags.DEFINE_bool('dvi', True, 'Run Differential Value Iteration')
_MDVI = flags.DEFINE_bool('mdvi', True,
                          'Run Multichain Differential Value Iteration')
_RVI = flags.DEFINE_bool('rvi', True, 'Run Relative Value Iteration')

# Environment flags
_GARET1 = flags.DEFINE_bool('garet1', True, 'Include GARET 1')
_GARET2 = flags.DEFINE_bool('garet2', True, 'Include GARET 2')
_MM1_1 = flags.DEFINE_bool('MM1_1', True, 'Include MM1 Queue 1')


def run(
    environments: Sequence[structure.MarkovRewardProcess],
    algorithm_constructors: Sequence[Callable[..., algorithm.Evaluation]],
    max_iters: int,
    convergence_tolerance: float,
    synchronized: bool):
  """Runs a list of algorithms on a list of environments and prints outcomes.
    Params:
      environments: Sequence of Markov Reward Processes to run.
      algorithm_constructors: Sequence of Callable algorithm constructors. If an
        algorithm has hyperparameters, it should have multiple entries in here
        with hypers preset using functools.partial.
      max_iters: Maximum number of iterations before declaring fail to converge.
      convergence_tolerance: Criteria for convergence.
      synchronized: Run algorithms in synchronized or asynchronous mode.
      """
  for environment in environments:
    initial_values = np.zeros(environment.num_states)
    inner_loop_range = 1 if synchronized else environment.num_states
    print(f'\nEnvironment {environment.name}\n----------------------')
    for algorithm_constructor in algorithm_constructors:
      alg = algorithm_constructor(mrp=environment,
                                  initial_values=initial_values,
                                  synchronized=synchronized)
      module_name = alg.__class__.__module__.split('.')[-1]
      alg_name = module_name + '::' + alg.__class__.__name__
      print(f'\nAlgorithm:{alg_name}')

      converged = False
      for i in range(max_iters):
        change_summary = 0.
        for _ in range(inner_loop_range):
          changes = alg.update()
          # Mean instead of sum so tolerance scales with num_states.
          change_summary += np.mean(np.abs(changes))
        # Basically divide by num_states if running async.
        change_summary /= inner_loop_range
        if alg.diverged():
          converged = False
          break

        if change_summary <= convergence_tolerance and i > 1:
          converged = True
          break
      print(
        f'step_size:\tConverged:{converged}\tafter {i} iterations\tMax Final Change:{np.max(np.abs(changes))}')


def main(argv):
  del argv  # Stop linter from complaining about unused argv.

  algorithm_constructors = []

  # Create constructors that only depends on params common to all algorithms.

  if _DVI.value:
    dvi_algorithm = functools.partial(dvi.Evaluation,
                                      step_size=1.,
                                      beta=1.,
                                      initial_r_bar=0.)
    algorithm_constructors.append(dvi_algorithm)

  if _MDVI.value:
    mdvi_algorithm = functools.partial(mdvi.Evaluation,
                                       step_size=1.,
                                       beta=1.,
                                       initial_r_bar=0.)
    algorithm_constructors.append(mdvi_algorithm)
  if _RVI.value:
    rvi_algorithm = functools.partial(rvi.Evaluation,
                                      step_size=1.,
                                      reference_index=0)
    algorithm_constructors.append(rvi_algorithm)

  if not algorithm_constructors:
    raise ValueError('No algorithms scheduled to run.')

  environments = []
  problem_dtype = np.float32 if _32bit.value else np.float64
  if _GARET1.value:
    mdp = garet.GARET1(dtype=problem_dtype)
    policy = (2, 1, 2, 2)
    mrp = mdp.as_markov_reward_process_from_deterministic_policy(policy)
    environments.append(mrp)
    policy = (2, 1, 2, 3)
    mrp = mdp.as_markov_reward_process_from_deterministic_policy(policy)
    environments.append(mrp)

  if _GARET2.value:
    mdp = garet.GARET2(dtype=problem_dtype)
    policy = (11, 15, 5, 14)
    mrp = mdp.as_markov_reward_process_from_deterministic_policy(policy)
    environments.append(mrp)
    policy = (11, 15, 7, 14)
    mrp = mdp.as_markov_reward_process_from_deterministic_policy(policy)
    environments.append(mrp)

  if _MM1_1.value:
    mdp = mm1_queue.MM1_QUEUE_1(dtype=problem_dtype)
    policy = [0] * mdp.num_states
    prefix = [0, 1, 0, 1, 0, 1, 0, 1]
    policy[:len(prefix)] = prefix
    mrp = mdp.as_markov_reward_process_from_deterministic_policy(policy)
    environments.append(mrp)
    policy = [0] * mdp.num_states
    prefix = [0, 1, 0, 1, 0, 1]
    policy[:len(prefix)] = prefix
    mrp = mdp.as_markov_reward_process_from_deterministic_policy(policy)
    environments.append(mrp)

  if not environments:
    raise ValueError('At least one environment required.')

  run(environments=environments,
      algorithm_constructors=algorithm_constructors,
      max_iters=_NUM_ITERS.value,
      convergence_tolerance=_CONVERGENCE_TOLERANCE.value,
      synchronized=_SYNCHRONIZED.value)


if __name__ == '__main__':
  app.run(main)
