"""Runs control algorithms a few times to generate timing in a few problems."""

import functools
import time
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
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import mm1_queue
from differential_value_iteration.environments import structure

FLAGS = flags.FLAGS
_NUM_ITERS = flags.DEFINE_integer('num_iters', 1000,
                                  'Number of iterations per algorithm.')
_SYNCHRONIZED = flags.DEFINE_bool('synchronized', True,
                                  'Run algorithms in synchronized mode.')
_32bit = flags.DEFINE_bool('32bit', False,
                           'Use 32 bit precision (default is 64 bit).')

_CONVERGENCE_TOLERANCE = flags.DEFINE_float('convergence_tolerance', 1e-5,
                                            'Tolerance for convergence.')

flags.DEFINE_bool('dvi', True, 'Run Differential Value Iteration')
flags.DEFINE_bool('mdvi', True, 'Run Multichain Differential Value Iteration')
flags.DEFINE_bool('rvi', True, 'Run Relative Value Iteration')

# Environment flags
_MDP1 = flags.DEFINE_bool('mdp1', True, 'Include MDP1 in benchmark.')
_MDP2 = flags.DEFINE_bool('mdp2', True, 'Include MDP2 in benchmark.')
_GARET1 = flags.DEFINE_bool('garet1', True, 'Include GARET 1 in benchmark.')
_GARET2 = flags.DEFINE_bool('garet2', True, 'Include GARET 2 in benchmark.')
_GARET3 = flags.DEFINE_bool('garet3', True, 'Include GARET 3 in benchmark.')
_GARET_100 = flags.DEFINE_bool('garet_100', True,
                               'Include GARET 100 in benchmark.')
_MM1_1 = flags.DEFINE_bool('MM1_1', True, 'Include MM1 Queue 1 in benchmark.')


def run(
    environments: Sequence[structure.MarkovRewardProcess],
    algorithm_constructors: Sequence[Callable[..., algorithm.Evaluation]],
    num_iters: int,
    convergence_tolerance: float,
    synchronized: bool):
  """Runs a list of algorithms on a list of environments and prints outcomes.
    Params:
      environments: Sequence of Markov Decision Processes to run.
      algorithm_constructors: Sequence of Callable algorithm constructors. If an
        algorithm has hyperparameters, it should have multiple entries in here
        with hypers preset using functools.partial.
      num_iters: Total number of iterations for benchmark.
      convergence_tolerance: Criteria for convergence.
      synchronized: Run algorithms in synchronized or asynchronous mode.
      """
  for environment in environments:
    print(f'\nEnvironment: {environment.name}\n----------------------')
    # print('\n', environment.rewards)
    initial_values = np.zeros(environment.num_states)
    inner_loop_range = 1 if synchronized else environment.num_states
    for algorithm_constructor in algorithm_constructors:
      total_time = 0.
      converged = False
      diverged = False
      alg = algorithm_constructor(mdp=environment,
                                  initial_values=initial_values,
                                  synchronized=synchronized)
      module_name = alg.__class__.__module__.split('.')[-1]
      alg_name = module_name + '::' + alg.__class__.__name__
      print(f'{alg_name}', end='\t')
      for i in range(num_iters):
        change_summary = 0.
        for _ in range(inner_loop_range):
          start_time = time.time()
          changes = alg.update()
          end_time = time.time()
          total_time += end_time - start_time
          # Mean instead of sum so tolerance scales with num_states.
          change_summary += np.mean(np.abs(changes))
        # Basically divide by num_states if running async.
        change_summary /= inner_loop_range
        if alg.diverged():
          diverged = True
          converged = False
          break

        if change_summary <= convergence_tolerance and i > 1:
          converged = True
          break
      converged_string = 'YES\t' if converged else 'NO\t'
      mean_return, std_return = estimate_policy_average_reward(alg, environment)

      if diverged:
        converged_string = 'DIVERGED'
      print(
        f'Average Time:{1000.*total_time/i:.3f} ms\tConverged:{converged_string}\t{i} iters\tMean final Change:{np.mean(np.abs(changes)):.5f} Avg Reward: {mean_return:.2f} ({std_return:.2f})')
      print(f'Policy: {alg.greedy_policy()}')
      # print(f'Estimates: {alg.get_estimates()}')

def estimate_policy_average_reward(alg, environment):
  policy = alg.greedy_policy()
  mc = environment.as_markov_chain_from_deterministic_policy(policy)
  sim_length = 1000
  num_reps = 100
  starting_state = 0
  returns = np.zeros(num_reps, dtype=np.float64)

  # Generatate sequences of states under greedy policy.
  simulation_results = mc.simulate(ts_length=sim_length,
                                   init=starting_state,
                                   num_reps=num_reps)
  for i, states_visited in enumerate(simulation_results):
    if max(states_visited) > len(policy):
      print(f'apparently visited states: {states_visited} but have policy: {policy}')
    # If this crashes, then mc.simulate is sampling invalid states b/c CDFS don't sum to 1.
    # In that case, check if any states_visited are == num_states, and if so, resample.
    actions_taken = policy[states_visited]
    rewards_seen = [environment.rewards[a, s] for a, s in zip (actions_taken, states_visited)]
    returns[i] = np.mean(rewards_seen)
  return np.mean(returns), np.std(returns)





def main(argv):
  del argv  # Stop linter from complaining about unused argv.

  algorithm_constructors = []

  # Create constructors that only depends on params common to all algorithms.
  if FLAGS.dvi:
    dvi_algorithm = functools.partial(dvi.Control,
                                      step_size=1.,
                                      beta=1.,
                                      initial_r_bar=0.)
    algorithm_constructors.append(dvi_algorithm)

  if FLAGS.mdvi:
    mdvi_algorithm_1 = functools.partial(mdvi.Control1,
                                         step_size=.5,
                                         beta=.5,
                                         initial_r_bar=0.,
                                         threshold=.1)
    algorithm_constructors.append(mdvi_algorithm_1)
    mdvi_algorithm_2 = functools.partial(mdvi.Control2,
                                         step_size=1.,
                                         beta=1.,
                                         initial_r_bar=0.,
                                         threshold=.01)
    algorithm_constructors.append(mdvi_algorithm_2)
  if FLAGS.rvi:
    rvi_algorithm = functools.partial(rvi.Control,
                                      step_size=.1,
                                      reference_index=0)
    algorithm_constructors.append(rvi_algorithm)

  if not algorithm_constructors:
    raise ValueError('No algorithms scheduled to run.')

  environments = []
  problem_dtype = np.float32 if _32bit.value else np.float64
  if _MDP1.value:
    environments.append(micro.create_mdp1(dtype=problem_dtype))
  if _MDP2.value:
    environments.append(micro.create_mdp2(dtype=problem_dtype))
  if _GARET1.value:
    environments.append(garet.GARET1(dtype=problem_dtype))
  if _GARET2.value:
    environments.append(garet.GARET2(dtype=problem_dtype))
  if _GARET3.value:
    environments.append(garet.GARET3(dtype=problem_dtype))
  if _GARET_100.value:
    environments.append(garet.GARET_100(dtype=problem_dtype))
  if _MM1_1.value:
    environments.append(mm1_queue.MM1_QUEUE_1(dtype=problem_dtype))

  if not environments:
    raise ValueError('At least one environment required.')

  run(environments=environments,
      algorithm_constructors=algorithm_constructors,
      num_iters=_NUM_ITERS.value,
      convergence_tolerance=_CONVERGENCE_TOLERANCE.value,
      synchronized=_SYNCHRONIZED.value)


if __name__ == '__main__':
  app.run(main)
