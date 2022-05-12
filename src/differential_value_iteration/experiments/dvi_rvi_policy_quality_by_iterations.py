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
from differential_value_iteration.algorithms import rvi
from differential_value_iteration.environments import garet
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import mm1_queue
from differential_value_iteration.environments import structure

FLAGS = flags.FLAGS
_NUM_ITERS = flags.DEFINE_integer('num_iters', 128,
                                  'Number of iterations per algorithm')

# Environment flags
_MDP1 = flags.DEFINE_bool('mdp1', True, 'Include MDP1 in benchmark')
_MDP4 = flags.DEFINE_bool('mdp4', True, 'Include MDP4 in benchmark')
_GARET1 = flags.DEFINE_bool('garet1', True, 'Include GARET 1 in benchmark')
_GARET2 = flags.DEFINE_bool('garet2', True, 'Include GARET 2 in benchmark')
_GARET3 = flags.DEFINE_bool('garet3', True, 'Include GARET 3 in benchmark')
_GARET_100 = flags.DEFINE_bool('garet_100', True,
                               'Include GARET 100 in benchmark.')
_MM1_1 = flags.DEFINE_bool('MM1_1', True, 'Include MM1 Queue 1 in benchmark')
_MM1_2 = flags.DEFINE_bool('MM1_2', True, 'Include MM1 Queue 2 in benchmark')
_MM1_3 = flags.DEFINE_bool('MM1_3', True, 'Include MM1 Queue 3 in benchmark')


def run(
    environments: Sequence[structure.MarkovRewardProcess],
    algorithm_constructors: Sequence[Callable[..., algorithm.Evaluation]],
    num_iters: int,
    measure_iters: Sequence[int]):
  """Runs a list of algorithms on a list of environments and prints outcomes.
    Params:
      environments: Sequence of Markov Decision Processes to run.
      algorithm_constructors: Sequence of Callable algorithm constructors. If an
        algorithm has hyperparameters, it should have multiple entries in here
        with hypers preset using functools.partial.
      num_iters: Total number of iterations for benchmark.
      measure_iters: Which iterations to evaluate policy on. Final policy is
        always evaluate.
      """
  for environment in environments:
    print(f'\nEnvironment: {environment.name}\n----------------------')
    initial_values = np.zeros(environment.num_states)
    for algorithm_constructor in algorithm_constructors:
      total_time = 0.
      converged = False
      diverged = False
      alg = algorithm_constructor(mdp=environment,
                                  initial_values=initial_values,
                                  synchronized=True)
      module_name = alg.__class__.__module__.split('.')[-1]
      alg_name = module_name + '::' + alg.__class__.__name__
      print(f'\n{alg_name}')
      changes = [0.]  # Dummy starting value.
      last_policy = None
      policy_switches = 0
      all_late_policies = set()
      for i in range(num_iters):
        if i in measure_iters:
          measure_policy(i, alg, environment, eval_all_states=False,
                         last_change=np.mean(np.abs(changes)), final=False)
        start_time = time.time()
        changes = alg.update()
        end_time = time.time()
        total_time += end_time - start_time
        # Mean instead of sum so tolerance scales with num_states.
        change_summary = np.mean(np.abs(changes))
        if i == 10000:
          last_policy = str(alg.greedy_policy())
          all_late_policies.add(last_policy)
        if i > 10000:
          this_policy = str(alg.greedy_policy())
          if this_policy != last_policy:
            policy_switches += 1
            last_policy = this_policy
            all_late_policies.add(last_policy)

        if alg.diverged():
          diverged = True
          converged = False
          break
        convergence_tolerance = .01
        if change_summary <= convergence_tolerance and i > 1:
          converged = True
          break
      converged_string = 'YES\t' if converged else 'NO\t'

      measure_policy(i, alg, environment, eval_all_states,
                     last_change=np.mean(np.abs(changes)), final=True)

      if diverged:
        converged_string = 'DIVERGED'
      print(
          f'Summary: Average Time:{1000. * total_time / i:.3f} ms\tConverged:{converged_string}\t{i} iters\tMean final Change:{np.mean(np.abs(changes)):.5f}')
      if not converged:
        print(
          f'After iter 10000, policy switched:{policy_switches} times and saw these policies:{all_late_policies}')


def measure_policy(iteration: int, alg: algorithm.Control,
    environment: structure.MarkovDecisionProcess, eval_all_states: bool,
    last_change: float, final: bool):
  policy = alg.greedy_policy()
  mean_returns, std_returns = estimate_policy_average_reward(policy,
                                                             environment,
                                                             eval_all_states)
  if final:
    print('Evaluation after final iter', end='\t')
  else:
    print('Evaluation at iteration:', iteration, end='\t')
  print('Returns', end=':')
  for mean_return, std_return in zip(mean_returns, std_returns):
    print(f'{mean_return:.2f} ({std_return:.2f})', end=' ')

  print(f'Last Change:{last_change:.5f}', end=' ')
  policy_entries = np.unique(policy)
  if len(policy_entries) == 1:
    print(f'\tPolicy: {policy_entries} everywhere.')
  else:
    if len(policy) > 15:
      end_actions = np.unique(policy[15:])
      if len(end_actions) == 1:
        print(f'Policy: {policy[:15]} ... rest all {end_actions[0]}')
      elif final:
        print(f'\nFinal Policy: {policy}')
      else:
        print(f'Policy: {policy[:15]}...')
    else:
      print(f'Policy: {policy}')


def sample_return(policy, environment, start_state, length):
  stochastic = True if policy.ndim == 2 else False
  state = start_state
  total_return = 0.

  for _ in range(length):
    if stochastic:
      action = np.random.choice(a=environment.num_actions, p=policy[:, state])
    else:
      action = policy[state]
    total_return += environment.rewards[action, state]
    state = np.random.choice(a=environment.num_states,
                             p=environment.transitions[action, state])
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


def main(argv):
  del argv  # Stop linter from complaining about unused argv.

  algorithm_constructors = []

  dvi_algorithm = functools.partial(dvi.Control,
                                    step_size=1.,
                                    beta=1.,
                                    divide_beta_by_num_states=True,
                                    initial_r_bar=0.)
  algorithm_constructors.append(dvi_algorithm)

  rvi_algorithm = functools.partial(rvi.Control,
                                    step_size=1.,
                                    reference_index=0)
  algorithm_constructors.append(rvi_algorithm)

  environments = []
  problem_dtype = np.float64
  if _MDP1.value:
    environments.append(micro.create_mdp1(dtype=problem_dtype))
  if _MDP4.value:
    environments.append(micro.create_mdp4(dtype=problem_dtype))
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
  if _MM1_2.value:
    environments.append(mm1_queue.MM1_QUEUE_2(dtype=problem_dtype))
  if _MM1_3.value:
    environments.append(mm1_queue.MM1_QUEUE_3(dtype=problem_dtype))

  measure_iters = [i for i in range(_NUM_ITERS.value)]
  if not environments:
    raise ValueError('At least one environment required.')

  run(environments=environments,
      algorithm_constructors=algorithm_constructors,
      num_iters=_NUM_ITERS.value,
      measure_iters=measure_iters)


if __name__ == '__main__':
  app.run(main)
