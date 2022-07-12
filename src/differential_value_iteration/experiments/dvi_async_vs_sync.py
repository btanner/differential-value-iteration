"""Compares performance of DVI sync vs async with various update strategies."""

import functools
import matplotlib
import numpy as np
from absl import app
from absl import flags

from differential_value_iteration.algorithms import async_strategies
from differential_value_iteration.algorithms import dvi
from differential_value_iteration.environments import garet
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import mm1_queue
from differential_value_iteration.experiments import simple_experiment_runner

matplotlib.use('Agg')
from matplotlib import pyplot as plt

FLAGS = flags.FLAGS
_MAX_ITERS = flags.DEFINE_integer('max_iters', 128,
                                  'Number of iterations per algorithm')
_STOCHASTIC_RUNS = flags.DEFINE_integer('stoch_runs', 100,
                                  'Number of runs for randomized algorithms')
_CONVERGENCE_TOL = flags.DEFINE_float('convergence_tol', .0001,
                                      'Maximum mean absolute change to be considered converged.')

_LOG_X_SCALE = flags.DEFINE_bool('log_x_scale', True, 'Use log scale of X in plots.')

# Environment flags
_GARET1 = flags.DEFINE_bool('garet_1', True, 'Include GARET 1 in benchmark')
_GARET2 = flags.DEFINE_bool('garet_2', True, 'Include GARET 2 in benchmark')
_GARET3 = flags.DEFINE_bool('garet_3', True, 'Include GARET 3 in benchmark')
_GARET_100 = flags.DEFINE_bool('garet_100', True,
                               'Include GARET 100 in benchmark.')
_MM1_1 = flags.DEFINE_bool('mm1_1', True, 'Include MM1 Queue 1 in benchmark')
_MM1_2 = flags.DEFINE_bool('mm2_2', True, 'Include MM1 Queue 2 in benchmark')
_MM1_3 = flags.DEFINE_bool('mm3_3', True, 'Include MM1 Queue 3 in benchmark')


def main(argv):
  del argv  # Stop linter from complaining about unused argv.

  environments = []
  problem_dtype = np.dtype(np.float64)
  if _GARET1.value:
    environments.append(('GARET 1', garet.GARET1(dtype=problem_dtype)))
  if _GARET2.value:
    environments.append(('GARET 2', garet.GARET2(dtype=problem_dtype)))
  if _GARET3.value:
    environments.append(('GARET 3', garet.GARET3(dtype=problem_dtype)))
  if _GARET_100.value:
    environments.append(('GARET 100', garet.GARET_100(dtype=problem_dtype)))
  if _MM1_1.value:
    environments.append(('Queueing 1', mm1_queue.MM1_QUEUE_1(dtype=problem_dtype)))
  if _MM1_2.value:
    environments.append(('Queueing 2', mm1_queue.MM1_QUEUE_2(dtype=problem_dtype)))
  if _MM1_3.value:
    environments.append(('Queueing 3', mm1_queue.MM1_QUEUE_3(dtype=problem_dtype)))

  if not environments:
    raise ValueError('At least one environment required.')

  async_strats = {'Round Robin':(async_strategies.RoundRobinASync, 1),
            'Random with Replacement':(async_strategies.RandomAsync, _STOCHASTIC_RUNS.value),
            'Random with no Replacement':(async_strategies.RandomAsyncWithoutReplacement, _STOCHASTIC_RUNS.value),
  }
  for env_name, environment in environments:
    print(env_name, environment.name)
    dvi_algorithm = dvi.Control(step_size=1.,
                                beta=1.,
                                mdp=environment,
                                initial_values=np.zeros(environment.num_states),
                                divide_beta_by_num_states=True,
                                initial_r_bar=0.,
                                synchronized=True)
    dvi_results = simple_experiment_runner.run_algorithm(dvi_algorithm,
                                                         max_iters=_MAX_ITERS.value,
                                                         converged_tol=_CONVERGENCE_TOL.value)

    async_results = {}
    max_async_iterations = 0
    for name, (async_strat, num_runs) in async_strats.items():
      async_results[name] = []
      for run in range(num_runs):
        #HACKY
        if num_runs > 1:
          seeded_strat = functools.partial(async_strat, seed=42+run)
        else:
          seeded_strat = async_strat
        dvi_async_algorithm = dvi.Control(step_size=1.,
                                          beta=1.,
                                          mdp=environment,
                                          initial_values=np.zeros(
                                              environment.num_states),
                                          divide_beta_by_num_states=True,
                                          initial_r_bar=0.,
                                          synchronized=False,
                                          async_manager_fn=seeded_strat)
        this_run_result = simple_experiment_runner.run_algorithm(
            dvi_async_algorithm,
            max_iters=_MAX_ITERS.value * environment.num_states,
            converged_tol=_CONVERGENCE_TOL.value)
        if len(this_run_result.policies) > max_async_iterations:
          max_async_iterations = len(this_run_result.policies)
        async_results[name].append(this_run_result)
    max_async_full_iterations = max_async_iterations // environment.num_states

    # # async_manager_fn = functools.partial(async_strategies.ConvergeRoundRobinASync, tol=_CONVERGENCE_TOL.value)
    # # async_manager_fn = async_strategies.RandomAsyncWithoutReplacement
    # async_manager_fn = async_strategies.RoundRobinASync
    # async_dvi_results = simple_experiment_runner.run_algorithm(
    #   dvi_async_algorithm,
    #   max_iters=_MAX_ITERS.value * environment.num_states,
    #   converged_tol=_CONVERGENCE_TOL.value)
    i = 0
    # print(async_dvi_results.state_values)
    # print(dvi_results.state_values)
    iterations_to_print = max(len(dvi_results.policies), max_async_full_iterations)

    value_distances = {'Sync DVI': []}
    policy_distances = {'Sync DVI': []}
    for async_name, results in async_results.items():
      value_distances[async_name] = tuple([] for _ in results)
      policy_distances[async_name] = tuple([] for _ in results)

    # Initial distance from 0.
    sync_val_distance = np.mean(np.abs(dvi_results.state_values[-1]))
    sync_policy_distance = np.mean(np.abs(dvi_results.policies[-1]))
    # sync_policy_distance = {name:np.mean(np.abs(dvi_results.policies[-1])) for name in async_results.keys()}
    # iterations_to_print = max(len(dvi_results.policies), len(
    #   async_dvi_results.policies) // environment.num_states)
    for iteration in range(iterations_to_print):
      print('Outer Iteration:', iteration)
      print('-------------------')
      for s in range(environment.num_states):
        for name, results in async_results.items():
          for r_id, result in enumerate(results):
            if i < len(result.policies):
              async_val_distance = np.mean(np.abs(dvi_results.state_values[-1] - result.state_values[i]))
              async_policy_distance = np.mean(np.abs(dvi_results.policies[-1] - result.policies[i]))
              print(
                f'\ts:{s}\tAsync Val Dist:{async_val_distance:.3f}\tAsync Policy Dist:{async_policy_distance:.3f}')
            else:
              async_val_distance = np.mean(np.abs(dvi_results.policies[-1] - result.policies[-1]))
            value_distances[name][r_id].append(async_val_distance)
            policy_distances[name][r_id].append(async_policy_distance)
        value_distances['Sync DVI'].append(sync_val_distance)
        policy_distances['Sync DVI'].append(sync_policy_distance)
        i += 1
      if iteration < len(dvi_results.policies):
        sync_val_distance = np.mean(np.abs(
          dvi_results.state_values[-1] - dvi_results.state_values[iteration]))
        sync_policy_distance = np.mean(
          np.abs(dvi_results.policies[-1] - dvi_results.policies[iteration]))
        print(
          f'Sync Val Dist:{sync_val_distance:.3f}\tSync Policy Dist:{sync_policy_distance:.3f}')
      else:
        print(f'Sync converged by now.')

    plt.figure(figsize=(10, 5))
    ax = plt.axes()
    for name, series in value_distances.items():
      print(f'Plotting: {name}')
      if isinstance(series, list) or len(series) == 1:
        if isinstance(series, tuple):
          series = series[0]
        plt.plot(series, label=name)
      else:
        # Multiple results, let's do error bars.
        series_np = np.array(series)
        num_samples = len(series)
        std_dev = np.std(series_np, axis=0)
        std_error = std_dev / np.sqrt(num_samples)
        y = np.mean(series_np, axis=0)
        # import pdb
        # pdb.set_trace()
        plt.errorbar(x=list(range(i)),y=y, yerr=std_error, label=name)
        # plt.plot(y, label=name)
        # plt.fill_between(list(range(i)), y-std_error, y+std_error)
        print(f'plotting series with len {len(series)} and name:{name}')
        # print(series)
    # plt.plot(value_distances['async'])

    plt.legend(prop={'size':16})
    plt.title(f'{env_name}: Distance From Final Values',size=18)
    x_scale = 'Log' if _LOG_X_SCALE.value else 'Linear'

    plt.xlabel(f'Number of Updates ({x_scale} Scale)', size=16)
    plt.ylabel('Mean Absolute Difference (Std. Error)', size=16)
    if _LOG_X_SCALE.value:
      ax.set_xscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print('saving...')
    plt.savefig(f'plots/{environment.name}_value_distance.png')
    # plt.figure()
    # for name, series in policy_distances.items():
    #   print(f'Plotting: {name}')
    #   plt.plot(series, label=name)
    # plt.legend()
    # # plt.plot(value_distances['async'])
    # plt.savefig(f'plots/{environment.name}_policy_distance.png')
    # # plt.plot(policy_distances['sync'])
    # # plt.plot(policy_distances['async'])
    # # plt.savefig('policy_distance.png')
    # # print(dvi_results.converged)
    # # print(async_dvi_results.converged)
    # # print('Sync vs Async final value diff:', np.mean(np.abs(
    # #   dvi_results.state_values[-1] - async_dvi_results.state_values[-1])))
    # # print('Sync vs Async final policy diff:', np.mean(
    # #   np.abs(dvi_results.policies[-1] - async_dvi_results.policies[-1])))


if __name__ == '__main__':
  app.run(main)
