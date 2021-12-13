'''Provides basic multi-processing experiment management.

Jobs are submitted to a job queue (pickled on disk).

This program spawns processes and executes those jobs in sequential order.

The results of the jobs are written to disk.

If the program is killed with Sigterm, it will take care to re-queue any active
jobs so that they will not be forgotten next time the program runs.

No effort is made to continue partial jobs.
'''
import dataclasses
import itertools
import multiprocessing
import os
import time
from datetime import datetime
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
from absl import app
from absl import logging
from absl import flags

FLAGS = flags.FLAGS
_LOAD_EXPERIMENT_ID = flags.DEFINE_string('experiment_name', '', '')


from differential_value_iteration.algorithms import dvi
from differential_value_iteration.environments import micro
from differential_value_iteration.experiments.minimanager import conductor

@dataclasses.dataclass
class Work:
  env_constructor: Callable[[Any, ...], Any]
  env_params: Dict[str, Any]

  agent_constructor: Callable[[Any, ...], Any]
  agent_params: Dict[str, Any]

  run_loop: Callable[[Any, ...], Any]
  run_params: Dict[str, Dict[str, Any]]



def experiment_fn(agent, num_iterations: int):
  for i in range(num_iterations):
    time.sleep(.01)
  return {'iters':i}


def do_work(work: Work):
  env = work.env_constructor(**work.env_params)
  agent_params = work.agent_params
  agent_params['initial_values'] = np.full(env.num_states, fill_value=0.)
  agent_params['mdp'] = env
  agent = work.agent_constructor(**agent_params)
  result = work.run_loop(agent=agent, **work.run_params)
  return result


def generate_work() -> Sequence[Work]:
  step_sizes = [1., .9, .5, .2, .3, .5]
  betas = [1., .1, .5, .2, .8, .1, .1, .1, .1]
  initial_r_bars = [0.]
  synchronized = [False]

  env_constructors = [micro.create_mdp1, micro.create_mdp2]
  agent_constructors = [dvi.Control]
  job_id = 0
  work = []
  for a, b, ir, s, e_fn, a_fn in itertools.product(step_sizes, betas, initial_r_bars, synchronized, env_constructors, agent_constructors):
    w = Work(env_params={'dtype':np.float64},
            env_constructor=e_fn,
            agent_params={'step_size':a,'beta':b, 'initial_r_bar':ir, 'synchronized': s},
            agent_constructor=a_fn,
            run_loop=experiment_fn,
            run_params={'num_iterations':100},
            )
    work.append(w)

  return work

def main(argv):

  # Should use a flag-settable prefix.
  results_dirname = 'results'
  results_path = os.path.join(os.getcwd(), results_dirname)
  print(f'Results path will be:{results_path}')
  if os.path.exists(results_path):
    print(f'Results path exists already.')
  else:
    print(f'Creating results path')
    os.makedirs(results_path, exist_ok=True)

  if _LOAD_EXPERIMENT_ID.value:
    experiment_id = _LOAD_EXPERIMENT_ID.value
    print(f'Trying to resume experiment: {experiment_id}')
    resume=True
  else:
    experiment_name = 'dvi_sweep'
    now = datetime.now().strftime('%Y_%d_%m_%H%M%S')
    experiment_id = experiment_name + '_' + now
    resume=False
    print(f'Starting experiment: {experiment_id}')

  experiment_params = conductor.ExperimentParams(results_path=results_path,
                                                 experiment_id=experiment_id,
                                                 work_fn=do_work)
  experiment_runner = conductor.Conductor(experiment_params=experiment_params)
  if resume:
    experiment_runner.resume()
  else:
    all_work = generate_work()
    experiment_runner.run(all_work)

  print('Control returned to main, program complete.')



if __name__ == '__main__':
  app.run(main)