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
import glob
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
_LOAD_EXPERIMENT_ID = flags.DEFINE_string('experiment_id', '', 'String for new name, or to resume existing experiment.')
_CLEAR_PAST_RESULTS = flags.DEFINE_bool('clear_past_results', False, 'Erase previous results.')

from differential_value_iteration.algorithms import dvi
from differential_value_iteration.environments import micro
from differential_value_iteration.experiments.minimanager import conductor

@dataclasses.dataclass
class Work:
  env_constructor: Callable[..., Any]
  env_params: Dict[str, Any]

  agent_constructor: Callable[..., Any]
  agent_params: Dict[str, Any]

  run_loop: Callable[..., Any]
  run_params: Dict[str, Dict[str, Any]]



def experiment_fn(agent, num_iterations: int):
  for i in range(num_iterations):
    time.sleep(.025)
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
  # step_sizes = [1., .9, .5,]
  # betas = [1., .1, .5, .2,]
  step_sizes = [1., .9, .5, .2, .3, .5]
  betas = [1., .1, .5, .2, .8, .1, .1, .1, .1]
  initial_r_bars = [0., 1.]
  synchronized = [False, True]

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
  logging.set_verbosity(logging.DEBUG)


  # Should use a flag-settable prefix.
  results_dirname = 'results'
  results_path = os.path.join(os.getcwd(), results_dirname)
  logging.debug('Results path will be:%s',results_path)

  if os.path.exists(results_path):
    logging.debug('Results path exists already.')
    if _CLEAR_PAST_RESULTS.value:
      logging.debug('Clearing existing results.')
      results_files_path = os.path.join(results_path, '*.results')
      results_files = glob.glob(results_files_path)
      for f in results_files:
        os.unlink(f)
      status_files_path = os.path.join(results_path, '*.status')
      status_files = glob.glob(status_files_path)
      for f in status_files:
        os.unlink(f)
  else:
    logging.debug("Creating results path")
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

  logging.info("Starting experiment: %s", experiment_id)

  experiment_params = conductor.ExperimentParams(save_path=results_path,
                                                 experiment_id=experiment_id,
                                                 work_fn=do_work)
  experiment_runner = conductor.Conductor(experiment_params=experiment_params)

  # Could also just call experiment_runner.resume() if we're sure we can resume.
  all_work = generate_work()
  experiment_runner.run(all_work, try_resume=resume)

  logging.info("Control returned to main, program complete.")



if __name__ == '__main__':
  app.run(main)