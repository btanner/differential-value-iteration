'''Provides basic multi-processing experiment management.

Jobs are submitted to a job queue (pickled on disk).

This program spawns processes and executes those jobs in sequential order.

The results of the jobs are written to disk.

If the program is killed with Sigterm, it will take care to re-queue any active
jobs so that they will not be forgotten next time the program runs.

No effort is made to continue partial jobs.
'''
import dataclasses
import multiprocessing
import os
import pickle
import queue
import signal
import sys
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Set

import numpy as np
from absl import logging

_EMPTY_SLEEP_TIME = 2


@dataclasses.dataclass
class Job:
  id: int
  work: Any

@dataclasses.dataclass
class Result:
  job: Job
  work_result: Any
  duration: float
  worker_name: str

@dataclasses.dataclass
class ExperimentParams:
  results_path: os.path
  experiment_id: str
  work_fn: Callable[[Any], Any]


@dataclasses.dataclass
class ExperimentDefinition:
  all_jobs: Dict[int, Job]
  incomplete_jobs: Set[int]
  complete_jobs: Set[int]


class Conductor:
  context: multiprocessing.context.BaseContext
  jobs: multiprocessing.Queue
  results: multiprocessing.Queue
  active_jobs: Dict[int, Job]
  lock: multiprocessing.Lock
  shutdown_signal: multiprocessing.Value
  workers_done: multiprocessing.Value
  experiment_params: ExperimentParams
  num_processes: int
  definition: Optional[ExperimentDefinition]

  def __init__(self, experiment_params: ExperimentParams,
      num_processes: Optional[int] = None):
    self.definition = None
    self.experiment_params = experiment_params
    self.num_processes = num_processes or multiprocessing.cpu_count()
    self.context = multiprocessing.get_context('spawn')
    manager = self.context.Manager()

    self.jobs = manager.Queue()
    self.results = manager.Queue()
    self.active_jobs = manager.dict()
    self._lock = self.context.Lock()
    self.shutdown_signal = manager.Value(typecode=bool, value=False)
    self.workers_done = manager.Value(typecode=bool, value=False)

  def lock(self):
    self._lock.acquire()

  def unlock(self):
    self._lock.release()

  def handle_jobs(self):
    process_name = 'Jobs_' + multiprocessing.process.current_process().name
    logging.info('%s Starting', process_name)
    while not self.shutdown_signal.value:
      try:
        job = self.jobs.get_nowait()
        # Note that we're working on this job.
        self.lock()
        self.active_jobs[job.id] = job
        self.unlock()

        result = self.do_work(job)
        self.results.put(result)
      except queue.Empty:
        # If something fails, active jobs could get re-queued. Don't die!
        print(f'{process_name}: Job Queue empty. Checking for unfinished jobs.')
        self.lock()
        num_active_jobs = len(self.active_jobs.keys())
        self.unlock()
        if num_active_jobs > 0:
          print(
            f'{process_name}: {num_active_jobs} jobs processing. Sleeping for {_EMPTY_SLEEP_TIME}s')
          time.sleep(_EMPTY_SLEEP_TIME)
          continue
        else:
          print(f'{process_name}: No active jobs.')
          break
      except Exception as e:
        print(f'{process_name}: Got exception: {e} {type(e)}.')
        break
    if self.shutdown_signal.value:
      print(f'{process_name}: Shutting down.')
    else:
      print(f'{process_name}: Completing.')

  def do_work(self, j: Job):
    print(f'run_job on job: {j.id}')
    start_time = time.time()
    work_results = self.experiment_params.work_fn(j.work)
    result = Result(job=j,
                    work_result=work_results,
                    duration=time.time() - start_time,
                    worker_name=multiprocessing.current_process().name)
    print(f'done job {j.id}')
    return result

  def handle_results(self):
    process_name = 'HandleResult_' + multiprocessing.process.current_process().name
    results_filename = self.experiment_params.experiment_id + '.results'
    definition_filename = self.experiment_params.experiment_id + '.def'
    print(f'{process_name}: Starting.')
    while not self.shutdown_signal.value:
      try:
        result = self.results.get_nowait()
        print(f'{process_name} : Beep boop. Processing result for Job:{result.job.id}')
        results_path = os.path.join(self.experiment_params.results_path,
                                    results_filename)
        with open(results_path, 'ab') as outfile:
          pickle.dump(result, outfile)
          outfile.close()

        # Note that we're done on this job.
        self.lock()
        self.definition.complete_jobs.add(result.job.id)
        self.definition.incomplete_jobs.remove(result.job.id)
        del self.active_jobs[result.job.id]
        self.unlock()
      except queue.Empty:
        print(
          f'{process_name}: Result Queue empty. Checking for unfinished jobs.')
        job_queue_has_anything = self.jobs.qsize() > 0
        self.lock()
        num_active_jobs = len(self.active_jobs.keys())
        self.unlock()
        if (job_queue_has_anything or num_active_jobs > 0):
          # Looks like there is still work todo or ongoing.
          if self.workers_done.value:
            # But, all the workers have finished or died.
            print(
              f'{process_name} Calling for shutdown because work to do but all workers stopped.')
            self.shutdown_signal.value = True
          else:
            # Workers still alive, so just wait for something to do.
            print(
              f'{process_name}: Still jobs todo or pending results. Good time to checkpoint. Sleeping.')
            self.lock()
            definition_path = os.path.join(self.experiment_params.results_path,
                                           definition_filename)
            with open(definition_path, 'wb') as outfile:
              pickle.dump(self.definition, outfile)
              outfile.close()
            self.unlock()

            time.sleep(_EMPTY_SLEEP_TIME)
            continue
        else:
          print(f'{process_name}: Job queue empty and active jobs empty.')
          break
      except Exception as exception:
        print(
          f'{process_name} threw exception:{exception}. Requesting shutdown.')
        self.shutdown_signal.value = True

    if self.shutdown_signal:
      print(f'{process_name}: Shutting down.')
    else:
      print(f'{process_name}: Completing.')

  def handle_shutdown(self):
    print(f'Handling shutdown.')
    self.lock()
    print(f'There are pending jobs? {len(self.active_jobs) > 0}')
    for k, v in self.active_jobs.items():
      print(f'Pushing job: {k} back into the queue')
      self.jobs.put(v)
    self.unlock()
    print('Still need to write these to a file before dying!')

  def handle_signals(self, sig, frame):
    del sig, frame
    print('You pressed Ctrl+C!')
    self.handle_shutdown()
    sys.exit(0)

  def run(self, work: Sequence[Any]):
    all_jobs = {}
    for job_id, w in enumerate(work):
      job = Job(id=job_id, work=w)
      self.jobs.put(job)
      all_jobs[job_id] = job
    self.definition = ExperimentDefinition(all_jobs=all_jobs,
                                           incomplete_jobs=set(all_jobs.keys()),
                                           complete_jobs=set())
    self._run()

  def resume(self):
    definition_filename = self.experiment_params.experiment_id + '.def'
    definition_path = os.path.join(self.experiment_params.results_path,
                                   definition_filename)
    with open(definition_path, 'rb') as infile:
      self.definition = pickle.load(infile)
      infile.close()
    for job_id in self.definition.incomplete_jobs:
      self.jobs.put(self.definition.all_jobs[job_id])
    self._run()

  def _run(self):
    signal.signal(signal.SIGINT, self.handle_signals)

    result_process = self.context.Process(target=self.handle_results,
                                          name='Results',
                                          daemon=True)
    result_process.start()
    processes = []
    for i in range(self.num_processes):
      p = self.context.Process(target=self.handle_jobs,
                               name=f'Worker_{i}',
                               daemon=True)
      processes.append(p)
      print(f'starting process:{p}')
      p.start()

    while not self.shutdown_signal.value:
      time.sleep(1)
      for p in processes:
        if not p.is_alive():
          print(f'RUN DETERMINED: DEAD PROCESS: {p}')
          # This will skip the next process, but that's ok.
          processes.remove(p)
      if not processes:
        print(f'No living workers left.')
        self.workers_done.value = True
        break

    if self.shutdown_signal.value:
      print(f'Shutdown Signal received in run.')
      self.handle_shutdown()

    for p in processes:
      print(f'joining process:{p}')
      p.join()

    print(f'All worker processes joined. Waiting on result processor process.')
    result_process.join()
    print('Conductor Complete.')