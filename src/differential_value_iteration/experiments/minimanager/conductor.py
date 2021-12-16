'''Provides basic multi-processing experiment management.

Work is submitted to the conductor.

This conductor spawns processes and executes that work in semi-sequential order.

Results will be pickled into a binary 
files. The conductor also periodically serializes the overall state to a file so that a long-running series of 
work can be resumed in case of the experiment being killed or crashing.

No effort is made to continue or resume partial jobs.

It is possible that rarely a piece of work may end with its results recorded twice because of timing when 
an experiment is stopped midway and later resumed.
'''
import array
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
import ctypes
import numpy as np
from absl import logging
import datetime
_EMPTY_SLEEP_TIME = 2

import enum

class JobStatus(enum.Enum):
  READY = 0
  ACTIVE = 1
  COMPLETE = 2

_STATUS_TYPE = 'H'  # Shared memory array type - (unsigned short).

@dataclasses.dataclass
class JobEvent:
  '''Store diagnostic info about jobs.'''
  timestamp: datetime.datetime
  worker_name: str
  event_type: str
  description: str


@dataclasses.dataclass
class Job:
  '''Container class for storing and tracking work internally.'''
  id: int
  work: Any  # TODO(tanno): Might be nice to constrain this somehow.
  history: Sequence[JobEvent]


@dataclasses.dataclass
class Result:
  '''Container class for storing results of work.'''
  job: Job
  work_result: Any  # TODO(tanno): Might be nice to constrain this somehow.
  duration: float


@dataclasses.dataclass
class ExperimentParams:
  results_path: os.path
  experiment_id: str
  work_fn: Callable[[Any], Any]


@dataclasses.dataclass
class ExperimentDefinition:
  all_jobs: Dict[int, Job]
  job_status: multiprocessing.Array

  def queue_incomplete_jobs(self, job_queue: multiprocessing.Queue):
    for job_id, status in enumerate(self.job_status):
      if status == JobStatus.READY.value:
        job_queue.put(self.all_jobs[job_id])

  @classmethod
  def from_work(cls,
      work: Sequence[Any],
      )->'ExperimentDefinition':

    all_jobs = {}
    for job_id, w in enumerate(work):
      job = Job(id=job_id, work=w)
      all_jobs[job_id] = job

    return cls(all_jobs=all_jobs, job_status=multiprocessing.Array(_STATUS_TYPE, len(all_jobs), lock=True))

  @classmethod
  def from_checkpoint(cls, definition_path: os.PathLike)->'ExperimentDefinition':
    with open(definition_path, 'rb') as infile:
      all_jobs = pickle.load(infile)
      job_status_local = pickle.load(infile)
      infile.close()
    status_shared = multiprocessing.Array(_STATUS_TYPE, job_status_local, lock=True)
    return cls(all_jobs=all_jobs, job_status=status_shared)

  def shutdown(self, save_file_path: os.PathLike):
    logging.info('Saving current progress b/c of shutdown.')
    # This is a recursive lock, so this is fine.
    self.job_status.get_lock().acquire()
    for i, status in enumerate(self.job_status):
      if status == JobStatus.ACTIVE.value:
        self.job_status[i] = JobStatus.READY.value
    self.save(save_file_path=save_file_path)
    self.job_status.get_lock().release()

  def save(self, save_file_path: os.PathLike):
    logging.debug('Saving current progress.')
    self.job_status.get_lock().acquire()
    # Convert any active jobs into ready jobs. Long term, this could create duplicate results.
    with open(save_file_path, 'wb') as outfile:
      pickle.dump(self.all_jobs, outfile)
      local_array = array.array(_STATUS_TYPE, self.job_status.get_obj())
      pickle.dump(local_array, outfile)
      outfile.close()
    self.job_status.get_lock().release()

  def num_active_jobs(self)->int:
    self.job_status.get_lock().acquire()
    # NOT A GREAT WAY TO DO THIS
    num_active_jobs = sum(status==JobStatus.ACTIVE.value for status in self.job_status)
    self.job_status.get_lock().release()
    return num_active_jobs


  def set_job_starting(self, job_id: int):
    self.job_status.get_lock().acquire()
    self.job_status[job_id] = JobStatus.ACTIVE.value
    self.job_status.get_lock().release()
    logging.debug('Worker set job %s starting, status are:%s', job_id, [status for status in self.job_status])

  def set_job_complete(self, job_id: int):
    self.job_status.get_lock().acquire()
    try:
      logging.debug('Trying set job %s complete. status:%s', job_id, [status for status in self.job_status])
      assert(self.job_status[job_id] == JobStatus.ACTIVE.value)
      self.job_status[job_id] = JobStatus.COMPLETE.value
    except Exception as e:
      logging.error('Exception setting job complete: %s %s', e, type(e))
    self.job_status.get_lock().release()


class Conductor:
  jobs: multiprocessing.Queue
  results: multiprocessing.Queue
  shutdown_signal: multiprocessing.Value
  workers_done: multiprocessing.Value
  experiment_params: ExperimentParams
  num_processes: int
  definition: Optional[ExperimentDefinition]
  main_process: bool

  def __init__(self,
               experiment_params: ExperimentParams,
               num_processes: Optional[int] = None):
    print(f'conductor init logging verbosity={logging.get_verbosity()}')
    self.main_process = True
    # This might be a good case to go back to using contexts.
    self.log_level = logging.get_verbosity()
    logging.info('info')
    logging.debug('debug')
    self.definition = None
    self.experiment_params = experiment_params
    self.num_processes = num_processes or multiprocessing.cpu_count()

    self.jobs = multiprocessing.Queue()
    self.results = multiprocessing.Queue()
    self.shutdown_signal = multiprocessing.Value(ctypes.c_bool, False)
    self.workers_done = multiprocessing.Value(ctypes.c_bool, False)
    self.num_sigints = 0


  def handle_jobs(self):
    self.main_process = False
    logging.set_verbosity(self.log_level)
    process_name = 'Jobs_' + multiprocessing.process.current_process().name
    logging.info('%s Starting', process_name)
    while not self.shutdown_signal.value:
      try:
        job = self.jobs.get_nowait()
        # Note that we're working on this job.
        self.definition.set_job_starting(job.id)
        result = self.do_work(job)
        self.results.put(result)
      except queue.Empty:
        # If something fails, active jobs could get re-queued. Don't die!
        print(f'{process_name}: Job Queue empty. Checking for unfinished jobs.')
        num_active_jobs = self.definition.num_active_jobs()
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
    self.main_process = False
    logging.set_verbosity(self.log_level)
    process_name = 'HandleResult_' + multiprocessing.process.current_process().name
    results_filename = self.experiment_params.experiment_id + '.results'
    results_path = os.path.join(self.experiment_params.results_path,results_filename)
    definition_filename = self.experiment_params.experiment_id + '.status'
    definition_path = os.path.join(self.experiment_params.results_path, definition_filename)

    print(f'{process_name}: Starting.')
    while not self.shutdown_signal.value:
      try:
        result = self.results.get_nowait()
        print(f'{process_name} : Beep boop. Processing result for Job:{result.job.id}')
        logging.debug('%s Opening results file for appending', process_name)
        with open(results_path, 'ab') as outfile:
          logging.debug('%s Dumping pickled result to outfile', process_name)
          pickle.dump(result, outfile)
          outfile.close()

        # Note that we're done on this job.
        self.definition.set_job_complete(result.job.id)
      except queue.Empty:
        logging.info('%s Result Queue empty. Checking for unfinished jobs.', process_name)
        job_queue_has_anything = self.jobs.qsize() > 0
        num_active_jobs = self.definition.num_active_jobs()
        if (job_queue_has_anything or num_active_jobs > 0):
          # Looks like there is still work todo or ongoing.
          if self.workers_done.value:
            # But, all the workers have finished or died.
            logging.info('%s Calling for shutdown because work to do but all workers stopped.', process_name)
            self.shutdown_signal.value = True
          else:
            # Workers still alive, so just wait for something to do.
            logging.info('%s Still jobs todo or pending results. Good time to checkpoint and sleep.', process_name)
            self.definition.save(definition_path)
            time.sleep(_EMPTY_SLEEP_TIME)
            continue
        else:
          print(f'{process_name}: Job queue empty and active jobs empty.')
          break
      except Exception as exception:
        logging.error('%s: threw exception: %s (%s}. Requesting shutdown.', process_name, exception, type(exception))
        self.shutdown_signal.value = True

    if self.shutdown_signal:
      logging.info('%s Shutting down.', process_name)
    else:
      logging.info('%s Completing.', process_name)

  def handle_shutdown(self):
    definition_filename = self.experiment_params.experiment_id + '.status'
    definition_path = os.path.join(self.experiment_params.results_path, definition_filename)
    logging.info('Saving current progress because of shutdown.')
    self.definition.shutdown(definition_path)

  def handle_signals(self, sig, frame):
    del sig, frame
    self.num_sigints += 1
    if self.num_sigints > 5:
      print(f'You really want to force quit, ok, ok!')
      sys.exit(0)
    if not self.main_process:
      return
    logging.warning('Received CTRL-C. Saving progress and exiting.')
    self.handle_shutdown()
    sys.exit(0)

  def run(self, work: Sequence[Any]):
    self.definition = ExperimentDefinition.from_work(work)
    self._run()

  def resume(self):
    definition_filename = self.experiment_params.experiment_id + '.status'
    definition_path = os.path.join(self.experiment_params.results_path,
                                   definition_filename)
    self.definition = ExperimentDefinition.from_checkpoint(definition_path)
    self._run()

  def _run(self):
    self.definition.queue_incomplete_jobs(self.jobs)
    signal.signal(signal.SIGINT, self.handle_signals)

    result_process = multiprocessing.Process(target=self.handle_results,
                                          name='Results',
                                          daemon=True)
    result_process.start()
    processes = []
    for i in range(self.num_processes):
      p = multiprocessing.Process(target=self.handle_jobs,
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
    self.handle_shutdown()
    print('Conductor Complete.')