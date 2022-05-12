"""Provides basic multi-processing experiment management.

Work is submitted to the conductor.

This conductor spawns processes and executes that work in semi-sequential order.

Results will be pickled into a binary
files. The conductor also periodically serializes the overall state to a file so that a long-running series of
work can be resumed in case of the experiment being killed or crashing.

No effort is made to continue or resume partial jobs.

It is possible that rarely a piece of work may end with its results recorded twice because of timing when
an experiment is stopped midway and later resumed.
"""
import ctypes
import dataclasses
import multiprocessing
import os
import pickle
import queue
import signal
import sys
import time
import traceback
from typing import Any, Callable, Optional, Sequence

from absl import logging
from differential_value_iteration.experiments.minimanager import job

_EMPTY_SLEEP_TIME = 2
_MAX_SIGNALS_FOR_FORCE_QUIT = 5
_Q_SIZE = 2 ** 15


@dataclasses.dataclass
class ExperimentParams:
    save_path: os.path
    experiment_name: str
    work_fn: Callable[[Any], Any]

    @property
    def results_file_path(self) -> os.path:
        results_filename = self.experiment_name + ".results"
        return os.path.join(self.save_path, results_filename)

    @property
    def status_file_path(self) -> os.path:
        status_filename = self.experiment_name + ".status"
        return os.path.join(self.save_path, status_filename)


class Conductor:
    """Manages load/save experiments and managing their workers and results.

    There are smelly variables here like main_process and log_verbosity.

    Depending on whether multiprocessing is spawn or fork, process details
    like log level, interrupt handling, etc. may or may not be copied to child
    processes. These variables help us handle those in a consistent way.
    """

    jobs: multiprocessing.Queue
    results: multiprocessing.Queue
    shutdown_signal: multiprocessing.Value
    workers_done: multiprocessing.Value
    experiment_params: ExperimentParams
    num_processes: int
    plan: Optional[job.WorkPlan]
    main_process: bool
    process_name: str

    def __init__(
        self, experiment_params: ExperimentParams, num_processes: Optional[int] = None
    ):
        self.main_process = True
        self.process_name = multiprocessing.current_process().name
        # This might be a good case to go back to using contexts.
        self.plan = None

        self.jobs = multiprocessing.Queue()
        # Ensure overloaded output pipe doesn't hang when we force shutdown.
        self.jobs.cancel_join_thread()

        self.results = multiprocessing.Queue()
        self.experiment_params = experiment_params
        self.num_processes = num_processes or multiprocessing.cpu_count()
        self.shutdown_signal = multiprocessing.Value(ctypes.c_bool, False)
        self.workers_done = multiprocessing.Value(ctypes.c_bool, False)
        self.num_sigints = 0
        self.log_verbosity = logging.get_verbosity()

    #  PUBLIC ENTRY POINTS
    def run(self, work: Sequence[Any], try_resume: bool = False):
        """External entry point for a new experiment."""
        if try_resume:
            logging.debug("Trying to resume from checkpoint.")
            self.plan = job.WorkPlan.from_checkpoint(
                self.experiment_params.status_file_path
            )

        if not self.plan:
            logging.fatal("DID NOT RESUME")

            self.plan = job.WorkPlan.from_work(
                work, self.experiment_params.status_file_path
            )
        self._run()

    def resume(self):
        """External entry point for an existing experiment."""
        definition_filename = self.experiment_params.experiment_name + ".status"
        definition_path = os.path.join(
            self.experiment_params.save_path, definition_filename
        )
        self.plan = job.WorkPlan.from_checkpoint(definition_path)
        self._run()

    # PRIVATE STUFF.

    def _setup_subprocess_details(self, process_name_prefix: str):
        logging.set_verbosity(self.log_verbosity)
        self.process_name = multiprocessing.current_process().name
        self.main_process = False
        base_process_name = multiprocessing.process.current_process().name
        self.process_name = process_name_prefix + base_process_name
        logging.info("%s Finished subprocess setup", self.process_name)

    def _handle_jobs(self):
        # Process entry point.
        self._setup_subprocess_details(process_name_prefix="Work")
        while not self.shutdown_signal.value:
            try:
                j = self.jobs.get_nowait()
                self.plan.set_job_starting(j.job_id)
                result = self._do_work(j)
                self.results.put(result)
            except queue.Empty:
                # Due to Multiprocess buffering, queue.Empty can be a false positive.
                qsize = self.jobs.qsize()
                if qsize == 0:
                    logging.info("%s: Queue empty.", self.process_name)
                    break
                else:
                    logging.debug(
                        "%s: Queue Empty but false positive b/c qsize:%s",
                        self.process_name,
                        qsize,
                    )
                    time.sleep(_EMPTY_SLEEP_TIME)
                    continue
            except Exception as exception:
                logging.error(
                    "%s: Got exception: %s (%s): %s",
                    self.process_name,
                    exception,
                    type(exception),
                    traceback.format_exc(),
                )
                break
        if self.shutdown_signal.value:
            logging.info("%s: Shutting down.", self.process_name)
        else:
            logging.info("%s: Completing.", self.process_name)

    def _handle_results(self):
        # Process entry point.
        self._setup_subprocess_details(process_name_prefix="Results")
        while not self.shutdown_signal.value:
            try:
                result = self.results.get_nowait()
                logging.info(
                    "%s: Processing result of job: %s", self.process_name, result.job_id
                )
                logging.debug(
                    "%s Opening results file for appending", self.process_name
                )
                with open(self.experiment_params.results_file_path, "ab") as outfile:
                    logging.debug("%s Appending result to file", self.process_name)
                    pickle.dump(result, outfile)

                # Note that we're done on this job.
                self.plan.set_job_complete(result.job_id)
            except queue.Empty:
                logging.debug(
                    "%s Result Queue empty. Checking for unfinished jobs.",
                    self.process_name,
                )
                # Even if queue.Empty, could be items in results queue b/c of pipes/buffering.
                results_are_pending = self.results.qsize() > 0
                job_queue_has_anything = self.jobs.qsize() > 0
                num_active_jobs = self.plan.num_active_jobs()
                if results_are_pending or job_queue_has_anything or num_active_jobs > 0:
                    # Looks like there is still work todo or ongoing.
                    if self.workers_done.value:
                        # But, all the workers have finished or died.
                        logging.error(
                            "%s Calling for shutdown because work to do but all workers stopped.",
                            self.process_name,
                        )
                        self.shutdown_signal.value = True
                    else:
                        # Workers still alive, wait for something to do.
                        logging.debug(
                            "%s Jobs todo or pending results. Checkpoint and sleep.",
                            self.process_name,
                        )
                        self.plan.save()
                        time.sleep(_EMPTY_SLEEP_TIME)
                        continue
                else:
                    logging.info(
                        "%s: Job queue empty and active jobs empty.", self.process_name
                    )
                    break
            except Exception as exception:
                logging.error(
                    "%s: threw exception: %s (%s}. Requesting shutdown.",
                    self.process_name,
                    exception,
                    type(exception),
                )
                self.shutdown_signal.value = True

        if self.shutdown_signal:
            logging.info("%s Shutting down.", self.process_name)
        else:
            logging.info("%s Completing.", self.process_name)

    def _do_work(self, j: job.Job):
        logging.debug("starting do_work on job %s", j.job_id)
        start_time = time.time()
        work_results = self.experiment_params.work_fn(j.work)
        result = job.Result(
            job_id=j.job_id,
            experiment_name=self.experiment_params.experiment_name,
            work_result=work_results,
            duration=time.time() - start_time,
        )
        logging.debug("completed do_work on job %s", j.job_id)
        return result

    def _handle_shutdown(self):
        """Catching SIGINT or main process ending will call this."""
        logging.info("Saving current progress because of shutdown.")
        self.plan.shutdown()

    def _handle_signals(self, sig, frame):
        """Catches a signal (like SIGINT [Ctrl-C]) and try to shutdown gracefully."""
        del sig, frame
        self.num_sigints += 1
        if self.num_sigints > _MAX_SIGNALS_FOR_FORCE_QUIT:
            logging.fatal(
                "More than %s SIGINTS. You really want to force quit, ok, ok!",
                _MAX_SIGNALS_FOR_FORCE_QUIT,
            )
        # If multiprocessing in FORK mode, this handler will exist on all processes.
        if not self.main_process:
            return
        logging.warning(
            "%s: Received CTRL-C. Trying to shut down gracefully.", self.process_name
        )
        self._handle_shutdown()
        logging.info("%s: Calling sys.exit", self.process_name)
        sys.exit(0)

    def _run(self):
        self.plan.queue_ready_jobs(self.jobs)
        # Register to handle CTRL-c.
        signal.signal(signal.SIGINT, self._handle_signals)

        result_process = multiprocessing.Process(
            target=self._handle_results, name="Results", daemon=True
        )
        result_process.start()
        worker_processes = []
        for i in range(self.num_processes):
            worker = multiprocessing.Process(
                target=self._handle_jobs, name=f"Worker_{i}", daemon=True
            )
            worker_processes.append(worker)
            logging.debug("Main process starting worker process: %s", worker)
            worker.start()

        # Main Process will spend most of lifecycle in this loop.
        main_loop_counter = 0
        while not self.shutdown_signal.value:
            time.sleep(1)
            main_loop_counter += 1
            for worker in worker_processes:
                if not worker.is_alive():
                    logging.info("Main process determined dead worker: %s", worker)
                    # Will skip the next process in loop iteration. That's ok.
                    worker_processes.remove(worker)
            if not worker_processes:
                logging.info("No living workers left.")
                self.workers_done.value = True
                break
            if main_loop_counter % 100 == 0:
                logging.info(
                    "Main Process: Loop%d\tApprox %d jobs to do",
                    main_loop_counter,
                    self.jobs.qsize(),
                )

        if self.shutdown_signal.value:
            logging.info("Shutdown from another process seen in main process.")
            self._handle_shutdown()

        for worker in worker_processes:
            logging.info("Waiting for process: %s to join.", worker)
            worker.join()

        logging.debug("Main process: All worker processes joined.")
        logging.info("Main process: Waiting on result processor process.")
        result_process.join()
        logging.info("Main process: Result processor jointed.")
        self._handle_shutdown()
        logging.info("Conductor complete.")
