"""Support classes for conductor-based experiments."""
import enum
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


class JobStatus(enum.Enum):
    READY = 0
    ACTIVE = 1
    COMPLETE = 2


_STATUS_TYPE = "H"  # Shared memory array type - (unsigned short).


@dataclasses.dataclass
class JobEvent:
    """Store diagnostic info about jobs."""

    timestamp: datetime.datetime
    worker_name: str
    event_type: str
    description: str

    @classmethod
    def create(cls, event_type: str, description: str = "") -> "JobEvent":
        return cls(
            timestamp=datetime.datetime.now(),
            worker_name=multiprocessing.current_process().name,
            event_type=event_type,
            description=description,
        )


@dataclasses.dataclass
class Job:
    """Container class for storing and tracking work internally."""

    id: int
    work: Any  # TODO(tanno): Might be nice to constrain this somehow.
    history: Sequence[JobEvent]

    @classmethod
    def create(cls, id: int, work: Any) -> "Job":
        return cls(id=id, work=work, history=[JobEvent.create(event_type="created")])

    def log_start(self, description: str = ""):
        self.history.append(
            JobEvent.create(event_type="work_started", description=description)
        )

    def log_complete(self, description: str = ""):
        self.history.append(
            JobEvent.create(event_type="work_complete", description=description)
        )

    def log_event(self, event_type: str, description: str = ""):
        self.history.append(
            JobEvent.create(event_type=event_type, description=description)
        )


@dataclasses.dataclass
class Result:
    """Container class for storing results of work."""

    job_id: int
    experiment_id: str

    work_result: Any  # TODO(tanno): Might be nice to constrain this somehow.
    duration: float



@dataclasses.dataclass
class WorkPlan:
    all_jobs: Dict[int, Job]
    job_status: multiprocessing.Array
    status_file_path: os.PathLike
    
    @classmethod
    def from_work(cls, work: Sequence[Any], status_file_path: os.PathLike) -> "WorkPlan":
        """Create an WorkPlan from a sequence of work."""
        all_jobs = {}
        for job_id, w in enumerate(work):
            all_jobs[job_id] = Job.create(id=job_id, work=w)

        return cls(
            all_jobs=all_jobs,
            job_status=multiprocessing.Array(_STATUS_TYPE, len(all_jobs), lock=True),
            status_file_path=status_file_path
        )

    @classmethod
    def from_checkpoint(cls, status_file_path: os.PathLike) -> Optional["WorkPlan"]:
        """Loads a WorkPlan from a saved checkpoint so we can resume it."""
        if not os.path.isfile(status_file_path):
          logging.info("No WorkPlan file found to load.")
          return None
        try:
          with open(status_file_path, "rb") as infile:
              all_jobs = pickle.load(infile)
              job_status_local = pickle.load(infile)
              infile.close()
          for job_id, job in all_jobs.items():
              this_job_status = JobStatus(job_status_local[job_id])
              job.log_event(
                  event_type="loaded_from_file",
                  description=f"Status: {this_job_status.name} Prior Events:{len(job.history)}",
              )
          status_shared = multiprocessing.Array(_STATUS_TYPE, job_status_local, lock=True)
          return cls(all_jobs=all_jobs, job_status=status_shared, status_file_path=status_file_path)
        except Exception as e:
          logging.error("Failed to load WorkPlan because %s (%s)", e, type(e))
          return None


    def queue_ready_jobs(self, job_queue: multiprocessing.Queue):
        self.job_status.get_lock().acquire()
        logging.info("Putting jobs into the queue.")
        added_count = 0
        for job_id, status in enumerate(self.job_status):
            if status == JobStatus.READY.value:
                job_queue.put(self.all_jobs[job_id])
                added_count += 1
                self.all_jobs[job_id].log_event(event_type="queued")
        logging.info("Added %s/%s jobs.", added_count, len(self.job_status))
        self.job_status.get_lock().release()

    def shutdown(self):
        """Changes status of active jobs to ready and forces a save."""
        logging.info("Saving current progress b/c of shutdown.")
        # This is a recursive lock, so this nesting with save is fine.
        self.job_status.get_lock().acquire()
        logging.debug("Resetting active jobs")
        for i, status in enumerate(self.job_status):
            if status == JobStatus.ACTIVE.value:
                self.job_status[i] = JobStatus.READY.value
                self.all_jobs[i].log_event(
                    event_type="rereadying", description="Active during shutdown"
                )
        logging.debug("Actually saving")
        self.save()
        logging.debug("Done saving")
        self.job_status.get_lock().release()

    def save(self):
        logging.debug("Saving current progress.")
        self.job_status.get_lock().acquire()
        with open(self.status_file_path, "wb") as outfile:
            pickle.dump(self.all_jobs, outfile)
            # Don't want to pickle a lock-wrapped shared-memory array.
            local_array = array.array(_STATUS_TYPE, self.job_status.get_obj())
            pickle.dump(local_array, outfile)
            outfile.close()
        self.job_status.get_lock().release()

    def num_active_jobs(self) -> int:
        self.job_status.get_lock().acquire()
        # This is not the cheapest way to do this.
        # Consider tracking with a variable if a bottleneck.
        num_active_jobs = sum(
            status == JobStatus.ACTIVE.value for status in self.job_status
        )
        self.job_status.get_lock().release()
        return num_active_jobs

    def set_job_starting(self, job_id: int):
        self.job_status.get_lock().acquire()
        self.job_status[job_id] = JobStatus.ACTIVE.value
        self.all_jobs[job_id].log_start()
        self.job_status.get_lock().release()
        logging.debug("Worker set job %s starting.", job_id)

    def set_job_complete(self, job_id: int):
        self.job_status.get_lock().acquire()
        try:
            logging.debug("Worker set job %s complete.", job_id)
            assert self.job_status[job_id] == JobStatus.ACTIVE.value
            self.job_status[job_id] = JobStatus.COMPLETE.value
            self.all_jobs[job_id].log_complete()
        except Exception as e:
            logging.error("Exception setting job complete: %s %s", e, type(e))
        self.job_status.get_lock().release()
