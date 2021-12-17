"""Provides basic multi-processing experiment management.

Jobs are submitted to a job queue (pickled on disk).

This program spawns processes and executes those jobs in sequential order.

The results of the jobs are written to disk.

If the program is killed with Sigterm, it will take care to re-queue any active
jobs so that they will not be forgotten next time the program runs.

No effort is made to continue partial jobs.
"""
import dataclasses
import itertools
import time
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from absl import app, flags, logging

FLAGS = flags.FLAGS
_LOAD_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name", "", "String for new name, or to resume existing experiment."
)
_CLEAR_PAST_RESULTS = flags.DEFINE_bool(
    "clear_past_results", False, "Erase previous results."
)

from differential_value_iteration.algorithms import dvi
from differential_value_iteration.environments import micro
from differential_value_iteration.experiments.minimanager import (conductor,
                                                                  utils)


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
        time.sleep(0.025)
    return {"iters": i}


def do_work(work: Work):
    env = work.env_constructor(**work.env_params)
    agent_params = work.agent_params
    agent_params["initial_values"] = np.full(env.num_states, fill_value=0.0)
    agent_params["mdp"] = env
    agent = work.agent_constructor(**agent_params)
    result = work.run_loop(agent=agent, **work.run_params)
    return result


def generate_work() -> Sequence[Work]:
    # step_sizes = [1., .9, .5,]
    # betas = [1., .1, .5, .2,]
    step_sizes = [1.0, 0.9, 0.5, 0.2, 0.3, 0.5]
    betas = [1.0, 0.1, 0.5, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1]
    initial_r_bars = [0.0, 1.0]
    synchronized = [False, True]

    env_constructors = [micro.create_mdp1, micro.create_mdp2]
    agent_constructors = [dvi.Control]
    job_id = 0
    work = []
    for a, b, ir, s, e_fn, a_fn in itertools.product(
        step_sizes,
        betas,
        initial_r_bars,
        synchronized,
        env_constructors,
        agent_constructors,
    ):
        w = Work(
            env_params={"dtype": np.float64},
            env_constructor=e_fn,
            agent_params={
                "step_size": a,
                "beta": b,
                "initial_r_bar": ir,
                "synchronized": s,
            },
            agent_constructor=a_fn,
            run_loop=experiment_fn,
            run_params={"num_iterations": 100},
        )
        work.append(w)

    return work


def main(argv):
    logging.set_verbosity(logging.DEBUG)

    # Should use a flag-settable prefix.
    save_dirname = "saves"
    save_path = utils.setup_save_directory(save_dirname=save_dirname)
    if _CLEAR_PAST_RESULTS.value:
        utils.clear_old_saves(save_path=save_path)

    experiment_name = utils.make_experiment_name(
        new_experiment_name_prefix="async_dvi_sweep",
        command_line_value=_LOAD_EXPERIMENT_NAME.value,
    )

    logging.info("Starting experiment: %s", experiment_name)

    experiment_params = conductor.ExperimentParams(
        save_path=save_path, experiment_name=experiment_name, work_fn=do_work
    )
    experiment_runner = conductor.Conductor(experiment_params=experiment_params)

    # Could also just call experiment_runner.resume() if we're sure we can resume.
    all_work = generate_work()
    experiment_runner.run(all_work, try_resume=True)

    logging.info("Control returned to main, program complete.")


if __name__ == "__main__":
    app.run(main)
