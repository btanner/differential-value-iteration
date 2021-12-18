"""Provides basic multi-processing experiment management.

Jobs are submitted to a job queue (pickled on disk).

This program spawns processes and executes those jobs in sequential order.

The results of the jobs are written to disk.

If the program is killed with Sigterm, it will take care to re-queue any active
jobs so that they will not be forgotten next time the program runs.

No effort is made to continue partial jobs.
"""
import dataclasses
import functools
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
from differential_value_iteration.environments import garet
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import structure
from differential_value_iteration.experiments import async_dvi_experiment
from differential_value_iteration.experiments.minimanager import conductor, utils


@dataclasses.dataclass
class Work:
    # env_constructor: Callable[..., Any]
    # env_params: Dict[str, Any]
    env_zero_arg_constructor: Callable[[None], structure.MarkovDecisionProcess]

    alg_constructor: Callable[..., Any]
    alg_params: Dict[str, Any]

    run_params: Dict[str, Dict[str, Any]]


def do_work(work: Work):
    # env = work.env_constructor(**work.env_params)
    env = work.env_zero_arg_constructor()
    alg_params = work.alg_params
    alg_params["initial_values"] = np.full(env.num_states, fill_value=0.0)
    alg_params["mdp"] = env
    alg = work.alg_constructor(**alg_params)
    result = async_dvi_experiment.run(
        environment=env,
        alg=alg,
        num_iters=work.run_params["num_iterations"],
        convergence_tolerance=0.001,
        synchronized=False,
        eval_all_states=True,
        measure_iters=[0, 500, 1000, 5000, 10000, 15000],
    )
    logging.info("Got results %s", result)
    return result


def make_garet_envs():
    seeds = [1 + x for x in range(100)]
    num_states = [2, 5, 10, 20, 50]
    num_actions = [2, 5]
    branch_factors = [2, 5, 10]
    garet_constructors = []
    for s, num_s, num_a, k in itertools.product(
        seeds, num_states, num_actions, branch_factors
    ):
        if k < num_s:
            garet_constructors.append(
                functools.partial(
                    garet.create,
                    seed=s,
                    num_states=num_s,
                    num_actions=num_a,
                    branching_factor=k,
                    dtype=np.float64,
                )
            )
    return garet_constructors


def generate_work() -> Sequence[Work]:
    step_sizes = [
        1.0,
        0.9,
        0.5,
    ]
    betas = [
        1.0,
        0.1,
        0.5,
        0.2,
    ]
    all_env_zero_arg_constructors = []
    micro_constructors = [micro.create_mdp1, micro.create_mdp3, micro.create_mdp4]
    for mc in micro_constructors:
        all_env_zero_arg_constructors.append(functools.partial(mc, dtype=np.float64))

    all_env_zero_arg_constructors.extend(make_garet_envs())

    work = []
    for a, b, e_fn in itertools.product(
        step_sizes,
        betas,
        all_env_zero_arg_constructors,
    ):
        w = Work(
            env_zero_arg_constructor=e_fn,
            alg_params={
                "step_size": a,
                "beta": b,
                "initial_r_bar": 0.0,
                "synchronized": False,
            },
            alg_constructor=dvi.Control,
            run_params={"num_iterations": 20000},
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
