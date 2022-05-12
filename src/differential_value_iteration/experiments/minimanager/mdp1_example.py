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

_NUM_ITERATIONS = 20000
_MEASURE_ITERS = tuple(np.arange(0, _NUM_ITERATIONS, step=100, dtype=np.int))

FLAGS = flags.FLAGS
_LOAD_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name", "", "String for new name, or to resume existing experiment."
)
_CLEAR_PAST_RESULTS = flags.DEFINE_bool(
    "clear_past_results", False, "Erase previous results."
)

from differential_value_iteration.algorithms import dvi
from differential_value_iteration.algorithms import rvi
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
    baseline: bool
    env_id: int


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
        convergence_tolerance=0.0000001,
        synchronized=False,
        eval_all_states=True,
        measure_iters=_MEASURE_ITERS,
    )
    result["env_id"] = work.env_id
    result["baseline"] = work.baseline
    logging.debug("Got results %s", result)
    return result


def make_dvi_work(env_constructors, env_ids):
    dvi_work = []
    async_manager_fns = [
      dvi.RoundRobinASync,
      functools.partial(dvi.RandomAsync, seed=42),
      functools.partial(dvi.ConvergeRoundRobinASync, tol=.001),
      functools.partial(dvi.ConvergeRandomASync, tol=.001, seed=42),
      ]
    for a, b, e_fn, async_manager_fn in itertools.product(
        [
            1.0,
            0.9,
            0.5,
        ],
        [1.0, 0.1, 0.5, 0.2],
        env_constructors,
        async_manager_fns
    ):
        env_id = env_ids[e_fn]
        w = Work(
            env_zero_arg_constructor=e_fn,
            alg_params={
                "step_size": a,
                "beta": b,
                "initial_r_bar": 0.0,
                "synchronized": False,
                "async_manager_fn": async_manager_fn,
                "divide_by_num_states": False,
            },
            alg_constructor=dvi.Control,
            run_params={"num_iterations": _NUM_ITERATIONS},
            env_id=env_id,
            baseline=False,
        )
        dvi_work.append(w)
    return dvi_work

def make_rvi_work(env_constructors, env_ids):
    rvi_work = []
    for e_fn in env_constructors:
        env_id = env_ids[e_fn]
        w = Work(
            env_zero_arg_constructor=e_fn,
            alg_params={
                "step_size": 1.,
                "reference_index": 0,
                "synchronized": True,
            },
            alg_constructor=rvi.Control,
            run_params={"num_iterations": _NUM_ITERATIONS},
            env_id=env_id,
            baseline=True,
        )
        rvi_work.append(w)
    return rvi_work

def generate_work() -> Sequence[Work]:
    all_env_zero_arg_constructors = []
    micro_constructors = [micro.create_mdp1]
    for mc in micro_constructors:
        all_env_zero_arg_constructors.append(functools.partial(mc, dtype=np.float64))

    # Map each env constructor to an id so we can compare algorithms to baselines later.
    env_ids = {}
    for e_fn in all_env_zero_arg_constructors:
        if e_fn not in env_ids:
            env_id = len(env_ids) + 1
            env_ids[e_fn] = env_id

    work = []
    rvi_baseline_work = make_rvi_work(all_env_zero_arg_constructors, env_ids)
    work.extend(rvi_baseline_work)
    dvi_work= make_dvi_work(all_env_zero_arg_constructors, env_ids)
    work.extend(dvi_work)
    return work


def main(argv):
    logging.set_verbosity(logging.INFO)

    # Should use a flag-settable prefix.
    save_dirname = "saves"
    save_path = utils.setup_save_directory(save_dirname=save_dirname)
    if _CLEAR_PAST_RESULTS.value:
        utils.clear_old_saves(save_path=save_path)

    experiment_name = utils.make_experiment_name(
        new_experiment_name_prefix="async_dvi_mdp1",
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
