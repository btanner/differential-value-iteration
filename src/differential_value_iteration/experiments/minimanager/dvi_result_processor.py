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
import pickle
import time
from typing import Any, Callable, Dict, Optional, Sequence
import pdb
import os
import inspect
import numpy as np
import pandas
from absl import app, flags, logging

FLAGS = flags.FLAGS
# _LOAD_EXPERIMENT_NAME = flags.DEFINE_string(
#     "experiment_name", None, "String for experiment name to load.", required=True
# )
_LOAD_DIR = flags.DEFINE_string("load_dir", None, "Subdirectory to look for data files to load.")
_VERBOSE = flags.DEFINE_bool("verbose_results", False, "Print lots of data for each result.")

# from differential_value_iteration.algorithms import dvi
# from differential_value_iteration.environments import garet
# from differential_value_iteration.environments import micro
# from differential_value_iteration.environments import structure
# from differential_value_iteration.experiments import async_dvi_experiment
from differential_value_iteration.experiments.minimanager import conductor, job, utils
from differential_value_iteration.experiments.minimanager import dvi_example  # This is bad but need to load jobs.

def make_pandas_non_baselines(all_results, all_jobs):
    result_map = {
        "env_id": "env_id",
        "diverged": "diverged",
        "converged": "converged",
        "average_time_ms": "average_time_ms",
        "last_iteration": "last_iteration",
        "late_policy_switches": "late_policy_switches",
    }

    df_dict={key: [] for key in result_map.values()}
    df_dict['non_converged'] = list()
    df_dict['last_policy'] = list()
    df_dict['last_policy_measurements'] = list()
    df_dict['last_value_change'] = list()
    df_dict['all_value_changes'] = list()
    df_dict['job_id'] = list()

    for _, result in all_results.items():
        if result.work_result["baseline"]:
            continue
        for work_result_key, pandas_key in result_map.items():
            df_dict[pandas_key].append(result.work_result[work_result_key])
        
        df_dict['job_id'].append(result.job_id)
        # env_id = result.work_result["env_id"]
        # diverged = result.work_result["diverged"]
        # converged = result.work_result["converged"]
        non_converged = not(result.work_result['diverged'] or result.work_result['converged'])
        df_dict['non_converged'].append(non_converged)

        # average_time_ms = result.work_result["average_time_ms"]
        # last_iter = result.work_result["last_iteration"]
        # late_policy_switches = result.work_result["late_policy_switches"]
        # late_policies = result.work_result["all_late_policies"]
        last_policy = result.work_result["last_policy"]
        df_dict['last_policy'].append(str(last_policy))
        last_policy_recording = max(list(result.work_result["policy_measurements"].keys()))
        last_policy_measurement =  result.work_result["policy_measurements"][last_policy_recording]
        df_dict['last_policy_measurements'].append(str(last_policy_measurement))
        last_value_change_recording = max(list(result.work_result["value_changes"].keys()))
        last_value_change =  result.work_result["value_changes"][last_value_change_recording]
        df_dict['last_value_change'].append(last_value_change)
        all_values_changes = result.work_result["value_changes"]
        df_dict['all_value_changes'].append(str(all_values_changes))
    df = pandas.DataFrame.from_dict(df_dict)
    logging.info('Saving main results dataframe.')
    df.to_pickle('temp_results_dataframe.pkl', protocol=4)

def make_pandas_baselines(all_results, all_jobs):
    result_map = {
        "env_id": "env_id",
        "diverged": "diverged",
        "converged": "converged",
        "average_time_ms": "average_time_ms",
        "last_iteration": "last_iteration",
        "late_policy_switches": "late_policy_switches",
    }

    df_dict={key: [] for key in result_map.values()}
    df_dict['non_converged'] = list()
    df_dict['last_policy'] = list()
    df_dict['last_policy_measurements'] = list()
    df_dict['last_value_change'] = list()
    df_dict['all_value_changes'] = list()
    df_dict['job_id'] = list()

    for _, result in all_results.items():
        if not result.work_result["baseline"]:
            continue
        for work_result_key, pandas_key in result_map.items():
            df_dict[pandas_key].append(result.work_result[work_result_key])
        
        df_dict['job_id'].append(result.job_id)
        # env_id = result.work_result["env_id"]
        # diverged = result.work_result["diverged"]
        # converged = result.work_result["converged"]
        non_converged = not(result.work_result['diverged'] or result.work_result['converged'])
        df_dict['non_converged'].append(non_converged)

        # average_time_ms = result.work_result["average_time_ms"]
        # last_iter = result.work_result["last_iteration"]
        # late_policy_switches = result.work_result["late_policy_switches"]
        # late_policies = result.work_result["all_late_policies"]
        last_policy = result.work_result["last_policy"]
        df_dict['last_policy'].append(str(last_policy))
        last_policy_recording = max(list(result.work_result["policy_measurements"].keys()))
        last_policy_measurement =  result.work_result["policy_measurements"][last_policy_recording]
        df_dict['last_policy_measurements'].append(str(last_policy_measurement))
        last_value_change_recording = max(list(result.work_result["value_changes"].keys()))
        last_value_change =  result.work_result["value_changes"][last_value_change_recording]
        df_dict['last_value_change'].append(last_value_change)
        all_values_changes = result.work_result["value_changes"]
        df_dict['all_value_changes'].append(str(all_values_changes))
    df = pandas.DataFrame.from_dict(df_dict)
    logging.info('Saving baseline dataframe.')
    df.to_pickle('temp_results_dataframe_baseline.pkl', protocol=4)


def divergences(results: Sequence[job.Result]):
    num_diverged = 0
    num_converged = 0
    num_nonconverged = 0
    num_baseline_diverged = 0
    num_baseline_converged = 0
    num_baseline_nonconverged = 0
    for _, result in results.items():
        if result.work_result["diverged"]:
            if result.work_result["baseline"]:
                num_baseline_diverged += 1
            else:
                num_diverged += 1
        if result.work_result["converged"]:
            if result.work_result["baseline"]:
                num_baseline_converged += 1
            else:
                num_converged += 1
        if not result.work_result["converged"] and not result.work_result["diverged"]:
            if result.work_result["baseline"]:
                num_baseline_nonconverged += 1
            else:
                num_nonconverged += 1

    logging.info("Quick Summary...\t%d results", len(results))
    logging.info("\tBaselines\t%d Converged\t%d NonConverged\t%d Diverged", num_baseline_converged, num_baseline_nonconverged, num_baseline_diverged)
    logging.info("\tAlgs\t%d Converged\t%d NonConverged\t%d Diverged", num_converged, num_nonconverged, num_diverged)

def policies(results: Sequence[job.Result], all_jobs):
    
    baseline_policies = {}
    baseline_result_indices = {}
    alg_policies = {}
    comparisons = 0
    issues = {}
    nonissues = {}
    env_ids = set()
    non_converged_mismatches = 0
    converged_mismatches = 0
    diverged_mismatches = 0
    all_alternate_policies = {}
    all_results_by_env_id = {}
    # RESULT IS ACTUALLY JOB_ID NOW
    for result_idx, result in results.items():
        # print(result)
        policy = result.work_result["last_policy"]
        env_id = result.work_result["env_id"]

        if env_id not in all_results_by_env_id:
            all_results_by_env_id[env_id] = set()
        all_results_by_env_id[env_id].add(result_idx)

        env_ids.add(env_id)
        if result.work_result["baseline"]:
            baseline_policies[env_id] = policy
            baseline_result_indices[env_id] = result_idx
            if _VERBOSE.value:
                print(f"env:{env_id} had baseline policy:{policy}")
        else:
            if _VERBOSE.value:
                print(f"env:{env_id} result:{result_idx} has policy:{policy}", end=" ")
                # print(result.work_result)
                pm = result.work_result['policy_measurements']
                max_step = max(pm.keys())
                print(f"final_step:{max_step} return:{pm[max_step][0]} with policy:{pm[max_step][2]}")
                # for step, val in result.work_result['policy_measurements'].items():
                #     print(f"{step}:{val}", end=" ")
                # print()

            if not env_id in alg_policies:
                alg_policies[env_id] = []
            alg_policies[env_id].append((result_idx, policy))
    for env_id, baseline_policy in baseline_policies.items():
        for (result_idx, alg_policy) in alg_policies[env_id]:
            comparisons += 1
            if alg_policy != baseline_policy:
                # print(f'Temp print for env_id:{env_id}')
                # print(alg_policy)
                # print(baseline_policy)

                def print_last_k_measures(job_result, k):                   
                    this_result = job_result.work_result
                    observed_values = this_result["policy_measurements"]
                    last_timesteps = list(observed_values.keys())[-k:]
                    for ts in last_timesteps:
                        print(f" t:{ts} = {observed_values[ts]}")

                k = 3
                # print(f'Policy measurements of last {k} records')
                # print_last_k_measures(results[result_idx], k)
                # print("Same for baseline")
                # print_last_k_measures(results[baseline_result_indices[env_id]], k)
                # import pdb
                # pdb.set_trace()
                # return
                if env_id not in all_alternate_policies:
                    all_alternate_policies[env_id] = set()
                all_alternate_policies[env_id].add(alg_policy)
                if env_id not in issues:
                    issues[env_id] = {}
                diverged = results[result_idx].work_result["diverged"]
                converged = results[result_idx].work_result["converged"]
                if diverged:
                    if 'diverged' not in issues[env_id]:
                        issues[env_id]['diverged'] = set()
                    issues[env_id]['diverged'].add(result_idx)
                    diverged_mismatches += 1
                if converged:
                    if 'converged' not in issues[env_id]:
                        issues[env_id]['converged'] = set()
                    issues[env_id]['converged'].add(result_idx)
                    converged_mismatches += 1
                else:
                    if 'nonconverged' not in issues[env_id]:
                        issues[env_id]['nonconverged'] = set()
                    issues[env_id]['nonconverged'].add(result_idx)
                    non_converged_mismatches += 1
            else:
                if env_id not in nonissues:
                    nonissues[env_id] = set()
                nonissues[env_id].add(result_idx)
                

            # logging.info("EnvId:%s alg_policy:%s not same as baseline:%s", env_id, alg_policy, baseline_policy)
    logging.info("Policies...\t%d Comparisons\t%d/%d envs had problems", comparisons, len(issues), len(env_ids))
    logging.info("Policies...\t%d Diverged problems\t%d NonConverged Probs\t%d Converged Probs", diverged_mismatches, non_converged_mismatches, converged_mismatches)
    converged_issues_env_count = 0
    outer_counter = 0
    for env_id, these_issues in issues.items():
        print(f"Looking at env_id:{env_id} which has {len(all_results_by_env_id[env_id])} results right now.")
        if 'converged' in these_issues:
            converged_issues_env_count += 1
            logging.info("\tEnv %d had %d/%d converged mismatches\t%d alternate policies", env_id, len(these_issues['converged']), len(alg_policies[env_id]), len(all_alternate_policies[env_id]))

        print(f'Mismatched policies')
        for issue_type, result_ids in these_issues.items():
            print(f'Issue Type: {issue_type}')
            for result_idx in result_ids:
                that_job = all_jobs[results[result_idx].job_id]
                p = that_job.alg_params
                print(f"\tBAD: {p['step_size']}\t{p['beta']}\t{p['divide_by_num_states']}\t{p['async_manager_fn']}")

        if env_id in nonissues:
            print(f'Converged AND matched policies')
            good_alg_result_indices = nonissues[env_id]
            for good_result_idx in good_alg_result_indices:
                that_job = all_jobs[results[good_result_idx].job_id]
                p = that_job.alg_params
                print(f"\tGood: {p['step_size']}\t{p['beta']}\t{p['divide_by_num_states']}\t{p['async_manager_fn']}")
        outer_counter += 1
        if outer_counter > 1:
            break
        else:
            print('\n\n')
    logging.info("Total of %d envs had mismatched policies with baseline even though they converged.", converged_issues_env_count)

def make_work_dataframe(all_jobs):
    df_dict = {'job_id': list(), 'env_id': list(), 'env_func': list(), 'alg_name': list() }
    for job_id, job in all_jobs.items():
        if job.baseline:
            continue
        df_dict['job_id'].append(job_id)
        alg_name = job.alg_constructor.__name__
        df_dict['alg_name'].append(alg_name)

        env_partial_func = job.env_zero_arg_constructor
        env_func_name = env_partial_func.func.__name__
        df_dict['env_func'].append(env_func_name)
        df_dict['env_id'].append(job.env_id)

        for param_name, param_value in job.alg_params.items():
            decorated_param_name = f'param_{param_name}'
            # This will only work if all algs have same params.
            if decorated_param_name not in df_dict:
                df_dict[decorated_param_name] = list()
            if isinstance(param_value, functools.partial):
                param_value = param_value.func.__name__
            if inspect.isclass(param_value):
                param_value = param_value.__name__
            df_dict[decorated_param_name].append(param_value)
    df = pandas.DataFrame.from_dict(df_dict)
    logging.info('Saving job params dataframe.')
    df.to_pickle('temp_jobs_dataframe.pkl', protocol=4)


def main(argv):
    logging.set_verbosity(logging.INFO)
    logging.info('Pandas version:%s', pandas.__version__)
    # Should use a flag-settable prefix.
    load_dirname = _LOAD_DIR.value
    load_path = utils.setup_load_directory(load_dirname=load_dirname)

    experiment_name = dvi_example._LOAD_EXPERIMENT_NAME.value
    utils.check_name_valid(experiment_name, raise_error=True)

    # definition_path = os.path.join(load_path, experiment_name + ".status")
    # logging.info("Trying to load work plan: %s", definition_path)
    # workplan = job.WorkPlan.from_checkpoint(definition_path)
    all_work = dvi_example.generate_work()
    # This is bad and dangerous.
    all_jobs = {}
    for job_id, w in enumerate(all_work):
        all_jobs[job_id] = w

    make_work_dataframe(all_jobs)
 
    logging.info("Loading experiment: %s", experiment_name)

    logging.info("%s and %s", load_path, experiment_name)
    experiment_params = conductor.ExperimentParams(
        save_path=load_path, experiment_name=experiment_name, work_fn=None)
    
    all_results = {}

 

    with open(experiment_params.results_file_path, 'rb') as results_file:
        while True:
            try:
                result = pickle.load(results_file)
            except EOFError:
                logging.debug("No more results in file.")
                break
            all_results[result.job_id]=result
            if len(all_results) % 1000 == 0:
                logging.debug("Loaded %s results so far", len(all_results))
    logging.info("Total results: %s", len(all_results))
    make_pandas_non_baselines(all_results, all_jobs)
    make_pandas_baselines(all_results, all_jobs)
    return

    filtered_results = filter_results(all_results, all_jobs)
    print(f"Went from: {len(all_results)} to filtered: {len(filtered_results)}")

    divergences(filtered_results)
    policies(filtered_results, all_jobs)

def filter_results(all_results, all_jobs):
    filtered_results = {}
    for job_id, result in all_results.items():
        if result.work_result["baseline"]:
            filtered_results[job_id] = result
        else:
            that_job = all_jobs[result.job_id]
            p = that_job.alg_params
            if p['divide_by_num_states']:
                # filtered_results[job_id] = result
                # This maybe doesn't work b/c experiments not done running.
                if p['beta']<1.:
                    filtered_results[job_id] = result
    return filtered_results





if __name__ == "__main__":
    app.run(main)
