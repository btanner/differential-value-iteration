import datetime
import glob
import os
import re
from absl import logging
from numpy.lib.npyio import load


def setup_save_directory(save_dirname: str) -> os.PathLike:
    save_path = os.path.join(os.getcwd(), save_dirname)
    logging.debug("Results path will be:%s", save_path)
    if os.path.exists(save_path):
        logging.debug("Save path exists already.")
    else:
        logging.debug("Creating save path")
        os.makedirs(save_path, exist_ok=True)
    return save_path

def setup_load_directory(load_dirname: str) -> os.PathLike:
    load_path = os.path.join(os.getcwd(), load_dirname)
    if not os.path.exists(load_path):
        raise ValueError(f"Load path does not exist:{load_path}")
    return load_path


def clear_old_saves(save_path: os.PathLike):
    status_removed = 0
    results_removed = 0
    logging.debug("Clearing existing results.")
    results_files_path = os.path.join(save_path, "*.results")
    results_files = glob.glob(results_files_path)
    for file in results_files:
        os.unlink(file)
        status_removed += 1
    status_files_path = os.path.join(save_path, "*.status")
    status_files = glob.glob(status_files_path)
    for file in status_files:
        os.unlink(file)
        results_removed += 1
    logging.debug(
        "Removed %d result files and %d status files.", results_removed, status_removed
    )

def check_name_valid(name: str, raise_error: bool=False)->bool:
    regex = r"[^A-Za-z0-9_\-\\]"
    scary_characters = re.findall(regex, name)
    if raise_error and scary_characters:
        raise ValueError(
            f"Please keep experiment names to numbers and underscores. Filter does not like:{scary_characters} in {name}"
        )
    return bool(scary_characters)


def make_experiment_name(
    new_experiment_name_prefix: str, command_line_value: str = None
) -> str:
    check_name_valid(new_experiment_name_prefix, raise_error=True)
    if command_line_value:
        check_name_valid(command_line_value, raise_error=True)
        experiment_name = command_line_value
    else:
        now = datetime.datetime.now().strftime("%Y_%d_%m_%H%M%S")
        experiment_name = new_experiment_name_prefix + "_" + now
    return experiment_name
