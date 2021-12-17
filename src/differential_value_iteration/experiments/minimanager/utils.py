import datetime
import glob
import os
import re
from absl import logging


def setup_save_directory(save_dirname: str) -> os.PathLike:
    save_path = os.path.join(os.getcwd(), save_dirname)
    logging.debug("Results path will be:%s", save_path)
    if os.path.exists(save_path):
        logging.debug("Results path exists already.")
    else:
        logging.debug("Creating results path")
        os.makedirs(save_path, exist_ok=True)
    return save_path


def clear_old_saves(save_path: os.PathLike):
    status_removed = 0
    results_removed = 0
    logging.debug("Clearing existing results.")
    results_files_path = os.path.join(save_path, "*.results")
    results_files = glob.glob(results_files_path)
    for f in results_files:
        os.unlink(f)
        status_removed += 1
    status_files_path = os.path.join(save_path, "*.status")
    status_files = glob.glob(status_files_path)
    for f in status_files:
        os.unlink(f)
        results_removed += 1
    logging.debug(
        "Removed %d result files and %d status files.", results_removed, status_removed
    )


def make_experiment_name(
    new_experiment_name_prefix: str, command_line_value: str = None
) -> str:
    regex = r"[^A-Za-z0-9_\-\\]"
    scary_characters = re.findall(regex, new_experiment_name_prefix)
    if scary_characters:
        raise ValueError(
            f"Please keep experiment name prefixes to letters, numbers and underscores. Filter does not like:{scary_characters}"
        )
    if command_line_value:
        scary_characters = re.findall(regex, command_line_value)
        if scary_characters:
            raise ValueError(
                f"Please keep experiment names to letters, numbers and underscores. Filter does not like:{scary_characters}"
            )
        experiment_name = command_line_value
    else:
        now = datetime.now().strftime("%Y_%d_%m_%H%M%S")
        experiment_name = new_experiment_name_prefix + "_" + now
    return experiment_name
