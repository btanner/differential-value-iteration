"""Provides simple way to run a Control algorithm to convergence or timeout."""
import dataclasses
from typing import Sequence

import numpy as np

from differential_value_iteration.algorithms import algorithm


@dataclasses.dataclass
class RunResults:
  policies: Sequence[Sequence[int]]
  state_values: Sequence[np.ndarray]
  mean_absolute_changes: Sequence[float]
  absolute_changes_per_state: Sequence[np.ndarray]
  converged: bool
  diverged: bool


def run_algorithm(control_algorithm: algorithm.Control, max_iters: int,
    converged_tol: float) -> RunResults:
  results = RunResults(policies=[], mean_absolute_changes=[],
                       absolute_changes_per_state=[], state_values=[],
                       converged=False, diverged=False)
  # results.state_values.append(control_algorithm.state_values().copy())
  for _ in range(max_iters):
    changes = control_algorithm.update()
    abs_changes = np.abs(changes)
    results.absolute_changes_per_state.append(abs_changes)
    results.mean_absolute_changes.append(float(np.mean(abs_changes)))
    results.policies.append(control_algorithm.greedy_policy())
    results.state_values.append(control_algorithm.state_values().copy())
    if control_algorithm.converged(tol=converged_tol):
      break
  results.diverged = control_algorithm.diverged()
  results.converged = control_algorithm.converged(tol=converged_tol)
  return results
