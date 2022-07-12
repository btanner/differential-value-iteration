"""Defines interface for algorithms."""
import abc

import numpy as np
from typing import Dict, Union

class Evaluation(abc.ABC):

  @abc.abstractmethod
  def reset(self) -> None:
    """Reset algorithm to initial conditions."""

  @abc.abstractmethod
  def update(self) -> np.ndarray:
    """Run one iteration of the algorithm and return the changes."""

  @abc.abstractmethod
  def diverged(self) -> bool:
    """Return true if diverged. Prefer false negative to false positives."""

  @abc.abstractmethod
  def types_ok(self) -> bool:
    """Sanity check returns False if something has gone wrong with precision."""

  @abc.abstractmethod
  def get_estimates(self) -> Dict[str, Union[np.ndarray, float]]:
    """Returns estimated quantities in a dictionary."""

  @property
  def pretty_name(self) -> str:
    module_name = self.__class__.__module__.split('.')[-1]
    return module_name + '::' + self.__class__.__name__

  def state_values(self) -> np.ndarray:
    raise NotImplementedError('state_values not defined for this algorithm.')


class Control(Evaluation):

  @abc.abstractmethod
  def greedy_policy(self) -> np.ndarray:
    """Returns the best action in each state."""

  def converged(self, tol: float) -> bool:
    """Checks if converged."""
    raise NotImplementedError(f'Algorithm does not implement converged.')