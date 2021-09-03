"""Defines interface for algorithms."""
import abc

import numpy as np


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
