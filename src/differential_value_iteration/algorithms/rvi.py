"""Evaluation and Control implementations of Relative Value Iteration."""

import numpy as np
from absl import logging
from differential_value_iteration.algorithms import algorithm
from differential_value_iteration.environments import structure


class Evaluation(algorithm.Evaluation):
  """Relative Value Iteration for prediction, section 1.1.1 in paper."""
  def __init__(
      self,
      mrp: structure.MarkovRewardProcess,
      initial_values: np.ndarray,
      step_size: float,
      reference_index: int,
      synchronized: bool):
    self.mrp = mrp
    # Ensure internal value types match environment precision.
    self.initial_values = initial_values.copy().astype(mrp.rewards.dtype)
    self.step_size = mrp.rewards.dtype.type(step_size)
    self.reference_index = reference_index
    self.index = 0
    self.synchronized = synchronized
    self.reset()

  def reset(self):
    self.current_values = self.initial_values.copy()

  def diverged(self) -> bool:
    if not np.isfinite(self.current_values).all():
      logging.warn('Current values not finite in RVI.')
      return True
    return False

  def types_ok(self) -> bool:
    return self.current_values.dtype == self.mrp.rewards.dtype

  def update(self) -> np.ndarray:
    if self.synchronized:
      return self.update_sync()
    return self.update_async()

  def update_sync(self):
    changes = self.mrp.rewards + np.dot(
        self.mrp.transitions,
        self.current_values - self.current_values[
          self.reference_index]) - self.current_values
    self.current_values += self.step_size * changes
    return changes

  def update_async(self):
    change = self.mrp.rewards[self.index] + np.dot(
        self.mrp.transitions[self.index],
        self.current_values - self.current_values[
          self.reference_index]) - self.current_values[self.index]
    self.current_values[self.index] += self.step_size * change
    self.index = (self.index + 1) % self.mrp.num_states
    return change
