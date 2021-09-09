"""Evaluation and Control Multichain Differential Value Iteration."""
from typing import Union

import numpy as np
from absl import logging
from differential_value_iteration.algorithms import algorithm
from differential_value_iteration.environments import structure


class Evaluation(algorithm.Evaluation):
  """Multichain DVI for prediction, section 3.1.1 in paper."""

  def __init__(
      self,
      mrp: structure.MarkovRewardProcess,
      initial_values: np.ndarray,
      initial_r_bar: Union[float, np.ndarray],
      step_size: float,
      beta: float,
      synchronized: bool):
    self.mrp = mrp
    # Ensure internal value types match environment precision.
    self.initial_values = initial_values.copy().astype(mrp.rewards.dtype)
    if isinstance(initial_r_bar, np.ndarray):
      self.initial_r_bar = initial_r_bar.copy().astype(mrp.rewards.dtype)
    else:
      self.initial_r_bar = np.full(shape=mrp.num_states,
                                   fill_value=initial_r_bar,
                                   dtype=mrp.rewards.dtype)
    self.step_size = mrp.rewards.dtype.type(step_size)
    self.beta = mrp.rewards.dtype.type(beta)
    self.index = 0
    self.synchronized = synchronized
    self.current_values = None
    self.r_bar = None
    self.reset()

  def reset(self):
    self.current_values = self.initial_values.copy()
    self.r_bar = self.initial_r_bar.copy()

  def diverged(self) -> bool:
    if not np.isfinite(self.current_values).all():
      logging.warn('Current values not finite in MDVI.')
      return True
    if not np.isfinite(self.r_bar).all():
      logging.warn('r_bar not finite in MDVI.')
      return True
    return False

  def types_ok(self) -> bool:
    return self.r_bar.dtype == self.mrp.rewards.dtype and self.current_values.dtype == self.mrp.rewards.dtype

  def update(self) -> np.ndarray:
    if self.synchronized:
      return self.update_sync()
    return self.update_async()

  def update_sync(self):
    self.r_bar = np.dot(self.mrp.transitions, self.r_bar)
    changes = self.mrp.rewards - self.r_bar + np.dot(self.mrp.transitions,
                                                     self.current_values) - self.current_values
    self.current_values += self.step_size * changes
    self.r_bar += self.beta * changes
    return changes

  def update_async(self):
    self.r_bar[self.index] = np.dot(self.mrp.transitions[self.index],
                                    self.r_bar)
    change = self.mrp.rewards[self.index] - self.r_bar[self.index] + np.dot(
        self.mrp.transitions[self.index],
        self.current_values) - self.current_values[self.index]
    self.current_values[self.index] += self.step_size * change
    self.r_bar += self.beta * change
    self.index = (self.index + 1) % self.mrp.num_states
    return change
