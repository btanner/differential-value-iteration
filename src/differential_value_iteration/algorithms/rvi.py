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

  def update_sync(self) -> np.ndarray:
    changes = self.mrp.rewards + np.dot(
        self.mrp.transitions,
        self.current_values - self.current_values[
          self.reference_index]) - self.current_values
    self.current_values += self.step_size * changes
    return changes

  def update_async(self) -> np.ndarray:
    change = self.mrp.rewards[self.index] + np.dot(
        self.mrp.transitions[self.index],
        self.current_values - self.current_values[
          self.reference_index]) - self.current_values[self.index]
    self.current_values[self.index] += self.step_size * change
    self.index = (self.index + 1) % self.mrp.num_states
    return change


class Control(algorithm.Evaluation):
  """Relative Value Iteration for control."""

  def __init__(
      self,
      mdp: structure.MarkovDecisionProcess,
      initial_values: np.ndarray,
      step_size: float,
      reference_index: int,
      synchronized: bool):
    self.mdp = mdp
    # Ensure internal value types match environment precision.
    self.initial_values = initial_values.copy().astype(mdp.rewards.dtype)
    self.step_size = mdp.rewards.dtype.type(step_size)
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
    return self.current_values.dtype == self.mdp.rewards.dtype

  def update(self) -> np.ndarray:
    if self.synchronized:
      return self.update_sync()
    return self.update_async()

  def update_sync(self) -> np.ndarray:
    # temp_s_by_a = np.zeros((self.mdp.num_states, self.mdp.num_actions),
    #                        dtype=self.mdp.rewards.dtype)
    # for a in range(self.mdp.num_actions):
    #   temp_s_by_a[:, a] = self.mdp.rewards[a] + np.dot(self.mdp.transitions[a],
    #                                                    self.current_values -
    #                                                    self.current_values[
    #                                                      self.reference_index]) - self.current_values
    # changes = np.max(temp_s_by_a, axis=1)

    temp_s_by_a = self.mdp.rewards + np.dot(self.mdp.transitions,
                                            self.current_values -
                                            self.current_values[
                                              self.reference_index]) - self.current_values
    changes = np.max(temp_s_by_a, axis=0)
    self.current_values += self.step_size * changes
    return changes

  def update_async(self) -> np.ndarray:
    # temp_a = np.zeros(self.mdp.num_actions)
    # for a in range(self.mdp.num_actions):
    #   temp_a[a] = self.mdp.rewards[a][self.index] + np.dot(
    #       self.mdp.transitions[a][self.index],
    #       self.current_values - self.current_values[self.reference_index]) - \
    #               self.current_values[self.index]
    temp_a = self.mdp.rewards[:, self.index] + np.dot(
        self.mdp.transitions[:, self.index],
        self.current_values - self.current_values[self.reference_index]) - \
             self.current_values[self.index]
    change = np.max(temp_a)
    self.current_values[self.index] += self.step_size * change
    self.index = (self.index + 1) % self.mdp.num_states
    return change
