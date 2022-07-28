"""Evaluation and Control implementations of Relative Value Iteration."""

from typing import Optional

import numpy as np
from absl import logging

from differential_value_iteration.algorithms import algorithm
from differential_value_iteration.algorithms import async_strategies
from differential_value_iteration.environments import structure


class Evaluation(algorithm.Evaluation):
  """Relative Value Iteration for prediction.

  Async methods here use an analagous strategy that is used in dvi.py. This is
  for comparison, not because async RVI is a real algorithm."""

  def __init__(
      self,
      mrp: structure.MarkovRewardProcess,
      initial_values: np.ndarray,
      step_size: float,
      reference_index: int,
      synchronized: bool,
      async_manager_fn: Optional[async_strategies.AsyncManager] = None,
  ):
    """Creates the evaluation algorithm.
    Args:
        mrp: The problem.
        initial_values: Initial state values.
        step_size: Step size. Will be scaled by number of states.
        reference_index: Which state to use as the reference index.
        synchronized: If True, execute synchronous updates of all states.
            Otherwise execute asynch updates according to async_strategy.
        async_manager_fn: Constructor for async manager.
    """
    self.mrp = mrp
    # Ensure internal value types match environment precision.
    self.initial_values = initial_values.copy().astype(mrp.rewards.dtype)
    self.step_size = mrp.rewards.dtype.type(step_size)
    self.reference_index = reference_index
    self.synchronized = synchronized
    if not async_manager_fn:
      async_manager_fn = async_strategies.RoundRobinASync
    self.async_manager = async_manager_fn(num_states=mrp.num_states,
                                          start_state=0)
    self.reset()

  def reset(self):
    self.current_values = self.initial_values.copy()

  def diverged(self) -> bool:
    if not np.isfinite(self.current_values).all():
      logging.warning('Current values not finite in RVI.')
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
    index = self.async_manager.next_state
    change = self.mrp.rewards[index] + np.dot(
        self.mrp.transitions[index],
        self.current_values - self.current_values[
          self.reference_index]) - self.current_values[index]
    self.current_values[index] += self.step_size * change
    self.async_manager.update(change)
    return change
  
  def get_estimates(self):
    return {'v': self.current_values, 'r_bar': self.current_values[self.reference_index]}


class Control(algorithm.Control):
  """Relative Value Iteration for control."""

  def __init__(
      self,
      mdp: structure.MarkovDecisionProcess,
      initial_values: np.ndarray,
      step_size: float,
      reference_index: int,
      synchronized: bool,
      async_manager_fn: Optional[async_strategies.AsyncManager] = None,
  ):
    """Creates the control algorithm.
    Args:
        mdp: The problem.
        initial_values: Initial state values.
        step_size: Step size.
        synchronized: If True, execute synchronous updates of all states.
            Otherwise execute asynch updates according to async_strategy.
        async_manager_fn: Constructor for async manager.
    """
    self.mdp = mdp
    # Ensure internal value types match environment precision.
    self.initial_values = initial_values.copy().astype(mdp.rewards.dtype)
    self.step_size = mdp.rewards.dtype.type(step_size)
    self.reference_index = reference_index
    self.synchronized = synchronized
    if not async_manager_fn:
      async_manager_fn = async_strategies.RoundRobinASync
    self.async_manager = async_manager_fn(num_states=mdp.num_states,
                                          start_state=0)
    self.reset()

  def reset(self):
    self.current_values = self.initial_values.copy()

  def diverged(self) -> bool:
    if not np.isfinite(self.current_values).all():
      logging.warning('Current values not finite in RVI.')
      return True
    return False

  def types_ok(self) -> bool:
    return self.current_values.dtype == self.mdp.rewards.dtype

  def update(self) -> np.ndarray:
    if self.synchronized:
      return self.update_sync()
    return self.update_async()

  def converged(self, tol: float) -> bool:
    sync_changes = self.calc_sync_changes()
    return np.mean(np.abs(sync_changes)) < tol

  def calc_sync_changes(self) -> np.ndarray:
    temp_s_by_a = self.mdp.rewards + np.dot(self.mdp.transitions,
                                            self.current_values -
                                            self.current_values[
                                              self.reference_index]) - self.current_values
    return np.max(temp_s_by_a, axis=0)


  def update_sync(self) -> np.ndarray:
    changes = self.calc_sync_changes()
    self.current_values += self.step_size * changes
    return changes

  def update_async(self) -> np.ndarray:
    index = self.async_manager.next_state
    temp_a = self.mdp.rewards[:, index] + np.dot(
        self.mdp.transitions[:, index],
        self.current_values - self.current_values[self.reference_index]) - \
             self.current_values[index]
    change = np.max(temp_a)
    self.current_values[index] += self.step_size * change
    self.async_manager.update(change)
    return change

  def greedy_policy(self) -> np.ndarray:
    temp_s_by_a = self.mdp.rewards + np.dot(self.mdp.transitions,
                                            self.current_values -
                                            self.current_values[
                                              self.reference_index])
    return np.argmax(temp_s_by_a, axis=0)

  def state_values(self) -> np.ndarray:
    return self.current_values

  def get_estimates(self):
    return {'v': self.current_values, 'r_bar': self.current_values[self.reference_index]}
