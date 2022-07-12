"""Evaluation and Control implementations of Differential Value Iteration."""

from typing import Optional

import numpy as np
from absl import logging

from differential_value_iteration.algorithms import algorithm
from differential_value_iteration.algorithms import async_strategies
from differential_value_iteration.environments import structure


class Evaluation(algorithm.Evaluation):
  """Differential Value Iteration for prediction."""

  def __init__(
      self,
      mrp: structure.MarkovRewardProcess,
      initial_values: np.ndarray,
      initial_r_bar: float,
      step_size: float,
      beta: float,
      divide_beta_by_num_states: bool,
      synchronized: bool,
      async_manager_fn: Optional[async_strategies.AsyncManager] = None,
  ):
    """Creates the evaluation algorithm.
    Args:
        mrp: The problem.
        initial_values: Initial state values.
        initial_r_bar: Initial r_bar.
        step_size: Step size.
        beta: Beta learning rate parameter.
        divide_beta_by_num_states: If true, divide beta by number of states.
        synchronized: If True, execute synchronous updates of all states.
            Otherwise execute asynch updates according to async_strategy.
        async_manager_fn: Constructor for async manager.
    """
    if divide_beta_by_num_states:
      beta /= mrp.num_states
    self.mrp = mrp
    # Ensure internal value types match environment precision.
    self.initial_values = initial_values.copy().astype(mrp.rewards.dtype)
    self.initial_r_bar = mrp.rewards.dtype.type(initial_r_bar)
    self.step_size = mrp.rewards.dtype.type(step_size)
    self.beta = mrp.rewards.dtype.type(beta)

    self.synchronized = synchronized
    if not async_manager_fn:
      async_manager_fn = async_strategies.RoundRobinASync
    self.async_manager = async_manager_fn(num_states=mrp.num_states,
                                          start_state=0)
    self.current_values = self.initial_values.copy()
    self.r_bar = self.initial_r_bar

  def reset(self):
    self.current_values = self.initial_values.copy()
    self.r_bar = self.initial_r_bar

  def diverged(self) -> bool:
    if not np.isfinite(self.current_values).all():
      logging.warning("Current values not finite in DVI.")
      return True
    if not np.isfinite(self.r_bar):
      logging.warning("r_bar not finite in DVI.")
      return True
    return False

  def types_ok(self) -> bool:
    return (
        self.r_bar.dtype == self.mrp.rewards.dtype
        and self.current_values.dtype == self.mrp.rewards.dtype
    )

  def update(self) -> np.ndarray:
    if self.synchronized:
      return self.update_sync()
    return self.update_async()

  def update_sync(self) -> np.ndarray:
    """Updates all state values and r_bar."""
    changes = (
        self.mrp.rewards
        - self.r_bar
        + np.dot(self.mrp.transitions, self.current_values)
        - self.current_values
    )
    self.current_values += self.step_size * changes
    self.r_bar += self.beta * np.sum(changes)
    return changes

  def update_async(self) -> np.ndarray:
    """Updates one state value and r_bar."""
    index = self.async_manager.next_state
    change = (
        self.mrp.rewards[index]
        - self.r_bar
        + np.dot(self.mrp.transitions[index], self.current_values)
        - self.current_values[index]
    )
    self.current_values[index] += self.step_size * change
    self.r_bar += self.beta * change
    self.async_manager.update(change)
    return change

  def get_estimates(self):
    return {"v": self.current_values, "r_bar": self.r_bar}


class Control(algorithm.Control):
  """Differential Value Iteration for control."""

  def __init__(
      self,
      mdp: structure.MarkovDecisionProcess,
      initial_values: np.ndarray,
      initial_r_bar: float,
      step_size: float,
      beta: float,
      divide_beta_by_num_states: bool,
      synchronized: bool,
      async_manager_fn: Optional[async_strategies.AsyncManager] = None,
  ):
    """Creates the control algorithm.
    Args:
        mdp: The problem.
        initial_values: Initial state values.
        initial_r_bar: Initial r_bar.
        step_size: Step size.
        beta: Beta learning rate parameter.
        divide_beta_by_num_states: If true, divide beta by number of states.
        synchronized: If True, execute synchronous updates of all states.
            Otherwise execute asynch updates according to async_strategy.
        async_manager_fn: Constructor for async manager.
    """
    self.mdp = mdp

    if divide_beta_by_num_states:
      beta /= mdp.num_states

    # Ensure internal value types match environment precision.
    self.initial_values = initial_values.copy().astype(mdp.rewards.dtype)
    self.initial_r_bar = mdp.rewards.dtype.type(initial_r_bar)
    self.step_size = mdp.rewards.dtype.type(step_size)
    self.beta = mdp.rewards.dtype.type(beta)

    self.synchronized = synchronized
    if not async_manager_fn:
      async_manager_fn = async_strategies.RoundRobinASync
    self.async_manager = async_manager_fn(num_states=mdp.num_states,
                                          start_state=0)
    self.current_values = self.initial_values.copy()
    self.r_bar = self.initial_r_bar
    self.reset()

  def reset(self):
    self.current_values = self.initial_values.copy()
    self.r_bar = self.initial_r_bar
    self.async_manager.reset()

  def update(self) -> np.ndarray:
    if self.synchronized:
      return self.update_sync()
    return self.update_async()

  def calc_sync_changes(self) -> np.ndarray:
    """Calculates changes that a sync update would do."""
    temp_s_by_a = (
        self.mdp.rewards
        - self.r_bar
        + np.dot(self.mdp.transitions, self.current_values)
        - self.current_values
    )
    return np.max(temp_s_by_a, axis=0)

  def update_sync(self) -> np.ndarray:
    """Updates all state values and r_bar."""
    changes = self.calc_sync_changes()
    self.current_values += self.step_size * changes
    self.r_bar += self.beta * np.sum(changes)

    return changes

  def update_async(self) -> np.ndarray:
    """Updates one state value and r_bar."""
    index = self.async_manager.next_state
    temp_a = (
        self.mdp.rewards[:, index]
        - self.r_bar
        + np.dot(self.mdp.transitions[:, index], self.current_values)
        - self.current_values[index]
    )
    change = np.max(temp_a)
    self.current_values[index] += self.step_size * change
    self.r_bar += self.beta * change
    self.async_manager.update(change)
    return change

  def converged(self, tol: float) -> bool:
    return np.mean(np.abs(self.calc_sync_changes())) < tol

  def state_values(self) -> np.ndarray:
    return self.current_values

  def diverged(self) -> bool:
    if not np.isfinite(self.current_values).all():
      logging.warning("Current values not finite in DVI.")
      return True
    if not np.isfinite(self.r_bar):
      logging.warning("r_bar not finite in DVI.")
      return True
    return False

  def types_ok(self) -> bool:
    return (
        self.r_bar.dtype == self.mdp.rewards.dtype
        and self.current_values.dtype == self.mdp.rewards.dtype
    )

  def greedy_policy(self) -> np.ndarray:
    temp_s_by_a = (
        self.mdp.rewards
        - self.r_bar
        + np.dot(self.mdp.transitions, self.current_values)
        - self.current_values
    )
    return np.argmax(temp_s_by_a, axis=0)

  def get_estimates(self):
    return {"v": self.current_values, "r_bar": self.r_bar}
