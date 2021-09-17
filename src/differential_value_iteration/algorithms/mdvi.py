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

  def update_sync(self) -> np.ndarray:
    self.r_bar = np.dot(self.mrp.transitions, self.r_bar)
    changes = self.mrp.rewards - self.r_bar + np.dot(self.mrp.transitions,
                                                     self.current_values) - self.current_values
    self.current_values += self.step_size * changes
    self.r_bar += self.beta * changes
    return changes

  def update_async(self) -> np.ndarray:
    self.r_bar[self.index] = np.dot(self.mrp.transitions[self.index],
                                    self.r_bar)
    change = self.mrp.rewards[self.index] - self.r_bar[self.index] + np.dot(
        self.mrp.transitions[self.index],
        self.current_values) - self.current_values[self.index]
    self.current_values[self.index] += self.step_size * change
    self.r_bar[self.index] += self.beta * change
    self.index = (self.index + 1) % self.mrp.num_states
    return change


class Control1(algorithm.Control):
  """Multichain DVI for prediction, section 3.1.1 in paper."""

  def __init__(
      self,
      mdp: structure.MarkovDecisionProcess,
      initial_values: np.ndarray,
      initial_r_bar: Union[float, np.ndarray],
      step_size: float,
      beta: float,
      threshold: float,
      synchronized: bool):
    self.mdp = mdp
    # Ensure internal value types match environment precision.
    self.initial_values = initial_values.copy().astype(mdp.rewards.dtype)
    if isinstance(initial_r_bar, np.ndarray):
      self.initial_r_bar = initial_r_bar.copy().astype(mdp.rewards.dtype)
    else:
      self.initial_r_bar = np.full(shape=mdp.num_states,
                                   fill_value=initial_r_bar,
                                   dtype=mdp.rewards.dtype)
    self.step_size = mdp.rewards.dtype.type(step_size)
    self.beta = mdp.rewards.dtype.type(beta)
    self.threshold = mdp.rewards.dtype.type(threshold)
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
    if self.r_bar.dtype != self.mdp.rewards.dtype:
      print(f'r_bar: {self.r_bar.dtype} vs {self.mdp.rewards.dtype}')
    if self.current_values.dtype != self.mdp.rewards.dtype:
      print(f'current_values: {self.current_values.dtype} vs {self.mdp.rewards.dtype}')
    return self.r_bar.dtype == self.mdp.rewards.dtype and self.current_values.dtype == self.mdp.rewards.dtype

  def update_orig(self) -> np.ndarray:
    if self.synchronized:
      return self.update_sync_orig()
    return self.update_async_orig()

  def update(self) -> np.ndarray:
    if self.synchronized:
      return self.update_sync()
    return self.update_async()

  def update_sync_orig(self) -> np.ndarray:
    temp_s_by_a = np.zeros((self.mdp.num_states, self.mdp.num_actions), dtype=self.mdp.rewards.dtype)
    for a in range(self.mdp.num_actions):
      temp_s_by_a[:, a] = np.dot(self.mdp.transitions[a], self.r_bar)
    self.r_bar = np.max(temp_s_by_a, axis=1)
    delta = np.zeros(self.mdp.num_states, dtype=self.mdp.rewards.dtype)
    for s in range(self.mdp.num_states):
      max_actions = np.where(temp_s_by_a[s] > self.r_bar[s] - self.threshold)[0]
      temp_a = np.zeros(len(max_actions), dtype=self.mdp.rewards.dtype)
      for i in range(len(max_actions)):
        temp_a[i] = self.mdp.rewards[max_actions[i]][s] - self.r_bar[s] + np.dot(
            self.mdp.transitions[max_actions[i]][s], self.current_values) - self.current_values[s]
      delta[s] = np.max(temp_a)
    self.current_values += self.step_size * delta
    self.r_bar += self.beta * delta
    return delta

  def update_sync(self) -> np.ndarray:
    temp_s_by_a = np.dot(self.mdp.transitions, self.r_bar).T
    self.r_bar = np.max(temp_s_by_a, axis=1)
    changes = np.zeros(self.mdp.num_states, dtype=self.mdp.rewards.dtype)
    for (s, action_vals), r_bar_s in zip(enumerate(temp_s_by_a), self.r_bar):
      max_actions = np.where(action_vals > r_bar_s - self.threshold)[0]
      temp_a = self.mdp.rewards[max_actions, s] - r_bar_s + np.dot(self.mdp.transitions[max_actions, s], self.current_values)-self.current_values[s]
      changes[s] = np.max(temp_a)
    self.current_values += self.step_size * changes
    self.r_bar += self.beta * changes
    return changes

  def update_async_orig(self) -> np.ndarray:
    temp_a = np.zeros(self.mdp.num_actions, self.mdp.rewards.dtype)
    for a in range(self.mdp.num_actions):
      temp_a[a] = np.dot(self.mdp.transitions[a][self.index], self.r_bar)
    self.r_bar[self.index] = np.max(temp_a)
    max_actions = np.where(temp_a > self.r_bar[self.index] - self.threshold)[0]
    temp_a = np.zeros(len(max_actions), dtype=self.mdp.rewards.dtype)
    for i in range(len(max_actions)):
      temp_a[i] = self.mdp.rewards[max_actions[i]][self.index] - self.r_bar[self.index] + np.dot(
          self.mdp.transitions[max_actions[i]][self.index], self.current_values) - self.current_values[self.index]
    delta = np.max(temp_a)
    self.current_values[self.index] += self.step_size * delta
    self.r_bar[self.index] += self.beta * delta
    self.index = (self.index + 1) % self.mdp.num_states
    return delta

  def update_async(self) -> np.ndarray:
    temp_a = np.dot(self.mdp.transitions[:, self.index], self.r_bar)
    self.r_bar[self.index] = np.max(temp_a)
    max_actions = np.where(temp_a > self.r_bar[self.index] - self.threshold)[0]
    temp_a = self.mdp.rewards[max_actions, self.index] - self.r_bar[self.index] + np.dot(self.mdp.transitions[max_actions, self.index], self.current_values) - self.current_values[self.index]
    change = np.max(temp_a)
    self.current_values[self.index] += self.step_size * change
    self.r_bar[self.index] += self.beta * change
    self.index = (self.index + 1) % self.mdp.num_states
    return change

  def greedy_policy(self) -> np.ndarray:
    # temp_s_by_a = np.dot(self.mdp.transitions, self.r_bar)
    # return np.argmax(temp_s_by_a, axis=0)
    temp_s_by_a = np.dot(self.mdp.transitions, self.r_bar).T
    self.r_bar = np.max(temp_s_by_a, axis=1)
    best_actions = np.zeros(self.mdp.num_states, dtype=np.int32)
    for (s, action_vals), r_bar_s in zip(enumerate(temp_s_by_a), self.r_bar):
      max_actions = np.where(action_vals > r_bar_s - self.threshold)[0]
      temp_a = self.mdp.rewards[max_actions, s] - r_bar_s + np.dot(self.mdp.transitions[max_actions, s], self.current_values)
      best_actions[s] = np.argmax(temp_a)
    return best_actions