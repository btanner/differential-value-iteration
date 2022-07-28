"""Evaluation and Control Multichain Differential Value Iteration.

As part of this project we did not complete fully this code because we did not
conduct any significant empirical exploration of multichain MDPs.

This implementation is mid-refactor between an initial implementation and a
faster, vectorized, NumPy version.

It should be functional, but no promises.

Control Algorithm 2 seems strictly better than Algorithm 1:
 - easier to pick parameters that converge
   (step_size = 1, beta = 1, divide_beta_by_num_states = True)
 - faster per iteration
 - converges in few iterations

Choosing the right combination of step size, beta, and threshold is tricky.

It has been suggested that these algorithms may work better with a decay
schedule on step sizes or state-dependent step sizes.
"""
from typing import Union

import numpy as np
from absl import logging
from differential_value_iteration.algorithms import algorithm
from differential_value_iteration.environments import structure


_DEBUG_ITER = -1

class Evaluation(algorithm.Evaluation):
  """Multichain DVI for prediction."""

  def __init__(
      self,
      mrp: structure.MarkovRewardProcess,
      initial_values: np.ndarray,
      initial_r_bar: Union[float, np.ndarray],
      step_size: float,
      beta: float,
      divide_beta_by_num_states: bool,
      synchronized: bool):

    if divide_beta_by_num_states:
      beta /= mrp.num_states

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

  def state_values(self) -> np.ndarray:
    return self.current_values

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

  def get_estimates(self):
    return {'v': self.current_values, 'r_bar': self.r_bar}


class Control1(algorithm.Control):
  """Multichain DVI for Control (Algorithm 1)."""

  def __init__(
      self,
      mdp: structure.MarkovDecisionProcess,
      initial_values: np.ndarray,
      initial_r_bar: Union[float, np.ndarray],
      step_size: float,
      beta: float,
      divide_beta_by_num_states: bool,
      threshold: float,
      synchronized: bool):
    self.mdp = mdp

    if divide_beta_by_num_states:
      beta /= mdp.num_states

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
    self.iteration = 0
    self.synchronized = synchronized
    self.current_values = None
    self.r_bar = None
    self.reset()

  def reset(self):
    self.current_values = self.initial_values.copy()
    self.r_bar = self.initial_r_bar.copy()
    self.iteration = 0

  def converged(self, tol: float) -> bool:
    raise NotImplementedError('Converged not implemented for MDVI.')

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
      print(
        f'current_values: {self.current_values.dtype} vs {self.mdp.rewards.dtype}')
    return self.r_bar.dtype == self.mdp.rewards.dtype and self.current_values.dtype == self.mdp.rewards.dtype

  def update_orig(self) -> np.ndarray:
    if self.synchronized:
      return self.update_sync_orig()
    return self.update_async_orig()

  def update(self) -> np.ndarray:
    self.iteration += 1
    if self.synchronized:
      return self.update_sync()
    return self.update_async()

  def update_sync_orig(self) -> (np.ndarray, np.ndarray, np.ndarray):
    """Straight Port of Yi's implementation. Returns new values and r_bar."""
    temp_s_by_a = np.zeros((self.mdp.num_states, self.mdp.num_actions),
                           dtype=self.mdp.rewards.dtype)
    for a in range(self.mdp.num_actions):
      temp_s_by_a[:, a] = np.dot(self.mdp.transitions[a], self.r_bar)
    r_bar = np.max(temp_s_by_a, axis=1)
    delta = np.zeros(self.mdp.num_states, dtype=self.mdp.rewards.dtype)
    for s in range(self.mdp.num_states):
      max_actions = np.where(temp_s_by_a[s] > r_bar[s] - self.threshold)[0]
      temp_a = np.zeros(len(max_actions), dtype=self.mdp.rewards.dtype)
      for i in range(len(max_actions)):
        temp_a[i] = self.mdp.rewards[max_actions[i]][s] - r_bar[
          s] + np.dot(
            self.mdp.transitions[max_actions[i]][s], self.current_values) - \
                    self.current_values[s]
      delta[s] = np.max(temp_a)
      if self.iteration == _DEBUG_ITER:
        print(f'orig: s:{s} {max(temp_s_by_a[s])-self.threshold} action_vals:{temp_s_by_a[s]} max_actions:{max_actions} temp_a shape:{temp_a.shape} temp_a:{temp_a} {np.max(temp_a)} ')
        print(f'next state probs: {self.mdp.transitions[:, s]}')

    new_current_values = self.current_values.copy() + self.step_size * delta
    new_r_bar = r_bar.copy() + self.beta * delta
    return delta, new_current_values, new_r_bar

  def update_sync_tanno(self) -> (np.ndarray, np.ndarray, np.ndarray):
    """Updated to be faster through vectorized ops. Work in progress."""
    temp_s_by_a = np.dot(self.mdp.transitions, self.r_bar).T
    r_bar = np.max(temp_s_by_a, axis=1)
    changes = np.zeros(self.mdp.num_states, dtype=self.mdp.rewards.dtype)
    for (s, action_vals), r_bar_s in zip(enumerate(temp_s_by_a), r_bar):
      max_actions = np.argwhere(action_vals > max(action_vals) - self.threshold)
      # immediate_rewards = self.mdp.rewards[max_actions, s]
      # this_state_reward_rate = r_bar_s
      # next_values_per_action = np.dot(self.mdp.transitions[max_actions, s], self.current_values)
      temp_a = self.mdp.rewards[max_actions, s] - r_bar_s + np.dot(
          self.mdp.transitions[max_actions, s], self.current_values) - \
               self.current_values[s]
      changes[s] = np.max(temp_a)
      if self.iteration == _DEBUG_ITER:
        print(f'tanno sync version')
        print(f's = {s}')
        print(f'next_r_bar := P[:, s={s}] * r_bar = {np.array2string(temp_s_by_a[s], precision=1)}')
        print(f'Max Action Indices = {max_actions[0]}')
        print(f'Immediate Rewards = {self.mdp.rewards[:, s]}')

        v2_r_bar = np.max(np.dot(self.mdp.transitions[:, s], self.r_bar).T)
        v2_next_vals = np.dot(self.mdp.transitions[:, s], self.current_values)
        v2_next_val_diffs = v2_next_vals - self.current_values[s]
        r_diffs = self.mdp.rewards[:, s] - v2_r_bar
        total_diffs = v2_next_val_diffs + r_diffs
        print(f'max(next_r_bar) = {np.array2string(v2_r_bar, precision=1)}')
        print(f'Next_Vals := P[:, s={s}] * V = {np.array2string(v2_next_vals, precision=1)}')
        print(f'V[s={s}] = {np.array2string(self.current_values[s], precision=1)}')
        print(f'V_diffs := Next_Vals - V[s={s}] = {np.array2string(v2_next_val_diffs, precision=1)}')
        print(f'R_diffs := R[:,s={s}] - max(next_r_bar) = {np.array2string(r_diffs, precision=1)}')
        print(f'Total_diffs = V_diffs + R_diffs = {np.array2string(total_diffs, precision=1)}')
        print(f'Arg Max total diff is {np.argmax(total_diffs)} while MDVI1 action set is: {max_actions[0]}')

    new_current_values = self.current_values.copy() + self.step_size * changes
    new_r_bar = r_bar.copy() + self.beta * changes
    return changes, new_current_values, new_r_bar

  def update_sync(self)-> np.ndarray:
    # orig_deltas, orig_new_values, orig_new_r_bar = self.update_sync_orig()
    deltas, new_values, new_r_bar = self.update_sync_tanno()
    # assert(np.allclose(orig_deltas, deltas))
    # assert(np.allclose(orig_new_values, new_values))
    # assert(np.allclose(orig_new_r_bar, new_r_bar))

    self.current_values = new_values
    self.r_bar = new_r_bar
    return deltas


  def update_async_orig(self) -> (float, int, float, float, int):
    i = self.index
    temp_a_i = np.zeros(self.mdp.num_actions, self.mdp.rewards.dtype)
    for a in range(self.mdp.num_actions):
      temp_a_i[a] = np.dot(self.mdp.transitions[a][i], self.r_bar)
    r_bar_i = np.max(temp_a_i)
    max_actions = np.where(temp_a_i > r_bar_i - self.threshold)[0]
    temp_a_i = np.zeros(len(max_actions), dtype=self.mdp.rewards.dtype)
    for a in range(len(max_actions)):
      temp_a_i[a] = self.mdp.rewards[max_actions[a]][i] - r_bar_i + np.dot(
          self.mdp.transitions[max_actions[a]][i],
          self.current_values) - self.current_values[i]
    delta = np.max(temp_a_i)

    new_val_i = self.current_values[i] + self.step_size * delta
    new_r_bar_i = r_bar_i + self.beta * delta
    # self.current_values[self.index] += self.step_size * delta
    # self.r_bar[self.index] += self.beta * delta
    # self.index = (self.index + 1) % self.mdp.num_states
    new_i = (i + 1) % self.mdp.num_states
    return delta, i, new_val_i, new_r_bar_i, new_i

  def update_async_tanno(self) -> (float, int, float, float, int):
    """This is not currently tested."""
    i = self.index
    temp_a_i = np.dot(self.mdp.transitions[:, i], self.r_bar)
    r_bar_i = np.max(temp_a_i)
    max_actions = np.argwhere(temp_a_i >= np.max(temp_a_i) - self.threshold)
    temp_a = self.mdp.rewards[max_actions, i] - r_bar_i + np.dot(self.mdp.transitions[max_actions, i],
                           self.current_values) - self.current_values[i]
    change = np.max(temp_a)
    new_val_i = self.current_values[i] + self.step_size * change
    new_r_bar_i = r_bar_i + self.beta * change
    # self.current_values[i] += self.step_size * change
    # self.r_bar[self.index] += self.beta * change
    # self.index = (self.index + 1) % self.mdp.num_states
    new_i = (i + 1) % self.mdp.num_states
    return change, i, new_val_i, new_r_bar_i, new_i

  def update_async(self)-> np.ndarray:
    orig_delta, orig_i, orig_new_value, orig_new_r_bar, orig_new_i = self.update_async_orig()
    delta, i, new_value, new_r_bar, new_i = self.update_async_tanno()
    assert(np.allclose(orig_delta, delta))
    assert(np.allclose(orig_new_value, new_value))
    assert(np.allclose(orig_i, i))
    assert(np.allclose(orig_new_r_bar, new_r_bar))
    assert(np.allclose(orig_new_i, new_i))

    self.current_values[i] = new_value
    self.r_bar[i] = new_r_bar
    self.index = new_i
    return delta

  def greedy_policy(self) -> np.ndarray:
    temp_s_by_a = np.dot(self.mdp.transitions, self.r_bar).T
    r_bar = np.max(temp_s_by_a, axis=1)
    best_actions = np.zeros(self.mdp.num_states, dtype=np.int32)
    for (s, action_vals), r_bar_s in zip(enumerate(temp_s_by_a), r_bar):
      max_actions = np.argwhere(action_vals > max(action_vals) - self.threshold)
      temp_a = self.mdp.rewards[max_actions, s] - r_bar_s + np.dot(
          self.mdp.transitions[max_actions, s], self.current_values)
      best_actions[s] = max_actions[np.argmax(temp_a)]

    return best_actions

  def get_estimates(self):
    return {'v': self.current_values, 'r_bar': self.r_bar}

  def state_values(self) -> np.ndarray:
    return self.current_values

class Control2(Control1):
  """Multichain DVI for Control (Algorithm 2)."""

  def update_sync_orig(self) -> (np.ndarray, np.ndarray, np.ndarray):
    temp_s_by_a = np.zeros((self.mdp.num_states, self.mdp.num_actions))
    for a in range(self.mdp.num_actions):
      temp_s_by_a[:, a] = np.dot(self.mdp.transitions[a], self.r_bar)
    r_bar = np.max(temp_s_by_a, axis=1)
    temp_s_by_a = np.zeros((self.mdp.num_states, self.mdp.num_actions))
    for a in range(self.mdp.num_actions):
      temp_s_by_a[:, a] = self.mdp.rewards[a] - r_bar + np.dot(self.mdp.transitions[a],
                                                          self.current_values) - self.current_values
    delta = np.max(temp_s_by_a, axis=1)
    new_current_values = self.current_values.copy() + self.step_size * delta
    new_r_bar = r_bar.copy() + self.beta * delta
    return delta, new_current_values, new_r_bar

  def update_sync_tanno(self) -> np.ndarray:
    r_bar = np.max(np.dot(self.mdp.transitions, self.r_bar).T, axis=1)
    next_vals = np.dot(self.mdp.transitions, self.current_values)
    next_val_diffs = next_vals - self.current_values
    r_diffs = self.mdp.rewards - r_bar
    delta = np.max(r_diffs + next_val_diffs, axis=0)
    new_current_values = self.current_values.copy() + self.step_size * delta
    new_r_bar = r_bar.copy() + self.beta * delta
    return delta, new_current_values, new_r_bar

  def update_async(self) -> np.ndarray:
    self.r_bar[self.index] = np.max(np.dot(self.mdp.transitions[:, self.index],
                                           self.r_bar))
    next_vals = np.dot(self.mdp.transitions[:, self.index], self.current_values)
    next_val_diffs = next_vals - self.current_values[self.index]
    r_diff = self.mdp.rewards[:, self.index] - self.r_bar[self.index]
    delta = np.max(r_diff + next_val_diffs)

    self.current_values[self.index] += self.step_size * delta
    self.r_bar[self.index] += self.beta * delta
    self.index = (self.index + 1) % self.mdp.num_states
    return delta

  def greedy_policy(self) -> np.ndarray:
    best_actions = np.zeros(self.mdp.num_states, dtype=np.int32)

    for s in range(self.mdp.num_states):
      temp_a = np.zeros(self.mdp.num_actions, dtype=self.mdp.rewards.dtype)
      for a in range(self.mdp.num_actions):
        immediate_r = self.mdp.rewards[a, s]
        future_r = np.dot(self.mdp.transitions[a, s], self.current_values)
        temp_a[a] = immediate_r + future_r - self.current_values[s]# - r_bar[s]
      best_actions[s] = np.argmax(temp_a)

    return best_actions