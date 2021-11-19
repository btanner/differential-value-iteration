"""Random control baseline."""

import numpy as np
from differential_value_iteration.algorithms import algorithm
from differential_value_iteration.environments import structure


class Control(algorithm.Control):
  """Random algorithm."""

  def __init__(
      self,
      mdp: structure.MarkovDecisionProcess,
      initial_values: np.ndarray,
      synchronized: bool,
  ):
    del initial_values
    self.mdp = mdp
    self.synchronized = synchronized

  def reset(self):
    pass

  def diverged(self) -> bool:
    return False

  def types_ok(self) -> bool:
    return True

  def update(self) -> np.ndarray:
    if self.synchronized:
      return self.update_sync()
    return self.update_async()

  def update_sync(self) -> np.ndarray:
    return np.zeros(self.mdp.num_states, dtype=self.mdp.rewards.dtype)

  def update_async(self) -> np.ndarray:
    return 0.

  def greedy_policy(self) -> np.ndarray:
    return np.random.randint(low=0, high=self.mdp.num_actions, size=self.mdp.num_states)

  def get_estimates(self):
    return None
