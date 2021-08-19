import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class MarkovRewardProcess:
  # |S| x |S| array of state to state transition probabilities.
  transitions: np.ndarray
  # |S| vector of rewards for entering each state.
  rewards: np.ndarray
  name: str

  def __post_init__(self):
    """Raises error if transition or reward matrices malformed."""
    # Check basic shapes.
    if self.transitions.ndim != 2:
      raise ValueError(
        f'mrp transitions should be 2 dimensional, not: {self.transitions.shape}')
    if self.transitions.shape[0] != self.transitions.shape[1]:
      raise ValueError(
        f'mrp transitions should be SxS, not: {self.transitions.shape}')
    if self.rewards.ndim != 1:
      raise ValueError(
        f'mrp rewards should be 1 dimensional, not: {self.rewards.shape}')
    if self.transitions.shape[0] != self.rewards.shape[0]:
      raise ValueError(
        f'mrp transition and reward states do not match: {self.transitions.shape} vs. {self.rewards.shape}')

    # Ensure transition probabilities sum to 1 for all states.
    state_probability_sums = self.transitions.sum(axis=-1)
    failed_unity = np.where(state_probability_sums != 1., True, False)
    num_invalid_states = np.sum(failed_unity)
    if num_invalid_states:
      bad_states = np.argwhere(failed_unity)
      raise ValueError(
          f'Invalid Reward Process, some states do not have transitions that sum to 1: {bad_states}')

  @property
  def num_states(self):
    return len(self.transitions)


@dataclasses.dataclass(frozen=True)
class MarkovDecisionProcess:
  # |A| x |S| x |S| array of (state, action) -> state transition probabilities.
  transitions: np.ndarray
  # |A| x |S| vector of rewards for each action.
  rewards: np.ndarray
  name: str

  def __post_init__(self):
    """Raises error if transition or reward matrices malformed."""
    # Check basic shapes.
    if self.transitions.ndim != 3:
      raise ValueError(
        f'mdp transitions should be 3 dimensional, not: {self.transitions.shape}')
    if self.transitions.shape[1] != self.transitions.shape[2]:
      raise ValueError(
        f'mdp transitions should be AxSxS, not: {self.transitions.shape}')
    if self.rewards.ndim != 2:
      raise ValueError(
        f'mdp rewards should be 2 dimensional, not: {self.rewards.shape}')
    if self.transitions.shape[0] != self.rewards.shape[0]:
      raise ValueError(
        f'mdp transition and reward actions do not match: {self.transitions.shape} vs. {self.rewards.shape}')
    if self.transitions.shape[1] != self.rewards.shape[1]:
      raise ValueError(
        f'mdp transition and reward states do not match: {self.transitions.shape} vs. {self.rewards.shape}')

    # Ensure transition probabilities sum to 1 for all actions and states.
    for action_idx, transitions in enumerate(self.transitions):
      state_probability_sums = transitions.sum(axis=-1)
      failed_unity = np.where(state_probability_sums != 1., True, False)
      num_invalid_states = np.sum(failed_unity)
      if num_invalid_states:
        bad_states = np.argwhere(failed_unity)
        raise ValueError(
            f'Invalid Decision Process, action:{action_idx}, some states do not have transitions that sum to 1: {bad_states}')

  @property
  def num_states(self):
    return self.transitions.shape[1]

  @property
  def num_actions(self):
    return len(self.transitions)


mrp1 = MarkovRewardProcess(
    transitions=np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=np.float32), rewards=np.array([0, 0, 3], dtype=np.float32),
    name='mrp1')

mrp2 = MarkovRewardProcess(
    transitions=np.array([
        [0, .9, .1, 0],
        [.1, 0, .9, 0],
        [.9, .1, 0, 0],
        [0, 0, 0, 1]], dtype=np.float32),
    rewards=np.array([0, 1, 8, 20], dtype=np.float32),
    name='mrp2')

mrp3 = MarkovRewardProcess(
    transitions=np.array([
        [.2, .8],
        [.2, .8]], dtype=np.float32),
    rewards=np.array([1, 1], dtype=np.float32),
    name='mrp3')

mdp1 = MarkovDecisionProcess(transitions=np.array([
    [[1, 0], [0, 1]],  # first action
    [[0, 1], [1, 0]]  # second action
], dtype=np.float32), rewards=np.array([
    [1, 1],  # first action
    [0, 0]  # second action
], dtype=np.float32), name='mdp1')

mdp2 = MarkovDecisionProcess(transitions=np.array([
    [[1, 0], [0, 1]],  # first action
    [[1, 0], [0, 1]]  # second action
], dtype=np.float32), rewards=np.array([
    [1, 1],  # first action
    [0, 2]  # second action
], dtype=np.float32), name='mdp2')
