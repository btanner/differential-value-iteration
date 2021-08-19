import dataclasses

import numpy as np


@dataclasses.dataclass
class MarkovRewardsProcess:
  # |S| x |S| array of state to state transition probabilities.
  transitions: np.ndarray
  # |S| vector of rewards for entering each state.
  rewards: np.ndarray
  name: str

  def num_states(self):
    return len(self.transitions)


@dataclasses.dataclass
class MarkovDecisionProcess:
  # |S| x |A| x |S| array of (state, action) -> state transition probabilities.
  transitions: np.ndarray
  # |S| x |A| or |S| x |A| vector of rewards actions? entering or leaving?
  rewards: np.ndarray
  name: str

  def num_states(self):
    return len(self.transitions)

  def num_actions(self):
    return self.transitions.shape[1]


class MDP_old:
  def __init__(self, P, r, name):
    self.P = P
    self.r = r
    self.name = name

  def is_multichain(self):
    raise NotImplementedError

  def num_states(self):
    return self.P.shape[0]

  def num_actions(self):
    return self.P.shape[1]


mrp1 = MarkovRewardsProcess(
    transitions=np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=np.float32), rewards=np.array([0, 0, 3], dtype=np.float32),
    name='mrp1')
#
# mrp1 = MRP(np.array([
#     [0, 1, 0],
#     [0, 0, 1],
#     [1, 0, 0]
# ], dtype=float), np.array([0, 0, 3], dtype=float), 'mrp1'
# )
#
# mrp2 = MRP(np.array([
#     [0.0, 0.9, 0.1, 0.0],
#     [0.1, 0.0, 0.9, 0.0],
#     [0.9, 0.1, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 1.0]
# ]), np.array([0.0, 1.0, 8.0, 20.0]), 'mrp2')
#
# mrp3 = MRP(np.array([
#     [0.2, 0.8],
#     [0.2, 0.8]
# ]), np.array([1.0, 1.0]), 'mrp3')

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

#
# mdp1 = MDP(np.array([
#     [[1.0, 0.0], [0.0, 1.0]],  # first action
#     [[0.0, 1.0], [1.0, 0.0]]  # second action
# ]), np.array([
#     [1.0, 1.0],  # first action
#     [0.0, 0.0]  # second action
# ]), 'mdp1')
#
# mdp2 = MDP(np.array([
#     [[1.0, 0.0], [0.0, 1.0]],  # first action
#     [[1.0, 0.0], [0.0, 1.0]]  # second action
# ]), np.array([
#     [1.0, 1.0],  # first action
#     [1.0, 2.0]  # second action
# ]), 'mdp2')
