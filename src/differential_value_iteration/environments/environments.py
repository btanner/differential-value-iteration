import numpy as np


class MRP:
  def __init__(self, P, r, name):
    self.P = P
    self.r = r
    self.name = name

  def is_multichain(self):
    raise NotImplementedError

  def num_states(self):
    return self.r.shape[0]


class MDP:
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


mrp1 = MRP(np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
], dtype=float), np.array([0, 0, 3], dtype=float), 'mrp1'
)

mrp2 = MRP(np.array([
    [0.0, 0.9, 0.1, 0.0],
    [0.1, 0.0, 0.9, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
]), np.array([0.0, 1.0, 8.0, 20.0]), 'mrp2')

mrp3 = MRP(np.array([
    [0.2, 0.8],
    [0.2, 0.8]
]), np.array([1.0, 1.0]), 'mrp3')

mdp1 = MDP(np.array([
    [[1.0, 0.0], [0.0, 1.0]],  # first action
    [[0.0, 1.0], [1.0, 0.0]]  # second action
]), np.array([
    [1.0, 1.0],  # first action
    [0.0, 0.0]  # second action
]), 'mdp1')

mdp2 = MDP(np.array([
    [[1.0, 0.0], [0.0, 1.0]],  # first action
    [[1.0, 0.0], [0.0, 1.0]]  # second action
]), np.array([
    [1.0, 1.0],  # first action
    [1.0, 2.0]  # second action
]), 'mdp2')
