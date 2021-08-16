# MRP with 3 states in a loop with a reward of 3 in one of the transitions
# (like in Exercise 10.7 of Sutton and Barto's (2018) textbook)

import numpy as np


class MRP:
    def __init__(self, P, r):
        self.P = P
        self.r = r
    
    def is_multichain(self):
        raise NotImplementedError
    
    def num_states(self):
        return self.r.shape[0]


class MDP:
    def __init__(self, P, r):
        self.P = P
        self.r = r
    
    def is_multichain(self):
        raise NotImplementedError
    
    def num_states(self):
        return self.P.shape[0]
    
    def num_actions(self):
        return self.P.shape[1]
    
    
three_loop_mrp = MRP(np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
    ], dtype=float), np.array([0, 0, 3], dtype=float)
)
three_one_mrp = MRP(np.array([
    [0.0, 0.9, 0.1, 0.0],
    [0.1, 0.0, 0.9, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
]), np.array([0.0, 1.0, 8.0, 20.0]))
two_state_mdp = MDP(np.array([
    [[1.0, 0.0], [0.0, 1.0]],  # first action
    [[0.0, 1.0], [1.0, 0.0]]   # second action
]), np.array([
    [1.0, 1.0],  # first action
    [0.0, 0.0]   # second action
]))
two_state_mdp2 = MDP(np.array([
    [[1.0, 0.0], [0.0, 1.0]],  # first action
    [[1.0, 0.0], [0.0, 1.0]]   # second action
]), np.array([
    [1.0, 1.0],  # first action
    [1.0, 2.0]   # second action
]))
