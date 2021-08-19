
import numpy as np
from differential_value_iteration.environments import structure

mrp1 = structure.MarkovRewardProcess(
    transitions=np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=np.float32), rewards=np.array([0, 0, 3], dtype=np.float32),
    name='mrp1')

mrp2 = structure.MarkovRewardProcess(
    transitions=np.array([
        [0, .9, .1, 0],
        [.1, 0, .9, 0],
        [.9, .1, 0, 0],
        [0, 0, 0, 1]], dtype=np.float32),
    rewards=np.array([0, 1, 8, 20], dtype=np.float32),
    name='mrp2')

mrp3 = structure.MarkovRewardProcess(
    transitions=np.array([
        [.2, .8],
        [.2, .8]], dtype=np.float32),
    rewards=np.array([1, 1], dtype=np.float32),
    name='mrp3')

mdp1 = structure.MarkovDecisionProcess(transitions=np.array([
    [[1, 0], [0, 1]],  # first action
    [[0, 1], [1, 0]]  # second action
], dtype=np.float32), rewards=np.array([
    [1, 1],  # first action
    [0, 0]  # second action
], dtype=np.float32), name='mdp1')

mdp2 = structure.MarkovDecisionProcess(transitions=np.array([
    [[1, 0], [0, 1]],  # first action
    [[1, 0], [0, 1]]  # second action
], dtype=np.float32), rewards=np.array([
    [1, 1],  # first action
    [0, 2]  # second action
], dtype=np.float32), name='mdp2')
