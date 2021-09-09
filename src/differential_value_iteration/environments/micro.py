import numpy as np
from differential_value_iteration.environments import structure


def create_mrp1(dtype: np.dtype) -> structure.MarkovRewardProcess:
  """Creates a cycling 3-state MRP from Sutton's book (Exercise 10.7)
   http://incompleteideas.net/book/RLbook2020.pdf

   Args:
     dtype: Dtype for reward/transition matrices: np.float32/np.float64

   Returns:
     The MRP.
   """
  return structure.MarkovRewardProcess(
      transitions=np.array([
          [0, 1, 0],
          [0, 0, 1],
          [1, 0, 0]
      ], dtype=dtype), rewards=np.array([0, 0, 1], dtype=dtype),
      name=f'mrp1 ({dtype.__name__})')


def create_mrp2(dtype: np.dtype):
  """Creates a multichain 4-state MRP.

   Args:
     dtype: Dtype for reward/transition matrices: np.float32/np.float64

   Returns:
     The MRP.
   """
  return structure.MarkovRewardProcess(
      transitions=np.array([
          [0, .9, .1, 0],
          [.1, 0, .9, 0],
          [.9, .1, 0, 0],
          [0, 0, 0, 1]], dtype=np.float32),
      rewards=np.array([0, 1, 8, 20], dtype=np.float32),
      name=f'mrp2 ({dtype.__name__})')


def create_mrp3(dtype: np.dtype):
  """Creates a unichain 2-state MRP.

   Args:
     dtype: Dtype for reward/transition matrices: np.float32/np.float64

   Returns:
     The MRP.
   """
  return structure.MarkovRewardProcess(
      transitions=np.array([
          [.2, .8],
          [.2, .8]], dtype=np.float32),
      rewards=np.array([1, 1], dtype=np.float32),
      name=f'mrp3 ({dtype.__name__})')


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
