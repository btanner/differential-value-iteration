"""Tests GARET MDP environment generator."""
from absl.testing import absltest
from absl.testing import parameterized

from differential_value_iteration.environments import garet

import numpy as np

class GaretTest(parameterized.TestCase):

  @parameterized.parameters(
      (42, 2, 1, 1),
      (42, 2, 2, 2),
      (42, 10, 2, 2),
      (42, 10, 2, 3),
      (42, 100, 2, 3),
  )
  def test_create_mdp(self, seed: int, num_states: int, num_actions: int, branching_factor: int):
    """MarkovDecisionProcess does thorough checks, ensures no errors raised."""
    mdp = garet.create(
        seed=seed,
        num_states=num_states,
        num_actions=num_actions,
        branching_factor=branching_factor,
        dtype=np.float64,
    )
    self.assertTrue(mdp is not None)


if __name__ == '__main__':
  absltest.main()
