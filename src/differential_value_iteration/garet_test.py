"""Tests GARET MDP environment generator."""
import jax
from absl.testing import absltest

from differential_value_iteration.algorithms import algorithms
from differential_value_iteration.environments import garet


class GaretTest(absltest.TestCase):

  def test_create_mdp(self):
    """MarkovDecisionProcess does thorough checks, ensurs no errors raised."""
    rng_key = jax.random.PRNGKey(42)
    mdp = garet.create(
        rng_key=rng_key,
        num_states=10,
        num_actions=2,
        branching_factor=2,
    )
    self.assertTrue(mdp is not None)


if __name__ == '__main__':
  absltest.main()
