"""Tests for basic ability to import subpackages."""
import jax.numpy as jnp
from absl.testing import absltest


class BasicJaxTest(absltest.TestCase):

  def test_jax_numpy_works(self):
    x = jnp.zeros(10)
    self.assertEqual(jnp.sum(x), 0)


if __name__ == '__main__':
  absltest.main()
