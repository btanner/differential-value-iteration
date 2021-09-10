"""Tests for basic functioning of DVI algorithms."""
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from differential_value_iteration.algorithms import dvi
from differential_value_iteration.environments import micro


class DVITest(parameterized.TestCase):

  @parameterized.parameters(np.float32, np.float64)
  def test_dvi_sync_converges(self, dtype: np.dtype):
    tolerance_places = 6 if dtype is np.float32 else 10
    environment = micro.create_mrp1(dtype)
    algorithm = dvi.Evaluation(
        mrp=environment,
        step_size=.5,
        beta=.5,
        initial_r_bar=.5,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)

    for _ in range(50):
      changes = algorithm.update()

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(np.sum(np.abs(changes)), 0.,
                             places=tolerance_places)

  @parameterized.parameters(np.float32, np.float64)
  def test_dvi_async_converges(self, dtype: np.dtype):
    tolerance_places = 6 if dtype is np.float32 else 10
    environment = micro.create_mrp1(dtype)
    algorithm = dvi.Evaluation(
        mrp=environment,
        step_size=.5,
        beta=.5,
        initial_r_bar=.5,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=False)

    for _ in range(50):
      change_sum = 0.
      for _ in range(environment.num_states):
        change = algorithm.update()
        change_sum += np.abs(change)

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(change_sum, 0., places=tolerance_places)


if __name__ == '__main__':
  absltest.main()
