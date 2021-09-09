"""Tests for basic functioning of Multichain DVI algorithms."""
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from differential_value_iteration.algorithms import mdvi
from differential_value_iteration.environments import micro


class MDVITest(parameterized.TestCase):

  @parameterized.parameters(
      (False, np.float32),
      (True, np.float32),
      (False, np.float64),
      (True, np.float64))
  def test_mdvi_sync_converges(self, r_bar_scalar: bool, dtype: np.dtype):
    tolerance_places = 6 if dtype is np.float32 else 10
    environment = micro.create_mrp1(dtype)
    initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
                                                    0., dtype)
    algorithm = mdvi.Evaluation(
        mrp=environment,
        step_size=.1,
        beta=.1,
        initial_r_bar=initial_r_bar,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)

    for _ in range(250):
      changes = algorithm.update()

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(np.sum(np.abs(changes)), 0.,
                             places=tolerance_places)

  @parameterized.parameters(
      (False, np.float32),
      (True, np.float32),
      (False, np.float64),
      (True, np.float64))
  def test_mdvi_async_converges(self, r_bar_scalar: bool, dtype: np.dtype):
    tolerance_places = 6 if dtype is np.float32 else 7
    environment = micro.create_mrp1(dtype)
    initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
                                                    0., dtype)
    algorithm = mdvi.Evaluation(
        mrp=environment,
        step_size=.15,
        beta=.15,
        initial_r_bar=initial_r_bar,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=False)

    for _ in range(75):
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
