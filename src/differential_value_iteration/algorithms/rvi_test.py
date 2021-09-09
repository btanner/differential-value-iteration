"""Tests for basic functioning of RVI algorithms."""
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from differential_value_iteration.algorithms import rvi
from differential_value_iteration.environments import micro


class RVITest(parameterized.TestCase):

  @parameterized.parameters(np.float32, np.float64)
  def test_rvi_sync_converges(self, dtype: np.dtype):
    tolerance_places = 6 if dtype is np.float32 else 10
    environment = micro.create_mrp1(dtype)
    algorithm = rvi.Evaluation(
        mrp=environment,
        step_size=.5,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        reference_index=0,
        synchronized=True)

    for _ in range(50):
      changes = algorithm.update()
    self.assertAlmostEqual(np.sum(np.abs(changes)), 0., places=tolerance_places)

  @parameterized.parameters(np.float32, np.float64)
  def test_rvi_async_converges(self, dtype: np.dtype):
    tolerance_places = 6 if dtype is np.float32 else 10
    environment = micro.create_mrp1(dtype)
    algorithm = rvi.Evaluation(
        mrp=environment,
        step_size=.5,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        reference_index=0,
        synchronized=False)

    for _ in range(50):
      change_sum = 0.
      for _ in range(environment.num_states):
        change = algorithm.update()
        change_sum += np.abs(change)
    self.assertAlmostEqual(change_sum, 0., places=tolerance_places)


if __name__ == '__main__':
  absltest.main()
