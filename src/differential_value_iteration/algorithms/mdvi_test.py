"""Tests for basic functioning of Multichain DVI algorithms."""
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from differential_value_iteration.algorithms import mdvi
from differential_value_iteration.environments import micro


class MDVITest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_mdvi_sync_converges(self, r_bar_scalar: bool):
    environment = micro.mrp1
    initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
                                                    0., np.float32)
    algorithm = mdvi.Evaluation(
        mrp=environment,
        step_size=.5,
        beta=.5,
        initial_r_bar=initial_r_bar,
        initial_values=np.zeros(environment.num_states, dtype=np.float32),
        synchronized=True)

    for _ in range(50):
      changes = algorithm.update()
    self.assertAlmostEqual(np.sum(np.abs(changes)), 0., places=7)

  @parameterized.parameters(True, False)
  def test_mdvi_async_converges(self, r_bar_scalar: bool):
    environment = micro.mrp1
    initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
                                                    0., np.float32)
    algorithm = mdvi.Evaluation(
        mrp=environment,
        step_size=.5,
        beta=.5,
        initial_r_bar=initial_r_bar,
        initial_values=np.zeros(environment.num_states, dtype=np.float32),
        synchronized=False)

    for _ in range(50):
      change_sum = 0.
      for _ in range(environment.num_states):
        change = algorithm.update()
        change_sum += np.abs(change)
    # This never goes all the way to 0.
    self.assertAlmostEqual(change_sum, 0., places=7)


if __name__ == '__main__':
  absltest.main()
