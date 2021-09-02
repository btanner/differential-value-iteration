"""Tests for basic functioning of M? DVI algorithms."""
from absl.testing import absltest

import numpy as np

from differential_value_iteration.algorithms import mdvi
from differential_value_iteration.environments import micro


class MDVITest(absltest.TestCase):

  def test_mdvi_sync_converges(self):
    environment = micro.mrp1
    algorithm = mdvi.Evaluation(
        mrp=environment,
        step_size=.5,
        beta=.5,
        initial_r_bar=np.zeros(environment.num_states, dtype=np.float32),
        initial_values=np.zeros(environment.num_states, dtype=np.float32),
        synchronized=True)

    for _ in range(50):
      changes = algorithm.update()
    self.assertAlmostEqual(np.sum(np.abs(changes)), 0.)

  def test_mdvi_async_converges(self):
    environment = micro.mrp1
    algorithm = mdvi.Evaluation(
        mrp=environment,
        step_size=.5,
        beta=.5,
        initial_r_bar=np.zeros(environment.num_states, dtype=np.float32),
        initial_values=np.zeros(environment.num_states, dtype=np.float32),
        synchronized=False)

    for _ in range(50):
      change_sum = 0.
      for _ in range(environment.num_states):
        change = algorithm.update()
        change_sum += np.abs(change)
    # This never goes all the way to 0.
    self.assertAlmostEqual(change_sum, 0., places=5)


if __name__ == '__main__':
  absltest.main()
