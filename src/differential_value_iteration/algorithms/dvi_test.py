"""Tests for basic functioning of RVI algorithms."""
from absl.testing import absltest

import numpy as np

from differential_value_iteration.algorithms import dvi
from differential_value_iteration.environments import micro



class RVITest(absltest.TestCase):

  def test_dvi_sync_converges(self):
    environment = micro.mrp1
    algorithm = dvi.Evaluation(
        mrp=environment,
        step_size=.5,
        beta=.5,
        initial_r_bar=.5,
        initial_values=np.zeros(environment.num_states, dtype=np.float32),
        synchronized=True)

    for _ in range(50):
      changes = algorithm.update()
    self.assertAlmostEqual(np.sum(np.abs(changes)), 0.)

  def test_rvi_async_converges(self):
    environment = micro.mrp1
    algorithm = dvi.Evaluation(
        mrp=environment,
        step_size=.5,
        beta=.5,
        initial_r_bar=.5,
        initial_values=np.zeros(environment.num_states, dtype=np.float32),
        synchronized=False)

    for _ in range(50):
      change_sum = 0.
      for _ in range(environment.num_states):
        change = algorithm.update()
        change_sum += np.abs(change)
    self.assertAlmostEqual(change_sum, 0.)


if __name__ == '__main__':
  absltest.main()
