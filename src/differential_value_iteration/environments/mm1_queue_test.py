"""Tests for mm1_queue."""
import functools

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from differential_value_iteration.environments import mm1_queue


class MM1QueueTest(parameterized.TestCase):

  def test_create_mdp(self):
    max_stored_jobs = 5
    cost_fn = functools.partial(mm1_queue.linear_cost_fn, cost_constant=5.)
    mdp = mm1_queue.create(arrival_rate=1., service_rate=1., admit_reward=5.,
                           cost_fn=cost_fn, max_stored_jobs=max_stored_jobs,
                           dtype=np.float64)
    self.assertTrue(mdp is not None)


if __name__ == '__main__':
  absltest.main()
