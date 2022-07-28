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

  def test_linear_cost_fn_checks_base_cost(self):
    with self.subTest('zero'):
      self.assertRaises(ValueError, mm1_queue.linear_cost_fn, 0., 0)
    with self.subTest('negative'):
      self.assertRaises(ValueError, mm1_queue.linear_cost_fn, -1., 0)
    # Positive is ok.
    mm1_queue.linear_cost_fn(.1, 0)

  def test_linear_cost_fn_checks_jobs_waiting(self):
    with self.subTest('negative'):
      self.assertRaises(ValueError, mm1_queue.linear_cost_fn, 1., -1)
    # Zero is ok.
    mm1_queue.linear_cost_fn(1., 0)

  @parameterized.parameters(1., 5.)
  def test_linear_cost_fn(self, base_cost: float):
    with self.subTest('no_jobs_no_cost'):
      self.assertEqual(mm1_queue.linear_cost_fn(base_cost, 0), 0)
    with self.subTest('1_jobs_1_cost'):
      self.assertEqual(mm1_queue.linear_cost_fn(base_cost, 1), base_cost)
    with self.subTest('1_job_positive_cost'):
      self.assertGreater(mm1_queue.linear_cost_fn(base_cost, 1), 0.)
    with self.subTest('k_jobs_k_cost'):
      self.assertEqual(mm1_queue.linear_cost_fn(base_cost, 10), 10 * base_cost)

  @parameterized.parameters(
      (0, 0, False), (1, 0, True), (2, 1, False), (3, 1, True))
  def test_global_state_to_paper_state(self, global_state: int,
      want_jobs_in_queue: int, want_new_job: bool):
    jobs_in_queue, new_job = mm1_queue.global_state_to_paper_state(global_state)
    with self.subTest('jobs_in_queue'):
      self.assertEqual(jobs_in_queue, want_jobs_in_queue)
    with self.subTest('new_job'):
      self.assertEqual(new_job, want_new_job)

  @parameterized.parameters(*tuple(range(10)))
  def test_state_conversion_reversible(self, global_state: int):
    jobs_in_queue, new_job = mm1_queue.global_state_to_paper_state(global_state)
    reversed_global_state = mm1_queue.to_global_state(jobs_in_queue, new_job)
    self.assertEqual(global_state, reversed_global_state)

  @parameterized.named_parameters(
      ('Empty No Job Admit', 0, False, mm1_queue.ADMIT, 1, False, 0., -1.),
      ('Empty No Job Continue', 0, False, mm1_queue.CONTINUE, 0, False, .5, 0.),
      ('Empty Yes Job Admit Processed', 0, True, mm1_queue.ADMIT, 0, False, .5, 2.),  # 2. for admitting with empty queue.
      ('Empty Yes Job Admit No Processed', 0, True, mm1_queue.ADMIT, 1, True, .5, 2.),  # 2. for admitting with empty queue.
      ('Empty Yes Job Continue', 0, True, mm1_queue.CONTINUE, 0, False, .5, 0.),  # 0 for continuing with empty queue.
      ('1 Yes Job Admit Processed', 1, True, mm1_queue.ADMIT, 1, False, .5, -2.),  # -2. for admitting with 1 queued.
      ('1 Yes Job Admit No Processed', 1, True, mm1_queue.ADMIT, 2, True, .5, -2.),  # -2. for admitting with 1 queued.
      ('1 Yes Job Continue Processed', 1, True, mm1_queue.CONTINUE, 0, False, .5, -4.),  # -6 for continuing with 1 queued.
      ('1 Yes Job Continue New Job', 1, True, mm1_queue.CONTINUE, 1, True, .5, -4.),  # -6 for continuing with 1 queued.
      ('Almost Full Yes Job Admit Processed', 18, True, mm1_queue.ADMIT, 18, False, .5, -70.),  # -70. for admitting.
      ('Almost Full Yes Job Admit No Processed', 18, True, mm1_queue.ADMIT, 19, True, .5, -70.),  # -70. for admitting.
      ('Almost Full Yes Job Continue Processed', 18, True, mm1_queue.CONTINUE, 17, False, .5, -72.),  # -74. for continuing.
      ('Almost Full Yes Job Continue Yes New Job', 18, True, mm1_queue.CONTINUE, 18, True, .5, -72.),  # -72. continuing.
      ('Full Yes Job Admit No New Job', 19, True, mm1_queue.ADMIT, 19, False, .5, -76.),  # -76. for admitting when full.
      ('Full Yes Job Admit Yes New Job', 19, True, mm1_queue.ADMIT, 19, True, .5, -76.),  # -76. for admitting when full.
      ('Full Yes Job Continue Processed', 19, True, mm1_queue.CONTINUE, 18, False, .5, -76.),  # -76. for continuing when full.
      ('Full Yes Job Continue Yes New Job', 19, True, mm1_queue.CONTINUE, 19, True, .5, -76.),  # -76. for continuing when full.
  )
  def test_structure_sensible(self, jobs_in_queue: int, new_job: bool,
      action: int, next_jobs_in_queue: int, next_new_job: bool,
      probability: float, reward: float):
    mdp = mm1_queue.create(
        arrival_rate=1.,
        service_rate=1.,
        admit_reward=3.,
        cost_fn=functools.partial(mm1_queue.linear_cost_fn,
                                  cost_constant=2.),
        max_stored_jobs=20, dtype=np.dtype(np.float64))
    start_state = mm1_queue.to_global_state(jobs_in_queue, new_job)
    next_state = mm1_queue.to_global_state(next_jobs_in_queue, next_new_job)
    with self.subTest('transition probability'):
      self.assertEqual(mdp.transitions[action, start_state, next_state],
                       probability)
    with self.subTest('reward'):
      self.assertEqual(mdp.rewards[action, start_state], reward)


if __name__ == '__main__':
  absltest.main()
