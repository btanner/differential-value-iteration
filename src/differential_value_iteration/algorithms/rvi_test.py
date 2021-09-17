"""Tests for basic functioning of RVI algorithms."""
import functools
import itertools
from typing import Callable

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from differential_value_iteration.algorithms import rvi
from differential_value_iteration.environments import garet
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import structure

_GARET1 = functools.partial(garet.create,
                            seed=42,
                            num_states=4,
                            num_actions=4,
                            branching_factor=3)
_GARET2 = functools.partial(garet.create,
                            seed=42,
                            num_states=4,
                            num_actions=20,
                            branching_factor=3)
_GARET3 = functools.partial(garet.create,
                            seed=42,
                            num_states=10,
                            num_actions=2,
                            branching_factor=3)


class RVIEvaluationTest(parameterized.TestCase):

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

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(np.sum(np.abs(changes)), 0.,
                             places=tolerance_places)

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

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(change_sum, 0., places=tolerance_places)


class RVIControlTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(
      (micro.create_mdp1, _GARET1, _GARET2, _GARET3),
      (np.float32, np.float64))
  )
  def test_rvi_sync_converges(self,
      mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
      dtype: np.dtype):
    tolerance_places = 10 if dtype == np.float64 else 6
    environment = mdp_constructor(dtype=dtype)
    algorithm = rvi.Control(
        mdp=environment,
        step_size=.75,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        reference_index=0,
        synchronized=True)

    for _ in range(100):
      changes = algorithm.update()

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(np.max(np.abs(changes)), 0.,
                             places=tolerance_places)

  @parameterized.parameters(itertools.product(
      (micro.create_mdp1, _GARET1, _GARET2, _GARET3),
      (np.float32, np.float64))
  )
  def test_rvi_async_converges(self,
      mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
      dtype: np.dtype):
    tolerance_places = 10 if dtype == np.float64 else 6
    environment = mdp_constructor(dtype=dtype)
    algorithm = rvi.Control(
        mdp=environment,
        step_size=.5,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        reference_index=0,
        synchronized=False)

    for _ in range(200):
      change_max = 0.
      for _ in range(environment.num_states):
        change = algorithm.update()
        change_max = np.maximum(change_max, np.abs(change))

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(change_max, 0., places=tolerance_places)


if __name__ == '__main__':
  absltest.main()
