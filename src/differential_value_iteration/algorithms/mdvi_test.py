"""Tests for basic functioning of Multichain DVI algorithms."""
import itertools
from typing import Callable

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from differential_value_iteration.algorithms import mdvi
from differential_value_iteration.environments import garet
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import structure

_GARET1, _GARET2, _GARET3 = garet.GARET1, garet.GARET2, garet.GARET3
_DTYPES = (np.float64, )

class MDVIEvaluationTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(
      (micro.create_mrp1, micro.create_mrp2),
      (False, True),
      _DTYPES))
  def test_mdvi_sync_converges(self,
      mrp_constructor: Callable[[np.dtype], structure.MarkovRewardProcess],
      r_bar_scalar: bool, dtype: np.dtype):
    if dtype == np.float64:
      tolerance_places = 8
    else:
      tolerance_places = 4 if mrp_constructor == micro.create_mrp2 else 6
    environment = mrp_constructor(dtype=dtype)
    initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
                                                    0., dtype)
    algorithm = mdvi.Evaluation(
        mrp=environment,
        step_size=1.,
        beta=1.,
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

  @parameterized.parameters(itertools.product(
      (micro.create_mrp1, micro.create_mrp2),
      (False, True),
      _DTYPES))
  def test_mdvi_async_converges(self,
      mrp_constructor: Callable[[np.dtype], structure.MarkovRewardProcess],
      r_bar_scalar: bool, dtype: np.dtype):
    if dtype == np.float64:
      tolerance_places = 8
    else:
      tolerance_places = 4 if mrp_constructor == micro.create_mrp2 else 6
    environment = mrp_constructor(dtype)
    initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
                                                    0., dtype)
    algorithm = mdvi.Evaluation(
        mrp=environment,
        step_size=1.,
        beta=1.,
        initial_r_bar=initial_r_bar,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=False)

    for _ in range(250):
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



class MDVIControlTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(
      (micro.create_mdp1, micro.create_mdp2, _GARET1, _GARET2, _GARET3),
      (mdvi.Control1, mdvi.Control2),
      (False, True),
      _DTYPES))
  def test_mdvi_sync_converges(self,
      mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
      alg_constructor,
      r_bar_scalar: bool, dtype: np.dtype):
    tolerance_places = 6 if dtype == np.float64 else 5
    environment = mdp_constructor(dtype=dtype)
    initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
                                                    0., dtype)
    algorithm = alg_constructor(
        mdp=environment,
        step_size=1.,
        beta=1.,
        threshold=.1,
        initial_r_bar=initial_r_bar,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)

    with self.subTest('initial_types'):
      self.assertTrue(algorithm.types_ok())

    for i in range(250):
      changes = algorithm.update()

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(np.max(np.abs(changes)), 0.,
                             places=tolerance_places)

  @parameterized.parameters(itertools.product(
      (micro.create_mdp1, micro.create_mdp2, _GARET1, _GARET2, _GARET3),
      (mdvi.Control1, mdvi.Control2),
      (False, True),
      _DTYPES))
  def test_mdvi_async_converges(self,
      mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
      alg_constructor,
      r_bar_scalar: bool, dtype: np.dtype):
    tolerance_places = 4 if dtype == np.float64 else 3
    environment = mdp_constructor(dtype=dtype)
    initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
                                                    0., dtype)
    algorithm = alg_constructor(
        mdp=environment,
        step_size=1.,
        beta=.1,
        threshold=.1,
        initial_r_bar=initial_r_bar,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=False)

    with self.subTest('initial_types'):
      self.assertTrue(algorithm.types_ok())

    for _ in range(1000):
      change_max = 0.
      for _ in range(environment.num_states):
        change = algorithm.update()
        change_max += np.maximum(change_max, np.abs(change))

    with self.subTest('did_not_diverge'):
      self.assertFalse(algorithm.diverged())
    with self.subTest('maintained_types'):
      self.assertTrue(algorithm.types_ok())
    with self.subTest('converged'):
      self.assertAlmostEqual(change_max, 0., places=tolerance_places)


if __name__ == '__main__':
  absltest.main()
