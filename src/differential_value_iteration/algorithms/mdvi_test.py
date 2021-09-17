"""Tests for basic functioning of Multichain DVI algorithms."""
import functools
import itertools
from typing import Callable

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from differential_value_iteration.algorithms import mdvi
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
_GARET_DEBUG = functools.partial(garet.create,
                            seed=42,
                            num_states=3,
                            num_actions=2,
                            branching_factor=3)
# _GARET2 = functools.partial(garet.create, seed=42, num_states=3, num_actions=20, branching_factor=3)
#
# class MDVIEvaluationTest(parameterized.TestCase):
#
#   @parameterized.parameters(itertools.product(
#       (micro.create_mrp1, micro.create_mrp2),
#       (False, True),
#       (np.float32, np.float64)))
#   def test_mdvi_sync_converges(self,
#       mrp_constructor: Callable[[np.dtype], structure.MarkovRewardProcess],
#       r_bar_scalar: bool, dtype: np.dtype):
#     if dtype == np.float64:
#       tolerance_places = 8
#     else:
#       tolerance_places = 4 if mrp_constructor == micro.create_mrp2 else 6
#     environment = mrp_constructor(dtype=dtype)
#     initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
#                                                     0., dtype)
#     algorithm = mdvi.Evaluation(
#         mrp=environment,
#         step_size=.1,
#         beta=.1,
#         initial_r_bar=initial_r_bar,
#         initial_values=np.zeros(environment.num_states, dtype=dtype),
#         synchronized=True)
#
#     for _ in range(250):
#       changes = algorithm.update()
#
#     with self.subTest('did_not_diverge'):
#       self.assertFalse(algorithm.diverged())
#     with self.subTest('maintained_types'):
#       self.assertTrue(algorithm.types_ok())
#     with self.subTest('converged'):
#       self.assertAlmostEqual(np.sum(np.abs(changes)), 0.,
#                              places=tolerance_places)
#
#   @parameterized.parameters(itertools.product(
#       (micro.create_mrp1, micro.create_mrp2),
#       (False, True),
#       (np.float32, np.float64)))
#   def test_mdvi_async_converges(self,
#       mrp_constructor: Callable[[np.dtype], structure.MarkovRewardProcess],
#       r_bar_scalar: bool, dtype: np.dtype):
#     if dtype == np.float64:
#       tolerance_places = 8
#     else:
#       tolerance_places = 4 if mrp_constructor == micro.create_mrp2 else 6
#     environment = mrp_constructor(dtype)
#     initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
#                                                     0., dtype)
#     algorithm = mdvi.Evaluation(
#         mrp=environment,
#         step_size=.1,
#         beta=.1,
#         initial_r_bar=initial_r_bar,
#         initial_values=np.zeros(environment.num_states, dtype=dtype),
#         synchronized=False)
#
#     for _ in range(250):
#       change_sum = 0.
#       for _ in range(environment.num_states):
#         change = algorithm.update()
#         change_sum += np.abs(change)
#
#     with self.subTest('did_not_diverge'):
#       self.assertFalse(algorithm.diverged())
#     with self.subTest('maintained_types'):
#       self.assertTrue(algorithm.types_ok())
#     with self.subTest('converged'):
#       self.assertAlmostEqual(change_sum, 0., places=tolerance_places)



class MDVIControlTest(parameterized.TestCase):
  # Seems like even control1 does not like GARET1 with the original updates.

  @parameterized.parameters(itertools.product(
      (micro.create_mdp1, micro.create_mdp2, _GARET1, _GARET2, _GARET3),
      (False, True),
      (np.float32, np.float64)))
  def test_mdvi_sync_converges(self,
      mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
      r_bar_scalar: bool, dtype: np.dtype):
    tolerance_places = 6 if dtype == np.float64 else 5
    environment = mdp_constructor(dtype=dtype)
    initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
                                                    0., dtype)
    algorithm = mdvi.Control1(
        mdp=environment,
        step_size=.1,
        beta=.1,
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
  #
  # @parameterized.parameters(itertools.product(
  #     (micro.create_mdp1, micro.create_mdp2),
  #     (False, True),
  #     (np.float32, np.float64)))
  # def test_mdvi_async_converges(self,
  #     mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
  #     r_bar_scalar: bool, dtype: np.dtype):
  #   tolerance_places = 8 if dtype == np.float64 else 5
  #   environment = mdp_constructor(dtype)
  #   initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
  #                                                   0., dtype)
  #   algorithm = mdvi.Control1(
  #       mdp=environment,
  #       step_size=.1,
  #       beta=.1,
  #       threshold=.01,
  #       initial_r_bar=initial_r_bar,
  #       initial_values=np.zeros(environment.num_states, dtype=dtype),
  #       synchronized=False)
  #
  #   with self.subTest('initial_types'):
  #     self.assertTrue(algorithm.types_ok())
  #
  #   for _ in range(250):
  #     change_sum = 0.
  #     for _ in range(environment.num_states):
  #       change = algorithm.update()
  #       change_sum += np.abs(change)
  #
  #   with self.subTest('did_not_diverge'):
  #     self.assertFalse(algorithm.diverged())
  #   with self.subTest('maintained_types'):
  #     self.assertTrue(algorithm.types_ok())
  #   with self.subTest('converged'):
  #     self.assertAlmostEqual(change_sum, 0., places=tolerance_places)
  #
  # @parameterized.parameters(itertools.product(
  #     (micro.create_mdp1, micro.create_mdp2,),
  #     (False, True),
  #     (np.float32, np.float64)))
  # def test_mdvi_sync_new_version_matches(self,
  #     mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
  #     r_bar_scalar: bool, dtype: np.dtype):
  #   tolerance_places = 8 if dtype == np.float64 else 5
  #   environment = mdp_constructor(dtype)
  #   initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
  #                                                   0., dtype)
  #   algorithm_orig = mdvi.Control1(
  #       mdp=environment,
  #       step_size=.1,
  #       beta=.1,
  #       threshold=.01,
  #       initial_r_bar=initial_r_bar,
  #       initial_values=np.zeros(environment.num_states, dtype=dtype),
  #       synchronized=True)
  #   algorithm_vector = mdvi.Control1(
  #       mdp=environment,
  #       step_size=.1,
  #       beta=.1,
  #       threshold=.01,
  #       initial_r_bar=initial_r_bar,
  #       initial_values=np.zeros(environment.num_states, dtype=dtype),
  #       synchronized=True)
  #
  #
  #   for _ in range(250):
  #     changes_orig = algorithm_orig.update_orig()
  #     changes = algorithm_vector.update()
  #     np.testing.assert_allclose(changes_orig, changes)
  #
  # @parameterized.parameters(itertools.product(
  #     (micro.create_mdp1, micro.create_mdp2,),
  #     (False, True),
  #     (np.float32, np.float64)))
  # def test_mdvi_async_new_version_matches(self,
  #     mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
  #     r_bar_scalar: bool, dtype: np.dtype):
  #   tolerance_places = 8 if dtype == np.float64 else 5
  #   environment = mdp_constructor(dtype)
  #   initial_r_bar = 0. if r_bar_scalar else np.full(environment.num_states,
  #                                                   0., dtype)
  #   algorithm_orig = mdvi.Control1(
  #       mdp=environment,
  #       step_size=.1,
  #       beta=.1,
  #       threshold=.01,
  #       initial_r_bar=initial_r_bar,
  #       initial_values=np.zeros(environment.num_states, dtype=dtype),
  #       synchronized=False)
  #   algorithm_vector = mdvi.Control1(
  #       mdp=environment,
  #       step_size=.1,
  #       beta=.1,
  #       threshold=.01,
  #       initial_r_bar=initial_r_bar,
  #       initial_values=np.zeros(environment.num_states, dtype=dtype),
  #       synchronized=False)
  #
  #
  #   for _ in range(250):
  #     change_orig_sum = 0.
  #     change_new_sum = 0.
  #     for _ in range(environment.num_states):
  #       change_orig = algorithm_orig.update_orig()
  #       change_orig_sum += np.abs(change_orig)
  #       change_new = algorithm_vector.update()
  #       change_new_sum += np.abs(change_new)
  #     np.testing.assert_allclose(change_orig_sum, change_new_sum)

if __name__ == '__main__':
  absltest.main()
