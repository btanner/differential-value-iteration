"""Tests that our control algorithms find same policy ontest problems."""
import itertools
from typing import Callable

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from differential_value_iteration.algorithms import dvi
from differential_value_iteration.algorithms import mdvi
from differential_value_iteration.algorithms import rvi
from differential_value_iteration.environments import garet
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import structure

_GARET1, _GARET2, _GARET3 = garet.GARET1, garet.GARET2, garet.GARET3


class PolicyTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(
      (micro.create_mdp1, micro.create_mdp2, _GARET1, _GARET2, _GARET3),
      (np.float32, np.float64)))
  def test_identical_policies_sync(self,
      mdp_constructor: Callable[[np.dtype], structure.MarkovDecisionProcess],
      dtype: np.dtype):
    environment = mdp_constructor(dtype=dtype)
    rvi_control = rvi.Control(
        mdp=environment,
        step_size=.75,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        reference_index=0,
        synchronized=True)
    dvi_control = dvi.Control(
        mdp=environment,
        step_size=.1,
        beta=.1,
        initial_r_bar=0.,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)
    mdvi_control_1 = mdvi.Control1(
        mdp=environment,
        step_size=.1,
        beta=.1,
        threshold=.1,
        initial_r_bar=0.,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)
    mdvi_control_2 = mdvi.Control2(
        mdp=environment,
        step_size=.1,
        beta=.1,
        threshold=.1,
        initial_r_bar=0.,
        initial_values=np.zeros(environment.num_states, dtype=dtype),
        synchronized=True)
    for i in range(500):
      rvi_control.update()
      dvi_control.update()
      mdvi_control_1.update()
    with self.subTest('rvi vs dvi'):
      np.testing.assert_array_equal(rvi_control.greedy_policy(),
                                    dvi_control.greedy_policy())
    with self.subTest('rvi vs mdvi1'):
      np.testing.assert_array_equal(rvi_control.greedy_policy(),
                                    mdvi_control_1.greedy_policy())
    with self.subTest('mdvi1 vs mdvi2'):
      np.testing.assert_array_equal(mdvi_control_1.greedy_policy(),
                                    mdvi_control_2.greedy_policy())


if __name__ == '__main__':
  absltest.main()
