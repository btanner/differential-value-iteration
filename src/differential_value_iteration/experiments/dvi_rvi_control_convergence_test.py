"""Compares policy and value convergence of sync DVI to RVI.

For a more configurable and in-depth investigation of these algorithms, look at:
dvi_rvi_policy_quality_by_iterations.py
"""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from differential_value_iteration.algorithms import dvi
from differential_value_iteration.algorithms import rvi
from differential_value_iteration.environments import garet
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import mm1_queue
from differential_value_iteration.experiments import simple_experiment_runner

_MAX_ITERS = 128
_CONVERGED_TOL = 1e-13
_SAME_ITERATION_TOL = 3

_MDPS = (micro.create_mdp1, micro.create_mdp4, garet.GARET1,
         garet.GARET2, garet.GARET3, garet.GARET_100, mm1_queue.MM1_QUEUE_1,
         mm1_queue.MM1_QUEUE_2, mm1_queue.MM1_QUEUE_3)


class SyncDVIConvergenceTest(parameterized.TestCase):

  @parameterized.parameters(*_MDPS)
  def test_convergence(self, mdp_constructor):
    environment = mdp_constructor(dtype=np.float64)
    rvi_algorithm = rvi.Control(step_size=1.,
                                mdp=environment,
                                initial_values=np.zeros(environment.num_states),
                                reference_index=0,
                                synchronized=True)
    dvi_algorithm = dvi.Control(step_size=1.,
                                beta=1.,
                                mdp=environment,
                                initial_values=np.zeros(environment.num_states),
                                divide_beta_by_num_states=True,
                                initial_r_bar=0.,
                                synchronized=True)
    rvi_results = simple_experiment_runner.run_algorithm(rvi_algorithm,
                                                         max_iters=_MAX_ITERS,
                                                         converged_tol=_CONVERGED_TOL)

    with self.subTest('RVI converged'):
      self.assertTrue(rvi_results.converged)
      self.assertFalse(rvi_results.diverged)

    dvi_results = simple_experiment_runner.run_algorithm(dvi_algorithm,
                                                         max_iters=_MAX_ITERS,
                                                         converged_tol=_CONVERGED_TOL)

    with self.subTest('DVI converged'):
      self.assertTrue(dvi_results.converged)
      self.assertFalse(dvi_results.diverged)

    rvi_iterations = len(rvi_results.policies)
    dvi_iterations = len(dvi_results.policies)

    iteration_difference = rvi_iterations - dvi_iterations
    with self.subTest('Similar Iterations'):
      self.assertLess(abs(iteration_difference), _SAME_ITERATION_TOL)

    with self.subTest('Final Policies Match'):
      match_per_state = rvi_results.policies[-1] == dvi_results.policies[-1]
      self.assertTrue(match_per_state.all())


if __name__ == '__main__':
  absltest.main()
