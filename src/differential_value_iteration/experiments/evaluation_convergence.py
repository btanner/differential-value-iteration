"""Runs a sweep over evaluation algorithms and prints results.

This experiment is far from ideal because it does not support sweeps over
different parameters for different algorithms at the moment.
"""

import functools
from typing import Any
from typing import Callable
from typing import Sequence

import numpy as np
from absl import app
from absl import flags
from differential_value_iteration.algorithms import dvi
from differential_value_iteration.algorithms import rvi
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import structure

FLAGS = flags.FLAGS
flags.DEFINE_integer('max_iters', 50000, 'Maximum iterations per algorithm.')
flags.DEFINE_float('minimum_step_size', .001, 'Minimum step size.')
flags.DEFINE_float('maximum_step_size', 1., 'Maximum step size.')
flags.DEFINE_integer('num_step_sizes', 10, 'Number of step sizes to try.')

flags.DEFINE_float('convergence_tolerance', 1e-7, 'Tolerance for convergence.')

# DVI-specific flags
flags.DEFINE_bool('dvi', True, 'Run Differential Value Iteration')
flags.DEFINE_float('dvi_minimum_beta', .001, 'Minimum DVI beta.')
flags.DEFINE_float('dvi_maximum_beta', 1., 'Maximum DVI beta.')
flags.DEFINE_integer('dvi_num_betas', 10, 'Number of DVI beta values to try.')
flags.DEFINE_float('dvi_initial_rbar', 0., 'Initial r_bar for DVI.')

# RVI-specific flags
flags.DEFINE_bool('rvi', True, 'Run Relative Value Iteration')
flags.DEFINE_integer('rvi_reference_index', 0, 'Reference index for RVI.')


def run(environments: Sequence[structure.MarkovRewardProcess],
    algorithm_constructors: Sequence[Callable[..., Any]],
    step_sizes: Sequence[float], max_iters: int, convergence_tolerance: float):
  for environment in environments:
    initial_values = np.zeros(environment.num_states)
    for algorithm_constructor in algorithm_constructors:
      print(f'Running {algorithm_constructor} on {environment.name}')
      for step_size in step_sizes:

        converged = False
        algorithm = algorithm_constructor(mrp=environment,
                                          initial_values=initial_values,
                                          step_size=step_size,
                                          synchronized=True)
        for i in range(max_iters):
          changes = algorithm.update()
          if np.sum(np.abs(changes)) <= convergence_tolerance and i > 1:
            converged = True
            break
        print(
          f'step_size:{step_size:.5f}\tConverged:{converged}\tafter {i} iterations\tFinal Changes:{changes}')


def main(argv):
  del argv  # Stop linter from complaining about unused argv.

  algorithm_constructors = []

  # Create constructors that only depends on params common to all algorithms.
  if FLAGS.dvi:
    betas = np.geomspace(start=FLAGS.dvi_minimum_beta,
                         stop=FLAGS.dvi_maximum_beta,
                         num=FLAGS.dvi_num_betas,
                         endpoint=True)
    for beta in betas:
      dvi_algorithm = functools.partial(dvi.Evaluation, beta=beta,
                                        initial_r_bar=FLAGS.dvi_initial_rbar)
      algorithm_constructors.append(dvi_algorithm)

  if FLAGS.rvi:
    rvi_algorithm = functools.partial(rvi.Evaluation,
                                      reference_index=FLAGS.rvi_reference_index)
    algorithm_constructors.append(rvi_algorithm)

  if not algorithm_constructors:
    raise ValueError('No algorithms scheduled to run.')

  # Generate stepsizes log-spaced minimum and maximum supplied.
  step_sizes = np.geomspace(
      start=FLAGS.minimum_step_size,
      stop=FLAGS.maximum_step_size,
      num=FLAGS.num_step_sizes,
      endpoint=True)

  environments = [micro.mrp1, micro.mrp2, micro.mrp3]

  run(environments=environments,
      algorithm_constructors=algorithm_constructors,
      step_sizes=step_sizes,
      max_iters=FLAGS.max_iters,
      convergence_tolerance=FLAGS.convergence_tolerance)


if __name__ == '__main__':
  app.run(main)
