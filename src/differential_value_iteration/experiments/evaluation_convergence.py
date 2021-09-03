"""Runs a sweep over evaluation algorithms and prints results."""
import functools
from typing import Sequence

import numpy as np
from absl import app
from absl import flags
from differential_value_iteration.algorithms import rvi
from differential_value_iteration.environments import micro
from differential_value_iteration.environments import structure

FLAGS = flags.FLAGS
flags.DEFINE_integer('max_iters', 50000, 'Maximum iterations per algorithm.')
flags.DEFINE_float('minimum_step_size', .001, 'Minimum step size.')
flags.DEFINE_float('maximum_step_size', 1., 'Maximum step size.')
flags.DEFINE_integer('num_step_sizes', 10, 'Number of step sizes to try.')
flags.DEFINE_float('convergence_tolerance', 1e-7, 'Tolerance for convergence.')


def run(environments: Sequence[structure.MarkovRewardProcess],
    step_sizes: Sequence[float], max_iters: int, convergence_tolerance: float):
  algorithm_constructors = []

  # Create a constructor that only depends on params common to all algorithms.
  rvi_algorithm = functools.partial(rvi.Evaluation, reference_index=0)
  algorithm_constructors.append(rvi_algorithm)

  for environment in environments:
    initial_values = np.zeros(environment.num_states)
    for algorithm_constructor in algorithm_constructors:
      print(f'Running {algorithm_constructor.func} on {environment.name}')
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
        print(f'step_size:{step_size:.5f}\tConverged:{converged}\tafter {i} iterations')


def main(argv):
  del argv  # Stop linter from complaining about unused argv.

  # Generate stepsizes log-spaced minimum and maximum supplied.
  step_sizes = np.geomspace(
      start=FLAGS.minimum_step_size,
      stop=FLAGS.maximum_step_size,
      num=FLAGS.num_step_sizes,
      endpoint=True)
  environments = [micro.mrp1, micro.mrp2, micro.mrp3]

  run(environments=environments,
      step_sizes=step_sizes,
      max_iters=FLAGS.max_iters,
      convergence_tolerance=FLAGS.convergence_tolerance)


if __name__ == '__main__':
  app.run(main)
