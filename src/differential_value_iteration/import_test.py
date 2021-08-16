"""Tests for basic ability to import subpackages."""
from absl.testing import absltest

from differential_value_iteration.algorithms import algorithms
from differential_value_iteration.environments import environments

class DifferentialValueIterationTest(absltest.TestCase):

  def test_algorithms_package_import(self):
    self.assertTrue(algorithms.rvi_sync)

  def test_environments_package_import(self):
    self.assertTrue(environments.ThreeLoopMRP)


if __name__ == '__main__':
    absltest.main()
