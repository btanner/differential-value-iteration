"""Tests for basic ability to import subpackages."""
from absl.testing import absltest

from differential_value_iteration.algorithms import algorithms
from differential_value_iteration.environments import micro


class DifferentialValueIterationTest(absltest.TestCase):

  def test_algorithms_package_import(self):
    self.assertTrue(algorithms.RVI_Evaluation)
    self.assertTrue(algorithms.DVI_Evaluation)
    self.assertTrue(algorithms.MDVI_Evaluation)
    self.assertTrue(algorithms.RVI_Control)
    self.assertTrue(algorithms.DVI_Control)
    self.assertTrue(algorithms.MDVI_Control1)
    self.assertTrue(algorithms.MDVI_Control2)

  def test_environments_package_import(self):
    self.assertTrue(micro.mrp1)
    self.assertTrue(micro.mrp2)
    self.assertTrue(micro.mdp1)
    self.assertTrue(micro.mdp2)


if __name__ == '__main__':
  absltest.main()
