"""Tests for basic ability to import subpackages."""
from absl.testing import absltest

from differential_value_iteration.algorithms import dvi
from differential_value_iteration.algorithms import rvi
from differential_value_iteration.algorithms import mdvi
from differential_value_iteration.environments import micro


class DifferentialValueIterationTest(absltest.TestCase):

  def test_algorithms_package_import(self):
    self.assertTrue(dvi.Control)
    self.assertTrue(dvi.Evaluation)

    self.assertTrue(rvi.Control)
    self.assertTrue(rvi.Evaluation)

    self.assertTrue(mdvi.Control1)
    self.assertTrue(mdvi.Control2)
    self.assertTrue(mdvi.Evaluation)

  def test_environments_package_import(self):
    self.assertTrue(micro.create_mrp1)
    self.assertTrue(micro.mdp1)
    self.assertTrue(micro.mdp2)


if __name__ == '__main__':
  absltest.main()
