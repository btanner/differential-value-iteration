"""Tests for basic ability to import subpackages."""
from absl.testing import absltest



class DifferentialValueIterationTest(absltest.TestCase):

  def test_algorithms_package_import(self):
    with self.subTest('dvi'):
      from differential_value_iteration.algorithms import dvi
      self.assertTrue(dvi.Control)
      self.assertTrue(dvi.Evaluation)

    with self.subTest('rvi'):
      from differential_value_iteration.algorithms import rvi
      self.assertTrue(rvi.Control)
      self.assertTrue(rvi.Evaluation)

    with self.subTest('mdvi'):
      from differential_value_iteration.algorithms import mdvi
      self.assertTrue(mdvi.Control1)
      self.assertTrue(mdvi.Control2)
      self.assertTrue(mdvi.Evaluation)

  def test_environments_package_import(self):
    from differential_value_iteration.environments import micro

  def test_quantecon_package_import(self):
    import quantecon


if __name__ == '__main__':
  absltest.main()
