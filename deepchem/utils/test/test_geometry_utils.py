import unittest
import numpy as np
from deepchem.utils.geometry_utils import unit_vector
from deepchem.utils.geometry_utils import angle_between
from deepchem.utils.geometry_utils import generate_random_unit_vector
from deepchem.utils.geometry_utils import generate_random_rotation_matrix

class TestGeometryUtils(unittest.TestCase):

  def test_generate_random_unit_vector(self):
    for _ in range(100):
      u = generate_random_unit_vector()
      # 3D vector with unit length
      self.assertEqual(u.shape, (3,))
      self.assertAlmostEqual(np.linalg.norm(u), 1.0)

  def test_generate_random_rotation_matrix(self):
    # very basic test, we check if rotations actually work in test_rotate_molecules
    for _ in range(100):
      m = generate_random_rotation_matrix()
      self.assertEqual(m.shape, (3, 3))

  def test_unit_vector(self):
    for _ in range(10):
      vector = np.random.rand(3)
      norm_vector = unit_vector(vector)
      self.assertAlmostEqual(np.linalg.norm(norm_vector), 1.0)

  def test_angle_between(self):
    for _ in range(10):
      v1 = np.random.rand(3,)
      v2 = np.random.rand(3,)
      angle = angle_between(v1, v2)
      self.assertLessEqual(angle, np.pi)
      self.assertGreaterEqual(angle, 0.0)
      self.assertAlmostEqual(angle_between(v1, v1), 0.0)
      self.assertAlmostEqual(angle_between(v1, -v1), np.pi)

