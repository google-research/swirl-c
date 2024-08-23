"""A basic test for Swirl-C to ensure it compiles."""

import unittest

from swirl_c.boundary import bc_types
from swirl_c.boundary import boundary
from swirl_c.common import types
from swirl_c.common import utils
from swirl_c.core import parameter
from swirl_c.numerics import gradient
from swirl_c.physics import constant
from swirl_c.physics import fluid
from swirl_c.physics.thermodynamics import generic


class SwirlCTest(unittest.TestCase):

  def test_swirl_c_modules_loaded_successfully(self):
    """Checks if all modeuls in swirl_c are loaded successfully."""
    self.assertTrue(bc_types)
    self.assertTrue(boundary)
    self.assertTrue(types)
    self.assertTrue(utils)
    self.assertTrue(parameter)
    self.assertTrue(gradient)
    self.assertTrue(constant)
    self.assertTrue(fluid)
    self.assertTrue(generic)


if __name__ == "__main__":
  unittest.main()
