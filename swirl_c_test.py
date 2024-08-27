# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
