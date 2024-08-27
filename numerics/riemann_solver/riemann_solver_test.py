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
"""Tests for riemann_solver.py."""

from swirl_c.core import parameter
from swirl_c.numerics.riemann_solver import hll
from swirl_c.numerics.riemann_solver import riemann_solver
from swirl_c.physics import physics_models as physics_models_lib
import tensorflow as tf


class RiemannSolverTest(tf.test.TestCase):

  def setUp(self):
    """Initializes the generic thermodynamics library object."""
    super().setUp()
    self.cfg = parameter.SwirlCParameters()
    self.physics_models = physics_models_lib.PhysicsModels(self.cfg)

  def test_riemann_connects_hll(self):
    """Function to check that the linker function correctly links to hll.py."""
    self.cfg.numeric_flux_scheme = 'HLL'
    result = riemann_solver.select_numeric_flux_fn(
        self.cfg,
    )
    self.assertIs(result, hll.hll_convective_flux)

  def test_riemann_raises_for_bad_opt(self):
    """Function to check ValueError is raised if scheme not implemented."""
    self.cfg.numeric_flux_scheme = 'bad_opt'
    msg = r'^("bad_opt" is not implemented)'
    with self.assertRaisesRegex(ValueError, msg):
      riemann_solver.select_numeric_flux_fn(self.cfg)


if __name__ == '__main__':
  tf.test.main()
