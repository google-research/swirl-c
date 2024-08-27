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
"""Tests for simple.py."""

import numpy as np
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.physics import constant
from swirl_c.physics import physics_models as physics_models_lib
from swirl_c.physics.transport import simple
import tensorflow as tf


class PhysicsModelsTest(tf.test.TestCase):

  def setUp(self):
    """Initializes the cfg, physics model, and thermodynamics model."""
    super().setUp()
    user_cfg = {
        'conservative_variable_names': list(types.BASE_CONSERVATIVE) + [
            'rho_y'
        ],
        'thermodynamics_model': 'generic',
        'transport_model': 'simple',
        'transport_parameter': {
            'nu': 1.0e-4,
            'pr': 0.7,
        },
    }
    self.cfg = parameter.SwirlCParameters(user_cfg)
    self.physics_model = physics_models_lib.PhysicsModels(self.cfg)
    self.transport_model = simple.TransportSimple(self.cfg)

  def test_dynamic_viscosity_computes_properly(self):
    """Tests that the dynamic viscosity is computed from density correctly."""
    nx = 16
    size = [nx, nx, nx]
    rho_np = np.random.uniform(size=size, low=1.0, high=2.0)
    rho_tf = tf.unstack(tf.convert_to_tensor(rho_np, dtype=types.DTYPE))
    states = {types.RHO: rho_tf}
    results = self.evaluate(self.transport_model.dynamic_viscosity(states))
    expected = rho_np * 1.0e-4
    self.assertAllClose(expected, results)

  def test_thermal_conductivity_computes_properly(self):
    """Tests that the thermal conductivity is computed correctly."""
    nx = 16
    size = [nx, nx, nx]
    rho_np = np.random.uniform(size=size, low=1.0, high=2.0)
    rho_tf = tf.unstack(tf.convert_to_tensor(rho_np, dtype=types.DTYPE))
    states = {types.RHO: rho_tf}
    results = self.evaluate(
        self.transport_model.thermal_conductivity(states, self.physics_model)
    )
    expected = (1.0 / 0.7) * constant.CP * 1.0e-4 * rho_np
    self.assertAllClose(expected, results)


if __name__ == '__main__':
  tf.test.main()
