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
"""Tests for rk3.py."""

import numpy as np
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.numerics.time_integration import rk3
from swirl_c.physics import physics_models as physics_models_lib
import tensorflow as tf


def _rhs_function_constant(
    replica_id: tf.Tensor,
    state: types.FlowFieldMap,
    helper_vars: types.FlowFieldMap,
) -> types.FlowFieldMap:
  """A fake RHS function for testing.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    state: A dictionary of flow field variables representing the initial
      condition for time integration.
    helper_vars: Helper variables used in the simulation. These variables are
      not updated by the Navier-Stokes equations.

  Returns:
    A specified derivative for testing.
  """
  del replica_id, helper_vars  # unused.
  df_dt = {
      var_name: tf.nest.map_structure(tf.zeros_like, val)
      for var_name, val in state.items()
  }
  return df_dt


def _rhs_function_specified(
    replica_id: tf.Tensor,
    state: types.FlowFieldMap,
    helper_vars: types.FlowFieldMap,
) -> types.FlowFieldMap:
  """A fake RHS function for testing.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    state: A dictionary of flow field variables representing the initial
      condition for time integration.
    helper_vars: Helper variables used in the simulation. These variables are
      not updated by the Navier-Stokes equations.

  Returns:
    A specified derivative for testing.
  """
  del replica_id, helper_vars
  df_dt = {
      var_name: tf.nest.map_structure(tf.ones_like, val)
      for var_name, val in state.items()
  }
  rho_test = tf.nest.map_structure(tf.reduce_mean, state['rho'])

  if rho_test[0] < 2.0:  # Stage 1.
    df_dt = {
        var_name: tf.nest.map_structure(lambda x: x * 20.0, val)
        for var_name, val in df_dt.items()
    }
  elif rho_test[0] > 2.75:  # Stage 2.
    df_dt = {
        var_name: tf.nest.map_structure(lambda x: x * 40.0, val)
        for var_name, val in df_dt.items()
    }
  else:  # Stage 3.
    df_dt = {
        var_name: tf.nest.map_structure(lambda x: x * -10.0, val)
        for var_name, val in df_dt.items()
    }
  return df_dt


class RK3Test(tf.test.TestCase):

  def test_integrate_constant_solution_in_time(self):
    """Tests the integrator when RHS is all zeros."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 4
    size = [nx, nx, nx]
    dim = 'x'
    cfg = parameter.SwirlCParameters({
        'dt': 0.1,
        'core_nx': nx,
        'core_ny': nx,
        'core_nz': nx,
        'halo_width': 0,
        'time_integration_scheme': 'rk3',
        'conservative_variable_names': list(types.BASE_CONSERVATIVE) + [
            'rho_y',
        ],
    })
    physics_model = physics_models_lib.PhysicsModels(cfg)
    conservative = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx), dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(np.ones(nx) * 2.0, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(np.ones(nx) * 3.0, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(np.ones(nx) * 4.0, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(np.ones(nx) * 5.0, dim, size),
        'rho_y': testing_utils.to_3d_tensor(np.ones(nx) * 6.0, dim, size),
    }
    expected = {
        types.RHO: testing_utils.to_3d_tensor(
            np.ones(nx), dim, size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx) * 2.0, dim, size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx) * 3.0, dim, size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx) * 4.0, dim, size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx) * 5.0, dim, size, as_tf_tensor=False
        ),
        'rho_y': testing_utils.to_3d_tensor(
            np.ones(nx) * 6.0, dim, size, as_tf_tensor=False
        ),
    }
    results = self.evaluate(
        rk3.integrate(
            replica_id,
            replicas,
            conservative,
            {},
            cfg,
            physics_model,
            _rhs_function_constant,
        )
    )
    self.assertDictEqual(results, expected)

  def test_integrate_arbitrary_rhs(self):
    """Tests the integrator for specified arbitrary RHS values."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 4
    size = [nx, nx, nx]
    dim = 'x'
    cfg = parameter.SwirlCParameters({
        'dt': 0.1,
        'core_nx': nx,
        'core_ny': nx,
        'core_nz': nx,
        'halo_width': 0,
        'time_integration_scheme': 'rk3',
        'conservative_variable_names': list(types.BASE_CONSERVATIVE) + [
            'rho_y',
        ],
    })
    physics_model = physics_models_lib.PhysicsModels(cfg)
    conservative = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx), dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(np.ones(nx) * 2.0, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(np.ones(nx) * 3.0, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(np.ones(nx) * 4.0, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(np.ones(nx) * 5.0, dim, size),
        'rho_y': testing_utils.to_3d_tensor(np.ones(nx) * 6.0, dim, size),
    }
    expected = {
        types.RHO: testing_utils.to_3d_tensor(
            np.ones(nx) + 1.0 / 3.0, dim, size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx) * 2.0 + 1.0 / 3.0, dim, size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx) * 3.0 + 1.0 / 3.0, dim, size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx) * 4.0 + 1.0 / 3.0, dim, size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx) * 5.0 + 1.0 / 3.0, dim, size, as_tf_tensor=False
        ),
        'rho_y': testing_utils.to_3d_tensor(
            np.ones(nx) * 6.0 + 1.0 / 3.0, dim, size, as_tf_tensor=False
        ),
    }
    results = self.evaluate(
        rk3.integrate(
            replica_id,
            replicas,
            conservative,
            {},
            cfg,
            physics_model,
            _rhs_function_specified,
        )
    )
    self.assertDictEqual(results, expected)


if __name__ == '__main__':
  tf.test.main()
