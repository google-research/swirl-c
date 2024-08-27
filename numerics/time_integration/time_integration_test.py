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
"""Tests for time_integration.py."""

from unittest import mock
import numpy as np
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.numerics.time_integration import rhs_type
from swirl_c.numerics.time_integration import time_integration
from swirl_c.physics import physics_models as physics_models_lib
import tensorflow as tf


def _mock_rk3_integrator(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    state: types.FlowFieldMap,
    helper_vars: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
    rhs_function: rhs_type.RHS,
) -> types.FlowFieldMap:
  """A fake RK3 integration function for testing.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    state: A dictionary of flow field variables representing the initial
      condition for time integration.
    helper_vars: Helper variables used in the simulation. These variables are
      not updated by the Navier-Stokes equations.
    cfg: The context object that stores parameters and information required by
      the simulation.
    physics_models: An object handler for physics models implemented in the
      current simulation.
    rhs_function: A method to compute the right hand side of the governing
      equations to be integrated, i.e. d(state)/dt = `rhs_function(replica_id,
      replicas, state_0, cfg)`. The method must take `replica_id`, `replicas`,
      `state_0`, `cfg`, and `physics_models` as positional arguments, and return
      the time derivative of all of the variables provided in `state_0`.

  Returns:
    A specified derivative for testing.
  """
  del replica_id, replicas, helper_vars, cfg, physics_models, rhs_function
  df_dt = {
      var_name: tf.nest.map_structure(tf.ones_like, val)
      for var_name, val in state.items()
  }
  return df_dt


def _mock_rk2_integrator(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    state: types.FlowFieldMap,
    helper_vars: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
    rhs_function: rhs_type.RHS,
) -> types.FlowFieldMap:
  """A fake RK2 integration function for testing.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    state: A dictionary of flow field variables representing the initial
      condition for time integration.
    helper_vars: Helper variables used in the simulation. These variables are
      not updated by the Navier-Stokes equations.
    cfg: The context object that stores parameters and information required by
      the simulation.
    physics_models: An object handler for physics models implemented in the
      current simulation.
    rhs_function: A method to compute the right hand side of the governing
      equations to be integrated, i.e. d(state)/dt = `rhs_function(replica_id,
      replicas, state_0, cfg)`. The method must take `replica_id`, `replicas`,
      `state_0`, `cfg`, and `physics_models` as positional arguments, and return
      the time derivative of all of the variables provided in `state_0`.

  Returns:
    A specified derivative for testing.
  """
  del replica_id, replicas, helper_vars, cfg, physics_models, rhs_function
  df_dt = {
      var_name: tf.nest.map_structure(lambda v: 6.0 * tf.ones_like(v), val)
      for var_name, val in state.items()
  }
  return df_dt


def _rhs_function_constant(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    state: types.FlowFieldMap,
    helper_vars: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
) -> types.FlowFieldMap:
  """A fake RHS function for testing.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    state: A dictionary of flow field variables representing the initial
      condition for time integration.
    helper_vars: Helper variables used in the simulation. These variables are
      not updated by the Navier-Stokes equations.
    cfg: The context object that stores parameters and information required by
      the simulation.
    physics_models: An object handler for physics models implemented in the
      current simulation.

  Returns:
    A specified derivative for testing.
  """
  del replica_id, replicas, helper_vars, cfg, physics_models
  df_dt = {
      var_name: tf.nest.map_structure(tf.ones_like, val)
      for var_name, val in state.items()
  }
  return df_dt


class TimeIntegrationTest(tf.test.TestCase):

  def test_time_integration_raise_on_bad_option(self):
    """Tests error is raised for not implemented time_integration."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    dim = 'x'
    cfg = parameter.SwirlCParameters(
        {'dt': 0.1, 'time_integration_scheme': 'bad_model_name'}
    )
    physics_model = physics_models_lib.PhysicsModels(cfg)
    conservative = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx), dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(np.ones(nx) * 2.0, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(np.ones(nx) * 3.0, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(np.ones(nx) * 4.0, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(np.ones(nx) * 5.0, dim, size),
        'rho_y': testing_utils.to_3d_tensor(np.ones(nx) * 6.0, dim, size),
    }
    msg = r'^("bad_model_name" is not implemented as an)'
    with self.assertRaisesRegex(NotImplementedError, msg):
      time_integration.time_integration(
          replica_id,
          replicas,
          conservative,
          {},
          cfg,
          physics_model,
          _rhs_function_constant,
      )

  @mock.patch(
      'swirl_c.numerics.time_integration.rk3.integrate',
      _mock_rk3_integrator,
  )
  def test_time_integration_selects_rk3_correctly(self):
    """Tests that the time integration switches to the rk3 integrator."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    dim = 'x'
    cfg = parameter.SwirlCParameters(
        {'dt': 0.1, 'time_integration_scheme': 'rk3'}
    )
    physics_model = physics_models_lib.PhysicsModels(cfg)
    conservative = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * 0.5, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(np.ones(nx) * 2.0, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(np.ones(nx) * 3.0, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(np.ones(nx) * 4.0, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(np.ones(nx) * 5.0, dim, size),
        'rho_y': testing_utils.to_3d_tensor(np.ones(nx) * 6.0, dim, size),
    }
    results = self.evaluate(
        time_integration.time_integration(
            replica_id,
            replicas,
            conservative,
            {},
            cfg,
            physics_model,
            _rhs_function_constant,
        )
    )
    expected = {
        types.RHO: testing_utils.to_3d_tensor(
            np.ones(nx), dim, size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx), dim, size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx), dim, size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx), dim, size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx), dim, size, as_tf_tensor=False
        ),
        'rho_y': testing_utils.to_3d_tensor(
            np.ones(nx), dim, size, as_tf_tensor=False
        ),
    }

    self.assertDictEqual(results, expected)

  @mock.patch(
      'swirl_c.numerics.time_integration.rk2.integrate',
      _mock_rk2_integrator,
  )
  def test_time_integration_selects_rk2_correctly(self):
    """Tests that the time integration switches to the RK2 integrator."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    dim = 'x'
    cfg = parameter.SwirlCParameters(
        {'dt': 0.1, 'time_integration_scheme': 'rk2'}
    )
    physics_model = physics_models_lib.PhysicsModels(cfg)
    conservative = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * 0.5, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(np.ones(nx) * 2.0, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(np.ones(nx) * 3.0, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(np.ones(nx) * 4.0, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(np.ones(nx) * 5.0, dim, size),
        'rho_y': testing_utils.to_3d_tensor(np.ones(nx) * 6.0, dim, size),
    }
    results = self.evaluate(
        time_integration.time_integration(
            replica_id,
            replicas,
            conservative,
            {},
            cfg,
            physics_model,
            _rhs_function_constant,
        )
    )
    expected = {
        types.RHO: testing_utils.to_3d_tensor(
            6.0 * np.ones(nx), dim, size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            6.0 * np.ones(nx), dim, size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            6.0 * np.ones(nx), dim, size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            6.0 * np.ones(nx), dim, size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            6.0 * np.ones(nx), dim, size, as_tf_tensor=False
        ),
        'rho_y': testing_utils.to_3d_tensor(
            6.0 * np.ones(nx), dim, size, as_tf_tensor=False
        ),
    }

    self.assertDictEqual(results, expected)


if __name__ == '__main__':
  tf.test.main()
