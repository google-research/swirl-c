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
"""A library to perform time integration using a third order Runge-Kutta."""

import logging
import numpy as np
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.numerics.time_integration import rhs_type
from swirl_c.physics import physics_models as physics_models_lib
import tensorflow as tf

# Coefficients in the 3rd order Runge Kutta time integration scheme.
# For du / dt = f(u), from u_{i} to u_{i+1}, the following steps are applied:
# u_1 = u_{i} + dt * f(u_{i})
# u_{i+1} = 0.5 * u_{i} + 0.5 * (u_1 + dt * f(u_1))


def integrate(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    state_0: types.FlowFieldMap,
    helper_vars: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
    rhs_fn: rhs_type.RHS,
) -> types.FlowFieldMap:
  """Integrates `rhs_function` with the 2nd order Runge-Kutta scheme.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    state_0: A dictionary of flow field variables representing the initial
      condition for time integration.
    helper_vars: Helper variables used in the simulation. These variables are
      not updated by the Navier-Stokes equations.
    cfg: The context object that stores parameters and information required by
      the simulation.
    physics_models: An object handler for physics models implemented in the
      current simulation.
    rhs_fn: A method to compute the right hand side of the governing equations
      to be integrated, i.e. d(state)/dt = `rhs_function(replica_id, state_0,
      helper_vars)`. The method must take `replica_id`, `state_0`, `helper_vars`
      as positional arguments, and return the time derivative of all of the
      variables provided in `state_0`.

  Returns:
    A dictionary of field variables after time integration over `cfg.dt` from
    the initial conditions `state_0`.
  """
  del replicas, physics_models

  logging.info('[RK2] Starting time integration.')

  step_fn = lambda u, f: u + cfg.dt * f

  df_dt = rhs_fn(replica_id, state_0, helper_vars)
  state_1 = {
      var_name: tf.nest.map_structure(step_fn, val, df_dt[var_name])
      for var_name, val in state_0.items()
  }

  df_dt = rhs_fn(replica_id, state_1, helper_vars)
  state_2 = {
      var_name: tf.nest.map_structure(step_fn, val, df_dt[var_name])
      for var_name, val in state_1.items()
  }

  return {
      var_name: tf.nest.map_structure(
          lambda a, b: 0.5 * (a + b), val, state_2[var_name]
      )
      for var_name, val in state_0.items()
  }
