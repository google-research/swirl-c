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
# u_2 = c11 * u_{i} + c12 * (u_1 + dt * f(u_1))
# u_{i+1} = c21 * u_{i} + c22 * (u_2 + dt * f(u_2))
_RK3_COEFFS = {'c11': 0.75, 'c21': 1.0 / 3.0}
_RK3_COEFFS['c12'] = 1.0 - _RK3_COEFFS['c11']
_RK3_COEFFS['c22'] = 1.0 - _RK3_COEFFS['c21']


def integrate(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    state_0: types.FlowFieldMap,
    helper_vars: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
    rhs_fn: rhs_type.RHS,
) -> types.FlowFieldMap:
  """Integrates `rhs_function` with the 3rd order Runge-Kutta scheme.

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

  def rk_fn(c0, f0, c1, f1, df1_dt, dt):
    """Computes the next rk stage from current, derivative, and weights."""
    if f0 is None:  # First stage of RK integrator.
      return tf.nest.map_structure(
          lambda f1, df1_dt: f1 + df1_dt * dt,
          f1,
          df1_dt,
      )
    else:  # Subsequent integrator stages.
      return tf.nest.map_structure(
          lambda f0, f1, df1_dt: f0 * c0 + (f1 + df1_dt * dt) * c1,
          f0,
          f1,
          df1_dt,
      )

  logging.info('[RK3] Starting time integration.')

  df_dt = rhs_fn(replica_id, state_0, helper_vars)
  state_1 = {
      var_name: rk_fn([], None, [], val, df_dt[var_name], cfg.dt)
      for var_name, val in state_0.items()
  }
  logging.info('[RK3] First stage completed.')

  df_dt = rhs_fn(replica_id, state_1, helper_vars)
  state_2 = {
      var_name: rk_fn(
          _RK3_COEFFS['c11'],
          val,
          _RK3_COEFFS['c12'],
          state_1[var_name],
          df_dt[var_name],
          cfg.dt,
      )
      for var_name, val in state_0.items()
  }
  logging.info('[RK3] Second stage completed.')

  df_dt = rhs_fn(replica_id, state_2, helper_vars)
  state_n = {
      var_name: rk_fn(
          _RK3_COEFFS['c21'],
          val,
          _RK3_COEFFS['c22'],
          state_2[var_name],
          df_dt[var_name],
          cfg.dt,
      )
      for var_name, val in state_0.items()
  }
  logging.info('[RK3] Third stage completed.')

  return state_n