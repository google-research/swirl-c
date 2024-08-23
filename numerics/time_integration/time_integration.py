"""A library to compute the time integration of the governing equations."""

import enum
import numpy as np
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.numerics.time_integration import rhs_type
from swirl_c.numerics.time_integration import rk2
from swirl_c.numerics.time_integration import rk3
from swirl_c.physics import physics_models as physics_models_lib
import tensorflow as tf


class TimeIntegrationScheme(enum.Enum):
  """Defines the time integrators."""
  # Second order Runge-Kutta time integrator.
  RK2 = 'rk2'
  # Third order Runge-Kutta time integrator.
  RK3 = 'rk3'


def time_integration(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    state_0: types.FlowFieldMap,
    helper_vars: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
    rhs_fn: rhs_type.RHS,
) -> types.FlowFieldMap:
  """Advances the simulation by integrating `rhs_function` for one timestep.

  This function approximates the time integration of the governing equations
  described by the method `rhs_function` using the time integration scheme
  specified by `cfg.time_integration_scheme`. The integration timestep is
  specified by `cfg.dt`.

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
      to be integrated, i.e. d(state)/dt = `rhs_function(replica_id, replicas,
      state_0, cfg)`. The method must take `replica_id`, `replicas`, `state_0`,
      `cfg`, and `physics_models` as positional arguments, and return the time
      derivative of all of the variables provided in `state_0`.

  Returns:
    A dictionary of field variables after time integration over `cfg.dt` from
    the initial conditions `state_0`.

  Raises:
    `NotImplementedError` if the specified time integration scheme is not
    implemented.
  """
  if cfg.time_integration_scheme == TimeIntegrationScheme.RK3.value:
    return rk3.integrate(
        replica_id, replicas, state_0, helper_vars, cfg, physics_models, rhs_fn
    )
  elif cfg.time_integration_scheme == TimeIntegrationScheme.RK2.value:
    return rk2.integrate(
        replica_id, replicas, state_0, helper_vars, cfg, physics_models, rhs_fn
    )
  else:
    raise NotImplementedError(
        f'"{cfg.time_integration_scheme}" is not implemented as an'
        ' "time_integration_scheme". Valid options are: '
        + str([scheme.value for scheme in TimeIntegrationScheme])
    )
