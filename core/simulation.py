"""Simulation setup for the Swirl-C CFD solver."""

import enum
import numpy as np
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.equations import compressible_navier_stokes
from swirl_c.numerics.time_integration import time_integration
from swirl_c.physics import physics_models as physics_models_lib
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf


class Solvers(enum.Enum):
  """Defines available solver/governing equation combinations."""
  COMPRESSIBLE_NS = 'compressible_navier_stokes'


class Simulation:
  """Defines the step function for the solver."""

  def __init__(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      cfg: parameter.SwirlCParameters,
  ):
    """Initializes the simulation step.

    Args:
      kernel_op: Unused argument provided for compatibility with Swirl-LM. In
        Swirl-C the `kernel_op` is contained in the `cfg` object and is
        specified when the `cfg` class instance is constructed.
      cfg: The context object that stores parameters and information required by
        the simulation.
    """
    del kernel_op  # Unused.
    self.cfg = cfg
    self.physics_models = physics_models_lib.PhysicsModels(cfg)

    computation_shape = np.array([cfg.cx, cfg.cy, cfg.cz])
    replicas = np.arange(np.prod(computation_shape), dtype=np.int32).reshape(
        computation_shape
    )

    if cfg.solver == Solvers.COMPRESSIBLE_NS.value:
      self.rhs_fn = compressible_navier_stokes.get_rhs_fn(
          replicas, self.cfg, self.physics_models
      )
    else:
      raise NotImplementedError(
          f'"{cfg.solver}" is not an implemented solver. Valid options are: '
          '"compressible_navier_stokes"'
      )

  def step(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      step_id: tf.Tensor,
      states: types.FlowFieldMap,
      additional_states: types.FlowFieldMap,
  ):
    """Simulation step update function.

    Args:
      replica_id: The tf.Tensor containing the replica id.
      replicas: The replicas. In particular, a numpy array that maps grid
        coordinates to replica id numbers.
      step_id: A `tf.Tensor` denoting the current step id.
      states: A dictionary of flow field variables representing the average
        value of conservative variables in the computational cell. The dict must
        include all conservative variables listed in
        `cfg.conservative_variable_names`.
      additional_states: A dictionary of flow field variables which contains
        additional states used during the simulation. In Swirl-C, no additional
        states are required at this time, and the argument is included for
        compatibility with Swirl-LM.

    Returns:
      A flow field dictionary of conservative variables after time integration
      for a single timestep from an initial condition given by `states`.
    """
    del step_id  # Unused.
    return time_integration.time_integration(
        replica_id,
        replicas,
        states,
        additional_states,
        self.cfg,
        self.physics_models,
        self.rhs_fn,
    )
