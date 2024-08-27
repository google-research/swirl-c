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
"""Defines a baseline class for transport models."""

from typing import Any
from swirl_c.common import types
from swirl_c.core import parameter
import tensorflow as tf


# BEGIN GOOGLE-INTERNAL
# TODO: b/311341665 - This circular dependence arises because we need to
# evaluate the thermodynamic state to evaluate transport properties. Creating
# a generic abstract transport class is a better solution than the any typing.
# END GOOGLE-INTERNAL
physics_models_lib = Any  # swirl_c.physics.physics_models imports this module.


class TransportSimple:
  """Defines function contracts for transport models."""

  def __init__(self, cfg: parameter.SwirlCParameters):
    """Constructor for the TransportSimple class.

    Args:
      cfg: The context object that stores parameters and information required by
        the simulation.
    """
    # BEGIN GOOGLE-INTERNAL
    # TODO: b/311341674 - Make use of proto to define transport model
    # parameters.
    # END GOOGLE-INTERNAL
    self._nu = cfg.transport_parameters['nu']
    self._pr = cfg.transport_parameters['pr']

  def kinematic_viscosity(
      self,
      states: types.FlowFieldMap,
  ) -> types.FlowFieldVar:
    """Computes the kinematic viscosity from the state and returns as a tensor.

    Args:
      states: A dictionary of flow field variables that must contain the
        density `RHO`.

    Returns:
      A flow field variable representing the kinematic viscosity across the
      computational domain.
    """
    return tf.nest.map_structure(
        lambda rho: self._nu * tf.ones_like(rho), states[types.RHO]
    )

  def dynamic_viscosity(
      self,
      states: types.FlowFieldMap,
  ) -> types.FlowFieldVar:
    """Computes the dynamic viscosity from the kinematic viscosity and density.

    Args:
      states: A dictionary of flow field variables that must contain the
        density `RHO`.

    Returns:
      A flow field variable representing the dynamic viscosity across the
      computational domain.
    """
    return tf.nest.map_structure(lambda rho: self._nu * rho, states[types.RHO])

  def thermal_conductivity(
      self,
      states: types.FlowFieldMap,
      physics_models: 'physics_models_lib.PhysicsModel',
  ) -> types.FlowFieldVar:
    """Computes the thermal conductivity from the state, Pr, and viscosity.

    Assuming a constant Prandtl number Pr, the thermal conductivity can be
    computed as,
    k = 1 / Pr * cp * mu,
    where k is the thermal conductivity, cp is the specific heat at constant
    pressure, and mu is the dynamic viscosity.

    Args:
      states: Dictionary of scalar flow field variables which must include the
        density `RHO`.
      physics_models: An object handler for physics models implemented in the
        current simulation.

    Returns:
      A flow field variable representing the thermal conductivity across the
      computational domain.
    """
    return tf.nest.map_structure(
        lambda mu, cp: (1.0 / self._pr) * cp * mu,
        self.dynamic_viscosity(states),
        physics_models.thermodynamics_model.cp(states),
    )
