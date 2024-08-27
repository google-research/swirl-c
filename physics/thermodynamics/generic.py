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
"""Defines a baseline class for thermodynamic models."""

import numpy as np
from swirl_c.common import types
from swirl_c.common import utils
from swirl_c.core import parameter
from swirl_c.physics import constant
import tensorflow as tf


class ThermodynamicsGeneric:
  """Defines function contracts for thermodynamic models."""

  def pressure(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: types.FlowFieldMap,
      cfg: parameter.SwirlCParameters,
  ) -> types.FlowFieldVar:
    """Computes the pressure from total energy.

    p = (γ-1)ρeᵢₙₜ,
    where eᵢₙₜ is the internal energy. Internal energy is first computed from
    total energy.

    Args:
      replica_id: The ID of the computatinal subdomain (replica).
      replicas: A numpy array that maps a replica's grid coordinate to its
        `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`.
      states: Dictionary of scalar flow field variables which must include the
        density `RHO`, three components of the velocity `U`, `V`, `W`, and the
        specific total energy `E`.
      cfg: The context object that stores parameters and information required by
        the simulation.

    Returns:
      The pressure.
    """
    return tf.nest.map_structure(
        lambda rho, e: (constant.GAMMA - 1.0) * rho * e,
        states[types.RHO],
        self.internal_energy(replica_id, replicas, states, cfg),
    )

  def total_enthalpy(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: types.FlowFieldMap,
      cfg: parameter.SwirlCParameters,
  ) -> types.FlowFieldVar:
    """Computes the total enthalpy.

    h = e + p/ρ.

    Args:
      replica_id: The ID of the computatinal subdomain (replica).
      replicas: A numpy array that maps a replica's grid coordinate to its
        `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`.
      states: Dictionary of scalar flow field variables which must include the
        density `RHO`, three components of the velocity `U`, `V`, `W`, and the
        specific total energy `E`.
      cfg: The context object that stores parameters and information required by
        the simulation.

    Returns:
      The total enthalpy.
    """
    return tf.nest.map_structure(
        lambda e, p, rho: e + p / rho,
        states[types.E],
        self.pressure(replica_id, replicas, states, cfg),
        states[types.RHO],
    )

  def sound_speed(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: types.FlowFieldMap,
      cfg: parameter.SwirlCParameters,
      opt: str = types.E,
  ) -> types.FlowFieldVar:
    """Computes the speed of sound for an ideal gas.

    For computing sound speed from internal energy:
    c = √γ(γ-1)eᵢₙₜ.
    If total energy is provided, internal energy is first calculated from total
    energy.

    For computing sound speed from sensible enthalpy:
    c = √(γ-1)hₛₑₙₛ
    If total enthalpy is provided, sensible enthalpy is first calculated from
    total enthalpy.

    For computing sound speed from pressure and density:
    c = √γp/ρ

    Args:
      replica_id: The ID of the computatinal subdomain (replica).
      replicas: A numpy array that maps a replica's grid coordinate to its
        `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`.
      states: Dictionary of scalar flow field variables. For `opt = E` and `opt
        = H`, the three components of the velocity `U`, `V`, `W` are required.
        For `opt = E`, the dictionary must include the total energy `E`. For
        `opt = H` the the dictionary must include the total enthalpy `H`. For
        `opt = 'p_rho'` only the gas pressure `P` and density `RHO` are
        required.
      cfg: The context object that stores parameters and information required by
        the simulation.
      opt: A string indicating the method to compute the sound speed. For `opt =
        E` (default) sound speed is computed from the provided total energy. For
        `opt = H` sound speed is computed from the provided total enthalpy. For
        `opt = 'p_rho' the sound speed is computed from the provided pressure
        and density.

    Returns:
      The speed of sound.

    Raises:
      ValueError if an incorrect `opt` string is provided.
    """
    if opt == types.E:
      return tf.nest.map_structure(
          lambda e_int: tf.math.sqrt(
              constant.GAMMA * (constant.GAMMA - 1.0) * e_int
          ),
          self.internal_energy(replica_id, replicas, states, cfg),
      )
    elif opt == types.H:
      return tf.nest.map_structure(
          lambda h_sens: tf.math.sqrt((constant.GAMMA - 1.0) * h_sens),
          self.sensible_enthalpy(replica_id, replicas, states, cfg),
      )
    elif opt == 'p_rho':
      return tf.nest.map_structure(
          lambda p, rho: tf.math.sqrt(constant.GAMMA * p / rho),
          states[types.P],
          states[types.RHO],
      )
    else:
      raise ValueError(
          f'"{opt}" is not a valid option for sound speed. Available options'
          f' are "{types.E}", "{types.H}", "p_rho".'
      )

  def potential_temperature(
      self,
      states: types.FlowFieldMap,
      cfg: parameter.SwirlCParameters,
  ) -> types.FlowFieldVar:
    """Computes the potential temperature from the pressure and temperature.

    θ = T (p/p₀)**-κ

    Args:
      states: Dictionary of scalar flow field variables containing the gas
        temperature `T` and pressure `P`.
      cfg: Simulation configuration variable. Must contain the reference
        pressure 'p_0'.

    Returns:
      The potential temperature θ.
    """
    return tf.nest.map_structure(
        lambda t, p: t * (p / cfg.p_0) ** -constant.KAPPA,
        states[types.T],
        states[types.P],
    )

  def temperature(
      self,
      states: types.FlowFieldMap,
      cfg: parameter.SwirlCParameters,
      opt: str,
  ) -> types.FlowFieldVar:
    """Computes the temperature from either potential temperature or EOS.

    For computing temperature from potential temperature:
    T = θ (p/p₀)**κ

    For computing temperature from the ideal gas equation of state:
    T = p/ρR

    Args:
      states: Dictionary of scalar flow field variables. For `opt = 'theta'`,
        `states` must include the pressure `P` and potential temperature
        `POTENTIAL_T`. For `opt = 'eos'`, `states` must include the pressure `P`
        and the density `RHO`.
      cfg: Simulation configuration variable. Must contain the reference
        pressure 'p_0' if `opt = 'theta'`.
      opt: String indicating method to calculate temperature. `opt = 'theta'`
        indicates temperature is computed from pressure and potential
        temperature. `opt = 'eos'` indicates temperature is calculated from
        pressure and density using the ideal gas equation of state.

    Returns:
      The temperature.

    Raises:
      ValueError if an incorrect `opt` string is provided.
    """
    if opt == 'theta':
      return tf.nest.map_structure(
          lambda theta, p: theta * (p / cfg.p_0) ** constant.KAPPA,
          states[types.POTENTIAL_T],
          states[types.P],
      )
    elif opt == 'eos':
      return tf.nest.map_structure(
          lambda p, rho: p / (rho * constant.R),
          states[types.P],
          states[types.RHO],
      )
    else:
      raise ValueError(
          f'"{opt}" is not a valid option for temperature. Available options'
          ' are "eos", "theta".'
      )

  def density(
      self,
      states: types.FlowFieldMap,
      cfg: parameter.SwirlCParameters,
      opt: str,
  ) -> types.FlowFieldVar:
    """Computes the density from either potential temperature or EOS.

    If temperature and pressure are provided:
    ρ = p/RT

    If potential temperature and pressure are provided, first the temperature is
      calculated:
    T = θ (p/p)₀**κ,
    Then the ideal gas equation of state is used to compute density.

    Args:
      states: Dictionary of scalar flow field variables. For `opt = 'theta'`,
        `states` must include the pressure `P` and potential temperature
        `POTENTIAL_T`. For `opt = 'eos'`, `states` must include the pressure `P`
        and the temperature `T`.
      cfg: Simulation configuration variable. Must contain the reference
        pressure 'p_0' if `opt = 'theta'`.
      opt: String indicating method to calculate temperature. `opt = 'theta'`
        indicates density is computed from pressure and potential temperature.
        `opt = 'eos'` indicates density is calculated from pressure and
        temperature using the ideal gas equation of state.

    Returns:
      The density.

    Raises:
      ValueError if an incorrect `opt` string is provided.
    """
    if opt == 'theta':
      t = self.temperature(states, cfg, opt)
    elif opt == 'eos':
      t = states[types.T]
    else:
      raise ValueError(
          f'"{opt}" is not a valid option for density. Available options'
          ' are "eos", "theta".'
      )
    return tf.nest.map_structure(
        lambda p, t: p / (constant.R * t), states[types.P], t
    )

  def gravitational_potential_energy(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      cfg: parameter.SwirlCParameters,
  ) -> types.FlowFieldVar:
    """Computes the specific gravitational potential energy of a fluid element.

    e_g = gh
    Where g is `constants.G` and h is the geopotential height.

    Args:
      replica_id: The ID of the computatinal subdomain (replica).
      replicas: A numpy array that maps a replica's grid coordinate to its
        `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`.
      cfg: The context object that stores parameters and information required by
        the simulation.

    Returns:
      The specific gravitational potential energy (J/kg).
    """
    # Expand the local mesh to get the local spatial postion.
    mesh = utils.mesh_local_expanded(replica_id, replicas, cfg)

    # If gravity is not present, we will return zeros like the 'z' mesh, which
    # will broadcast to the correct shape when added/subtracted from a
    # FlowFieldVar.
    if utils.gravity_direction(cfg) == -1:
      return tf.nest.map_structure(tf.zeros_like, mesh['z'])

    dims = ('x', 'y', 'z')
    g_vec = constant.G * np.fromiter((cfg.g[dim] for dim in dims), dtype=float)
    # The standard definition of potential energy is:
    # (potential energy) = - (work done to move object),
    # Therefore, subtract dot product of gravitional accelration and position
    # vectors.
    pe_fn = lambda x, y, z: -g_vec[0] * x - g_vec[1] * y - g_vec[2] * z
    return tf.nest.map_structure(pe_fn, mesh['x'], mesh['y'], mesh['z'])

  def total_energy(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: types.FlowFieldMap,
      cfg: parameter.SwirlCParameters,
  ) -> types.FlowFieldVar:
    """Computes the total energy for an ideal gas.

    eₜ = cᵥT + ½(u² + v² + w²) + gz
    Where gz is the contribution of gravitational potential.

    Args:
      replica_id: The ID of the computatinal subdomain (replica).
      replicas: A numpy array that maps a replica's grid coordinate to its
        `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`.
      states: Dictionary of scalar flow field variables which must include the
        temperature `T` and three components of the velocity `U`, `V`, `W`.
      cfg: The context object that stores parameters and information required by
        the simulation.

    Returns:
      The gas density.
    """
    e_t = tf.nest.map_structure(
        lambda t, u, v, w, pe: constant.CV * t  # pylint: disable=g-long-lambda
        + 0.5 * (u**2 + v**2 + w**2)
        + pe,
        states[types.T],
        states[types.U],
        states[types.V],
        states[types.W],
        self.gravitational_potential_energy(replica_id, replicas, cfg),
    )
    return e_t

  def internal_energy(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: types.FlowFieldMap,
      cfg: parameter.SwirlCParameters,
  ) -> types.FlowFieldVar:
    """Computes the internal energy from the total energy.

    eᵢₙₜ = eₜ - ½(u² + v² + w²) - gz,
    where gz is the contribution of gravitational potential.

    Args:
      replica_id: The ID of the computatinal subdomain (replica).
      replicas: A numpy array that maps a replica's grid coordinate to its
        `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`.
      states: Dictionary of scalar flow field variables which must include the
        total energy `E` and three components of the velocity `U`, `V`, `W`.
      cfg: The context object that stores parameters and information required by
        the simulation.

    Returns:
      The gas internal energy.
    """
    e_int = tf.nest.map_structure(
        lambda e_t, u, v, w, pe: e_t - 0.5 * (u**2 + v**2 + w**2) - pe,
        states[types.E],
        states[types.U],
        states[types.V],
        states[types.W],
        self.gravitational_potential_energy(replica_id, replicas, cfg),
    )
    return e_int

  def sensible_enthalpy(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: types.FlowFieldMap,
      cfg: parameter.SwirlCParameters,
  ) -> types.FlowFieldVar:
    """Computes the sensible enthalpy from the total enthalpy.

    hₛₑₙₛ = hₜ - ½(u² + v² + w²) - gz,
    where gz is the contribution of gravitational potential.

    Args:
      replica_id: The ID of the computatinal subdomain (replica).
      replicas: A numpy array that maps a replica's grid coordinate to its
        `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`.
      states: Dictionary of scalar flow field variables which must include the
        total enthalpy `H` and three components of the velocity `U`, `V`, `W`.
      cfg: The context object that stores parameters and information required by
        the simulation.

    Returns:
      The gas sensible enthalpy.
    """
    h_sens = tf.nest.map_structure(
        lambda h_t, u, v, w, pe: h_t - 0.5 * (u**2 + v**2 + w**2) - pe,
        states[types.H],
        states[types.U],
        states[types.V],
        states[types.W],
        self.gravitational_potential_energy(replica_id, replicas, cfg),
    )
    return h_sens

  def cp(
      self,
      states: types.FlowFieldMap,
  ) -> types.FlowFieldVar:
    """Returns the constant pressure specific heat of the gas.

    Args:
      states: Dictionary of scalar flow field variables which are required to
        compute `cp`. Here, only `RHO` is required.

    Returns:
      The mixture specific heat at constant pressure as a flow field variable.
    """
    return tf.nest.map_structure(
        lambda rho: constant.CP * tf.ones_like(rho), states[types.RHO]
    )
