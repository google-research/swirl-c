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
"""A library of fluid related variables."""

from typing import Dict, List

from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.numerics import gradient
from swirl_c.numerics import interpolation
from swirl_c.physics import fluid
from swirl_c.physics import physics_models as physics_models_lib
import tensorflow as tf


# BEGIN GOOGLE-INTERNAL
# Note that we do not support setting a face boundary condition to use when
# computing the diffusive fluxes. As such, any boundary condition must be
# specified through either the cell average values, or by overwriting the net
# diffusive flux. See b/310692602 for additional details.
# END GOOGLE-INTERNAL
def shear_stress(
    states: types.FlowFieldMap,
    states_fx: types.FlowFieldMap,
    states_fy: types.FlowFieldMap,
    states_fz: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
) -> List[List[tf.Tensor]]:
  R"""Computes the shear stress tensor (3 x 3).

  Args:
    states: A dictionary of cell-averaged/centered 3D flow-field variables.
    states_fx: A dictionary of 3D flow-field variables on the x faces.
    states_fy: A dictionary of 3D flow-field variables on the y faces.
    states_fz: A dictionary of 3D flow-field variables on the z faces.
    cfg: The context object that stores parameters and information required by
      the simulation.
    physics_models: An object which contains the physics models used by the
      simulation.

  Returns:
    The strain rate tensor:
    \tau_{ij} = \mu S_{ij},
    where elements with subscript j are stored on the j face.
  """
  s = fluid.strain_rate(states, cfg)
  rho = (states_fx['rho'], states_fy['rho'], states_fz['rho'])
  nu = physics_models.transport_model.kinematic_viscosity(states)

  tau_fn = lambda rho_j, nu_j, s_ij: 2.0 * rho_j * nu_j * s_ij

  return [
      [tf.nest.map_structure(tau_fn, rho[j], nu, s[i][j]) for j in range(3)]
      for i in range(3)
  ]


# BEGIN GOOGLE-INTERNAL
# Note that we do not support setting a face boundary condition for diffusive
# fluxes. As such, any boundary condition must be specified through either the
# cell average values, or by overwriting the net diffusive flux. See b/310692602
# for additional details.
# END GOOGLE-INTERNAL
def single_component_heat_flux(
    primitive: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
) -> Dict[str, types.FlowFieldVar]:
  """Computes the heat flux of a single component gas.

  Args:
    primitive: A dictionary of all primitive flow field variables, including all
      variables in `cfg.primitive_variable_names`, as well as pressure `P`.
    cfg: The context object that stores parameters and information required by
      the simulation.
    physics_models: An object which contains the physics models used by the
      simulation.

  Returns:
    A dictionary of flow field variables where keys represent the
    direction of the flux and values are the flux values across the i - 1/2 face
    of the computational cell.
  """
  t = physics_models.thermodynamics_model.temperature(primitive, cfg, 'eos')
  kappa = physics_models.transport_model.thermal_conductivity(
      primitive, physics_models
  )

  flux = {}
  for dim in types.DIMS:
    delta = (cfg.dx, cfg.dy, cfg.dz)[types.DIMS.index(dim)]
    kappa_f = interpolation.linear_interpolation(kappa, dim, cfg.kernel_op)
    dtdx = gradient.backward_1(t, delta, dim, cfg.kernel_op)
    flux[dim] = tf.nest.map_structure(
        lambda kappa, dtdx: -kappa * dtdx, kappa_f, dtdx
    )
  return flux
