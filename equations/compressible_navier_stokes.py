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
"""A library for computing the RHS of the compressible Navier-Stokes eqn."""

import enum
import functools
import logging
from typing import Dict, Optional, Tuple

import numpy as np
from swirl_c.boundary import bc_types
from swirl_c.boundary import boundary
from swirl_c.common import types
from swirl_c.common import utils
from swirl_c.core import parameter
from swirl_c.numerics import gradient
from swirl_c.numerics import interpolation
from swirl_c.numerics.riemann_solver import riemann_solver
from swirl_c.numerics.time_integration import rhs_type
from swirl_c.physics import diffusion
from swirl_c.physics import physics_models as physics_models_lib
from swirl_lm.numerics import interpolation as swirl_lm_interpolation
import tensorflow as tf

_CELL_FACE_BC = bc_types.BoundaryDictionaryType.CELL_FACES.value


class InterpolationSchemes(enum.Enum):
  """Defines the available face interpolation schemes."""
  WENO_3 = 'WENO_3'
  WENO_5 = 'WENO_5'
  MUSCL = 'MUSCL'


def get_rhs_fn(
    replicas: np.ndarray,
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
) -> rhs_type.RHS:
  """Creates and returns a new tf.function that wraps the rhs function.

  WARNING: DON'T CALL THIS FUNCTION INSIDE A LOOP, WHICH WILL CAUSE RETRACING.

  Args:
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    cfg: The context object that stores parameters and information required by
      the simulation.
    physics_models: An object handler for physics models implemented in the
      current simulation.

  Returns:
    A function that takes all state variables in conservative form, and returns
    the right-hand side function of the compressible Navier-Stokes equation.
  """

  @tf.function()
  def rhs(
      replica_id: tf.Tensor,
      conservative: types.FlowFieldMap,
      helper_vars: types.FlowFieldMap,
  ) -> types.FlowFieldMap:
    """Computes the rhs of the governing equation for conservative variables.

    Note that RHS assumes a sign convention for fluxes such that a positive flux
    along a given dimension indicates flow from left (negative) to right
    (positive) along that dimension.

    Args:
      replica_id: The ID of the computatinal subdomain (replica).
      conservative: A dictionary of flow field variables representing the
        average value of conservative variables in the computational cell. The
        dictionary must include all conservative variables listed in
        `cfg.conservative_variable_names`.
      helper_vars: Helper variables used in the simulation. These variables are
        not updated by the Navier-Stokes equations.

    Returns:
      A dictionary of flow field variables representing the total volumetric
      time
      rate of change of the conserved variables for each computational cell.
    """
    logging.info(
        'Entering the right-hand side computation for the compressible'
        ' Navier-Stokes equations.'
    )

    # Step 1: Update cell average values based on boundary conditions and halo
    # exchange for ghost cells.
    conservative = boundary.update_conservative_cell_averages(
        replica_id, replicas, conservative.copy(), cfg
    )
    logging.info('Cell average BC update completed.')

    # Step 2: Compute required primitive values and helper variables.
    primitive = utils.conservative_to_primitive_variables(conservative)
    primitive[types.P] = physics_models.thermodynamics_model.pressure(
        replica_id, replicas, primitive, cfg
    )
    primitive[types.H] = physics_models.thermodynamics_model.total_enthalpy(
        replica_id, replicas, primitive, cfg
    )
    logging.info('Primitive variable update completed.')

    helper_vars, face_interp_var_names = _update_helper_variables(
        replica_id, replicas, primitive, helper_vars, cfg
    )

    # Step 3: Interpolate cell average values to face values.
    # Note that interpolations are performed for primitive variables only.
    # Conservative variables are converted back from the interpolated primitive
    # variables for physical consistency. Here we choose to interpolate
    # primitive variables instead of conservative variables to ensure
    # consistencies with the reconstructed helper variables on faces, because
    # `helper_vars` are treated as primitive variables.
    primitive_neg, primitive_pos = _face_interpolation(primitive, cfg)
    helper_var_neg, helper_var_pos = _face_interpolation(
        helper_vars, cfg, face_interp_var_names
    )
    conservative_neg = {
        dim: utils.primitive_to_conservative_variables({
            var_name: primitive_face[var_name]
            for var_name in cfg.primitive_variable_names
        })
        for dim, primitive_face in primitive_neg.items()
    }
    conservative_pos = {
        dim: utils.primitive_to_conservative_variables({
            var_name: primitive_face[var_name]
            for var_name in cfg.primitive_variable_names
        })
        for dim, primitive_face in primitive_pos.items()
    }
    logging.info('Face interpolation completed.')

    # Step 4: Update face values based on boundary condtions.
    conservative_neg, conservative_pos, primitive_neg, primitive_pos = (
        boundary.update_faces(
            replica_id,
            replicas,
            conservative_neg,
            conservative_pos,
            primitive_neg,
            primitive_pos,
            cfg,
        )
    )
    logging.info('Face BC update completed.')

    # Step 5: Compute convective fluxes.
    flux_conv = _intercell_convective_flux(
        replica_id,
        replicas,
        conservative_neg,
        conservative_pos,
        primitive_neg,
        primitive_pos,
        helper_var_neg,
        helper_var_pos,
        cfg,
        physics_models,
    )
    flux_conv = boundary.update_fluxes(
        replica_id,
        replicas,
        flux_conv,
        cfg,
        bc_types.BoundaryFluxType.CONVECTIVE,
    )
    logging.info('Convective flux computation completed.')

    # Step 6: Compute diffusive fluxes.
    if cfg.include_diffusion:
      # BEGIN GOOGLE-INTERNAL
      # TODO(b/310692602): Currently, we do not support a face boundary
      # condition in the diffusive flux calculation. Only cell average BCs are
      # considered.
      # END GOOGLE-INTERNAL
      if _CELL_FACE_BC in cfg.bc:
        raise NotImplementedError(
            'Face boundary conditions are not supported with diffusion'
            ' enabled at'
            ' this time.'
            # BEGIN GOOGLE-INTERNAL
            ' See b/310692602 for more details.'
            # END GOOGLE-INTERNAL
        )
      flux_diff = _intercell_diffusive_flux(primitive, cfg, physics_models)
      flux_diff = boundary.update_fluxes(
          replica_id,
          replicas,
          flux_diff,
          cfg,
          bc_types.BoundaryFluxType.DIFFUSIVE,
      )
      logging.info('Diffusive flux computation completed.')
    else:
      flux_diff = {
          dim: {
              var_name: tf.nest.map_structure(tf.zeros_like, val)
              for var_name, val in flux_conv[dim].items()
          }
          for dim in types.DIMS
      }
      logging.info('Diffusive flux placeholder initialization completed.')

    # Step 7: Compute total flux.
    flux = {
        dim: {
            var_name: tf.nest.map_structure(
                tf.add, val, flux_diff[dim][var_name]
            )
            for var_name, val in flux_conv[dim].items()
        }
        for dim in types.DIMS
    }
    flux = boundary.update_fluxes(
        replica_id,
        replicas,
        flux,
        cfg,
        bc_types.BoundaryFluxType.TOTAL,
    )
    logging.info('Total flux computation completed.')

    # Step 8: Compute source terms from models.
    source = physics_models.source_function(
        replica_id, replicas, conservative, helper_vars
    )

    # Step 9: Update source terms with flux gradient.
    # Regarding subtraction of the gradient, consider the typical control volume
    # budget equation, df/dt  = sum(f''_in - f''_out) * (A / V) + f'''_src.
    # Here, df/dt is the time rate change of f per unit volume, f''_in and
    # f''_out are the total fluxes of f in and out of the control volume along a
    # given dimension, A is the area of the face the flux is crossing, V is the
    # volume of the control volume, and f'''_src is the additional volumetric
    # sources of f. Note that (A / V) is equal to the grid size (e.g. dx) along
    # a dimension. The gradient calculated by `forward_1` is equivalent to,
    # (f''_out - f''_in)* (A / V), which is the negative of the flux difference
    # in the control volume budget above. Therefore, we add the negative (i.e.
    # subtract) of the gradient along each dimension to the source term to get
    # the RHS (i.e. df/dt).
    delta = {'x': cfg.dx, 'y': cfg.dy, 'z': cfg.dz}
    for var_name in cfg.conservative_variable_names:
      for dim in types.DIMS:
        source[var_name] = tf.nest.map_structure(
            tf.subtract,
            source[var_name],
            gradient.forward_1(
                flux[dim][var_name], delta[dim], dim, cfg.kernel_op
            ),
        )
        logging.info(
            'Source term update for %s in dim %s completed.', var_name, dim
        )
    return source

  return rhs


def _update_helper_variables(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    primitive: types.FlowFieldMap,
    helper_var: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
) -> Tuple[types.FlowFieldMap, list[str]]:
  """Updates helper variables required by the simulation.

  Args:
    replica_id: The ID of the computational subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    primitive: A dictionary of primitive flow field which contains all variables
      in `cfg.primitive_variable_names`. This must include at minimum the
      density `RHO`, three components of velocity `U`, `V`, `W`, the enthalpy
      `H`, and the pressure `P`. Any additional transported scalars must also be
      included.
    helper_var: A dictionary of helper variables.
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    A tuple with the first argument being the updated dictionary of helper
    variables, and the second argument being a list of variable names that will
    be interpolated on to cell faces.
  """
  helper_var = helper_var.copy()
  face_interp_variables = []

  # With the presence of reference pressure ('p_ref') and density ('rho_ref'),
  # we are considering there to be a hydrostatic balance. Therefore we will use
  # the hydrodynamic pressure p_dyn = p - p_ref when computing the convective
  # fluxes for the momentum and energy equations. The buoyancy term will be
  # computed with the density difference from the hydrostatic state, which is
  # d_rho = rho - rho_ref.
  if 'p_ref' in helper_var and 'rho_ref' in helper_var:
    p_dyn = tf.nest.map_structure(
        tf.math.subtract, primitive[types.P], helper_var['p_ref']
    )
    d_rho = tf.nest.map_structure(
        tf.math.subtract, primitive[types.RHO], helper_var['rho_ref']
    )
    # We assume that the fluid remains balanced hydrostatically on the
    # boundaries.
    bc = {
        dim: {
            face: (bc_types.BoundaryCondition.DIRICHLET, 0.0)
            for face in range(2)
        }
        for dim in types.DIMS
    }
    helper_var.update({
        'p_dyn': boundary.exchange_halos(
            replica_id, replicas, p_dyn, bc, cfg.halo_width
        ),
        'd_rho': boundary.exchange_halos(
            replica_id, replicas, d_rho, bc, cfg.halo_width
        ),
    })
    face_interp_variables.append('p_dyn')

  return helper_var, face_interp_variables


def _compute_physical_flux_1d(
    conservative: types.FlowFieldMap,
    primitive: types.FlowFieldMap,
    helper_var: types.FlowFieldMap,
    dim: str,
    conservative_var_name: str,
) -> types.FlowFieldVar:
  """Computes the physical convective flux along `dim`.

  Here, the physical convective flux is computed as the product of the mass flow
  rate along `dim` (`RHO_U` for `dim = 'x'`, `RHO_V` for `dim = 'y'`, or `RHO_W`
  for `dim = 'z'`), and the primitive quantity being transported. When the
  primitive quantity `U`, `V`, or `W` is aligned with `dim` (i.e. `dim = 'x'`
  and `U`), the flux also includes the pressure contribution. For the energy
  equation, pressure work is included through the definition of enthalpy. The
  sign of the returned flux is positive if the flow is from left (negative) to
  right (positive).

  Args:
    conservative: A dictionary of conservative flow field variables. It is only
      required to contain the momentum term aligned with `dim` (e.g. `RHO_U` for
      `dim = 'x'`, `RHO_V` for `dim = 'y'`, or `RHO_W` for `dim = 'z'`). Other
      terms are ignored.
    primitive: A dictionary of primitive flow field which contains all variables
      in `cfg.primitive_variable_names`. This must include at minimum the
      density `RHO`, three components of velocity `U`, `V`, `W`, the enthalpy
      `H`, and the pressure `P`. Any additional transported scalars must also be
      included.
    helper_var: A dictionary of helper variables.
    dim: An string indicating the dimension to compute the flux along, either
      'x', 'y', or 'z'.
    conservative_var_name: The name of the conservative variable for which the
      flux is computed. It should be included in
      `cfg.conservative_variable_names`.

  Returns:
    A dictionary of flow field variables `flux`, containing fluxes of all
    variables listed in `cfg.primitive_variable_names`.
  """
  # Prepares states updated by helper variables.
  p_updated = helper_var.get('p_dyn', primitive[types.P])

  var_name = types.conservative_to_primitive_name(conservative_var_name)

  # Setup: Get the name of the velocity component that is in the same direction
  # as the `dim` of the flux being computed and the associated consevative
  # variable name.
  aligned_velocity_name = types.VELOCITY[types.DIMS.index(dim)]
  aligned_momentum_name = types.primitive_to_conservative_name(
      aligned_velocity_name
  )

  if var_name == aligned_velocity_name:
    # First special case to consider: Momentum flux along `dim`.
    # The aligned momentum term includes the pressure.
    flux_fn = lambda m_flux, u, p: m_flux * u + p
    return tf.nest.map_structure(
        flux_fn,
        conservative[aligned_momentum_name],
        primitive[aligned_velocity_name],
        p_updated,
    )
  elif var_name == types.E:
    # Second special case to consider: Energy flux.
    # Here, we use total enthalpy rather than total energy, thus including the
    # pressure work on `E`. Note that the exact pressure `p` instead of the
    # perturbed-from-reference pressure `p_updated` is used here. If `p_updated`
    # is used, additional terms are introduced based on the derivation from the
    # updated momentum equation, which makes the implementation difficult to
    # maintain.
    return tf.nest.map_structure(
        lambda rho_u, rho, e, p: rho_u * (e + p / rho),
        conservative[aligned_momentum_name],
        primitive[types.RHO],
        primitive[types.E],
        primitive[types.P],
    )
  elif var_name == types.RHO:
    # Third special case: Mass flux.
    # Here, just pass through the mass flux directly.
    return conservative[aligned_momentum_name]
  else:
    # Otherwise, standard convective flux applies.
    return (
        tf.nest.map_structure(
            tf.math.multiply,
            conservative[aligned_momentum_name],
            primitive[var_name],
        )
    )


def _intercell_convective_flux(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    conservative_neg: Dict[str, types.FlowFieldMap],
    conservative_pos: Dict[str, types.FlowFieldMap],
    primitive_neg: Dict[str, types.FlowFieldMap],
    primitive_pos: Dict[str, types.FlowFieldMap],
    helper_var_neg: Dict[str, types.FlowFieldMap],
    helper_var_pos: Dict[str, types.FlowFieldMap],
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
) -> Dict[str, types.FlowFieldMap]:
  """Computes the total intercell flux across the i - 1/2 face of the cell.

  The total intercell flux across the i - 1/2 face of the computation cell is
  computed from the interpolated face values provided. The numeric flux is
  approximated using the scheme specified in `cfg.numeric_flux_scheme`. The
  sign of the returned flux for a given `dim` is positive if the flow is from
  left (negative) to right (positive) along the given dimension.

  Note that this function assumes all provided values (physical domain and halo
  cells) are physical and correct, and computes the flux from them accordingly.
  Boundary conditions are not considered here, and the returned flux must be
  corrected to consider boundary conditions.

  Args:
    replica_id: The ID of the computational subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    conservative_neg: A dictionary with keys 'x', 'y', and 'z' representing each
      of the three directions, where each item contains a dictionary of
      conservative flow field variables interpolated onto the left (i - 1/2)
      face of the computational cell normal to the indicated dimension. The
      interpolation is performed with a negative stencil. All variables in
      `cfg.conservative_variable_names` are required, which at minimum will
      contain the density `RHO`, three components of momentum `RHO_U` `RHO_V`
      and `RHO_W`, the total energy `RHO_E`, and any additional transported
      scalars.
    conservative_pos: A dictionary with keys 'x', 'y', and 'z' representing each
      of the three directions, where each item contains a dictionary of
      conservative flow field variables interpolated onto the left (i - 1/2)
      face of the computational cell normal to the indicated dimension. The
      interpolation is performed with a positive stencil. `conservative_pos`
      requires the same variables as `conservative_neg`
    primitive_neg: A dictionary with keys 'x', 'y', and 'z' representing each of
      the three directions, where each item contains a dictionary of primitive
      flow field variables interpolated onto the left (i - 1/2) face of the
      computational cell normal to the indicated dimension. The interpolation is
      performed with a negative stencil. In additional to the variables listed
      in `cfg.primitive_variable_names`, both the pressure `P` and total
      enthalpy `H` are required. `cfg.primitive_variable_names` must include at
      minimum the density `RHO`, three components of velocity `U`, `V`, `W`, the
      total energy `E`, and any additional transported scalars.
    primitive_pos: A dictionary with keys 'x', 'y', and 'z' representing each of
      the three directions, where each item contains a dictionary of primitive
      flow field variables interpolated onto the left (i - 1/2) face of the
      computational cell normal to the indicated dimension. The interpolation is
      performed with a positive stencil. `primitive_pos` requires the same
      variables asl `primitive_neg`.
    helper_var_neg: A dictionary with keys 'x', 'y', and 'z' representing each
      of the three directions, where each item contains a dictionary of helper
      variables interpolated onto the left (i - 1/2) face of the computational
      cell normal to the indicated dimension. The interpolation is performed
      with a negative stencil. In additional to the variables listed in
      `cfg.helper_variable_names`.
    helper_var_pos: A dictionary with keys 'x', 'y', and 'z' representing each
      of the three directions, where each item contains a dictionary of helper
      variables interpolated onto the left (i - 1/2) face of the computational
      cell normal to the indicated dimension. The interpolation is performed
      with a positive stencil. `helper_var_pos` requires the same
      variables asl `helper_var_neg`.
    cfg: The context object that stores parameters and information required by
      the simulation.
    physics_models: An object handler for physics models implemented in the
      current simulation. At minimum, a thermodynamics model is required here

  Returns:
    A dictionary of dictionaries of flow field variables where each variable
    represents the total convective intercell flux of the conservative variables
    across the i - 1/2 face of the computational cell in the indicated
    dimension.
  """
  numeric_flux_fn = riemann_solver.select_numeric_flux_fn(cfg)

  # Compute the convective flux.
  # Note that depending on the Riemann scheme, the computed flux can be very
  # sensitive to floating point error. See b/302341111 for more details.
  flux = {}
  for dim in types.DIMS:
    flux[dim] = numeric_flux_fn(
        replica_id,
        replicas,
        conservative_neg[dim],
        conservative_pos[dim],
        primitive_neg[dim],
        primitive_pos[dim],
        helper_var_neg[dim],
        helper_var_pos[dim],
        _compute_physical_flux_1d,
        dim,
        cfg,
        physics_models,
    )
  return flux


def _face_interpolation(
    cell_avg: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
    interp_filter: Optional[list[str]] = None,
) -> Tuple[
    Dict[str, types.FlowFieldMap],
    Dict[str, types.FlowFieldMap],
]:
  """Computes variables on the i - 1/2 faces.

  This function interpolates the cell average variables onto the three i - 1/2
  faces of each computation cell using the interpolation scheme specified in
  `cfg.interpolation_scheme`. All variables provided are interpolated onto the
  three i - 1/2 faces. This interpolation returns two values for each face,
  one computed with a positive stencil and a second computed using a negative
  stencil.

  Note that this function assumes all provided values (physical domain and halo
  cells) are physical and correct, and computes the interpolation as such.
  Boundary conditions are not considered here, and the returned face value must
  be corrected to consider boundary conditions.

  Args:
    cell_avg: A dictionary of flow field variables representing the average
      value of variables in the computational cell.
    cfg: The context object that stores parameters and information required by
      the simulation.
    interp_filter: A list of variable names that need to be interpolated onto
      the faces. All variables are interpolated if not provided. Note that an
      empty list means that no variables will be interpolated.

  Returns:
    A tuple `(face_neg, face_pos)` where each item is a dictionary. Each
    dictionary has three keys 'x', 'y', 'z', representing the dimension of the
    interpolation. The value of each key is a dictionary of flow field variables
    interpolated onto the i - 1/2 face normal to the specified interpolation
    dimension. The suffix `_pos` and `_neg` indicate if a positive or negative
    stencil is used to interpolate the values. Example: `face_pos['x']['rho_v']`
    is the y-momentum interpolated onto the i - 1/2 face normal to 'x' using a
    positive stencil.

  Raises:
    NotImplementedError if the specified interpolation scheme is not
    implemented.
  """
  if cfg.interpolation_scheme == InterpolationSchemes.WENO_3.value:
    face_interp_fn = functools.partial(swirl_lm_interpolation.weno, k=2)
  elif cfg.interpolation_scheme == InterpolationSchemes.WENO_5.value:
    face_interp_fn = functools.partial(swirl_lm_interpolation.weno, k=3)
  elif cfg.interpolation_scheme == InterpolationSchemes.MUSCL.value:
    face_interp_fn = functools.partial(
        swirl_lm_interpolation.flux_limiter,
        limiter_type=swirl_lm_interpolation.FluxLimiterType.MUSCL,
    )
  else:
    raise NotImplementedError(
        f'"{cfg.interpolation_scheme}" is not implemented as an'
        ' "interpolation_scheme". Valid options are: '
        + str([scheme.value for scheme in InterpolationSchemes])
    )

  face_interp = {
      var_name: {dim: face_interp_fn(val, dim) for dim in types.DIMS}
      for var_name, val in cell_avg.items()
      if interp_filter is None or var_name in interp_filter
  }
  face_neg = {
      dim: {
          var_name: val[dim][0] for var_name, val in face_interp.items()
      }
      for dim in types.DIMS
  }
  face_pos = {
      dim: {
          var_name: val[dim][1] for var_name, val in face_interp.items()
      }
      for dim in types.DIMS
  }

  return (face_neg, face_pos)


def _intercell_diffusive_flux(
    primitive: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
) -> Dict[str, types.FlowFieldMap]:
  """Computes the intercell diffusive fluxes across the i - 1/2 cell face.

  The sign of the returned flux for a particular dimension (as indicated by the
  key of the returned dictionary) is positive if the flow is from left
  (negative) to right (positive) along the indicated dimension. The flux is
  negative if the flow is from right (positive) to left (negative) along the
  indicated dimension.

  Note that we do not support setting a face boundary condition for diffusive
  fluxes. As such, any boundary condition must be specified through either the
  cell average values, or by overwriting the net diffusive flux. 
  # BEGIN GOOGLE-INTERNAL
  See b/310692602 for additional details.
  # END GOOGLE-INTERNAL

  Args:
    primitive: A dictionary of all primitive flow field variables, i.e.
      all variables listed in `cfg.primitive_variable_names`.
    cfg: The context object that stores parameters and information required by
      the simulation.
    physics_models: An object which contains the physics models used by the
      simulation.

  Returns:
    A dictionary of dictionaries of flow field variables where each variable
    represents the total convective intercell flux of the conservative variables
    across the i - 1/2 face of the computational cell in the indicated
    dimension.
  """
  primitive_f = {
      dim: {
          var_name: interpolation.linear_interpolation(
              primitive[var_name], dim, cfg.kernel_op
          )
          for var_name in list(types.VELOCITY) + [types.RHO]
      }
      for dim in types.DIMS
  }
  tau = diffusion.shear_stress(
      primitive,
      primitive_f['x'],
      primitive_f['y'],
      primitive_f['z'],
      cfg,
      physics_models,
  )
  q = diffusion.single_component_heat_flux(primitive, cfg, physics_models)

  fluxes = {
      dim: {
          var_name: tf.nest.map_structure(tf.zeros_like, primitive['rho'])
          for var_name in cfg.conservative_variable_names
      }
      for dim in types.DIMS
  }

  # Add viscous stress to momentum.
  for j, dim in enumerate(types.DIMS):
    for i in range(3):
      fluxes[dim][types.MOMENTUM[i]] = tf.nest.map_structure(
          tf.negative, tau[i][j]
      )

  # Add viscous dissipation to energy.
  for j, dim in enumerate(types.DIMS):
    vel_buf = [primitive_f[dim][var_name] for var_name in types.VELOCITY]
    tau_buf = [tau[i][j] for i in range(3)]
    dot_prod = tf.nest.map_structure(
        lambda u, v, w, tx, ty, tz: u * tx + v * ty + w * tz,
        vel_buf[0],
        vel_buf[1],
        vel_buf[2],
        tau_buf[0],
        tau_buf[1],
        tau_buf[2],
    )
    fluxes[dim][types.RHO_E] = tf.nest.map_structure(tf.negative, dot_prod)

  # Add heat flux to energy flux.
  for dim in types.DIMS:
    fluxes[dim][types.RHO_E] = tf.nest.map_structure(
        tf.add, q[dim], fluxes[dim][types.RHO_E]
    )

  return fluxes
