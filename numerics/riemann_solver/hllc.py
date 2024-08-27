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
"""Library to compute the numerical flux using the HLLC approximation.

This library is implemented following the reference:
[1] Toro, E. F. (2013). Riemann solvers and numerical methods for fluid
dynamics: a practical introduction. Springer Science & Business Media.
"""

from typing import Tuple

import numpy as np
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.numerics.riemann_solver import hll
from swirl_c.physics import physics_models as physics_models_lib
import tensorflow as tf

FluxFn = hll.FluxFn


def hllc_convective_flux(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    conservative_neg: types.FlowFieldMap,
    conservative_pos: types.FlowFieldMap,
    primitive_neg: types.FlowFieldMap,
    primitive_pos: types.FlowFieldMap,
    helper_var_neg: types.FlowFieldMap,
    helper_var_pos: types.FlowFieldMap,
    flux_fn: FluxFn,
    dim: str,
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
) -> types.FlowFieldMap:
  """Computes the flux based on the Harten-Lax-van Leer-Contact approximation.

  The implementation follows reference [1].

  Note that this function is naive, and does not perform any interpolation to
  the cell face. Thus, the computed intercell flux is representative of the
  i - 1/2 face of the cell ONLY IF the provided states (e.g `conservative_pos`,
  `conservative_neg`, etc.) are properly interpolated to the i - 1/2 face of the
  computational cell. Values on the i - 1/2 face are stored at index i.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    conservative_neg: A dictionary of flow field variables interpolated onto the
      left (i - 1/2) face of the computational cell using a negative stencil.
      The dictionary must include all conservative variables listed in
      `cfg.conservative_variable_names`.
    conservative_pos: A dictionary of flow field variables interpolated onto the
      left (i - 1/2) face of the computational cell using a positive stencil.
      The dictionary must include all conservative variables listed in
      `cfg.conservative_variable_names`.
    primitive_neg: A dictionary of flow field variables interpolated onto the
      left (i - 1/2) face of the computational cell using a negative stencil.
      The dictionary must include the density `RHO`, three components of
      velocity `U`, `V`, `W`, the pressure `P`, and the total enthalpy `H`.
    primitive_pos: A dictionary of flow field variables interpolated onto the
      left (i - 1/2) face of the computational cell using a positive stencil.
      The dictionary must include the density `RHO`, three components of
      velocity `U`, `V`, `W`, the pressure `P`, and the total enthalpy `H`.
    helper_var_neg: A dictionary with keys 'x', 'y', and 'z' representing each
      of the three directions, where each item contains a dictionary of helper
      variables interpolated onto the left (i - 1/2) face of the computational
      cell normal to the indicated dimension. The interpolation is performed
      with a negative stencil. Each dictionary contains all variables listed in
      `cfg.helper_variable_names`.
    helper_var_pos: A dictionary with keys 'x', 'y', and 'z' representing each
      of the three directions, where each item contains a dictionary of helper
      variables interpolated onto the left (i - 1/2) face of the computational
      cell normal to the indicated dimension. The interpolation is performed
      with a positive stencil. `helper_var_pos` requires the same variables as
      `helper_var_neg`.
    flux_fn: A function that computes the convective intercell flux across the
      left (i - 1/2) face of the computational cell using a negative stencil.
      The function must consider the convective flux of all variables listed in
      `cfg.conservative_variable_names`.
    dim: The dimension ('x', 'y', or 'z') that the HLL flux is computed along.
    cfg: The context object that stores parameters and information required by
      the simulation.
    physics_models: An object handler for physics models implemented in the
      current simulation. At minimum, a thermodynamics model is required here.

  Returns:
    A dictionary of flow field variables representing the approximated numerical
    intercell flux of the variables listed in
    `cfg.conservative_variable_names` across the i - 1/2 face of the
    computational cell.
  """
  # Update the pressure in primitive variables by using the hydrodynamic
  # pressure if it is provided as a helper variable. When we do this, the
  # updated pressure should be used to compute the sound speeds and energy flux
  # in the intermediate states to ensure consistency with the physical flux.
  def update_primitives_with_helper_vars(primitive, helper_var):
    primitive_updated = {var_name: val for var_name, val in primitive.items()}
    if 'p_dyn' in helper_var:
      primitive_updated[types.P] = helper_var['p_dyn']
    return primitive_updated

  primitive_neg_updated = update_primitives_with_helper_vars(
      primitive_neg, helper_var_neg
  )
  primitive_pos_updated = update_primitives_with_helper_vars(
      primitive_pos, helper_var_pos
  )

  s_l, s_m, s_r = _compute_wave_speeds(
      replica_id,
      replicas,
      primitive_neg_updated,
      primitive_pos_updated,
      dim,
      cfg,
      physics_models,
  )

  aligned_velocity_name = types.VELOCITY[types.DIMS.index(dim)]

  def intermediate_fluxes(primitive_k, s_k, flux_k, conservative_var_name):
    """Computes the intermediate fluxes (k = L, R) following eq (10.38) [1]."""
    # Note that eq (10.39) is substituted into eq (10.38) and rearranged to
    # minimize floating point error. The conservative variablees are recomputed
    # from primitive variables with eq (10.5).
    primitive_var_name = types.conservative_to_primitive_name(
        conservative_var_name
    )
    if primitive_var_name == types.RHO:
      return tf.nest.map_structure(
          lambda rho_k, v_k, s_k, s_m, f_k: (
              f_k + rho_k * (s_m - v_k) * tf.math.divide_no_nan(s_k, s_k - s_m)
          ),
          primitive_k[types.RHO],
          primitive_k[aligned_velocity_name],
          s_k,
          s_m,
          flux_k[types.RHO],
      )
    elif primitive_var_name == types.E:
      return tf.nest.map_structure(
          lambda rho_k, v_k, s_k, s_m, e_k, p_k, f_k: (
              f_k
              + rho_k
              * tf.math.divide_no_nan(s_k, s_k - s_m)
              * (s_m - v_k)
              * (e_k + p_k / rho_k + (s_k - v_k) * s_m)
          ),
          primitive_k[types.RHO],
          primitive_k[aligned_velocity_name],
          s_k,
          s_m,
          primitive_k[types.E],
          primitive_k[types.P],
          flux_k[types.RHO_E],
      )
    elif primitive_var_name == aligned_velocity_name:
      return tf.nest.map_structure(
          lambda rho_k, v_k, s_k, s_m, f_k: f_k
          + rho_k
          * s_k
          * (tf.math.divide_no_nan(s_k - v_k, s_k - s_m) * s_m - v_k),
          primitive_k[types.RHO],
          primitive_k[aligned_velocity_name],
          s_k,
          s_m,
          flux_k[conservative_var_name],
      )
    else:
      return tf.nest.map_structure(
          lambda rho_k, v_k, s_k, s_m, phi_k, f_k: f_k
          + rho_k * (s_m - v_k) * phi_k * tf.math.divide_no_nan(s_k, s_k - s_m),
          primitive_k[types.RHO],
          primitive_k[aligned_velocity_name],
          s_k,
          s_m,
          primitive_k[primitive_var_name],
          flux_k[conservative_var_name],
      )

  flux_neg = {
      var_name: flux_fn(
          conservative_neg,
          primitive_neg,
          helper_var_neg,
          dim,
          var_name,
      )
      for var_name in cfg.conservative_variable_names
  }
  flux_pos = {
      var_name: flux_fn(
          conservative_pos,
          primitive_pos,
          helper_var_pos,
          dim,
          var_name,
      )
      for var_name in cfg.conservative_variable_names
  }
  flux_m_neg = {
      var_name: intermediate_fluxes(
          primitive_neg_updated, s_l, flux_neg, var_name
      )
      for var_name in cfg.conservative_variable_names
  }
  flux_m_pos = {
      var_name: intermediate_fluxes(
          primitive_pos_updated, s_r, flux_pos, var_name
      )
      for var_name in cfg.conservative_variable_names
  }

  def hllc_flux_fn(f_l, f_r, f_ml, f_mr, s_l, s_m, s_r):
    """Computes the HLLC numerical flux following eq (10.26)."""
    return tf.where(
        tf.greater_equal(s_l, 0.0),
        f_l,
        tf.where(
            tf.logical_and(tf.less_equal(s_l, 0.0), tf.greater_equal(s_m, 0.0)),
            f_ml,
            tf.where(
                tf.logical_and(
                    tf.less_equal(s_m, 0.0), tf.greater_equal(s_r, 0.0)
                ),
                f_mr,
                f_r,
            ),
        ),
    )

  return {
      var_name: tf.nest.map_structure(
          hllc_flux_fn,
          flux_neg[var_name],
          flux_pos[var_name],
          flux_m_neg[var_name],
          flux_m_pos[var_name],
          s_l,
          s_m,
          s_r,
      )
      for var_name in cfg.conservative_variable_names
  }


def _compute_wave_speeds(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    primitive_neg: types.FlowFieldMap,
    primitive_pos: types.FlowFieldMap,
    dim: str,
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
) -> Tuple[types.FlowFieldVar, types.FlowFieldVar, types.FlowFieldVar]:
  """Computes the acoustic and contact wave speeds.

  Note that this function is naive, and does not perform any interpolation to
  the cell face. Thus, the computed wave speed is representative of the i - 1/2
  face of the cell ONLY IF the provided states `primitive_pos` and
  `primitive_neg` are properly interpolated to the i - 1/2 face of the
  computational cell.

  Reference:
  Toro, E. F. (2013). Riemann solvers and numerical methods for fluid dynamics:
  a practical introduction. Springer Science & Business Media.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    primitive_neg: A dictionary of flow field variables interpolated onto the
      left (i - 1/2) face of the computational cell using a negative stencil.
      The dictionary must include the density `RHO`, three components of
      velocity `U`, `V`, `W`, the pressure `P`, and the total enthalpy `H`.
    primitive_pos: A dictionary of flow field variables interpolated onto the
      left (i - 1/2) face of the computational cell using a positive stencil.
      The dictionary must include the density `RHO`, three components of
      velocity `U`, `V`, `W`, the pressure `P`, and the total enthalpy `H`.
    dim: The dimension ('x', 'y', or 'z') that the approximated characteristic
      waves are traveling along.
    cfg: The context object that stores parameters and information required by
      the simulation.
    physics_models: An object handler for physics models implemented in the
      current simulation. At minimum, a thermodynamics model is required here.

  Returns:
    A tuple of flow field variables `(s_l, s_m, s_r)` where `s_l` is the
    approximate speed of the left traveling acoustic wave, `s_m` is the
    approximate speed of the contact wave, and `s_r` is the approximate speed of
    the right traveling acoustic wave. Both waves are approximated at the
    i - 1/2 face of the computational cell.
  """
  aligned_velocity_name = types.VELOCITY[types.DIMS.index(dim)]

  v_l = primitive_neg[aligned_velocity_name]
  v_r = primitive_pos[aligned_velocity_name]

  rho_l = primitive_neg[types.RHO]
  rho_r = primitive_pos[types.RHO]

  p_l = primitive_neg[types.P]
  p_r = primitive_pos[types.P]

  s_l, s_r = hll.compute_hll_roe_wave_speed_estimates(
      replica_id,
      replicas,
      primitive_neg,
      primitive_pos,
      dim,
      cfg,
      physics_models,
  )

  def s_m_fn(rho_l, v_l, s_l, p_l, rho_r, v_r, s_r, p_r):
    """Computes the contact wave speed following eq (10.37) in ref [1]."""
    # Note that `divide_no_nan` is used here just to prevent nans from division
    # by 0, which leads to unexpected behavior during conditioning (i.e. calling
    # `tf.cond`). The actual returned value when the denominator is 0 will not
    # affect the final result because the solution from this regime will not be
    # selected.
    s_m = tf.math.divide_no_nan(
        p_r - p_l + rho_l * v_l * (s_l - v_l) - rho_r * v_r * (s_r - v_r),
        rho_l * (s_l - v_l) - rho_r * (s_r - v_r),
    )
    return tf.where(
        tf.math.less(tf.math.abs(s_m), 1e-6), tf.zeros_like(s_m), s_m
    )

  s_m = tf.nest.map_structure(
      s_m_fn, rho_l, v_l, s_l, p_l, rho_r, v_r, s_r, p_r
  )

  return s_l, s_m, s_r
