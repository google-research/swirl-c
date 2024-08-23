"""Library to compute the numerical flux using the HLL approximation."""

from typing import Callable, Tuple
import numpy as np
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.physics import physics_models as physics_models_lib
import tensorflow as tf


FluxFn = Callable[
    [
        types.FlowFieldMap,  # Conservative variables.
        types.FlowFieldMap,  # Primitive variables.
        types.FlowFieldMap,  # Helper variables.
        str,  # Dimension of the flux.
        str,  # The name of the conservative variable.
    ],
    types.FlowFieldVar,  # The convective flux of the conservative variable.
]


def hll_convective_flux(
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
  """Computes the intercell flux based on the Harten-Lax-van Leer approximation.

  Note that this function is naive, and does not perform any interpolation to
  the cell face. Thus, the computed intercell flux is representative of the
  i - 1/2 face of the cell ONLY IF the provided states (e.g `conservative_pos`,
  `conservative_neg`, etc.) are properly interpolated to the i - 1/2 face of the
  computational cell.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    conservative_neg: A dictionary of flow field variables interpolated onto the
      left (i - 1/2) face of the computations cell using a negative stencil. The
      dictionary must include all conservative variables listed in
      `cfg.conservative_variable_names`.
    conservative_pos: A dictionary of flow field variables interpolated onto the
      left (i - 1/2) face of the computations cell using a positive stencil. The
      dictionary must include all conservative variables listed in
      `cfg.conservative_variable_names`.
    primitive_neg: A dictionary of flow field variables interpolated onto the
      left (i - 1/2) face of the computations cell using a negative stencil. The
      dictionary must include the density `RHO`, three components of velocity
      `U`, `V`, `W`, the pressure `P`, and the total enthalpy `H`.
    primitive_pos: A dictionary of flow field variables interpolated onto the
      left (i - 1/2) face of the computations cell using a positive stencil. The
      dictionary must include the density `RHO`, three components of velocity
      `U`, `V`, `W`, the pressure `P`, and the total enthalpy `H`.
    helper_var_neg: A dictionary with keys 'x', 'y', and 'z' representing each
      of the three directions, where each item contains a dictionary of helper 
      variables interpolated onto the left (i - 1/2) face of the computational
      cell normal to the indicated dimension. The interpolation is performed
      with a negative stencil. Each dictionary in `helper_var_neg` contains all
      variables listed in `cfg.helper_variable_names`.
    helper_var_pos: A dictionary with keys 'x', 'y', and 'z' representing each
      of the three directions, where each item contains a dictionary of helper
      variables interpolated onto the left (i - 1/2) face of the computational
      cell normal to the indicated dimension. The interpolation is performed
      with a positive stencil. `helper_var_pos` requires the same
      variables as `helper_var_neg`.
    flux_fn: A function that computes the convective intercell flux across the
      left (i - 1/2) face of the computations cell using a negative stencil. The
      function must consider the convective flux of all variables listed in
      `cfg.conservative_variable_names`.
    dim: The dimension ('x', 'y', or 'z') that the HLL flux is computed along.
    cfg: The context object that stores parameters and information required by
      the simulation.
    physics_models: An object handler for physics models implemented in the
      current simulation. At minimum, a thermodynamics model is required here.

  Returns:
    A dictionary of flow field variables representing the approximated numerical
    intercell flux of the variables listing in
    `cfg.conservative_variable_names across the i - 1/2 face of the
    computational cell.
  """

  s_l, s_r = compute_hll_roe_wave_speed_estimates(
      replica_id,
      replicas,
      primitive_neg,
      primitive_pos,
      dim,
      cfg,
      physics_models,
  )

  ds = tf.nest.map_structure(tf.math.subtract, s_r, s_l)
  t1 = tf.nest.map_structure(
      lambda s_r, s_l, ds: tf.math.divide_no_nan(
          tf.minimum(s_r, 0.0) - tf.minimum(s_l, 0.0), ds
      ),
      s_r,
      s_l,
      ds,
  )
  t2 = tf.nest.map_structure(
      lambda t1: tf.math.subtract(tf.ones_like(t1), t1), t1
  )
  t3 = tf.nest.map_structure(
      lambda s_r, s_l, ds: tf.math.divide_no_nan(
          s_r * tf.abs(s_l) - s_l * tf.abs(s_r), 2.0 * ds
      ),
      s_r,
      s_l,
      ds,
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

  flux = {}
  for var_name in cfg.conservative_variable_names:
    flux[var_name] = tf.nest.map_structure(
        lambda fp, fn, sp, sn, t1, t2, t3: t1 * fp + t2 * fn - t3 * (sp - sn),
        flux_pos[var_name],
        flux_neg[var_name],
        conservative_pos[var_name],
        conservative_neg[var_name],
        t1,
        t2,
        t3,
    )
  # Note that the flux calculation here can be very sensitive to floating point
  # error, particularly `t3 * (sp - sn)`. In quiescent flow, error in
  # interpolation of `sp` and `sn` can be amplified by `t3`, which is the order
  # of the speed of sound in the medium. See b/302341111 for more details.
  return flux


def compute_hll_roe_wave_speed_estimates(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    primitive_neg: types.FlowFieldMap,
    primitive_pos: types.FlowFieldMap,
    dim: str,
    cfg: parameter.SwirlCParameters,
    physics_models: physics_models_lib.PhysicsModels,
) -> Tuple[types.FlowFieldVar, types.FlowFieldVar]:
  """Computes the approximate left and right characteristic wave speeds.

  Note that this function is naive, and does not perform any interpolation to
  the cell face. Thus, the computed wave speed is representative of the i - 1/2
  face of the cell ONLY IF the provided states `primitive_pos` and
  `primitive_neg` are properly interpolated to the i - 1/2 face of the
  computational cell.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`
    primitive_neg: A dictionary of flow field variables interpolated onto the
      left (i - 1/2) face of the computations cell using a negavie stencil. The
      dictionary must include the density `RHO`, three components of velocity
      `U`, `V`, `W`, the pressure `P`, and the total enthalpy `H`.
    primitive_pos: A dictionary of flow field variables interpolated onto the
      left (i - 1/2) face of the computations cell using a positive stencil. The
      dictionary must include the density `RHO`, three components of velocity
      `U`, `V`, `W`, the pressure `P`, and the total enthalpy `H`.
    dim: The dimension ('x', 'y', or 'z') that the approximated characteristic
      waves are traveling along.
    cfg: The context object that stores parameters and information required by
      the simulation.
    physics_models: An object handler for physics models implemented in the
      current simulation. At minimum, a thermodynamics model is required here.

  Returns:
    A tuple of flow field variables `(s_l, s_r)` where `s_l` is the approximated
    speed of the left traveling characteristic wave, and `s_r` is the
    approximate speed of the right traveling characteristic wave. Both waves are
    approximated at the i - 1/2 face of the computational cell.
  """
  # Setup: Get the name of the velocity component that is in the same direction
  # as the `dim` of the flux being computed.
  aligned_velocity_name = types.VELOCITY[types.DIMS.index(dim)]

  # Step 1: Compute the required thermodynamic quantities.
  c_neg = physics_models.thermodynamics_model.sound_speed(
      replica_id, replicas, primitive_neg, cfg, opt='p_rho'
  )
  c_pos = physics_models.thermodynamics_model.sound_speed(
      replica_id, replicas, primitive_pos, cfg, opt='p_rho'
  )

  # Step 2: Compute the Roe averaged primitive quantities.
  rho_neg_s = tf.nest.map_structure(tf.math.sqrt, primitive_neg[types.RHO])
  rho_pos_s = tf.nest.map_structure(tf.math.sqrt, primitive_pos[types.RHO])
  inv_rho_sum = tf.nest.map_structure(
      lambda rho_neg_s, rho_pos_s: tf.math.reciprocal_no_nan(
          rho_neg_s + rho_pos_s
      ),
      rho_neg_s,
      rho_pos_s,
  )

  def roe_avg_fn(f_neg, f_pos):
    """Computes the Roe averaged state."""
    return tf.nest.map_structure(
        lambda rho_neg_s, rho_pos_s, f_neg, f_pos, inv_rho_sum: (
            rho_neg_s * f_neg + rho_pos_s * f_pos
        )
        * inv_rho_sum,
        rho_neg_s,
        rho_pos_s,
        f_neg,
        f_pos,
        inv_rho_sum,
    )

  roe_avg_states = {}
  for var_name in list(types.VELOCITY) + [types.H]:
    roe_avg_states.update(
        {var_name: roe_avg_fn(primitive_neg[var_name], primitive_pos[var_name])}
    )

  roe_avg_c = physics_models.thermodynamics_model.sound_speed(
      replica_id, replicas, roe_avg_states, cfg, opt=types.H
  )

  # Step 3: Compute the wave speed estimates.
  s_l = tf.nest.map_structure(
      lambda u_neg, c_neg, u, c: tf.minimum(u_neg - c_neg, u - c),
      primitive_neg[aligned_velocity_name],
      c_neg,
      roe_avg_states[aligned_velocity_name],
      roe_avg_c,
  )
  s_r = tf.nest.map_structure(
      lambda u_pos, c_pos, u, c: tf.maximum(u_pos + c_pos, u + c),
      primitive_pos[aligned_velocity_name],
      c_pos,
      roe_avg_states[aligned_velocity_name],
      roe_avg_c,
  )

  return s_l, s_r
