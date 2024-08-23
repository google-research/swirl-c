"""A library of fluid related variables."""

from typing import List

import numpy as np
from swirl_c.boundary import bc_types
from swirl_c.boundary import boundary
from swirl_c.common import types
from swirl_c.common import utils
from swirl_c.core import parameter
from swirl_c.numerics import gradient
from swirl_c.numerics import interpolation
from swirl_c.physics import constant
from swirl_lm.utility import common_ops
import tensorflow as tf


def strain_rate(
    states: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
) -> List[List[tf.Tensor]]:
  R"""Computes the strain rate tensor (3 x 3).

  Args:
    states: A dictionary of cell-averaged/centered 3D flow-field variables.
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    The strain rate tensor:
    S_{ij} = 0.5 * (\partial_{x_j} u_i + \partial_{x_i} u_j) -
        1 / 3 \partial_{x_k} u_k \delta_{ij},
    where elements with subscript j are stored on the j face.
  """
  div_fn = lambda a, b, c: a + b + c
  s_diag_fn = lambda g, d: g - 1.0 / 3.0 * d
  s_off_diag_fn = lambda g_0, g_1: 0.5 * (g_0 + g_1)

  def grad(f, grad_dim, face_dim):
    """Computes the gradient of `f` in `grad_dim` on faces in `face_dim`."""
    h = (cfg.dx, cfg.dy, cfg.dz)[types.DIMS.index(grad_dim)]
    if grad_dim == face_dim:
      grad_f = gradient.backward_1(f, h, grad_dim, cfg.kernel_op)
    else:
      grad_f_cell = gradient.central_2(f, h, grad_dim, cfg.kernel_op)
      grad_f = interpolation.linear_interpolation(
          grad_f_cell, face_dim, cfg.kernel_op
      )

    return grad_f

  u, v, w = [states[var_name] for var_name in types.VELOCITY]

  dudx_x = grad(u, 'x', 'x')
  dudx_y = grad(u, 'x', 'y')
  dudx_z = grad(u, 'x', 'z')

  dvdy_x = grad(v, 'y', 'x')
  dvdy_y = grad(v, 'y', 'y')
  dvdy_z = grad(v, 'y', 'z')

  dwdz_x = grad(w, 'z', 'x')
  dwdz_y = grad(w, 'z', 'y')
  dwdz_z = grad(w, 'z', 'z')

  div_x = tf.nest.map_structure(div_fn, dudx_x, dvdy_x, dwdz_x)
  div_y = tf.nest.map_structure(div_fn, dudx_y, dvdy_y, dwdz_y)
  div_z = tf.nest.map_structure(div_fn, dudx_z, dvdy_z, dwdz_z)

  return [
      [
          tf.nest.map_structure(s_diag_fn, dudx_x, div_x),
          tf.nest.map_structure(
              s_off_diag_fn, grad(u, 'y', 'y'), grad(v, 'x', 'y')
          ),
          tf.nest.map_structure(
              s_off_diag_fn, grad(u, 'z', 'z'), grad(w, 'x', 'z')
          ),
      ],
      [
          tf.nest.map_structure(
              s_off_diag_fn, grad(v, 'x', 'x'), grad(u, 'y', 'x')
          ),
          tf.nest.map_structure(s_diag_fn, dvdy_y, div_y),
          tf.nest.map_structure(
              s_off_diag_fn, grad(v, 'z', 'z'), grad(w, 'y', 'z')
          ),
      ],
      [
          tf.nest.map_structure(
              s_off_diag_fn, grad(w, 'x', 'x'), grad(u, 'z', 'x')
          ),
          tf.nest.map_structure(
              s_off_diag_fn, grad(w, 'y', 'y'), grad(v, 'z', 'y')
          ),
          tf.nest.map_structure(s_diag_fn, dwdz_z, div_z),
      ],
  ]


def pressure_hydrostatic(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    states: types.FlowFieldMap,
    cfg: parameter.SwirlCParameters,
) -> types.FlowFieldVar:
  """Computes the pressure under hydrostatic conditions.

  The hydrostatic pressure is computed by integrating ∂p/dz = -gρ. The density
  is computed from the ideal gas equation of state and the potential
  potential temperature. In the absence of gravity, the hydrostatic pressure
  returned is the reference pressure `cfg.p_0`.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    replicas: A numpy array that maps a replica's grid coordinate to its
      `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`.
    states: Dictionary of scalar flow field variables which must include the
      potential temperature `POTENTIAL_T`.
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    The hydrostatic pressure.

  Raises:
    ValueError if the gravity vector is not aligned with the computational
    mesh.
  """
  dims = ('x', 'y', 'z')
  g_dim = utils.gravity_direction(cfg)
  if g_dim == 3:
    g_vec = np.fromiter((cfg.g[dim] for dim in dims), dtype=float)
    raise ValueError(
        'Hydrostatic pressure is only available when the computational mesh'
        ' is aligned with the gravity unit vector. The provided gravity'
        f' vector was [{g_vec[0]}, {g_vec[1]}, {g_vec[2]}].'
    )

  if g_dim == -1:
    return tf.nest.map_structure(
        lambda theta: cfg.p_0 * tf.ones_like(theta),
        states[types.POTENTIAL_T],
    )

  def strip_halos(f):
    """Removes ghost cells in the vertical direction."""
    vertical_halos = [0, 0, 0]
    vertical_halos[g_dim] = cfg.halo_width

    return common_ops.strip_halos(f, vertical_halos)

  def integration_fn(theta):
    """Computes the integrant for the pressure evaluation."""
    return -constant.G * constant.KAPPA / constant.R / theta

  # Performs integration to points in the interior domain only.
  integrant = tf.nest.map_structure(
      integration_fn, strip_halos(states[types.POTENTIAL_T])
  )
  delta = (cfg.dx, cfg.dy, cfg.dz)[g_dim]
  buf, _ = common_ops.integration_in_dim(
      replica_id, replicas, integrant, delta, g_dim
  )

  p_interior = tf.nest.map_structure(
      lambda b: cfg.p_0 * (1.0 + b) ** (1.0 / constant.KAPPA), buf
  )

  # Performs integration in the ghost cells.
  def get_pressure_bc(face):
    """Computes the pressure at the boundaries in the vertical direction."""
    # Because the integration is performed from the two ends of the domain
    # outwards, the integral needs to be reversed on the lower end.
    sign = -1.0 if face == 0 else 1.0

    p_0 = common_ops.get_face(p_interior, g_dim, face, 0)[0]
    p_bc = []

    for i in range(cfg.halo_width):
      theta_lim = [
          common_ops.get_face(
              states[types.POTENTIAL_T], g_dim, face, cfg.halo_width - i - j
          )[0]
          for j in range(2)
      ]
      integrant_lim = [
          tf.nest.map_structure(integration_fn, theta_i)
          for theta_i in theta_lim
      ]
      integral = tf.nest.map_structure(
          lambda a, b: 0.5 * (a + b) * delta, *integrant_lim
      )
      p_1 = tf.nest.map_structure(
          lambda p_0_i, int_i: (  # pylint: disable=g-long-lambda
              p_0_i**constant.KAPPA + sign * cfg.p_0**constant.KAPPA * int_i
          )
          ** (1.0 / constant.KAPPA),
          p_0,
          integral,
      )
      p_bc.append(p_1)
      p_0 = p_1

    # The order of the ghost cell values need to follow the coordinates.
    # Because the integration is performed outwards, the sequence needs to be
    # reversed on the lower end.
    if face == 0:
      p_bc.reverse()

    return p_bc

  # Update pressure in the ghost cells.
  vertical_paddings = [(0, 0)] * 3
  vertical_paddings[g_dim] = [
      cfg.halo_width,
  ] * 2
  p = common_ops.pad(p_interior, vertical_paddings, 0.0)

  bc = [
      [
          (bc_types.BoundaryCondition.NEUMANN, 0.0),
      ]
      * 2
  ] * 3
  bc[g_dim] = [
      (bc_types.BoundaryCondition.DIRICHLET, get_pressure_bc(i))
      for i in range(2)
  ]

  dims = ('x', 'y', 'z')
  p_bc_dict = {
      dim: {face: val for face, val in enumerate(bc[dims.index(dim)])}
      for dim in dims
  }
  cfg_p = parameter.SwirlCParameters(
      {'halo_width': cfg.halo_width, 'bc': {'p': p_bc_dict}}
  )
  return boundary.update_boundary(replica_id, replicas, {'p': p}, cfg_p)['p']
