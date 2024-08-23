"""A library of helper functions to be used in the simulations."""

from typing import Dict
from absl import logging
import numpy as np
from swirl_c.common import types
from swirl_c.core import parameter
import tensorflow as tf


def mesh_local_size(
    cfg: parameter.SwirlCParameters,
) -> list[int]:
  """Determines the local mesh size from the global mesh and `replicas` shape.

  Args:
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    The mesh size `[nx, ny, nz]` as a list of `int`s excluding the halo cells.
  """
  n_core = [cfg.cx, cfg.cy, cfg.cz]
  n = [cfg.x.shape[0], cfg.y.shape[0], cfg.z.shape[0]]
  # Get the number of mesh points in each replica without halo cells.
  n_local = [int(n_i / n_core_i) for n_i, n_core_i in zip(n, n_core)]
  return n_local


def mesh_local(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    cfg: parameter.SwirlCParameters,
) -> Dict[str, tf.Tensor]:
  """Determines the mesh grid local to the present replica."""
  n_local = mesh_local_size(cfg)

  # Get the topological location of the present replica.
  core_idx = tf.where(replicas == replica_id)[0]
  mesh = {
      'x': tf.cast(cfg.x, dtype=types.DTYPE),
      'y': tf.cast(cfg.y, dtype=types.DTYPE),
      'z': tf.cast(cfg.z, dtype=types.DTYPE),
  }
  halo_lo = tf.cast(
      tf.linspace(-cfg.halo_width, -1, cfg.halo_width), dtype=types.DTYPE
  )
  halo_hi = tf.cast(
      tf.linspace(1, cfg.halo_width, cfg.halo_width), dtype=types.DTYPE
  )

  mesh_full = {
      dim: tf.concat(  # pylint: disable=g-complex-comprehension
          [
              (mesh[dim][1] - mesh[dim][0]) * halo_lo,
              mesh[dim],
              mesh[dim][-1] + (mesh[dim][-1] - mesh[dim][-2]) * halo_hi,
          ],
          axis=0,
      )
      for dim in types.DIMS
  }

  indices = {
      dim: tf.cast(  # pylint: disable=g-complex-comprehension
          tf.linspace(
              core_idx[i] * n_local[i],
              (core_idx[i] + 1) * n_local[i] + 2 * cfg.halo_width - 1,
              n_local[i] + 2 * cfg.halo_width,
          ),
          dtype=tf.int32,
      )
      for i, dim in enumerate(types.DIMS)
  }

  return {dim: tf.gather(mesh_full[dim], indices[dim]) for dim in types.DIMS}


def mesh_local_expanded(
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    cfg: parameter.SwirlCParameters,
) -> Dict[str, tf.Tensor]:
  """Expands the local mesh to 3D to conform with the flow field variables."""
  mesh = mesh_local(replica_id, replicas, cfg)

  nz = mesh['z'].shape[0]

  mesh['x'] = mesh['x'][tf.newaxis, :, tf.newaxis]
  mesh['y'] = mesh['y'][tf.newaxis, tf.newaxis, :]
  mesh['z'] = mesh['z'][:, tf.newaxis, tf.newaxis]

  if not cfg.use_3d_tf_tensor:
    mesh['x'] = tf.unstack(mesh['x']) * nz
    mesh['y'] = tf.unstack(mesh['y']) * nz
    mesh['z'] = tf.unstack(mesh['z'])

  return mesh


def gravity_direction(cfg: parameter.SwirlCParameters) -> int:
  """Finds the direction of gravity.

  The function checks the gravity unit vector `cfg.g` to determine if
  gravity is present in the simulation, and if the gravity vector is aligned
  with the computational grid.

  Args:
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    An integer `g_dim` indicating the gravity present. `g_dim = -1` denotes no
    gravity present. `g_dim = 0`, `1`, or `2` indicates gravit aligned to the
    computational grid in the `x`, `y`, or `z` direction respectively.
    `g_dim = 3` indicates gravity is present but is not aligned with the
    computational grid.
  """
  g_vec = np.fromiter((cfg.g[dim] for dim in types.DIMS), dtype=float)
  magnitude_g_vec = np.linalg.norm(g_vec)
  if np.abs(magnitude_g_vec - 1.0) >= types.SMALL:
    return -1

  g_dim = 3
  for i in range(len(types.DIMS)):
    if np.abs(np.abs(cfg.g[types.DIMS[i]]) - 1.0) < types.SMALL:
      g_dim = i
      break
  return g_dim


def conservative_to_primitive_variables(
    states: types.FlowFieldMap,
) -> types.FlowFieldMap:
  """Converts conservative variables to primitive variables.

    Consider an intensive property 'b' of the flow field. We solve for the
    conservative form of the variable 'ρb'. This utility is intended to convert
    from 'ρb' back to 'b'.

    All flow field variables provided in the dictionary are divided by density
    and returned, excluding density. Density is returned without modification.
    Variables are renamed according to the defined `CONS_VAR_PREFIX`.

  Args:
    states: Dictionary of scalar flow field variables to be normalized by the
      density. The dictionary must include the density `RHO`.

  Returns:
    A dictionary of primitive scalar flow field variables, including density.
  """
  inv_rho = tf.nest.map_structure(tf.math.reciprocal, states[types.RHO])
  primitives = {types.RHO: states[types.RHO]}
  for key in states.keys():
    if key == types.RHO:
      continue
    elif types.is_conservative_name(key):
      prim_name = types.conservative_to_primitive_name(key)
      primitives[prim_name] = tf.nest.map_structure(
          tf.math.multiply, inv_rho, states[key]
      )
    else:
      logging.info(
          '%s was passed as a conservative variable but did not match'
          ' %s"var_name" format, and was skipped.',
          key,
          types.CONS_VAR_PREFIX,
      )

  return primitives


def primitive_to_conservative_variables(
    states: types.FlowFieldMap,
) -> types.FlowFieldMap:
  """Converts primitive variables to conservative variables.

    Consider an intensive property 'b' of the flow field. We solve for the
    conservative form of the variable 'ρb'. This utility is intended to convert
    from 'b' back to 'ρb'.

    All flow field variables in the dictionary are multiplied by density and
    returned, excluding density. Density is returned without modification.
    Variables are renamed according to the defined `CONS_VAR_PREFIX`.

  Args:
    states: Dictionary of scalar flow field variables to be scaled by the
      density. The dictionary must include the density `RHO`.

  Returns:
    A dictionary of conservative scalar flow field variables, including density.
  """
  conservatives = {}
  for key in states.keys():
    if key == types.RHO:
      conservatives[types.RHO] = states[types.RHO]
    else:
      cons_name = types.primitive_to_conservative_name(key)
      conservatives[cons_name] = tf.nest.map_structure(
          tf.math.multiply, states[types.RHO], states[key]
      )

  return conservatives
