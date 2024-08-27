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
"""Library for initializing the variables on TPU cores."""

from typing import Callable, Dict, Optional, Text, Tuple, Union

import numpy as np
from swirl_c.common import types
from swirl_c.core import parameter
import tensorflow as tf

TensorOrArray = Union[tf.Tensor, np.ndarray]
ThreeIntTuple = Union[np.ndarray, tf.Tensor, Tuple[int, int, int]]
ValueFunction = Callable[
    [
        TensorOrArray,
        TensorOrArray,
        TensorOrArray,
        float,
        float,
        float,
        ThreeIntTuple,
    ],
    tf.Tensor,
]

_DEFAULT_PERMUTATION = (2, 0, 1)


def default_initializer(replica_id, coordinates, cfg) -> Dict[str, tf.Tensor]:
  """Generates the default initial field for all conservative variables.

  This initializer method is intended to only be used when restarting from a
  checkpoint or a save file, where the states for each subdomain are overwritten
  by the save file. Each local mesh is filled with 0 for all conservative
  variables listed in `cfg.conservative_variable_names`.

  Args:
    replica_id: The ID of the computatinal subdomain (replica).
    coordinates: A vector/sequence of integer with length 3 representing the
      logical coordinate of the core in the logical mesh [x, y, z].
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    A dictionary of flow field variables containing all variables listed in
    `cfg.conservative_variable_names`. Each field is set to 0 everywhere.
  """
  del replica_id

  def constant_field(val):
    return partial_field_for_core(
        cfg, coordinates, constant_initial_state_fn(val)
    )

  return {
      var_name: constant_field(0.0)
      for var_name in cfg.conservative_variable_names
  }


def partial_field_for_core(
    cfg: parameter.SwirlCParameters,
    coordinate: ThreeIntTuple,
    value_fn: ValueFunction,
    perm: Optional[ThreeIntTuple] = _DEFAULT_PERMUTATION,
    pad_mode: Optional[Text] = 'SYMMETRIC',
) -> tf.Tensor:
  """Generates a specified flow field variable on the local core mesh.

  The full grid spec is provided by `cfg.x`, `cfg.y` or `cfg.z`.
  The value function `value_fn` takes a 3-D mesh grid and corresponding lengths
  in three different dimensions as arguments. The value function then maps the
  given flow field variable onto the interior of the local mesh, excluding halo
  cells. The coordinates of the local mesh are determined by the processor
  topology specified in `cfg` and the processor coordinate `coordinate`.

  NB: `perm` and `pad_mode` have defaults if the parameters are not provided.
  This is in contrast to passing the value `None`, which means, `do not
  transpose` and `do not pad`, respectively.

  # BEGIN GOOGLE-INTERNAL
  Note that this method replaces the Swirl-LM method
  `initializer.partial_mesh_for_core`. This is necessary because Swirl-C and
  Swirl-LM use different mesh definitions in the parameter class. Resolution of
  b/309863073 will render this method redundant.
  # END GOOGLE-INTERNAL

  Args:
    cfg: The context object that stores parameters and information required by
      the simulation.
    coordinate: A vector/sequence of integer with length 3 representing the
      logical coordinate of the core in the logical mesh [x, y, z].
    value_fn: A function that takes the local mesh_grid tensor for the core (in
      order x, y, z), the global characteristic length floats (in order x, y, z)
      and the local core coordinate, and returns a 3-D tensor representing the
      value for the local core (without including the margin/overlap between the
      cores).
    perm: A 3-tuple that defines the permutation ordering for the returned
      tensor. The default is (2, 0, 1). If `None`, permutation is not applied.
    pad_mode: Defines the padding applied the returned tensor. Must be
      'CONSTANT', 'REFLECT', 'SYMMETRIC' or `None`. The default is 'CONSTANT'.
      If `None`, padding is not applied.

  Returns:
    A 3-D tensor representing the value of a flow field variable on the local
    core mesh. The values are valid only within the 'core' portion of the
    sub-grid (i.e. excluding halo cells). Halos are filled according to the
    specified `pad_mode`. By default, halos are filled with 0.

  Raises:
    ValueError: If arguments are incorrect.
  """
  # BEGIN GOOGLE-INTERNAL
  # TODO(b/309863073): Once Swirl-C is modified to use the Swirl-LM grid
  # definition, `initializer.partial_mesh_for_core` from Swirl-LM can be used
  # in place of this method.
  # END GOOGLE-INTERNAL
  lx = cfg.lx
  ly = cfg.ly
  lz = cfg.lz

  core_nx = cfg.core_nx
  core_ny = cfg.core_ny
  core_nz = cfg.core_nz

  gx = coordinate[0]
  gy = coordinate[1]
  gz = coordinate[2]

  # These assert ops will be ignored on TPU. Force to place on CPU in case the
  # function is used outside initialization stage (which is already on CPU).
  with tf.device('CPU'):
    tf.debugging.assert_greater_equal(
        gx,
        0,
        'Invalid subgrid coordinate specified with negative x core index.',
    )
    tf.debugging.assert_greater(
        cfg.cx,
        gx,
        'Invalid subgrid coordinate specified with x core index. Must be '
        'smaller than total number of core partitioning in x direction.',
    )
    tf.debugging.assert_greater_equal(
        gy,
        0,
        'Invalid subgrid coordinate specified with negative y core index.',
    )
    tf.debugging.assert_greater(
        cfg.cy,
        gy,
        'Invalid subgrid coordinate specified with y core index. Must be '
        'smaller than total number of core partitioning in y direction.',
    )
    tf.debugging.assert_greater_equal(
        gz,
        0,
        'Invalid subgrid coordinate specified with negative z core index.',
    )
    tf.debugging.assert_greater(
        cfg.cz,
        gz,
        'Invalid subgrid coordinate specified with z core index. Must be '
        'smaller than total number of core partitioning in z direction.',
    )
    tf.debugging.assert_greater_equal(
        core_nx,
        cfg.halo_width,
        f'Local mesh size `core_nx = {core_nx}` must be at least the halo width'
        f' `cfg.halo_width = {cfg.halo_width}`.',
    )
    tf.debugging.assert_greater_equal(
        core_ny,
        cfg.halo_width,
        f'Local mesh size `core_ny = {core_ny}` must be at least the halo width'
        f' `cfg.halo_width = {cfg.halo_width}`.',
    )
    tf.debugging.assert_greater_equal(
        core_nz,
        cfg.halo_width,
        f'Local mesh size `core_nz = {core_nz}` must be at least the halo width'
        f' `cfg.halo_width = {cfg.halo_width}`.',
    )

  xs = _get_local_1d_mesh(core_nx, gx, cfg.x)
  ys = _get_local_1d_mesh(core_ny, gy, cfg.y)
  zs = _get_local_1d_mesh(core_nz, gz, cfg.z)

  xx, yy, zz = tf.meshgrid(xs, ys, zs, indexing='ij')
  val = value_fn(
      xx,
      yy,
      zz,
      types.NP_DTYPE(lx),
      types.NP_DTYPE(ly),
      types.NP_DTYPE(lz),  # pytype: disable=wrong-arg-types  # numpy-scalars
      coordinate,
  )
  if pad_mode:
    val = tf.pad(
        val,
        paddings=[
            [cfg.halo_width, cfg.halo_width],
            [cfg.halo_width, cfg.halo_width],
            [cfg.halo_width, cfg.halo_width],
        ],
        mode=pad_mode,
    )
  if perm:
    val = tf.transpose(val, perm=perm)
  return val


def _get_local_1d_mesh(
    core_n: int,
    core_id: tf.Tensor,
    global_1d_mesh: tf.Tensor,
) -> tf.Tensor:
  """Returns the portion of the global mesh along the dimension given.

  All arguments are implicitly in the given dimension corresponding to the
  context in which this helper is being called.

  Args:
    core_n: The number of grid points along a given dimension per core, i.e. the
      length of `local_1d_mesh` for a given dimension.
    core_id: The index of the core in {0, 1, ... num_cores - 1} along the given
      dimension.
    global_1d_mesh: A 1D `tf.tensor` giving the global mesh (excluding halo
      cells) along a given dimension.

  Returns:
    The subgrid mesh along a given dimension, excluding halo cells.
  """
  return tf.gather(
      global_1d_mesh,
      tf.cast(
          tf.linspace(core_id * core_n, (core_id + 1) * core_n - 1, core_n),
          tf.int32,
      ),
  )


# Some typical `value_fn` are provided below.
def constant_initial_state_fn(value: float) -> ValueFunction:
  """Defines function to generate a constant initial state.

  Args:
    value: The constant value that will be applied to the field across the local
      mesh.

  Returns:
    A value function which will generate a constant initial state with the value
    specified by `value`.
  """

  def init_fn(xx, yy, zz, lx, ly, lz, coord):
    """Generates a field with a constant."""
    del yy, zz, lx, ly, lz, coord  # Unused.
    return value * tf.ones_like(xx)

  return init_fn


def step_function_initial_state_fn(
    left_value: float, right_value: float, step_loc: float, dim: str
) -> ValueFunction:
  """Defines a function to generate an initial state from a step function.

  The global mesh will be split in two along the specified `dim` at `step_loc`,
  with the left (negative) region of the domain having a constant value
  `left_value`, and the right (positive) region of the domain having a constant
  value `right_value`. The local mesh is then extracted from the global mesh
  accordingly.

  Args:
    left_value: The constant value that the left (negative) region of the domain
      will take.
    right_value: The constant value that the right (positive) region of the
      domain will take.
    step_loc: The normalized location along `dim` where the global mesh will be
      sectioned. A value of 0.5 indicates the domain is split at the center.
      Values of 0.25 and 0.75 indicate a split to the left (negative) and right
      (positive) of center respectively.
    dim: A string indicating dimension along which to section the domain. `dim`
      must be either 'x', 'y', or 'z'.

  Returns:
    A value function which will generate a domain specified by a step function
    along the indicated dimension.
  """

  def init_fn(xx, yy, zz, lx, ly, lz, coord):
    del coord  # Unused.
    if dim == 'x':
      l = lx
      v = xx
    elif dim == 'y':
      l = ly
      v = yy
    else:
      l = lz
      v = zz
    return tf.compat.v1.where(
        tf.math.abs(v / l) <= step_loc,
        left_value * tf.ones_like(v),
        right_value * tf.ones_like(v),
    )

  return init_fn
