"""A library of helper functions to be used with unit tests."""

from typing import Sequence, Union

import numpy as np
from swirl_c.common import types
import tensorflow as tf


def to_3d_tensor(
    f: np.ndarray,
    dim: str,
    size: list[int],
    as_tf_tensor: bool = True,
) -> Union[Sequence[tf.Tensor], np.ndarray]:
  """Tiles a 1D `np.ndarray` to make a `tf.Tensor` or `np.ndarray`.

  Args:
    f: The 1D array to be tiled.
    dim: The dimension that the 1D array is mapped to. Options are 'x', 'y', or
      'z'.
    size: List that specifies the size of the target `tf.Tensor` or `np.ndarray`
      as `[nz, nx, ny]`. The size of the specified dimension (e.g. `nx` for `dim
      = 'x'`) must match the length of `f`.
    as_tf_tensor: Boolean to switch function return from `tf.Tensor` (`True`,
      default) to a numpy `ndarray` of equal size (`False`).

  Returns:
    Either a `tf.Tensor` or `np.ndarray` produced by tiling `f`. The 3D field is
    represented as a list of 2D `tf.Tensor`s.

  Raises:
    ValueError if `dim` is not 'x', 'y', or 'z'.
  """
  if dim == 'x':
    f_3d = np.tile(f[np.newaxis, :, np.newaxis], (size[0], 1, size[2]))
  elif dim == 'y':
    f_3d = np.tile(f[np.newaxis, np.newaxis, :], (size[0], size[1], 1))
  elif dim == 'z':
    f_3d = np.tile(f[:, np.newaxis, np.newaxis], (1, size[1], size[2]))
  else:
    raise ValueError(
        f'"{dim}" is invalid dimension option. Valid options are "x", "y", "z".'
    )

  if as_tf_tensor:
    return tf.unstack(tf.convert_to_tensor(f_3d, dtype=types.DTYPE))
  else:
    return f_3d
