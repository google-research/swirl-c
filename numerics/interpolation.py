"""A library of interpolation methods."""

from swirl_c.common import types
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf


def linear_interpolation(
    f: types.FlowFieldVar,
    dim: str,
    kernel_op: get_kernel_fn.ApplyKernelOp,
) -> types.FlowFieldVar:
  """Linearly interpolates `f` onto the i - 1/2 face assuming a uniform mesh.

  Args:
    f: A flow field variable representing the cell average of a quantity which
      will be linearly interpolated onto the i - 1/2 face.
    dim: The direction 'x', 'y', or 'z' along which to perform the
      interpolation.
    kernel_op: An instance of `ApplyKernelOp` indicating the particular
      operation to compute the interpolation on TPU.

  Returns:
    A flow field variable representing the left (i - 1/2) interpolated
    face value of `f` across the domain.
  """

  def neighbor_sum(f):
    if dim == 'x':
      return kernel_op.apply_kernel_op_x(f, 'ksx')
    elif dim == 'y':
      return kernel_op.apply_kernel_op_y(f, 'ksy')
    elif dim == 'z':
      return kernel_op.apply_kernel_op_z(f, 'ksz', 'kszsh')
    else:
      raise ValueError(
          f'"{dim}" is not a valid dimension. Available options are: "x", "y",'
          ' "z".'
      )
  return tf.nest.map_structure(lambda f: 0.5 * f, neighbor_sum(f))
