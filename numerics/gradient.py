"""A library for gradient computations."""

from swirl_c.common import types
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf


def forward_1(
    f: types.FlowFieldVar,
    h: float,
    dim: str,
    kernel_op: get_kernel_fn.ApplyKernelOp,
) -> types.FlowFieldVar:
  """Computes the gradient with 1st order forward difference."""

  # pylint: disable=g-long-lambda
  op = {
      'x': lambda f: tf.nest.map_structure(
          lambda df: df / h, kernel_op.apply_kernel_op_x(f, 'kdx+')
      ),
      'y': lambda f: tf.nest.map_structure(
          lambda df: df / h, kernel_op.apply_kernel_op_y(f, 'kdy+')
      ),
      'z': lambda f: tf.nest.map_structure(
          lambda df: df / h, kernel_op.apply_kernel_op_z(f, 'kdz+', 'kdz+sh')
      ),
  }
  # pylint: enable=g-long-lambda

  return op[dim](f)


def backward_1(
    f: types.FlowFieldVar,
    h: float,
    dim: str,
    kernel_op: get_kernel_fn.ApplyKernelOp,
) -> types.FlowFieldVar:
  """Computes the gradient with 1st order backward difference."""

  # pylint: disable=g-long-lambda
  op = {
      'x': lambda f: tf.nest.map_structure(
          lambda df: df / h, kernel_op.apply_kernel_op_x(f, 'kdx')
      ),
      'y': lambda f: tf.nest.map_structure(
          lambda df: df / h, kernel_op.apply_kernel_op_y(f, 'kdy')
      ),
      'z': lambda f: tf.nest.map_structure(
          lambda df: df / h, kernel_op.apply_kernel_op_z(f, 'kdz', 'kdzsh')
      ),
  }
  # pylint: enable=g-long-lambda

  return op[dim](f)


def central_2(
    f: types.FlowFieldVar,
    h: float,
    dim: str,
    kernel_op: get_kernel_fn.ApplyKernelOp,
) -> types.FlowFieldVar:
  """Computes the gradient with 2nd order central difference."""

  # pylint: disable=g-long-lambda
  op = {
      'x': lambda f: tf.nest.map_structure(
          lambda df: df / (2.0 * h), kernel_op.apply_kernel_op_x(f, 'kDx')
      ),
      'y': lambda f: tf.nest.map_structure(
          lambda df: df / (2.0 * h), kernel_op.apply_kernel_op_y(f, 'kDy')
      ),
      'z': lambda f: tf.nest.map_structure(
          lambda df: df / (2.0 * h),
          kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh'),
      ),
  }
  # pylint: enable=g-long-lambda

  return op[dim](f)
