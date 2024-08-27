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
