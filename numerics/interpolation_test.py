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
"""Tests for interpolation."""

from absl.testing import parameterized
import numpy as np
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.numerics import interpolation
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf

_DIMS = ('x', 'y', 'z')


class InterpolationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(*zip(_DIMS))
  def test_linear_interpolation_computes_correctly(self, dim):
    """Checks that the linear interpolation computes correctly."""
    kernel_op = get_kernel_fn.ApplyKernelConvOp(8)
    nx = 16
    size = [nx, nx, nx]
    f_vec = np.logspace(0.0, 2.0, nx)
    f = testing_utils.to_3d_tensor(f_vec, dim, size)
    for face_dim in types.DIMS:
      results = self.evaluate(
          interpolation.linear_interpolation(f, face_dim, kernel_op)
      )
      expected_vec = np.zeros(nx)
      with self.subTest(name=f'face_dim: {face_dim}'):
        if face_dim == dim:
          expected_vec[1:] = f_vec[:-1] + 0.5 * np.diff(f_vec)
        else:
          expected_vec = f_vec
        expected = testing_utils.to_3d_tensor(
            expected_vec, dim, size, as_tf_tensor=False
        )
        self.assertAllClose(
            expected[1:, 1:, 1:], np.array(results)[1:, 1:, 1:]
        )


if __name__ == '__main__':
  tf.test.main()
