"""Tests for gradient."""

from absl.testing import parameterized
import numpy as np
from swirl_c.common import testing_utils
from swirl_c.numerics import gradient
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf

_DIMS = ('x', 'y', 'z')


class GradientTest(tf.test.TestCase, parameterized.TestCase):

  def gen_sine_fn_along_dim(self, dim, n=16, n_normal=4):
    """Generates a sine profile from 0 to 2pi along `dim`."""
    x = np.linspace(0, 2.0 * np.pi, n)
    y = np.sin(x)
    if dim == 'x':
      y = np.tile(y[np.newaxis, :, np.newaxis], [n_normal, 1, n_normal])
    elif dim == 'y':
      y = np.tile(y[np.newaxis, np.newaxis, :], [n_normal, n_normal, 1])
    else:  # dim == 'z':
      y = np.tile(y[:, np.newaxis, np.newaxis], [1, n_normal, n_normal])

    return tf.unstack(tf.convert_to_tensor(y, dtype=tf.float32)), x[1] - x[0]

  def extract_profile_along_dim(self, u, dim):
    """Extracts a 1D profile at the center of `u` along `dim`."""
    u = np.stack(u) if isinstance(u, list) else u
    nz, nx, ny = u.shape
    if dim == 'x':
      return u[nz // 2, :, ny // 2]
    elif dim == 'y':
      return u[nz // 2, nx // 2, :]
    else:  # dim == 'z':
      return u[:, nx // 2, ny // 2]

  @parameterized.parameters(*zip(_DIMS))
  def test_forward_1_provides_correct_1st_order_derivative(self, dim):
    """Checks if the 1st order derivative is computed correctly."""
    u, h = self.gen_sine_fn_along_dim(dim)
    kernel_op = get_kernel_fn.ApplyKernelSliceOp()
    dudh = self.evaluate(gradient.forward_1(u, h, dim, kernel_op))

    expected = [
        0.97101221,
        0.80311538,
        0.4963526,
        0.10376595,
        -0.30676278,
        -0.66424943,
        -0.90688133,
        -0.9927052,
        -0.90688133,
        -0.66424943,
        -0.30676278,
        0.10376595,
        0.4963526,
        0.80311538,
        0.97101221,
    ]

    self.assertAllClose(
        expected, self.extract_profile_along_dim(dudh, dim)[:-1]
    )

  @parameterized.parameters(*zip(_DIMS))
  def test_central_2_provides_correct_1st_order_derivative(self, dim):
    """Checks if the 1st order derivative is computed correctly."""
    u, h = self.gen_sine_fn_along_dim(dim)
    kernel_op = get_kernel_fn.ApplyKernelSliceOp()
    dudh = self.evaluate(gradient.central_2(u, h, dim, kernel_op))

    expected = [
        0.88706379,
        0.64973399,
        0.30005927,
        -0.10149841,
        -0.4855061,
        -0.78556538,
        -0.94979326,
        -0.94979326,
        -0.78556538,
        -0.4855061,
        -0.10149841,
        0.30005927,
        0.64973399,
        0.88706379,
    ]

    self.assertAllClose(
        expected, self.extract_profile_along_dim(dudh, dim)[1:-1]
    )

  @parameterized.parameters(*zip(_DIMS))
  def test_backward_1_provides_correct_1st_order_derivative(self, dim):
    """Checks if the 1st order derivative is computed correctly."""
    nx = 16
    size = [nx, nx, nx]
    x = np.linspace(0.0, 1.0, nx)
    f_vec = np.sin(x)
    f = testing_utils.to_3d_tensor(f_vec, dim, size)
    kernel_op = get_kernel_fn.ApplyKernelSliceOp()
    h = np.diff(x)[0]
    results = self.evaluate(gradient.backward_1(f, h, dim, kernel_op))
    df_dx = np.zeros(nx)
    df_dx[1:] = np.diff(f_vec) / np.diff(x)
    expected = testing_utils.to_3d_tensor(df_dx, dim, size, as_tf_tensor=False)
    self.assertAllClose(expected[1:, 1:, 1:], np.array(results)[1:, 1:, 1:])


if __name__ == '__main__':
  tf.test.main()
