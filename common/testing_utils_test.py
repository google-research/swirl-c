"""Tests for testing_utils.py."""

from absl.testing import parameterized
import numpy as np
from swirl_c.common import testing_utils
import tensorflow as tf


class TestingUtilsTest(tf.test.TestCase, parameterized.TestCase):
  _DIMS = ('x', 'y', 'z')

  @parameterized.parameters(*zip(_DIMS))
  def test_to_3d_tensor_returns_tf_tensor(self, dim):
    """Checks that to_3d_tensor returns a dictionary of 2D `tf.Tensor`s."""
    size = [12, 14, 17]
    if dim == 'x':
      f = np.linspace(5.0, 5.0, size[1])
      f_3d = np.tile(f[np.newaxis, :, np.newaxis], (size[0], 1, size[2]))
    elif dim == 'y':
      f = np.linspace(5.0, 5.0, size[2])
      f_3d = np.tile(f[np.newaxis, np.newaxis, :], (size[0], size[1], 1))
    else:  # dim == 'z':
      f = np.linspace(5.0, 5.0, size[0])
      f_3d = np.tile(f[:, np.newaxis, np.newaxis], (1, size[1], size[2]))

    result = testing_utils.to_3d_tensor(f, dim, size)
    # First, check that the length of the list of 2D tf.Tensors is correct.
    self.assertLen(result, size[0])
    # Loop over 2D tensors and check for correct data type, size, and values.
    for i_slice in range(0, size[0]):
      self.assertDTypeEqual(result[i_slice], tf.float32)
      self.assertAllEqual(f_3d[i_slice, :, :], self.evaluate(result[i_slice]))

  @parameterized.parameters(*zip(_DIMS))
  def test_to_3d_tensor_returns_ndarray(self, dim):
    """Checks that to_3d_tensor returns a `np.ndarray`."""
    size = [12, 14, 17]
    if dim == 'x':
      f = np.linspace(5.0, 5.0, size[1])
      f_3d = np.tile(f[np.newaxis, :, np.newaxis], (size[0], 1, size[2]))
    elif dim == 'y':
      f = np.linspace(5.0, 5.0, size[2])
      f_3d = np.tile(f[np.newaxis, np.newaxis, :], (size[0], size[1], 1))
    else:  # dim == 'z':
      f = np.linspace(5.0, 5.0, size[0])
      f_3d = np.tile(f[:, np.newaxis, np.newaxis], (1, size[1], size[2]))

    result = testing_utils.to_3d_tensor(f, dim, size, as_tf_tensor=False)
    # First, check the type.
    self.assertDTypeEqual(result, f.dtype)
    # Second, check the shape and values match
    self.assertAllEqual(f_3d, result)

  def test_to_3d_tensor_invalid_dimension(self):
    """Checks that to_3d_tensor raises error if invalid `dim` is specified."""
    size = [12, 14, 17]
    dim = 'w'
    f = np.linspace(5.0, 5.0, 10)
    msg = r'"w" is invalid dimension option. Valid options are "x", "y", "z".'
    with self.assertRaisesRegex(ValueError, msg):
      testing_utils.to_3d_tensor(f, dim, size)


if __name__ == '__main__':
  tf.test.main()
