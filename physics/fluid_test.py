"""Tests for fluid."""

import os

from absl.testing import parameterized
import numpy as np
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.numerics import kernel_op_types
from swirl_c.physics import constant
from swirl_c.physics import fluid
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework import tpu_runner


_TEST_DATA_DIR = 'google3/third_party/py/swirl_c/physics/test_data'


class FluidTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes the generic thermodynamics library object."""
    super().setUp()
    self.cfg = parameter.SwirlCParameters()

  def run_tpu_test(self, replicas, device_fn, inputs):
    """Runs `device` function on TPU."""
    device_inputs = [list(x) for x in zip(*inputs)]
    computation_shape = replicas.shape
    runner = tpu_runner.TpuRunner(computation_shape=computation_shape)
    return runner.run(device_fn, *device_inputs)

  def save_results(self, filename, output):
    """Writes an numpy array to file."""
    write_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')
    filename = os.path.join(write_dir, filename)

    with tf.io.gfile.GFile(filename, 'bw') as f:
      np.save(f, output)

  def gen_taylor_green_vortex(self, nx=16, ny=16, nz=16):
    """Generates `u`, `v`, and `w` at cell center and faces with TGV."""
    n_tot = [2 * n for n in (nx, ny, nz)]
    h = [np.pi / (n - 1) for n in (nx, ny, nz)]
    x, y, z = [h_i * np.arange(n) for h_i, n in zip(h, n_tot)]

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    u = np.cos(xx) * np.sin(yy) * np.sin(zz)
    v = np.sin(xx) * np.cos(yy) * np.sin(zz)
    w = np.sin(xx) * np.sin(yy) * np.cos(zz)
    rho = 0.5 * (np.sin(xx) * np.sin(yy) * np.sin(yy) + 2.0)

    # Compute the strain rate analytically. Note that in this configuration:
    # dudx = dvdy = dwdz, dudy = dvdx, dudz = dwdx, dvdz = dwdy.
    s_kk = np.zeros(n_tot, dtype=np.float32)
    s_12 = np.cos(xx) * np.cos(yy) * np.sin(zz)
    s_13 = np.cos(xx) * np.sin(yy) * np.cos(zz)
    s_23 = np.sin(xx) * np.cos(yy) * np.cos(zz)

    def get_face_val(s_full, dim):
      """Get values on the specific face in `dim` with node values on `::2`."""
      # Note that all face values at i - 1/2 are saved at index i.
      s_face = np.zeros((nx, ny, nz), dtype=np.float32)
      if dim == 'x':
        s_face[1:, :, :] = s_full[1:-1:2, ::2, ::2]
      elif dim == 'y':
        s_face[:, 1:, :] = s_full[::2, 1:-1:2, ::2]
      elif dim == 'z':
        s_face[:, :, 1:] = s_full[::2, ::2, 1:-1:2]
      return s_face.transpose((2, 0, 1))

    s = [
        [
            get_face_val(s_kk, 'x'),
            get_face_val(s_12, 'y'),
            get_face_val(s_13, 'z'),
        ],
        [
            get_face_val(s_12, 'x'),
            get_face_val(s_kk, 'y'),
            get_face_val(s_23, 'z'),
        ],
        [
            get_face_val(s_13, 'x'),
            get_face_val(s_23, 'y'),
            get_face_val(s_kk, 'z'),
        ],
    ]

    # Rearrange the coordinates from x-y-z to z-x-y
    u, v, w, rho = [np.transpose(f, (2, 0, 1)) for f in [u, v, w, rho]]

    states = {
        'u': tf.unstack(tf.convert_to_tensor(u[::2, ::2, ::2])),
        'v': tf.unstack(tf.convert_to_tensor(v[::2, ::2, ::2])),
        'w': tf.unstack(tf.convert_to_tensor(w[::2, ::2, ::2])),
        'rho': tf.unstack(tf.convert_to_tensor(rho[::2, ::2, ::2])),
    }

    cfg = {f'd{dim}': 2.0 * h_i for dim, h_i in zip(('x', 'y', 'z'), h)}
    cfg.update({'nu': 1e-1})

    return states, s, cfg

  _DIMS = ('x', 'y', 'z')

  def test_strain_rate_computed_correctly(self):
    """Checks if the strain rate is computed correctly."""
    states, expected, cfg = self.gen_taylor_green_vortex()
    cfg.update(
        {'kernel_op_type': kernel_op_types.KernelOpType.KERNEL_OP_SLICE.value}
    )
    del cfg['nu']  # Unused.
    s = self.evaluate(
        fluid.strain_rate(
            states,
            parameter.SwirlCParameters(cfg),
        )
    )

    for i in range(3):
      for j in range(3):
        with self.subTest(name=f'S{i}{j}'):
          s_ij = np.stack(s[i][j])
          self.assertAllClose(
              expected[i][j][1:-1, 1:-1, 1:-1],
              s_ij[1:-1, 1:-1, 1:-1],
              atol=5e-2,  # O(h^3)
          )

  @parameterized.parameters(*_DIMS)
  def test_pressure_hydrostatic_computed_correctly(self, dim):
    """Checks if the pressure is computed correctly at hydrostatic condition."""
    n = 16
    size = [16, 16, 16]
    dims = ('x', 'y', 'z')
    computation_shape = [1, 1, 1]
    computation_shape[dims.index(dim)] = 2
    replicas = np.reshape(np.arange(2), computation_shape)
    self.cfg.cx = computation_shape[0]
    self.cfg.cy = computation_shape[1]
    self.cfg.cz = computation_shape[2]
    self.cfg.dx = 100.0
    self.cfg.dy = 100.0
    self.cfg.dz = 100.0
    self.cfg.g[dim] = -1.0
    t_vec = 300.0 * np.ones(n)
    states = {
        types.POTENTIAL_T: testing_utils.to_3d_tensor(t_vec, dim, size),
    }

    inputs = [
        [tf.constant(0), states],
        [tf.constant(1), states],
    ]

    def device_fn(replica_id, states):
      """Wraps the hydrostatic pressure function."""
      return fluid.pressure_hydrostatic(
          replica_id,
          replicas,
          states,
          self.cfg,
      )

    output = self.run_tpu_test(replicas, device_fn, inputs)
    delta = (self.cfg.dx, self.cfg.dy, self.cfg.dz)[types.DIMS.index(dim)]
    # Generate the expected values from an analytical solution.
    p_fn = lambda z: self.cfg.p_0 * (
        1.0 - constant.G * constant.KAPPA * z / constant.R / 300.0
    ) ** (1.0 / constant.KAPPA)
    z_global = delta * (
        np.arange(2 * (n - self.cfg.halo_width)) - self.cfg.halo_width
    )
    z = [z_global[:n], z_global[-n:]]

    def expand_dims(f):
      """Expands the dimension of 1D tensor along directions normal to `dim`."""
      if dim == 'x':
        return np.tile(f[np.newaxis, :, np.newaxis], (n, 1, n))
      elif dim == 'y':
        return np.tile(f[np.newaxis, np.newaxis, :], (n, n, 1))
      elif dim == 'z':
        return np.tile(f[:, np.newaxis, np.newaxis], (1, n, n))

    self.save_results(
        f'replicas_{computation_shape[0]}{computation_shape[1]}'
        f'{computation_shape[2]}.npy',
        output,
    )

    for i in range(2):
      expected = p_fn(expand_dims(z[i]))
      with self.subTest(name=f'Replica{i}'):
        self.assertAllClose(expected, np.stack(output[i]), rtol=1e-4, atol=10)

  def test_pressure_hydrostatic_raises_error_for_misaligned_gravity(self):
    """Checks if an error is raised for gravity vector which is not aligned."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    states = {}
    theta = np.pi / 3
    phi = -np.pi / 3
    g_vec = np.zeros(3)
    g_vec[0] = np.sin(phi) * np.cos(theta)
    g_vec[1] = np.sin(phi) * np.sin(theta)
    g_vec[2] = np.cos(phi)
    dims = ('x', 'y', 'z')
    for dim in dims:
      self.cfg.g[dim] = g_vec[dims.index(dim)]
    msg = r'^(Hydrostatic pressure)'
    with self.assertRaisesRegex(ValueError, msg):
      fluid.pressure_hydrostatic(
          replica_id,
          replicas,
          states,
          self.cfg,
      )

  def test_pressure_hydrostatic_no_gravity(self):
    """Checks that hydrostatic pressure returns p_0 if no gravity is present."""
    n = 16
    size = [n, n, n]
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    self.cfg.dx = 100.0
    self.cfg.dy = 100.0
    self.cfg.dz = 100.0
    t_vec = 300.0 * np.ones(n)
    states = {
        types.POTENTIAL_T: testing_utils.to_3d_tensor(t_vec, 'x', size),
    }
    expected = self.cfg.p_0 * np.ones([n, n, n])
    res = self.evaluate(
        fluid.pressure_hydrostatic(
            replica_id,
            replicas,
            states,
            self.cfg,
        )
    )
    self.assertAllClose(expected, res)


if __name__ == '__main__':
  tf.test.main()
