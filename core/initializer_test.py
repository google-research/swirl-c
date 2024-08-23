"""Tests for initializer.py."""

from absl.testing import parameterized
import numpy as np
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.core import initializer
from swirl_c.core import parameter
import tensorflow as tf

_DIMS = ('x', 'y', 'z')


class InitializerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes the default cfg object."""
    super().setUp()
    self.cfg = parameter.SwirlCParameters()

  def test_default_initializer_returns_zeros_one_replica(self):
    """Tests that the default initializer returns zeros for all variables."""
    self.cfg.conservative_variable_names = list(types.BASE_CONSERVATIVE) + [
        'rho_y'
    ]
    replica_id = tf.constant(0)
    coordinates = [0, 0, 0]

    results = self.evaluate(
        initializer.default_initializer(replica_id, coordinates, self.cfg)
    )
    expected = {
        'rho': np.zeros([16, 16, 16]),
        'rho_u': np.zeros([16, 16, 16]),
        'rho_v': np.zeros([16, 16, 16]),
        'rho_w': np.zeros([16, 16, 16]),
        'rho_e': np.zeros([16, 16, 16]),
        'rho_y': np.zeros([16, 16, 16]),
    }
    self.assertDictEqual(expected, results)

  def test_default_initializer_returns_zeros_four_replicas(self):
    """Tests that the default initializer returns zeros for four replicas."""
    self.cfg.conservative_variable_names = list(types.BASE_CONSERVATIVE) + [
        'rho_y'
    ]
    replica_id = tf.constant(0)
    self.cfg.cx = 2
    self.cfg.cy = 2
    self.cfg.cz = 1
    self.cfg.core_nx = 5
    self.cfg.core_ny = 5
    self.cfg.core_nz = 10
    expected = {
        'rho': np.zeros([16, 11, 11]),
        'rho_u': np.zeros([16, 11, 11]),
        'rho_v': np.zeros([16, 11, 11]),
        'rho_w': np.zeros([16, 11, 11]),
        'rho_e': np.zeros([16, 11, 11]),
        'rho_y': np.zeros([16, 11, 11]),
    }
    for ic in range(2):
      for jc in range(2):
        coordinates = [ic, jc, 0]
        results = self.evaluate(
            initializer.default_initializer(replica_id, coordinates, self.cfg)
        )
        with self.subTest(name=f'cx: {ic}, cy: {jc}, cz: 0'):
          self.assertDictEqual(expected, results)

  @parameterized.parameters(*_DIMS)
  def test_partial_field_for_core_one_replica(self, dim):
    """Tests that the returned partial mesh values are correct for 1 replica."""
    coordinate = [0, 0, 0]

    def init_fn(xx, yy, zz, lx, ly, lz, coord):
      """Generates an initial field for testing."""
      del lx, ly, lz, coord
      match dim:
        case 'x':
          return xx
        case 'y':
          return yy
        case _:  # case 'z'
          return zz

    results = self.evaluate(
        initializer.partial_field_for_core(self.cfg, coordinate, init_fn)
    )
    expected = np.zeros([16, 16, 16])
    match dim:
      case 'x':
        interior = testing_utils.to_3d_tensor(
            self.cfg.x, 'x', [10, 10, 10], as_tf_tensor=False
        )
      case 'y':
        interior = testing_utils.to_3d_tensor(
            self.cfg.y, 'y', [10, 10, 10], as_tf_tensor=False
        )
      case _:  # case 'z'
        interior = testing_utils.to_3d_tensor(
            self.cfg.z, 'z', [10, 10, 10], as_tf_tensor=False
        )
    expected[3:-3, 3:-3, 3:-3] = interior
    for i in range(3):
      expected[3:-3, 3 - i - 1, 3:-3] = expected[3:-3, 3 + i, 3:-3]
      expected[3:-3, -(3 - i), 3:-3] = expected[3:-3, -(3 + 1 + i), 3:-3]
    for j in range(3):
      expected[3:-3, :, 3 - j - 1] = expected[3:-3, :, 3 + j]
      expected[3:-3, :, -(3 - j)] = expected[3:-3, :, -(3 + 1 + j)]
    for k in range(3):
      expected[3 - k - 1, :, :] = expected[3 + k, :, :]
      expected[-(3 - k), :, :] = expected[-(3 + 1 + k), :, :]

    self.assertAllClose(expected, results)

  @parameterized.product(dim=_DIMS, comp_shape_dim=_DIMS)
  def test_partial_field_for_core_four_replicas(self, dim, comp_shape_dim):
    """Tests that the partial mesh values are correct for four replicas."""

    def init_fn(xx, yy, zz, lx, ly, lz, coord):
      """Generates an initial field for testing."""
      del lx, ly, lz, coord
      match dim:
        case 'x':
          return xx
        case 'y':
          return yy
        case _:  # case 'z'
          return zz

    self.cfg.cx = 2
    self.cfg.cy = 2
    self.cfg.cz = 2
    self.cfg.core_nx = 5
    self.cfg.core_ny = 5
    self.cfg.core_nz = 5
    match comp_shape_dim:
      case 'x':
        self.cfg.cx = 1
        self.cfg.core_nx = 10
      case 'y':
        self.cfg.cy = 1
        self.cfg.core_ny = 10
      case _:  # 'z'
        self.cfg.cz = 1
        self.cfg.core_nz = 10

    match dim:
      case 'x':
        interior = testing_utils.to_3d_tensor(
            self.cfg.x, 'x', [10, 10, 10], as_tf_tensor=False
        )
      case 'y':
        interior = testing_utils.to_3d_tensor(
            self.cfg.y, 'y', [10, 10, 10], as_tf_tensor=False
        )
      case _:  # case 'z'
        interior = testing_utils.to_3d_tensor(
            self.cfg.z, 'z', [10, 10, 10], as_tf_tensor=False
        )
    for ic in range(self.cfg.cx):
      for jc in range(self.cfg.cy):
        for kc in range(self.cfg.cz):
          coordinates = [ic, jc, kc]
          results = self.evaluate(
              initializer.partial_field_for_core(self.cfg, coordinates, init_fn)
          )
          if self.cfg.cx == 1:
            expected = np.zeros([11, 16, 11])
          elif self.cfg.cy == 1:
            expected = np.zeros([11, 11, 16])
          else:  # self.cfg.cz == 1
            expected = np.zeros([16, 11, 11])
          i_s = 0
          i_e = 10
          j_s = 0
          j_e = 10
          k_s = 0
          k_e = 10
          if self.cfg.cx > 1:
            if ic == 0:
              i_s = 0
              i_e = 5
            else:
              i_s = 5
              i_e = 10
          if self.cfg.cy > 1:
            if jc == 0:
              j_s = 0
              j_e = 5
            else:
              j_s = 5
              j_e = 10
          if self.cfg.cz > 1:
            if kc == 0:
              k_s = 0
              k_e = 5
            else:
              k_s = 5
              k_e = 10
          expected[3:-3, 3:-3, 3:-3] = interior[k_s:k_e, i_s:i_e, j_s:j_e]
          for i in range(3):
            expected[3:-3, 3 - i - 1, 3:-3] = expected[3:-3, 3 + i, 3:-3]
            expected[3:-3, -(3 - i), 3:-3] = expected[3:-3, -(3 + 1 + i), 3:-3]
          for j in range(3):
            expected[3:-3, :, 3 - j - 1] = expected[3:-3, :, 3 + j]
            expected[3:-3, :, -(3 - j)] = expected[3:-3, :, -(3 + 1 + j)]
          for k in range(3):
            expected[3 - k - 1, :, :] = expected[3 + k, :, :]
            expected[-(3 - k), :, :] = expected[-(3 + 1 + k), :, :]

          with self.subTest(name=f'cx: {ic}, cy: {jc}, cz: {kc}'):
            self.assertAllClose(expected, results)

  def test_get_local_1d_mesh_returns_correct_mesh(self):
    """Confirms that the correct portion of the global mesh is returned."""
    xx = np.linspace(0.0, 1.0, 64)
    mesh = {
        0: xx[0:16],
        1: xx[16:32],
        2: xx[32:48],
        3: xx[48:64],
    }
    xx = tf.convert_to_tensor(xx, dtype=types.DTYPE)
    core_n = 16
    num_cores = 4
    for core_id in range(num_cores):
      with self.subTest(name=core_id):
        self.assertAllClose(
            mesh[core_id],
            self.evaluate(initializer._get_local_1d_mesh(core_n, core_id, xx)),
        )

  def test_constant_initial_state_fn_evaluates_correctly(self):
    """Tests that the init_fn returned produces a constant initial state."""
    init_fn = initializer.constant_initial_state_fn(1.45)
    xs = np.linspace(0.0, 1.0, 16)
    ys = np.linspace(0.0, 1.0, 16)
    zs = np.linspace(0.0, 1.0, 16)
    xx, yy, zz = tf.meshgrid(xs, ys, zs, indexing='ij')
    results = self.evaluate(init_fn(xx, yy, zz, 1.0, 1.0, 1.0, [0, 0, 0]))
    expected = 1.45 * np.ones([16, 16, 16])
    self.assertAllClose(expected, results)

  @parameterized.parameters(*_DIMS)
  def test_step_function_initial_state_fn_evaluates_correctly(self, dim):
    """Tests that the init_fn returned produces the expected step function."""
    init_fn = initializer.step_function_initial_state_fn(1.45, 2.5, 0.4, dim)
    xs = np.linspace(0.25, 0.75, 16)
    ys = np.linspace(0.25, 0.75, 16)
    zs = np.linspace(0.25, 0.75, 16)
    xx, yy, zz = tf.meshgrid(xs, ys, zs, indexing='ij')
    results = self.evaluate(init_fn(xx, yy, zz, 1.0, 1.0, 1.0, [0, 0, 0]))
    expected = 1.45 * np.ones([16, 16, 16])
    match dim:
      case 'x':
        expected[5:, :, :] = 2.5
      case 'y':
        expected[:, 5:, :] = 2.5
      case _:  # case 'z'
        expected[:, :, 5:] = 2.5
    self.assertAllClose(expected, results)


if __name__ == '__main__':
  tf.test.main()
