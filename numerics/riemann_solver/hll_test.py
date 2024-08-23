"""Tests for hll.py."""

from absl.testing import parameterized
import numpy as np
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.numerics.riemann_solver import hll
from swirl_c.physics import constant
from swirl_c.physics import physics_models
import tensorflow as tf


def _compute_convective_flux(
    conservative: types.FlowFieldMap,
    primitive: types.FlowFieldMap,
    helper_vars: types.FlowFieldMap,
    dim: str,
    conserrvative_var_name: str,
) -> types.FlowFieldVar:
  """Computes the convective flux for `var_name`."""
  del conservative, helper_vars

  var_name = types.conservative_to_primitive_name(conserrvative_var_name)
  aligned_velocity_name = types.VELOCITY[types.DIMS.index(dim)]

  if var_name == types.RHO:
    return tf.nest.map_structure(
        tf.math.multiply, primitive[types.RHO], primitive[aligned_velocity_name]
    )
  elif var_name == aligned_velocity_name:
    return tf.nest.map_structure(
        lambda rho, u, p: rho * u**2 + p,
        primitive[types.RHO],
        primitive[aligned_velocity_name],
        primitive[types.P],
    )
  elif var_name == types.E:
    return tf.nest.map_structure(
        lambda rho, u, h: rho * u * h,
        primitive[types.RHO],
        primitive[aligned_velocity_name],
        primitive[types.H],
    )
  else:
    return tf.nest.map_structure(
        lambda rho, u, phi: rho * u * phi,
        primitive[types.RHO],
        primitive[aligned_velocity_name],
        primitive[var_name],
    )


class HLLTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes the generic thermodynamics library object."""
    super().setUp()
    self.cfg = parameter.SwirlCParameters(
        {
            'conservative_variable_names': list(types.BASE_CONSERVATIVE) + [
                'rho_y'
            ]
        }
    )
    self.physics_models = physics_models.PhysicsModels(self.cfg)

  def normal_shock_relations(self, m):
    """Computes the shock jump conditions for a normal shock Mach number M."""
    p1_p0 = (2.0 * constant.GAMMA * m**2 - (constant.GAMMA - 1.0)) / (
        constant.GAMMA + 1.0
    )
    r1_r0 = (
        (constant.GAMMA + 1.0) * m**2 / ((constant.GAMMA - 1.0) * m**2 + 2.0)
    )
    t1_t0 = p1_p0 / r1_r0
    return p1_p0, r1_r0, t1_t0

  _DIMS = ('x', 'y', 'z')

  @parameterized.parameters(*_DIMS)
  def test_compute_hll_roe_wave_speed_estimates_uniform_field(self, dim):
    """Tests the wave speeds in a uniform field."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]

    u = 10.0 * np.ones(nx)
    v = 20.0 * np.ones(nx)
    w = -5.0 * np.ones(nx)
    p = 101325.0 * np.ones(nx)
    t = 500.0 * np.ones(nx)
    rho = p / (constant.R * t)
    h = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)

    primitive_neg = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        types.H: testing_utils.to_3d_tensor(h, dim, size),
    }
    primitive_pos = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        types.H: testing_utils.to_3d_tensor(h, dim, size),
    }

    results = hll.compute_hll_roe_wave_speed_estimates(
        replica_id,
        replicas,
        primitive_neg,
        primitive_pos,
        dim,
        self.cfg,
        self.physics_models,
    )

    c = np.sqrt(constant.GAMMA * constant.R * t)
    match dim:
      case 'x':
        s_l = testing_utils.to_3d_tensor(u - c, dim, size, as_tf_tensor=False)
        s_r = testing_utils.to_3d_tensor(u + c, dim, size, as_tf_tensor=False)
      case 'y':
        s_l = testing_utils.to_3d_tensor(v - c, dim, size, as_tf_tensor=False)
        s_r = testing_utils.to_3d_tensor(v + c, dim, size, as_tf_tensor=False)
      case _:  # case 'z'
        s_l = testing_utils.to_3d_tensor(w - c, dim, size, as_tf_tensor=False)
        s_r = testing_utils.to_3d_tensor(w + c, dim, size, as_tf_tensor=False)

    with self.subTest(name='s_l'):
      self.assertAllClose(s_l, self.evaluate(results[0]))

    with self.subTest(name='s_r'):
      self.assertAllClose(s_r, self.evaluate(results[1]))

  @parameterized.parameters(*_DIMS)
  def test_compute_hll_roe_wave_speed_estimates_strong_shock_left(self, dim):
    """Tests the wave speed against a specified left running shock."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]

    # Compute the post shock state from pre shock and Mach number.
    m = 6.0
    t0 = 300.0
    p0 = 101325.0
    u0 = m * np.sqrt(constant.GAMMA * constant.R * t0)
    p1_p0, r1_r0, t1_t0 = self.normal_shock_relations(m)
    u1_u0 = 1 / r1_r0
    frame_vel = -500.0
    u = np.array([1.0, u1_u0]) * u0 + frame_vel
    v = 100.0 * np.ones(2)
    w = -200.0 * np.ones(2)
    p = np.array([1.0, p1_p0]) * p0
    t = np.array([1.0, t1_t0]) * t0
    rho = p / (constant.R * t)
    h = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)

    def roe_avg(rho, f):
      rho_s = np.sqrt(rho)
      roe_avg_f = np.sum(rho_s * f) / np.sum(rho_s)
      return roe_avg_f

    roe_u = roe_avg(rho, u)
    roe_v = roe_avg(rho, v)
    roe_w = roe_avg(rho, w)
    roe_h = roe_avg(rho, h)
    roe_h_int = roe_h - 0.5 * (roe_u**2 + roe_v**2 + roe_w**2)
    roe_c = np.sqrt(constant.GAMMA * constant.R * roe_h_int / constant.CP)
    # Rotate the velocity components depending on `dim` to align shock
    u_field = u
    v_field = v
    w_field = w
    if dim == 'y':
      u_field = v
      v_field = u
    elif dim == 'z':
      u_field = w
      w_field = u
    primitive_neg = {
        types.RHO: testing_utils.to_3d_tensor(rho[0] * np.ones(nx), dim, size),
        types.U: testing_utils.to_3d_tensor(
            u_field[0] * np.ones(nx), dim, size
        ),
        types.V: testing_utils.to_3d_tensor(
            v_field[0] * np.ones(nx), dim, size
        ),
        types.W: testing_utils.to_3d_tensor(
            w_field[0] * np.ones(nx), dim, size
        ),
        types.P: testing_utils.to_3d_tensor(p[0] * np.ones(nx), dim, size),
        types.H: testing_utils.to_3d_tensor(h[0] * np.ones(nx), dim, size),
    }
    primitive_pos = {
        types.RHO: testing_utils.to_3d_tensor(rho[1] * np.ones(nx), dim, size),
        types.U: testing_utils.to_3d_tensor(
            u_field[1] * np.ones(nx), dim, size
        ),
        types.V: testing_utils.to_3d_tensor(
            v_field[1] * np.ones(nx), dim, size
        ),
        types.W: testing_utils.to_3d_tensor(
            w_field[1] * np.ones(nx), dim, size
        ),
        types.P: testing_utils.to_3d_tensor(p[1] * np.ones(nx), dim, size),
        types.H: testing_utils.to_3d_tensor(h[1] * np.ones(nx), dim, size),
    }
    results = hll.compute_hll_roe_wave_speed_estimates(
        replica_id,
        replicas,
        primitive_neg,
        primitive_pos,
        dim,
        self.cfg,
        self.physics_models,
    )

    s_l = testing_utils.to_3d_tensor(
        np.ones(nx) * frame_vel, dim, size, as_tf_tensor=False
    )
    s_r = testing_utils.to_3d_tensor(
        np.ones(nx) * (roe_c + roe_u),
        dim,
        size,
        as_tf_tensor=False,
    )

    with self.subTest(name='s_l'):
      self.assertAllClose(s_l, self.evaluate(results[0]), rtol=types.SMALL)

    with self.subTest(name='s_r'):
      self.assertAllClose(s_r, self.evaluate(results[1]), rtol=types.SMALL)

  @parameterized.parameters(*_DIMS)
  def test_compute_hll_roe_wave_speed_estimates_strong_shock_right(self, dim):
    """Tests the wave speed against a specified right running shock."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]

    # Compute the post shock state from pre shock and Mach number.
    m = 6.0
    t0 = 300.0
    p0 = 101325.0
    u0 = -m * np.sqrt(constant.GAMMA * constant.R * t0)
    p1_p0, r1_r0, t1_t0 = self.normal_shock_relations(m)
    u1_u0 = 1 / r1_r0
    frame_vel = -500.0
    u = np.array([u1_u0, 1.0]) * u0 + frame_vel
    v = 100.0 * np.ones(2)
    w = -200.0 * np.ones(2)
    p = np.array([p1_p0, 1.0]) * p0
    t = np.array([t1_t0, 1.0]) * t0
    rho = p / (constant.R * t)
    h = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)

    def roe_avg(rho, f):
      rho_s = np.sqrt(rho)
      roe_avg_f = np.sum(rho_s * f) / np.sum(rho_s)
      return roe_avg_f

    roe_u = roe_avg(rho, u)
    roe_v = roe_avg(rho, v)
    roe_w = roe_avg(rho, w)
    roe_h = roe_avg(rho, h)
    roe_h_int = roe_h - 0.5 * (roe_u**2 + roe_v**2 + roe_w**2)
    roe_c = np.sqrt(constant.GAMMA * constant.R * roe_h_int / constant.CP)
    # Rotate the velocity components depending on `dim` to align shock
    u_field = u
    v_field = v
    w_field = w
    if dim == 'y':
      u_field = v
      v_field = u
    elif dim == 'z':
      u_field = w
      w_field = u
    primitive_neg = {
        types.RHO: testing_utils.to_3d_tensor(rho[0] * np.ones(nx), dim, size),
        types.U: testing_utils.to_3d_tensor(
            u_field[0] * np.ones(nx), dim, size
        ),
        types.V: testing_utils.to_3d_tensor(
            v_field[0] * np.ones(nx), dim, size
        ),
        types.W: testing_utils.to_3d_tensor(
            w_field[0] * np.ones(nx), dim, size
        ),
        types.P: testing_utils.to_3d_tensor(p[0] * np.ones(nx), dim, size),
        types.H: testing_utils.to_3d_tensor(h[0] * np.ones(nx), dim, size),
    }
    primitive_pos = {
        types.RHO: testing_utils.to_3d_tensor(rho[1] * np.ones(nx), dim, size),
        types.U: testing_utils.to_3d_tensor(
            u_field[1] * np.ones(nx), dim, size
        ),
        types.V: testing_utils.to_3d_tensor(
            v_field[1] * np.ones(nx), dim, size
        ),
        types.W: testing_utils.to_3d_tensor(
            w_field[1] * np.ones(nx), dim, size
        ),
        types.P: testing_utils.to_3d_tensor(p[1] * np.ones(nx), dim, size),
        types.H: testing_utils.to_3d_tensor(h[1] * np.ones(nx), dim, size),
    }
    results = hll.compute_hll_roe_wave_speed_estimates(
        replica_id,
        replicas,
        primitive_neg,
        primitive_pos,
        dim,
        self.cfg,
        self.physics_models,
    )

    s_l = testing_utils.to_3d_tensor(
        np.ones(nx) * (roe_u - roe_c),
        dim,
        size,
        as_tf_tensor=False,
    )
    s_r = testing_utils.to_3d_tensor(
        np.ones(nx) * frame_vel, dim, size, as_tf_tensor=False
    )

    with self.subTest(name='s_l'):
      self.assertAllClose(s_l, self.evaluate(results[0]), rtol=types.SMALL)

    with self.subTest(name='s_r'):
      self.assertAllClose(s_r, self.evaluate(results[1]), rtol=types.SMALL)

  @parameterized.parameters(*_DIMS)
  def test_compute_hll_roe_wave_speed_estimates_heavyside_function(self, dim):
    """Tests the wave speed against the expected for a single shock."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]

    # Compute the post shock state from pre shock and Mach number.
    m = 6.0
    t0 = 300.0
    p0 = 101325.0
    u0 = m * np.sqrt(constant.GAMMA * constant.R * t0)
    p1_p0, r1_r0, t1_t0 = self.normal_shock_relations(m)
    u1_u0 = 1 / r1_r0
    frame_vel = -500.0
    u = np.array([1.0, u1_u0]) * u0 + frame_vel
    v = 100.0 * np.ones(2)
    w = -200.0 * np.ones(2)
    p = np.array([1.0, p1_p0]) * p0
    t = np.array([1.0, t1_t0]) * t0
    rho = p / (constant.R * t)
    h = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    c = np.sqrt(constant.GAMMA * constant.R * t)

    def heavyside_vector(f, nx):
      f_vec = np.zeros(nx + 1)
      i_mid = int((nx + 1) / 2)
      f_vec[:i_mid] = f[0]
      f_vec[i_mid:] = f[1]
      return f_vec

    rho_vec = heavyside_vector(rho, nx)
    u_vec = heavyside_vector(u, nx)
    v_vec = heavyside_vector(v, nx)
    w_vec = heavyside_vector(w, nx)
    p_vec = heavyside_vector(p, nx)
    h_vec = heavyside_vector(h, nx)

    def roe_avg(rho, f):
      rho_s = np.sqrt(rho)
      roe_avg_f = np.sum(rho_s * f) / np.sum(rho_s)
      return roe_avg_f

    roe_u = roe_avg(rho, u)
    roe_v = roe_avg(rho, v)
    roe_w = roe_avg(rho, w)
    roe_h = roe_avg(rho, h)
    roe_h_int = roe_h - 0.5 * (roe_u**2 + roe_v**2 + roe_w**2)
    roe_c = np.sqrt(constant.GAMMA * constant.R * roe_h_int / constant.CP)

    # Rotate the velocity components depending on `dim` to align shock
    u_field = u_vec
    v_field = v_vec
    w_field = w_vec
    if dim == 'y':
      u_field = v_vec
      v_field = u_vec
    elif dim == 'z':
      u_field = w_vec
      w_field = u_vec

    primitive_neg = {
        types.RHO: testing_utils.to_3d_tensor(rho_vec[:-1], dim, size),
        types.U: testing_utils.to_3d_tensor(u_field[:-1], dim, size),
        types.V: testing_utils.to_3d_tensor(v_field[:-1], dim, size),
        types.W: testing_utils.to_3d_tensor(w_field[:-1], dim, size),
        types.P: testing_utils.to_3d_tensor(p_vec[:-1], dim, size),
        types.H: testing_utils.to_3d_tensor(h_vec[:-1], dim, size),
    }
    primitive_pos = {
        types.RHO: testing_utils.to_3d_tensor(rho_vec[1:], dim, size),
        types.U: testing_utils.to_3d_tensor(u_field[1:], dim, size),
        types.V: testing_utils.to_3d_tensor(v_field[1:], dim, size),
        types.W: testing_utils.to_3d_tensor(w_field[1:], dim, size),
        types.P: testing_utils.to_3d_tensor(p_vec[1:], dim, size),
        types.H: testing_utils.to_3d_tensor(h_vec[1:], dim, size),
    }
    results = hll.compute_hll_roe_wave_speed_estimates(
        replica_id,
        replicas,
        primitive_neg,
        primitive_pos,
        dim,
        self.cfg,
        self.physics_models,
    )

    s_l_vec = np.zeros(nx)
    s_l_vec[:7] = u[0] - c[0]
    s_l_vec[8:] = u[1] - c[1]
    s_l_vec[7] = frame_vel
    s_l = testing_utils.to_3d_tensor(s_l_vec, dim, size, as_tf_tensor=False)

    s_r_vec = np.zeros(nx)
    s_r_vec[:7] = u[0] + c[0]
    s_r_vec[8:] = u[1] + c[1]
    s_r_vec[7] = roe_c + roe_u
    s_r = testing_utils.to_3d_tensor(s_r_vec, dim, size, as_tf_tensor=False)

    with self.subTest(name='s_l'):
      self.assertAllClose(s_l, self.evaluate(results[0]), rtol=types.SMALL)

    with self.subTest(name='s_r'):
      self.assertAllClose(s_r, self.evaluate(results[1]), rtol=types.SMALL)

  @parameterized.parameters(*_DIMS)
  def test_hll_convective_flux_correct_quiescent_gas(self, dim):
    """Checks that the convective flux only pressure for quiescent gas."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    nx = 16
    size = [nx, nx, nx]
    u = np.zeros(nx)
    v = np.zeros(nx)
    w = np.zeros(nx)
    p = 101325.0 * np.ones(nx)
    t = 300.0 * np.ones(nx)
    rho = p / (constant.R * t)
    ke = 0.5 * (u**2 + v**2 + w**2)
    e = ke + constant.CV * t
    h = ke + constant.CP * t
    y = np.ones(nx)
    p_eff = np.zeros(3)
    match dim:
      case 'x':
        p_eff[0] = p[0]
      case 'y':
        p_eff[1] = p[1]
      case _:  # case 'z'
        p_eff[2] = p[2]

    conservative_neg = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
        'rho_y': testing_utils.to_3d_tensor(rho * y, dim, size),
    }
    conservative_pos = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
        'rho_y': testing_utils.to_3d_tensor(rho * y, dim, size),
    }
    primitive_neg = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.H: testing_utils.to_3d_tensor(h, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        'y': testing_utils.to_3d_tensor(y, dim, size),
    }
    primitive_pos = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.H: testing_utils.to_3d_tensor(h, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        'y': testing_utils.to_3d_tensor(y, dim, size),
    }
    flux_pos = {
        types.RHO: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.zeros(nx) + p_eff[0], dim, size
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.zeros(nx) + p_eff[1], dim, size
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.zeros(nx) + p_eff[2], dim, size
        ),
        types.RHO_E: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        'rho_y': testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
    }
    results = self.evaluate(
        hll.hll_convective_flux(
            replica_id,
            replicas,
            conservative_neg,
            conservative_pos,
            primitive_neg,
            primitive_pos,
            {},
            {},
            _compute_convective_flux,
            dim,
            self.cfg,
            self.physics_models,
        )
    )

    expected = self.evaluate(flux_pos)
    with self.subTest(name='keys'):
      self.assertSequenceEqual(expected.keys(), results.keys())

    for key, val in expected.items():
      with self.subTest(name=key):
        self.assertAllClose(val, results[key])

  @parameterized.parameters(*_DIMS)
  def test_hll_convective_flux_correct_plug_flow(self, dim):
    """Checks that the convective flux is correct for plug flow."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    u = np.ones(nx) * 10.0
    v = np.ones(nx) * -15.0
    w = np.ones(nx) * 25.0
    p = 101325.0 * np.ones(nx)
    p_eff = np.zeros(3)
    match dim:
      case 'x':
        v_conv = u[0]
        p_eff[0] = p[0]
      case 'y':
        v_conv = v[0]
        p_eff[1] = p[0]
      case _:  # case 'z'
        v_conv = w[0]
        p_eff[2] = p[0]
    t = np.linspace(300.0, 3000.0, nx)
    rho = p / (constant.R * t)
    ke = 0.5 * (u**2 + v**2 + w**2)
    e = ke + constant.CV * t
    h = ke + constant.CP * t
    y = np.linspace(0.0, 1.0, nx)

    conservative_neg = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
        'rho_y': testing_utils.to_3d_tensor(rho * y, dim, size),
    }
    conservative_pos = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
        'rho_y': testing_utils.to_3d_tensor(rho * y, dim, size),
    }
    primitive_neg = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.H: testing_utils.to_3d_tensor(h, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        'y': testing_utils.to_3d_tensor(y, dim, size),
    }
    primitive_pos = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.H: testing_utils.to_3d_tensor(h, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        'y': testing_utils.to_3d_tensor(y, dim, size),
    }
    flux_pos = {
        types.RHO: testing_utils.to_3d_tensor(rho * v_conv, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(
            rho * u * v_conv + p_eff[0], dim, size
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rho * v * v_conv + p_eff[1], dim, size
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rho * w * v_conv + p_eff[2], dim, size
        ),
        types.RHO_E: testing_utils.to_3d_tensor(rho * h * v_conv, dim, size),
        'rho_y': testing_utils.to_3d_tensor(rho * y * v_conv, dim, size),
    }
    results = self.evaluate(
        hll.hll_convective_flux(
            replica_id,
            replicas,
            conservative_neg,
            conservative_pos,
            primitive_neg,
            primitive_pos,
            {},
            {},
            _compute_convective_flux,
            dim,
            self.cfg,
            self.physics_models,
        )
    )

    expected = self.evaluate(flux_pos)
    with self.subTest(name='keys'):
      self.assertSequenceEqual(expected.keys(), results.keys())

    for key, val in expected.items():
      with self.subTest(name=key):
        self.assertAllClose(val, results[key])

  @parameterized.parameters(*_DIMS)
  def test_hll_convective_flux_correct_s_l_positive(self, dim):
    """Checks that the convective flux is correct when s_l > 0."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    self.cfg.conservative_variable_names = list(types.BASE_CONSERVATIVE)
    nx = 16
    size = [nx, nx, nx]

    # To ensure s_r > s_l > 0.0, we consider a left running shock in crossflow
    # faster than the shock speed, thus the shock moves right in the
    # computational frame.
    m = 6.0
    t0 = 300.0
    p0 = 101325.0
    u0 = m * np.sqrt(constant.GAMMA * constant.R * t0)
    p1_p0, r1_r0, t1_t0 = self.normal_shock_relations(m)
    u1_u0 = 1 / r1_r0
    frame_vel = u0 * 0.5
    u = 250.0 * np.ones(2)
    v = 100.0 * np.ones(2)
    w = -200.0 * np.ones(2)
    p = np.array([1.0, p1_p0]) * p0
    p_eff = np.zeros([3, 2])
    match dim:
      case 'x':
        u = np.array([1.0, u1_u0]) * u0 + frame_vel
        v_conv = u
        p_eff[0, :] = p
      case 'y':
        v = np.array([1.0, u1_u0]) * u0 + frame_vel
        v_conv = v
        p_eff[1, :] = p
      case _:  # case 'z'
        w = np.array([1.0, u1_u0]) * u0 + frame_vel
        v_conv = w
        p_eff[2, :] = p
    t = np.array([1.0, t1_t0]) * t0
    rho = p / (constant.R * t)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    h = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)

    conservative_neg = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[0], dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * u[0], dim, size
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * v[0], dim, size
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * w[0], dim, size
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * e[0], dim, size
        ),
    }
    conservative_pos = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[1], dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * u[1], dim, size
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * v[1], dim, size
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * w[1], dim, size
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * e[1], dim, size
        ),
    }
    primitive_neg = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[0], dim, size),
        types.U: testing_utils.to_3d_tensor(np.ones(nx) * u[0], dim, size),
        types.V: testing_utils.to_3d_tensor(np.ones(nx) * v[0], dim, size),
        types.W: testing_utils.to_3d_tensor(np.ones(nx) * w[0], dim, size),
        types.H: testing_utils.to_3d_tensor(np.ones(nx) * h[0], dim, size),
        types.P: testing_utils.to_3d_tensor(np.ones(nx) * p[0], dim, size),
    }
    primitive_pos = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[1], dim, size),
        types.U: testing_utils.to_3d_tensor(np.ones(nx) * u[1], dim, size),
        types.V: testing_utils.to_3d_tensor(np.ones(nx) * v[1], dim, size),
        types.W: testing_utils.to_3d_tensor(np.ones(nx) * w[1], dim, size),
        types.H: testing_utils.to_3d_tensor(np.ones(nx) * h[1], dim, size),
        types.P: testing_utils.to_3d_tensor(np.ones(nx) * p[1], dim, size),
    }
    flux_neg = {
        types.RHO: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * v_conv[0], dim, size
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * u[0] * v_conv[0] + p_eff[0, 0], dim, size
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * v[0] * v_conv[0] + p_eff[1, 0], dim, size
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * w[0] * v_conv[0] + p_eff[2, 0], dim, size
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * h[0] * v_conv[0], dim, size
        ),
    }
    results = self.evaluate(
        hll.hll_convective_flux(
            replica_id,
            replicas,
            conservative_neg,
            conservative_pos,
            primitive_neg,
            primitive_pos,
            {},
            {},
            _compute_convective_flux,
            dim,
            self.cfg,
            self.physics_models,
        )
    )

    # For s_r > s_l > 0.0, t1 = 0.0, t3 = 0.0, t2 = 1.0
    # Therefore, expected is flux_neg
    expected = self.evaluate(flux_neg)

    with self.subTest(name='keys'):
      self.assertSequenceEqual(expected.keys(), results.keys())

    for key, val in expected.items():
      with self.subTest(name=key):
        self.assertAllClose(val, results[key])

  @parameterized.parameters(*_DIMS)
  def test_hll_convective_flux_correct_s_r_negative(self, dim):
    """Checks that the convective flux is correct when s_r < 0."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    self.cfg.conservative_variable_names = list(types.BASE_CONSERVATIVE)
    nx = 16
    size = [nx, nx, nx]

    # To ensure s_l < s_r < 0.0, we consider a right running shock in crossflow
    # faster than the shock speed, thus the shock moves left in the
    # computational frame.
    m = 6.0
    t0 = 300.0
    p0 = 101325.0
    u0 = -m * np.sqrt(constant.GAMMA * constant.R * t0)
    p1_p0, r1_r0, t1_t0 = self.normal_shock_relations(m)
    u1_u0 = 1 / r1_r0
    frame_vel = u0 * 0.5
    u = 250.0 * np.ones(2)
    v = 100.0 * np.ones(2)
    w = -200.0 * np.ones(2)
    p = np.array([p1_p0, 1.0]) * p0
    p_eff = np.zeros([3, 2])
    match dim:
      case 'x':
        u = np.array([u1_u0, 1.0]) * u0 + frame_vel
        v_conv = u
        p_eff[0, :] = p
      case 'y':
        v = np.array([u1_u0, 1.0]) * u0 + frame_vel
        v_conv = v
        p_eff[1, :] = p
      case _:  # case 'z'
        w = np.array([u1_u0, 1.0]) * u0 + frame_vel
        v_conv = w
        p_eff[2, :] = p
    t = np.array([t1_t0, 1.0]) * t0
    rho = p / (constant.R * t)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    h = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)

    conservative_neg = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[0], dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * u[0], dim, size
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * v[0], dim, size
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * w[0], dim, size
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * e[0], dim, size
        ),
    }
    conservative_pos = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[1], dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * u[1], dim, size
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * v[1], dim, size
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * w[1], dim, size
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * e[1], dim, size
        ),
    }
    primitive_neg = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[0], dim, size),
        types.U: testing_utils.to_3d_tensor(np.ones(nx) * u[0], dim, size),
        types.V: testing_utils.to_3d_tensor(np.ones(nx) * v[0], dim, size),
        types.W: testing_utils.to_3d_tensor(np.ones(nx) * w[0], dim, size),
        types.H: testing_utils.to_3d_tensor(np.ones(nx) * h[0], dim, size),
        types.P: testing_utils.to_3d_tensor(np.ones(nx) * p[0], dim, size),
    }
    primitive_pos = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[1], dim, size),
        types.U: testing_utils.to_3d_tensor(np.ones(nx) * u[1], dim, size),
        types.V: testing_utils.to_3d_tensor(np.ones(nx) * v[1], dim, size),
        types.W: testing_utils.to_3d_tensor(np.ones(nx) * w[1], dim, size),
        types.H: testing_utils.to_3d_tensor(np.ones(nx) * h[1], dim, size),
        types.P: testing_utils.to_3d_tensor(np.ones(nx) * p[1], dim, size),
    }
    flux_pos = {
        types.RHO: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * v_conv[1], dim, size
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * u[1] * v_conv[1] + p_eff[0, 1], dim, size
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * v[1] * v_conv[1] + p_eff[1, 1], dim, size
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * w[1] * v_conv[1] + p_eff[2, 1], dim, size
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * h[1] * v_conv[1], dim, size
        ),
    }
    results = self.evaluate(
        hll.hll_convective_flux(
            replica_id,
            replicas,
            conservative_neg,
            conservative_pos,
            primitive_neg,
            primitive_pos,
            {},
            {},
            _compute_convective_flux,
            dim,
            self.cfg,
            self.physics_models,
        )
    )

    # For s_l < s_r < 0.0, t1 = 1.0, t3 = 0.0, t2 = 0.0
    # Therefore, expected is flux_pos
    expected = self.evaluate(flux_pos)

    with self.subTest(name='keys'):
      self.assertSequenceEqual(expected.keys(), results.keys())

    for key, val in expected.items():
      with self.subTest(name=key):
        self.assertAllClose(val, results[key])

  @parameterized.parameters(*_DIMS)
  def test_hll_convective_flux_correct_star_region(self, dim):
    """Checks that the convective flux is correct when s_l < 0 and s_r > 0."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    self.cfg.conservative_variable_names = list(types.BASE_CONSERVATIVE)
    nx = 16
    size = [nx, nx, nx]

    # To ensure s_r > 0.0 and s_l < 0.0, we consider a left running shock in
    # crossflow less than the shock speed.
    m = 6.0
    t0 = 300.0
    p0 = 101325.0
    u0 = m * np.sqrt(constant.GAMMA * constant.R * t0)
    p1_p0, r1_r0, t1_t0 = self.normal_shock_relations(m)
    u1_u0 = 1 / r1_r0
    frame_vel = -u0 * 0.40
    u = 250.0 * np.ones(2)
    v = 100.0 * np.ones(2)
    w = -200.0 * np.ones(2)
    p = np.array([1.0, p1_p0]) * p0
    p_eff = np.zeros([3, 2])
    match dim:
      case 'x':
        u = np.array([1.0, u1_u0]) * u0 + frame_vel
        v_conv = u
        p_eff[0, :] = p
      case 'y':
        v = np.array([1.0, u1_u0]) * u0 + frame_vel
        v_conv = v
        p_eff[1, :] = p
      case _:  # case 'z'
        w = np.array([1.0, u1_u0]) * u0 + frame_vel
        v_conv = w
        p_eff[2, :] = p
    t = np.array([1.0, t1_t0]) * t0
    rho = p / (constant.R * t)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    h = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)

    conservative_neg = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[0], dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * u[0], dim, size
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * v[0], dim, size
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * w[0], dim, size
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * e[0], dim, size
        ),
    }
    conservative_pos = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[1], dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * u[1], dim, size
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * v[1], dim, size
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * w[1], dim, size
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * e[1], dim, size
        ),
    }
    primitive_neg = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[0], dim, size),
        types.U: testing_utils.to_3d_tensor(np.ones(nx) * u[0], dim, size),
        types.V: testing_utils.to_3d_tensor(np.ones(nx) * v[0], dim, size),
        types.W: testing_utils.to_3d_tensor(np.ones(nx) * w[0], dim, size),
        types.H: testing_utils.to_3d_tensor(np.ones(nx) * h[0], dim, size),
        types.P: testing_utils.to_3d_tensor(np.ones(nx) * p[0], dim, size),
    }
    primitive_pos = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[1], dim, size),
        types.U: testing_utils.to_3d_tensor(np.ones(nx) * u[1], dim, size),
        types.V: testing_utils.to_3d_tensor(np.ones(nx) * v[1], dim, size),
        types.W: testing_utils.to_3d_tensor(np.ones(nx) * w[1], dim, size),
        types.H: testing_utils.to_3d_tensor(np.ones(nx) * h[1], dim, size),
        types.P: testing_utils.to_3d_tensor(np.ones(nx) * p[1], dim, size),
    }
    flux_pos = {
        types.RHO: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * v_conv[1], dim, size
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * u[1] * v_conv[1] + p_eff[0, 1], dim, size
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * v[1] * v_conv[1] + p_eff[1, 1], dim, size
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * w[1] * v_conv[1] + p_eff[2, 1], dim, size
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * h[1] * v_conv[1], dim, size
        ),
    }
    results = self.evaluate(
        hll.hll_convective_flux(
            replica_id,
            replicas,
            conservative_neg,
            conservative_pos,
            primitive_neg,
            primitive_pos,
            {},
            {},
            _compute_convective_flux,
            dim,
            self.cfg,
            self.physics_models,
        )
    )

    # Because the initial conditions yield a psuedo-steady shock, the right wave
    # is an acousitic wave, and does not perturb the flow in the star region.
    # Thus the expected flux is the positive flux.
    expected = self.evaluate(flux_pos)
    with self.subTest(name='keys'):
      self.assertSequenceEqual(expected.keys(), results.keys())

    for key, val in expected.items():
      with self.subTest(name=key):
        self.assertAllClose(val, results[key], rtol=types.SMALL)

  @parameterized.parameters(*_DIMS)
  def test_hll_convective_flux_correct_shock_reflection(self, dim):
    """Checks that the convective flux is correct for shock reflection."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    self.cfg.conservative_variable_names = list(types.BASE_CONSERVATIVE)
    nx = 16
    size = [nx, nx, nx]

    # Set the initial conditions as the reflection of two Mach 2.5 shocks.
    m = 2.5
    t0 = 300.0
    p0 = 101325.0
    u0 = m * np.sqrt(constant.GAMMA * constant.R * t0)
    p1_p0, r1_r0, t1_t0 = self.normal_shock_relations(m)
    u1_u0 = 1 / r1_r0
    u = 250.0 * np.ones(2)
    v = 100.0 * np.ones(2)
    w = -200.0 * np.ones(2)
    p = np.array([1.0, 1.0]) * p1_p0 * p0
    p_eff = np.zeros([3, 2])
    match dim:
      case 'x':
        u = np.array([1.0, -1.0]) * (1.0 - u1_u0) * u0
        v_conv = u
        p_eff[0, :] = p
      case 'y':
        v = np.array([1.0, -1.0]) * (1.0 - u1_u0) * u0
        v_conv = v
        p_eff[1, :] = p
      case _:  # case 'z'
        w = np.array([1.0, -1.0]) * (1.0 - u1_u0) * u0
        v_conv = w
        p_eff[2, :] = p
    t = np.array([1.0, 1.0]) * t1_t0 * t0
    rho = p / (constant.R * t)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    h = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    conservative_neg = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[0], dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * u[0], dim, size
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * v[0], dim, size
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * w[0], dim, size
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[0] * e[0], dim, size
        ),
    }
    conservative_pos = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[1], dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * u[1], dim, size
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * v[1], dim, size
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * w[1], dim, size
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx) * rho[1] * e[1], dim, size
        ),
    }
    primitive_neg = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[0], dim, size),
        types.U: testing_utils.to_3d_tensor(np.ones(nx) * u[0], dim, size),
        types.V: testing_utils.to_3d_tensor(np.ones(nx) * v[0], dim, size),
        types.W: testing_utils.to_3d_tensor(np.ones(nx) * w[0], dim, size),
        types.H: testing_utils.to_3d_tensor(np.ones(nx) * h[0], dim, size),
        types.P: testing_utils.to_3d_tensor(np.ones(nx) * p[0], dim, size),
    }
    primitive_pos = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx) * rho[1], dim, size),
        types.U: testing_utils.to_3d_tensor(np.ones(nx) * u[1], dim, size),
        types.V: testing_utils.to_3d_tensor(np.ones(nx) * v[1], dim, size),
        types.W: testing_utils.to_3d_tensor(np.ones(nx) * w[1], dim, size),
        types.H: testing_utils.to_3d_tensor(np.ones(nx) * h[1], dim, size),
        types.P: testing_utils.to_3d_tensor(np.ones(nx) * p[1], dim, size),
    }

    results = self.evaluate(
        hll.hll_convective_flux(
            replica_id,
            replicas,
            conservative_neg,
            conservative_pos,
            primitive_neg,
            primitive_pos,
            {},
            {},
            _compute_convective_flux,
            dim,
            self.cfg,
            self.physics_models,
        )
    )

    # Compute flux based on reflected shock.
    roe_h = constant.CP * t[0] + 0.5 * v_conv[0] ** 2
    roe_c = np.sqrt(constant.GAMMA * constant.R * roe_h / constant.CP)
    s = roe_c

    # For the shock reflection, the flux is zero for everything except the
    # momentum aligned with `dim`.
    expected = {
        types.RHO: testing_utils.to_3d_tensor(
            np.zeros(nx), dim, size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.zeros(nx), dim, size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            np.zeros(nx), dim, size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            np.zeros(nx), dim, size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.zeros(nx), dim, size, as_tf_tensor=False
        ),
    }
    match dim:
      case 'x':
        expected.update(
            {
                types.RHO_U: testing_utils.to_3d_tensor(
                    (rho[0] * v_conv[0] ** 2 + p[0] + rho[0] * v_conv[0] * s)
                    * np.ones(nx),
                    dim,
                    size,
                    as_tf_tensor=False,
                )
            }
        )
      case 'y':
        expected.update(
            {
                types.RHO_V: testing_utils.to_3d_tensor(
                    (rho[0] * v_conv[0] ** 2 + p[0] + rho[0] * v_conv[0] * s)
                    * np.ones(nx),
                    dim,
                    size,
                    as_tf_tensor=False,
                )
            }
        )
      case _:  # case 'z'
        expected.update(
            {
                types.RHO_W: testing_utils.to_3d_tensor(
                    (rho[0] * v_conv[0] ** 2 + p[0] + rho[0] * v_conv[0] * s)
                    * np.ones(nx),
                    dim,
                    size,
                    as_tf_tensor=False,
                )
            }
        )

    with self.subTest(name='keys'):
      self.assertSequenceEqual(expected.keys(), results.keys())

    for key, val in expected.items():
      with self.subTest(name=key):
        self.assertAllClose(val, results[key], rtol=types.SMALL)


if __name__ == '__main__':
  tf.test.main()
