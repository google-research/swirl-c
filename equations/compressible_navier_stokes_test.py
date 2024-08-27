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
"""Tests for compressible_navier_stokes.py."""

from unittest import mock
from absl.testing import parameterized
import numpy as np
from swirl_c.boundary import bc_types
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.equations import compressible_navier_stokes
from swirl_c.numerics import kernel_op_types
from swirl_c.physics import constant
from swirl_c.physics import physics_models as physics_models_lib
import tensorflow as tf


def _mock_gradient_function_returns_zeros(f, h, dim, cfg):
  """Mocks the gradient function and returns zeros."""
  del h, dim, cfg  # Unused.
  return tf.nest.map_structure(tf.zeros_like, f)


def _mock_interpolation_function_different(v, dim, k):
  """Mocks WENO interpolation and returns values where pos != neg."""
  del k  # Unused.
  match dim:
    case 'x':
      f_neg = 0.9
      f_pos = 1.1
    case 'y':
      f_neg = 0.8
      f_pos = 1.2
    case _:  # 'z'
      f_neg = 0.7
      f_pos = 1.3
  neg = tf.nest.map_structure(lambda v: v * f_neg, v)
  pos = tf.nest.map_structure(lambda v: v * f_pos, v)
  return neg, pos


def _mock_interpolation_function(v, dim, k):
  """Mocks WENO interpolation and returns node values."""
  # Here, we mock the WENO interpolation to avoid issue with floating point
  # error in WENO interpolation. See b/302341111 for more details.
  del dim, k  # Unused.
  return v, v


def _mock_boundary_function(
    replica_id, replicas, conservative, cfg
):
  """Mocks boundary and leave halo unchanged."""
  del replica_id, replicas, cfg  # Unused.
  return conservative


def _mock_intercell_flux_function_arbitrary(
    replica_id,
    replicas,
    conservative_neg,
    conservative_pos,
    primitive_neg,
    primative_pos,
    helper_var_neg,
    helper_var_pos,
    cfg,
    physics_models,
):
  """Mocks the intercell flux function."""
  del (
      replica_id,
      replicas,
      conservative_neg,
      conservative_pos,
      primitive_neg,
      primative_pos,
      helper_var_neg,
      helper_var_pos,
      cfg,
      physics_models,
  )  # Unused.
  t_x = np.linspace(0.0, 1.0, 16) * 2.0 * np.pi
  flux_x_vec = np.sin(t_x)
  t_y = np.linspace(0.0, 1.0, 16) * 2.0 * np.pi + np.pi / 3.0
  flux_y_vec = np.sin(t_y)
  t_z = np.linspace(0.0, 1.0, 16)
  flux_z_vec = t_z**2.0

  flux_factor = {
      types.RHO: 1.0,
      types.RHO_U: 2.0,
      types.RHO_V: 3.0,
      types.RHO_W: 4.0,
      types.RHO_E: 5.0,
      'rho_y': 6.0,
  }

  flux = {
      'x': {
          var_name: testing_utils.to_3d_tensor(
              flux_x_vec * flux_factor[var_name], 'x', [16, 16, 16]
          )
          for var_name in flux_factor
      },
      'y': {
          var_name: testing_utils.to_3d_tensor(
              flux_y_vec * flux_factor[var_name], 'y', [16, 16, 16]
          )
          for var_name in flux_factor
      },
      'z': {
          var_name: testing_utils.to_3d_tensor(
              flux_z_vec * flux_factor[var_name], 'z', [16, 16, 16]
          )
          for var_name in flux_factor
      },
  }

  return flux


def _mock_intercell_flux_function_return_conservative(
    replica_id,
    replicas,
    conservative_neg,
    conservative_pos,
    primitive_neg,
    primative_pos,
    helper_var_neg,
    helper_var_pos,
    cfg,
    physics_models,
):
  """Mocks the intercell flux function to return conservative values."""
  del (
      replica_id,
      replicas,
      conservative_pos,
      primitive_neg,
      primative_pos,
      helper_var_neg,
      helper_var_pos,
      cfg,
      physics_models,
  )  # Unused.
  flux = {}
  for dim in types.DIMS:
    flux[dim] = {
        var_name: val for var_name, val in conservative_neg['x'].items()
    }
  return flux


def _mock_intercell_flux_function_return_zeros(
    replica_id,
    replicas,
    conservative_neg,
    conservative_pos,
    primitive_neg,
    primative_pos,
    helper_var_neg,
    helper_var_pos,
    cfg,
    physics_models,
):
  """Mocks the intercell flux function to return zeros."""
  del (
      replica_id,
      replicas,
      conservative_pos,
      primitive_neg,
      primative_pos,
      helper_var_neg,
      helper_var_pos,
      cfg,
      physics_models,
  )  # Unused.
  flux = {}
  for dim in types.DIMS:
    flux[dim] = {
        var_name: tf.nest.map_structure(tf.zeros_like, val)
        for var_name, val in conservative_neg['x'].items()
    }
  return flux


def _mock_intercell_diffusive_flux_function_returns_sin_profiles(
    primitive,
    cfg,
    physics_models,
):
  """Mocks the intercell flux function."""
  del primitive, cfg, physics_models
  xx = np.linspace(0.0, 1.0, 16) * 2.0 * np.pi
  flux_x_vec = np.sin(xx)
  yy = np.linspace(0.0, 1.0, 16) * 2.0 * np.pi + np.pi / 3.0
  flux_y_vec = np.sin(yy)
  zz = np.linspace(0.0, 1.0, 16)
  flux_z_vec = np.sin(zz**2.0)

  flux_factor = {
      types.RHO: 1.0,
      types.RHO_U: 2.0,
      types.RHO_V: 3.0,
      types.RHO_W: 4.0,
      types.RHO_E: 5.0,
      'rho_y': 6.0,
  }

  flux = {
      'x': {
          var_name: testing_utils.to_3d_tensor(
              flux_x_vec * flux_factor[var_name], 'x', [16, 16, 16]
          )
          for var_name in flux_factor
      },
      'y': {
          var_name: testing_utils.to_3d_tensor(
              flux_y_vec * flux_factor[var_name], 'y', [16, 16, 16]
          )
          for var_name in flux_factor
      },
      'z': {
          var_name: testing_utils.to_3d_tensor(
              flux_z_vec * flux_factor[var_name], 'z', [16, 16, 16]
          )
          for var_name in flux_factor
      },
  }

  return flux


def _mock_shear_stress(
    primitive,
    primitive_fx,
    primitive_fy,
    primitive_fz,
    cfg,
    physics_models,
):
  """Mocks the shear stress function to return specified tau."""
  del primitive_fx, primitive_fy, primitive_fz, cfg, physics_models
  tau = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
  return [
      [
          tf.nest.map_structure(
              lambda rho: tau[i][j] * tf.ones_like(rho), primitive[types.RHO]  # pylint: disable=cell-var-from-loop
          )
          for j in range(3)
      ]
      for i in range(3)
  ]


class NavierStokesTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes the cfg object."""
    super().setUp()
    self.cfg = parameter.SwirlCParameters({
        'interpolation_scheme': 'WENO_3',
        'numeric_flux_scheme': 'HLL',
        'include_diffusion': False,
        'conservative_variable_names': list(types.BASE_CONSERVATIVE) + [
            'rho_y'
        ],
        'primitive_variable_names': list(types.BASE_PRIMITIVES) + ['y'],
        'kernel_op_type': kernel_op_types.KernelOpType.KERNEL_OP_SLICE.value,
    })
    self.physics_models = physics_models_lib.PhysicsModels(self.cfg)

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
  def test_compute_physical_flux_1d_computes_correctly(self, dim):
    """Confirms that the physical fluxes are correct along each dim."""
    size = [12, 14, 16]
    match dim:
      case 'x':
        nx = size[1]
      case 'y':
        nx = size[2]
      case _:  # case 'z'
        nx = size[0]
    u = np.linspace(-3.0, 12.0, nx)
    v = np.linspace(6.0, -24.0, nx)
    w = np.linspace(-9.0, 36.0, nx)
    p = np.logspace(3.0, 6.0, nx)
    t = np.linspace(250.0, 3000.0, nx)
    y = np.linspace(0.0, 1.0, nx)
    rho = p / (constant.R * t)
    h = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    primitive = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        types.E: testing_utils.to_3d_tensor(e, dim, size),
        types.H: testing_utils.to_3d_tensor(h, dim, size),
        'y': testing_utils.to_3d_tensor(y, dim, size),
    }
    conservative = {
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
    }
    results = {
        var_name: self.evaluate(
            compressible_navier_stokes._compute_physical_flux_1d(
                conservative,
                primitive,
                {},
                dim,
                var_name,
            )
        )
        for var_name in [
            types.RHO,
            types.RHO_U,
            types.RHO_V,
            types.RHO_W,
            types.RHO_E,
            'rho_y',
        ]
    }
    flux_vectors = np.zeros([nx, 6])
    match dim:
      case 'x':
        flux_vectors[:, 0] = rho * u
        flux_vectors[:, 1] = rho * u * u + p
        flux_vectors[:, 2] = rho * v * u
        flux_vectors[:, 3] = rho * w * u
        flux_vectors[:, 4] = rho * (e + p / rho) * u
        flux_vectors[:, 5] = rho * y * u
      case 'y':
        flux_vectors[:, 0] = rho * v
        flux_vectors[:, 1] = rho * u * v
        flux_vectors[:, 2] = rho * v * v + p
        flux_vectors[:, 3] = rho * w * v
        flux_vectors[:, 4] = rho * (e + p / rho) * v
        flux_vectors[:, 5] = rho * y * v
      case _:  # case 'z'
        flux_vectors[:, 0] = rho * w
        flux_vectors[:, 1] = rho * u * w
        flux_vectors[:, 2] = rho * v * w
        flux_vectors[:, 3] = rho * w * w + p
        flux_vectors[:, 4] = rho * (e + p / rho) * w
        flux_vectors[:, 5] = rho * y * w
    expected = {
        types.RHO: testing_utils.to_3d_tensor(flux_vectors[:, 0], dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(
            flux_vectors[:, 1], dim, size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            flux_vectors[:, 2], dim, size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            flux_vectors[:, 3], dim, size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            flux_vectors[:, 4], dim, size, as_tf_tensor=False
        ),
        'rho_y': testing_utils.to_3d_tensor(
            flux_vectors[:, 5], dim, size, as_tf_tensor=False
        ),
    }
    with self.subTest(name='keys'):
      self.assertSequenceEqual(expected.keys(), results.keys())
    for var_name, expected_val in expected.items():
      with self.subTest(name=f'{var_name}'):
        self.assertAllClose(expected_val, results[var_name])

  @parameterized.parameters(*_DIMS)
  def test_intercell_flux_quiescent_gas_returns_zeros(self, dim):
    """Confirms that the intercell flux is only pressure in quiescent gas."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    u = np.zeros(nx)
    v = np.zeros(nx)
    w = np.zeros(nx)
    p = 101325.0 * np.ones(nx)
    t = 500.0 * np.ones(nx)
    rho = p / (constant.R * t)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    h = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    y = np.ones(nx) * 0.5

    conservative = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
        'rho_y': testing_utils.to_3d_tensor(rho * y, dim, size),
    }
    primitive = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.E: testing_utils.to_3d_tensor(e, dim, size),
        'y': testing_utils.to_3d_tensor(y, dim, size),
        types.H: testing_utils.to_3d_tensor(h, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
    }

    results = self.evaluate(
        compressible_navier_stokes._intercell_convective_flux(
            replica_id,
            replicas,
            {dim: conservative for dim in types.DIMS},
            {dim: conservative for dim in types.DIMS},
            {dim: primitive for dim in types.DIMS},
            {dim: primitive for dim in types.DIMS},
            {dim: {} for dim in types.DIMS},
            {dim: {} for dim in types.DIMS},
            self.cfg,
            self.physics_models,
        )
    )

    for flux_dim in types.DIMS:
      with self.subTest(name=f'flux direction:{flux_dim}, keys'):
        self.assertSequenceEqual(conservative.keys(), results[flux_dim].keys())

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
          'rho_y': testing_utils.to_3d_tensor(
              np.zeros(nx), dim, size, as_tf_tensor=False
          ),
      }
      # Add pressure to momentum flux normal to computational cell face.
      expected[
          types.primitive_to_conservative_name(
              types.VELOCITY[types.DIMS.index(flux_dim)]
          )
      ] += testing_utils.to_3d_tensor(p, dim, size, as_tf_tensor=False)
      for var_name, field in expected.items():
        with self.subTest(name=f'flux direction:{flux_dim}, {var_name}'):
          hw = self.cfg.halo_width
          domain_expected = field[
              hw : -(hw - 1), hw : -(hw - 1), hw : -(hw - 1)
          ]
          domain_results = np.array(results[flux_dim][var_name])[
              hw : -(hw - 1), hw : -(hw - 1), hw : -(hw - 1)
          ]
          self.assertAllClose(
              domain_expected,
              domain_results,
          )

  @parameterized.parameters(*_DIMS)
  def test_intercell_flux_plug_flow_euler_computes_correctly(self, dim):
    """Confirm intercell flux calculation for inviscid plug flow."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    u = np.zeros(nx)
    v = np.zeros(nx)
    w = np.zeros(nx)
    match dim:
      case 'x':
        v_conv = 10.0
        u = np.ones(nx) * v_conv
      case 'y':
        v_conv = -25.0
        v = np.ones(nx) * v_conv
      case _:
        v_conv = 15.0
        w = np.ones(nx) * v_conv

    p = 101325.0 * np.ones(nx)
    t = np.linspace(250.0, 3000.0, nx)
    rho = p / (constant.R * t)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    h = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    y = np.linspace(0.0, 1.0, nx)

    conservative = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
        'rho_y': testing_utils.to_3d_tensor(rho * y, dim, size),
    }
    primitive = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.E: testing_utils.to_3d_tensor(e, dim, size),
        'y': testing_utils.to_3d_tensor(y, dim, size),
        types.H: testing_utils.to_3d_tensor(h, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
    }
    results = self.evaluate(
        compressible_navier_stokes._intercell_convective_flux(
            replica_id,
            replicas,
            {dim: conservative for dim in types.DIMS},
            {dim: conservative for dim in types.DIMS},
            {dim: primitive for dim in types.DIMS},
            {dim: primitive for dim in types.DIMS},
            {dim: {} for dim in types.DIMS},
            {dim: {} for dim in types.DIMS},
            self.cfg,
            self.physics_models,
        )
    )

    for flux_dim in types.DIMS:
      with self.subTest(name=f'flux direction:{flux_dim}, keys'):
        self.assertSequenceEqual(conservative.keys(), results[flux_dim].keys())

      if flux_dim == dim:
        # If flux is aligned with the convective velocity, then plug flow.
        expected = {
            types.RHO: testing_utils.to_3d_tensor(
                rho * v_conv, dim, size, as_tf_tensor=False
            ),
            types.RHO_U: testing_utils.to_3d_tensor(
                rho * u * v_conv, dim, size, as_tf_tensor=False
            ),
            types.RHO_V: testing_utils.to_3d_tensor(
                rho * v * v_conv, dim, size, as_tf_tensor=False
            ),
            types.RHO_W: testing_utils.to_3d_tensor(
                rho * w * v_conv, dim, size, as_tf_tensor=False
            ),
            types.RHO_E: testing_utils.to_3d_tensor(
                rho * h * v_conv, dim, size, as_tf_tensor=False
            ),
            'rho_y': testing_utils.to_3d_tensor(
                rho * y * v_conv, dim, size, as_tf_tensor=False
            ),
        }
      else:  # flux_dim != dim
        # Otherwise, no convective flux.
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
            'rho_y': testing_utils.to_3d_tensor(
                np.zeros(nx), dim, size, as_tf_tensor=False
            ),
        }
      # Add pressure to momentum flux normal to computational cell face.
      expected[
          types.primitive_to_conservative_name(
              types.VELOCITY[types.DIMS.index(flux_dim)]
          )
      ] += testing_utils.to_3d_tensor(p, dim, size, as_tf_tensor=False)
      for var_name, field in expected.items():
        with self.subTest(name=f'flux direction:{flux_dim}, {var_name}'):
          hw = self.cfg.halo_width
          domain_expected = field[
              hw : -(hw - 1), hw : -(hw - 1), hw : -(hw - 1)
          ]
          domain_results = np.array(results[flux_dim][var_name])[
              hw : -(hw - 1), hw : -(hw - 1), hw : -(hw - 1)
          ]
          self.assertAllClose(
              domain_expected,
              domain_results,
          )

  @parameterized.parameters(*_DIMS)
  def test_intercell_flux_standing_shock_computes_correctly(self, dim):
    """Confirm intercell flux calculation for a standing shock."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]

    m = 6.0
    t0 = 300.0
    p0 = 101325.0
    rho0 = p0 / (constant.R * t0)
    u0 = m * np.sqrt(constant.GAMMA * constant.R * t0)
    h0 = constant.CP * t0 + 0.5 * u0**2

    _, r1_r0, t1_t0 = self.normal_shock_relations(m)
    u1_u0 = 1 / r1_r0

    def heavyside_vector(f, nx):
      """Maps heavyside function to a vector."""
      f_vec = np.zeros(nx)
      i_mid = int((nx) / 2)
      f_vec[:i_mid] = f[0]
      f_vec[i_mid:] = f[1]
      return f_vec

    rho = heavyside_vector(np.array([1.0, r1_r0]) * rho0, nx)
    t = heavyside_vector(np.array([1.0, t1_t0]) * t0, nx)
    p = rho * constant.R * t
    u = np.zeros(nx)
    v = np.zeros(nx)
    w = np.zeros(nx)
    match dim:
      case 'x':
        u = heavyside_vector(np.array([1.0, u1_u0]) * u0, nx)
      case 'y':
        v = heavyside_vector(np.array([1.0, u1_u0]) * u0, nx)
      case _:
        w = heavyside_vector(np.array([1.0, u1_u0]) * u0, nx)

    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    h = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    y = np.ones(nx) * 0.5

    conservative = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
        'rho_y': testing_utils.to_3d_tensor(rho * y, dim, size),
    }
    primitive = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.E: testing_utils.to_3d_tensor(e, dim, size),
        'y': testing_utils.to_3d_tensor(y, dim, size),
        types.H: testing_utils.to_3d_tensor(h, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
    }
    results = self.evaluate(
        compressible_navier_stokes._intercell_convective_flux(
            replica_id,
            replicas,
            {dim: conservative for dim in types.DIMS},
            {dim: conservative for dim in types.DIMS},
            {dim: primitive for dim in types.DIMS},
            {dim: primitive for dim in types.DIMS},
            {dim: {} for dim in types.DIMS},
            {dim: {} for dim in types.DIMS},
            self.cfg,
            self.physics_models,
        )
    )

    for flux_dim in types.DIMS:
      with self.subTest(name=f'flux direction:{flux_dim}, keys'):
        self.assertSequenceEqual(conservative.keys(), results[flux_dim].keys())

      if flux_dim == dim:
        # If flux is aligned with the convective velocity, then steady flow.
        expected = {
            types.RHO: testing_utils.to_3d_tensor(
                np.ones(nx) * rho0 * u0, dim, size, as_tf_tensor=False
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
                np.ones(nx) * rho0 * h0 * u0, dim, size, as_tf_tensor=False
            ),
            'rho_y': testing_utils.to_3d_tensor(
                np.ones(nx) * rho0 * y * u0, dim, size, as_tf_tensor=False
            ),
        }
        # Add momentum aligned with convective flow.
        expected[
            types.primitive_to_conservative_name(
                types.VELOCITY[types.DIMS.index(flux_dim)]
            )
        ] += testing_utils.to_3d_tensor(
            np.ones(nx) * (rho0 * u0**2 + p0), dim, size, as_tf_tensor=False
        )
      else:  # flux_dim != dim
        # Otherwise, no convective flux.
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
            'rho_y': testing_utils.to_3d_tensor(
                np.zeros(nx), dim, size, as_tf_tensor=False
            ),
        }
        # Add pressure to momentum flux normal to computational cell face.
        expected[
            types.primitive_to_conservative_name(
                types.VELOCITY[types.DIMS.index(flux_dim)]
            )
        ] += testing_utils.to_3d_tensor(p, dim, size, as_tf_tensor=False)
      for var_name, field in expected.items():
        with self.subTest(name=f'flux direction:{flux_dim}, {var_name}'):
          hw = self.cfg.halo_width
          domain_expected = field[
              hw : -(hw - 1), hw : -(hw - 1), hw : -(hw - 1)
          ]
          domain_results = np.array(results[flux_dim][var_name])[
              hw : -(hw - 1), hw : -(hw - 1), hw : -(hw - 1)
          ]
          self.assertAllClose(
              domain_expected,
              domain_results,
          )

  @mock.patch(
      'swirl_lm.numerics.interpolation.weno',
      _mock_interpolation_function_different,
  )
  def test_face_interpolation_computes_conservative_variables_correctly(self):
    """Confirm that the face interpolation returns expected values."""
    nx = 16
    size = [nx, nx, nx]
    dim = 'x'

    rho = np.ones(nx)
    t = np.ones(nx) * 300.0
    u = np.ones(nx) * 10.0
    v = np.ones(nx) * -15.0
    w = np.ones(nx) * 25.0

    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    y = np.ones(nx) * 0.5

    conservative = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
        'rho_y': testing_utils.to_3d_tensor(rho * y, dim, size),
    }

    results = self.evaluate(
        compressible_navier_stokes._face_interpolation(
            conservative,
            self.cfg,
        )
    )

    before_interp = {
        types.RHO: rho,
        types.RHO_U: rho * u,
        types.RHO_V: rho * v,
        types.RHO_W: rho * w,
        types.RHO_E: rho * e,
        'rho_y': rho * y,
    }
    for dim in types.DIMS:
      match dim:
        case 'x':
          pos_factor = 1.1
          neg_factor = 0.9
        case 'y':
          pos_factor = 1.2
          neg_factor = 0.8
        case _:  # 'z'
          pos_factor = 1.3
          neg_factor = 0.7
      conservative_pos = {
          var_name: testing_utils.to_3d_tensor(
              val * pos_factor, dim, size, as_tf_tensor=False
          )
          for var_name, val in before_interp.items()
      }
      conservative_neg = {
          var_name: testing_utils.to_3d_tensor(
              val * neg_factor, dim, size, as_tf_tensor=False
          )
          for var_name, val in before_interp.items()
      }
      with self.subTest(f'{dim}, conservative_neg'):
        self.assertDictEqual(conservative_neg, results[0][dim])
      with self.subTest(f'{dim}, conservative_pos'):
        self.assertDictEqual(conservative_pos, results[1][dim])

  @mock.patch(
      'swirl_lm.numerics.interpolation.weno',
      _mock_interpolation_function_different,
  )
  def test_face_interpolation_computes_primitive_variables_correctly(self):
    """Confirm that the face interpolation returns expected values."""
    nx = 16
    size = [nx, nx, nx]
    dim = 'x'

    rho = np.ones(nx)
    t = np.ones(nx) * 300.0
    u = np.ones(nx) * 10.0
    v = np.ones(nx) * -15.0
    w = np.ones(nx) * 25.0

    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    y = np.ones(nx) * 0.5

    primitive = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.E: testing_utils.to_3d_tensor(e, dim, size),
        'y': testing_utils.to_3d_tensor(y, dim, size),
    }

    results = self.evaluate(
        compressible_navier_stokes._face_interpolation(
            primitive,
            self.cfg,
        )
    )

    before_interp = {
        types.RHO: rho,
        types.U: u,
        types.V: v,
        types.W: w,
        types.E: e,
        'y': y,
    }
    for dim in types.DIMS:
      match dim:
        case 'x':
          pos_factor = 1.1
          neg_factor = 0.9
        case 'y':
          pos_factor = 1.2
          neg_factor = 0.8
        case _:  # 'z'
          pos_factor = 1.3
          neg_factor = 0.7
      primitive_pos = {
          var_name: testing_utils.to_3d_tensor(
              val * pos_factor, dim, size, as_tf_tensor=False
          )
          for var_name, val in before_interp.items()
      }
      primitive_neg = {
          var_name: testing_utils.to_3d_tensor(
              val * neg_factor, dim, size, as_tf_tensor=False
          )
          for var_name, val in before_interp.items()
      }
      with self.subTest(f'{dim}, primitive_neg'):
        self.assertDictEqual(primitive_neg, results[0][dim])
      with self.subTest(f'{dim}, primitive_pos'):
        self.assertDictEqual(primitive_pos, results[1][dim])

  @parameterized.parameters(*_DIMS)
  @mock.patch(
      'swirl_lm.numerics.interpolation.weno', _mock_interpolation_function
  )
  @mock.patch(
      'swirl_c.boundary.boundary.update_conservative_cell_averages',
      _mock_boundary_function,
  )
  def test_rhs_quiescent_gas_returns_zeros(self, dim):
    """Confirms that the rhs in quiescent gas is approximately zero."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    self.cfg.bc = {}
    u = np.zeros(nx)
    v = np.zeros(nx)
    w = np.zeros(nx)
    p = 101325.0 * np.ones(nx)
    t = 500.0 * np.ones(nx)
    rho = p / (constant.R * t)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    y = np.ones(nx) * 0.5

    conservative = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
        'rho_y': testing_utils.to_3d_tensor(rho * y, dim, size),
    }

    rhs_fn = compressible_navier_stokes.get_rhs_fn(
        replicas, self.cfg, self.physics_models
    )

    results = self.evaluate(rhs_fn(replica_id, conservative, {}))

    with self.subTest(name='keys'):
      self.assertSequenceEqual(conservative.keys(), results.keys())

    for var_name in conservative:
      hw = self.cfg.halo_width
      self.assertAllClose(
          np.array(results[var_name])[hw:-hw, hw:-hw, hw:-hw],
          np.zeros(size)[hw:-hw, hw:-hw, hw:-hw],
      )

  @parameterized.parameters(*_DIMS)
  @mock.patch(
      'swirl_lm.numerics.interpolation.weno', _mock_interpolation_function
  )
  @mock.patch(
      'swirl_c.boundary.boundary.update_conservative_cell_averages',
      _mock_boundary_function,
  )
  def test_rhs_plug_flow_euler_computes_correctly(self, dim):
    """Confirm RHS calculation for inviscid plug flow."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]

    self.cfg.bc = {}
    dx = self.cfg.dx
    u = np.zeros(nx)
    v = np.zeros(nx)
    w = np.zeros(nx)
    match dim:
      case 'x':
        v_conv = 10.0
        u = np.ones(nx) * v_conv
      case 'y':
        v_conv = -25.0
        v = np.ones(nx) * v_conv
      case _:
        v_conv = 15.0
        w = np.ones(nx) * v_conv

    p = 101325.0 * np.ones(nx)
    t = np.ones(nx) * 300.0
    rho = p / (constant.R * t)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    y = np.linspace(0.0, 1.0, nx)

    conservative = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
        'rho_y': testing_utils.to_3d_tensor(rho * y, dim, size),
    }

    rhs_fn = compressible_navier_stokes.get_rhs_fn(
        replicas, self.cfg, self.physics_models
    )

    results = self.evaluate(rhs_fn(replica_id, conservative, {}))
    flux_y = rho * y * v_conv
    rhs_y = np.zeros(nx)
    rhs_y[:-1] = -np.diff(flux_y) / dx
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
        'rho_y': testing_utils.to_3d_tensor(
            rhs_y, dim, size, as_tf_tensor=False
        ),
    }

    with self.subTest(name='keys'):
      self.assertSequenceEqual(conservative.keys(), results.keys())

    for var_name, val in expected.items():
      hw = self.cfg.halo_width
      with self.subTest(name=var_name):
        self.assertAllClose(
            np.array(results[var_name])[hw:-hw, hw:-hw, hw:-hw],
            val[hw:-hw, hw:-hw, hw:-hw],
            rtol=2.0 * types.SMALL,
        )

  @mock.patch(
      'swirl_c.equations.compressible_navier_stokes._intercell_convective_flux',
      _mock_intercell_flux_function_arbitrary,
  )
  @mock.patch(
      'swirl_c.boundary.boundary.update_conservative_cell_averages',
      _mock_boundary_function,
  )
  def test_rhs_specified_intercell_convective_flux_computes_correctly(self):
    """Confirm RHS calculation for arbitrary imposed intercell flux."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    dim = 'x'
    nx = 16
    size = [nx, nx, nx]
    self.cfg.bc = {}
    dx = 1.0
    self.cfg.dx = dx
    self.cfg.dy = dx
    self.cfg.dz = dx
    u = np.ones(nx) * 10.0
    v = np.ones(nx) * -25.0
    w = np.ones(nx) * 15.0
    p = 101325.0 * np.ones(nx)
    t = np.ones(nx) * 300.0
    rho = p / (constant.R * t)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    y = np.linspace(0.0, 1.0, nx)
    conservative = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
        'rho_y': testing_utils.to_3d_tensor(rho * y, dim, size),
    }

    rhs_fn = compressible_navier_stokes.get_rhs_fn(
        replicas, self.cfg, self.physics_models
    )

    results = self.evaluate(rhs_fn(replica_id, conservative, {}))

    # Now compute the expected RHS from mock fluxes.
    def compute_forward_diff(flux_vec):
      rhs_vec = np.zeros(nx)
      rhs_vec[:-1] = -np.diff(flux_vec)
      return rhs_vec

    t_x = np.linspace(0.0, 1.0, 16) * 2.0 * np.pi
    flux_x_vec = np.sin(t_x)
    rhs_x_vec = compute_forward_diff(flux_x_vec)

    t_y = np.linspace(0.0, 1.0, 16) * 2.0 * np.pi + np.pi / 3.0
    flux_y_vec = np.sin(t_y)
    rhs_y_vec = compute_forward_diff(flux_y_vec)

    t_z = np.linspace(0.0, 1.0, 16)
    flux_z_vec = t_z**2.0
    rhs_z_vec = compute_forward_diff(flux_z_vec)

    rhs_x = testing_utils.to_3d_tensor(rhs_x_vec, 'x', size, as_tf_tensor=False)
    rhs_y = testing_utils.to_3d_tensor(rhs_y_vec, 'y', size, as_tf_tensor=False)
    rhs_z = testing_utils.to_3d_tensor(rhs_z_vec, 'z', size, as_tf_tensor=False)
    rhs_mtx = (rhs_x + rhs_y + rhs_z) / dx

    expected = {
        types.RHO: rhs_mtx,
        types.RHO_U: rhs_mtx * 2.0,
        types.RHO_V: rhs_mtx * 3.0,
        types.RHO_W: rhs_mtx * 4.0,
        types.RHO_E: rhs_mtx * 5.0,
        'rho_y': rhs_mtx * 6.0,
    }

    with self.subTest(name='keys'):
      self.assertSequenceEqual(conservative.keys(), results.keys())

    for var_name, val in expected.items():
      hw = self.cfg.halo_width
      with self.subTest(name=var_name):
        self.assertAllClose(
            np.array(results[var_name])[hw:-hw, hw:-hw, hw:-hw],
            val[hw:-hw, hw:-hw, hw:-hw],
            rtol=2.0 * types.SMALL,
        )

  @mock.patch(
      'swirl_lm.numerics.interpolation.weno', _mock_interpolation_function
  )
  @mock.patch(
      'swirl_c.equations.compressible_navier_stokes._intercell_convective_flux',
      _mock_intercell_flux_function_return_conservative,
  )
  def test_rhs_bc_updates_cell_averages_correctly_x_dim_only(self):
    """Confirms that the cell average BC update is applied correctly.

    Here we only test applying a 'x' boundary condition to veryify that the
    call to `boundary.py` to apply the boundary condition is behaving as
    expected. Testing of the update on all dimensions is performed in
    `boundary_test.py`.
    """
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    self.cfg.bc = {}
    dx = self.cfg.dx
    size = [nx, nx, nx]

    conservative = {
        types.RHO: testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
        types.RHO_U: testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
        types.RHO_V: testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
        types.RHO_W: testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
        types.RHO_E: testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
        'rho_y': testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
    }

    self.cfg.bc['cell_averages'] = {
        types.RHO: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
        },
        types.RHO_U: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
        },
        types.RHO_V: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
        },
        types.RHO_W: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
        },
        types.RHO_E: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
        },
        'rho_y': {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
        },
    }

    rhs_fn = compressible_navier_stokes.get_rhs_fn(
        replicas, self.cfg, self.physics_models
    )

    results = self.evaluate(rhs_fn(replica_id, conservative, {}))

    cons_vec = 0.5 * np.ones(nx)
    cons_vec[:3] = 1.0
    cons_vec[-3:] = 2.0
    rhs_vec = np.zeros(nx)
    rhs_vec[:-1] = -np.diff(cons_vec) / dx
    expected = {
        types.RHO: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        'rho_y': testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
    }

    hw = self.cfg.halo_width
    with self.subTest(name='keys'):
      self.assertSequenceEqual(results.keys(), expected.keys())
    for var_name, val in expected.items():
      with self.subTest(name=var_name):
        self.assertAllClose(
            np.array(results[var_name])[
                hw - 1 : -hw, hw - 1 : -hw, hw - 1 : -hw
            ],
            val[hw - 1 : -hw, hw - 1 : -hw, hw - 1 : -hw],
            rtol=2.0 * types.SMALL,
        )

  @mock.patch(
      'swirl_lm.numerics.interpolation.weno', _mock_interpolation_function
  )
  @mock.patch(
      'swirl_c.equations.compressible_navier_stokes._intercell_convective_flux',
      _mock_intercell_flux_function_return_conservative,
  )
  @mock.patch(
      'swirl_c.boundary.boundary.update_conservative_cell_averages',
      _mock_boundary_function,
  )
  def test_rhs_bc_updates_cell_faces_correctly_x_dim_only(self):
    """Confirms that the cell face BC update is applied correctly.

    Here we only test applying a 'x' boundary condition to veryify that the
    call to `boundary.py` to apply the boundary condition is behaving as
    expected. Testing of the update on all dimensions is performed in
    `boundary_test.py`.
    """
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    self.cfg.bc = {}
    self.cfg.nx = 10
    self.cfg.ny = 10
    self.cfg.nz = 10
    self.cfg.dx = 1.0
    self.cfg.dy = 1.0
    self.cfg.dz = 1.0

    conservative = {
        types.RHO: testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
        types.RHO_U: testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
        types.RHO_V: testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
        types.RHO_W: testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
        types.RHO_E: testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
        'rho_y': testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
    }

    self.cfg.bc['cell_faces'] = {
        types.RHO: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        types.RHO_U: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        types.RHO_V: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        types.RHO_W: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        types.RHO_E: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        'rho_y': {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
    }

    rhs_fn = compressible_navier_stokes.get_rhs_fn(
        replicas, self.cfg, self.physics_models
    )

    results = self.evaluate(rhs_fn(replica_id, conservative, {}))

    flux_vec = 0.5 * np.ones(nx)
    flux_vec[3] = 1.0
    flux_vec[13] = 2.0
    rhs_vec = np.zeros(nx)
    rhs_vec[:-1] = -np.diff(flux_vec)
    expected = {
        types.RHO: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        'rho_y': testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
    }

    hw = self.cfg.halo_width
    with self.subTest(name='keys'):
      self.assertSequenceEqual(results.keys(), expected.keys())
    for var_name, val in expected.items():
      with self.subTest(name=var_name):
        self.assertAllClose(
            np.array(results[var_name])[
                hw - 1 : -hw, hw - 1 : -hw, hw - 1 : -hw
            ],
            val[hw - 1 : -hw, hw - 1 : -hw, hw - 1 : -hw],
            rtol=2.0 * types.SMALL,
        )

  @mock.patch(
      'swirl_lm.numerics.interpolation.weno', _mock_interpolation_function
  )
  @mock.patch(
      'swirl_c.boundary.boundary.update_conservative_cell_averages',
      _mock_boundary_function,
  )
  def test_rhs_bc_updates_cell_convective_fluxes_correctly_x_dim_only(self):
    """Confirms that the cell convective flux BC update is applied correctly.

    Here we only test applying a 'x' boundary condition to veryify that the
    call to `boundary.py` to apply the boundary condition is behaving as
    expected. Testing of the update on all dimensions is performed in
    `boundary_test.py`.
    """
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    self.cfg.nx = 10
    self.cfg.ny = 10
    self.cfg.nz = 10
    self.cfg.dx = 1.0
    self.cfg.dy = 1.0
    self.cfg.dz = 1.0
    self.cfg.bc = {}

    t = 300.0
    e = constant.CV * t + 3.0 / 2.0
    h = constant.CP * t + 3.0 / 2.0

    conservative = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx), 'x', size),
        types.RHO_U: testing_utils.to_3d_tensor(np.ones(nx), 'x', size),
        types.RHO_V: testing_utils.to_3d_tensor(np.ones(nx), 'x', size),
        types.RHO_W: testing_utils.to_3d_tensor(np.ones(nx), 'x', size),
        types.RHO_E: testing_utils.to_3d_tensor(e * np.ones(nx), 'x', size),
        'rho_y': testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
    }

    self.cfg.bc['intercell_fluxes'] = {}
    self.cfg.bc['intercell_fluxes']['convective'] = {
        types.RHO: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 4.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 5.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        types.RHO_U: {
            'x': {
                0: (None, None),
                1: (None, None),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        types.RHO_V: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 4.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 5.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        types.RHO_W: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 4.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 5.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        types.RHO_E: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, h + 3.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, h + 4.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        'rho_y': {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 3.5),
                1: (bc_types.BoundaryCondition.DIRICHLET, 4.5),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
    }

    rhs_fn = compressible_navier_stokes.get_rhs_fn(
        replicas, self.cfg, self.physics_models
    )

    results = self.evaluate(rhs_fn(replica_id, conservative, {}))

    flux_vec = np.ones(nx)
    flux_vec[3] = 4.0
    flux_vec[13] = 5.0
    rhs_vec = np.zeros(nx)
    rhs_vec[:-1] = -np.diff(flux_vec)
    expected = {
        types.RHO: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.zeros(nx), 'x', size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        'rho_y': testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
    }

    hw = self.cfg.halo_width
    with self.subTest(name='keys'):
      self.assertSequenceEqual(results.keys(), expected.keys())
    for var_name, val in expected.items():
      with self.subTest(name=var_name):
        self.assertAllClose(
            np.array(results[var_name])[
                hw - 1 : -hw, hw - 1 : -hw, hw - 1 : -hw
            ],
            val[hw - 1 : -hw, hw - 1 : -hw, hw - 1 : -hw],
            rtol=2.0 * types.SMALL,
        )

  @mock.patch(
      'swirl_lm.numerics.interpolation.weno', _mock_interpolation_function
  )
  @mock.patch(
      'swirl_c.boundary.boundary.update_conservative_cell_averages',
      _mock_boundary_function,
  )
  def test_rhs_bc_updates_cell_total_fluxes_correctly(self):
    """Confirms that the cell total flux BC update is applied correctly."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    self.cfg.nx = 10
    self.cfg.ny = 10
    self.cfg.nz = 10
    self.cfg.dx = 1.0
    self.cfg.dy = 1.0
    self.cfg.dz = 1.0
    self.cfg.bc = {}

    t = 300.0
    e = constant.CV * t + 3.0 / 2.0
    h = constant.CP * t + 3.0 / 2.0

    conservative = {
        types.RHO: testing_utils.to_3d_tensor(np.ones(nx), 'x', size),
        types.RHO_U: testing_utils.to_3d_tensor(np.ones(nx), 'x', size),
        types.RHO_V: testing_utils.to_3d_tensor(np.ones(nx), 'x', size),
        types.RHO_W: testing_utils.to_3d_tensor(np.ones(nx), 'x', size),
        types.RHO_E: testing_utils.to_3d_tensor(e * np.ones(nx), 'x', size),
        'rho_y': testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
    }

    self.cfg.bc['intercell_fluxes'] = {}
    self.cfg.bc['intercell_fluxes']['total'] = {
        types.RHO: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 4.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 5.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        types.RHO_U: {
            'x': {
                0: (None, None),
                1: (None, None),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        types.RHO_V: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 4.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 5.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        types.RHO_W: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 4.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 5.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        types.RHO_E: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, h + 3.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, h + 4.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        'rho_y': {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 3.5),
                1: (bc_types.BoundaryCondition.DIRICHLET, 4.5),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
    }

    rhs_fn = compressible_navier_stokes.get_rhs_fn(
        replicas, self.cfg, self.physics_models
    )

    results = self.evaluate(rhs_fn(replica_id, conservative, {}))

    flux_vec = np.ones(nx)
    flux_vec[3] = 4.0
    flux_vec[13] = 5.0
    rhs_vec = np.zeros(nx)
    rhs_vec[:-1] = -np.diff(flux_vec)
    expected = {
        types.RHO: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            np.zeros(nx), 'x', size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
        'rho_y': testing_utils.to_3d_tensor(
            rhs_vec, 'x', size, as_tf_tensor=False
        ),
    }

    hw = self.cfg.halo_width
    with self.subTest(name='keys'):
      self.assertSequenceEqual(results.keys(), expected.keys())
    for var_name, val in expected.items():
      with self.subTest(name=var_name):
        self.assertAllClose(
            np.array(results[var_name])[
                hw - 1 : -hw, hw - 1 : -hw, hw - 1 : -hw
            ],
            val[hw - 1 : -hw, hw - 1 : -hw, hw - 1 : -hw],
            rtol=2.0 * types.SMALL,
        )

  @mock.patch(
      'swirl_lm.numerics.interpolation.weno', _mock_interpolation_function
  )
  @mock.patch(
      'swirl_c.equations.compressible_navier_stokes._intercell_convective_flux',
      _mock_intercell_flux_function_return_conservative,
  )
  @mock.patch(
      'swirl_c.numerics.gradient.forward_1',
      _mock_gradient_function_returns_zeros,
  )
  def test_rhs_source_function_updates_correctly(self):
    """Confirms that the source function is correctly applied."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    self.cfg.bc = {}

    conservative = {
        types.RHO: testing_utils.to_3d_tensor(0.5 * np.ones(nx), 'x', size),
        types.RHO_U: testing_utils.to_3d_tensor(1.5 * np.ones(nx), 'x', size),
        types.RHO_V: testing_utils.to_3d_tensor(2.5 * np.ones(nx), 'x', size),
        types.RHO_W: testing_utils.to_3d_tensor(3.5 * np.ones(nx), 'x', size),
        types.RHO_E: testing_utils.to_3d_tensor(np.zeros(nx), 'x', size),
        'rho_y': testing_utils.to_3d_tensor(4.5 * np.ones(nx), 'x', size),
    }

    self.cfg.bc['cell_averages'] = {
        types.RHO: {
            'x': {
                0: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
        },
        types.RHO_U: {
            'x': {
                0: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
        },
        types.RHO_V: {
            'x': {
                0: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
        },
        types.RHO_W: {
            'x': {
                0: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
        },
        types.RHO_E: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
        },
        'rho_y': {
            'x': {
                0: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
        },
    }

    def fake_source_fn(replica_id, replicas, conservative, helper_vars):
      del replica_id, replicas, helper_vars  # Unused.
      return {
          var_name: tf.nest.map_structure(lambda x: x + 1.0, val)
          for var_name, val in conservative.items()
      }

    self.physics_models.source_function = fake_source_fn

    rhs_fn = compressible_navier_stokes.get_rhs_fn(
        replicas, self.cfg, self.physics_models
    )

    results = self.evaluate(rhs_fn(replica_id, conservative, {}))

    expected = {
        types.RHO: testing_utils.to_3d_tensor(
            1.5 * np.ones(nx), 'x', size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            2.5 * np.ones(nx), 'x', size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            3.5 * np.ones(nx), 'x', size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            4.5 * np.ones(nx), 'x', size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            np.ones(nx), 'x', size, as_tf_tensor=False
        ),
        'rho_y': testing_utils.to_3d_tensor(
            5.5 * np.ones(nx), 'x', size, as_tf_tensor=False
        ),
    }

    hw = self.cfg.halo_width
    with self.subTest(name='keys'):
      self.assertSequenceEqual(results.keys(), expected.keys())
    for var_name, val in expected.items():
      with self.subTest(name=var_name):
        self.assertAllClose(
            val[hw:-hw, hw:-hw, hw:-hw],
            np.array(results[var_name])[hw:-hw, hw:-hw, hw:-hw],
            rtol=2.0 * types.SMALL,
        )

  def test_intercell_diffusive_flux_is_zero_in_isothermal_gas(self):
    """Confirms that the intercell diffusive flux is zero in isothermal gas."""
    dim = 'x'
    nx = 16
    size = [nx, nx, nx]
    u = 10.0 * np.ones(nx)
    v = -20.0 * np.ones(nx)
    w = 25.0 * np.ones(nx)
    p = 101325.0 * np.ones(nx)
    t = 500.0 * np.ones(nx)
    rho = p / (constant.R * t)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    y = np.ones(nx) * 0.5

    primitive = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.E: testing_utils.to_3d_tensor(e, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        'y': testing_utils.to_3d_tensor(y, dim, size),
    }
    results = self.evaluate(
        compressible_navier_stokes._intercell_diffusive_flux(
            primitive,
            self.cfg,
            self.physics_models,
        )
    )

    for flux_dim in types.DIMS:
      with self.subTest(name=f'flux direction:{flux_dim}, keys'):
        self.assertSequenceEqual(
            [types.RHO] + list(types.MOMENTUM) + [types.RHO_E, 'rho_y'],
            list(results[flux_dim].keys()),
        )
      for var_name, val in results[flux_dim].items():
        with self.subTest(name=f'flux direction:{flux_dim}, {var_name}'):
          hw = self.cfg.halo_width
          domain_expected = np.zeros([11, 11, 11])
          domain_results = np.array(val)[
              hw : -(hw - 1), hw : -(hw - 1), hw : -(hw - 1)
          ]
          self.assertAllClose(
              domain_expected,
              domain_results,
          )

  @parameterized.parameters(*_DIMS)
  def test_intercell_diffusive_flux_heat_conduction_only(self, dim):
    """Confirms the diffusive flux is only heat conduction in stationary gas."""
    nx = 16
    size = [nx, nx, nx]
    h = 0.5
    self.cfg.dx = h
    self.cfg.dy = h
    self.cfg.dz = h
    u = np.zeros(nx)
    v = np.zeros(nx)
    w = np.zeros(nx)
    p = 101325.0 * np.ones(nx)
    t = 400.0 + np.linspace(0.0, 100.0, nx)
    rho = p / (constant.R * t)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    y = np.ones(nx) * 0.5

    primitive = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.E: testing_utils.to_3d_tensor(e, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        'y': testing_utils.to_3d_tensor(y, dim, size),
    }
    results = self.evaluate(
        compressible_navier_stokes._intercell_diffusive_flux(
            primitive,
            self.cfg,
            self.physics_models,
        )
    )

    for flux_dim in types.DIMS:
      with self.subTest(name=f'flux direction:{flux_dim}, keys'):
        self.assertSequenceEqual(
            [types.RHO] + list(types.MOMENTUM) + [types.RHO_E, 'rho_y'],
            list(results[flux_dim].keys()),
        )
      # The only nonzero flux is energy, which we treat here.
      expected_rho_e = np.zeros(size)
      if flux_dim == dim:
        kappa_centers = (
            self.cfg.transport_parameters['nu']
            / self.cfg.transport_parameters['pr']
            * rho
            * constant.CP
        )
        kappa_faces = kappa_centers[:-1] + 0.5 * np.diff(kappa_centers)
        dtdx = np.diff(t) / h
        flux_vec = np.zeros(nx)
        flux_vec[1:] = -kappa_faces * dtdx
        expected_rho_e = testing_utils.to_3d_tensor(
            flux_vec, dim, size, as_tf_tensor=False
        )
      for var_name, val in results[flux_dim].items():
        with self.subTest(name=f'flux direction:{flux_dim}, {var_name}'):
          hw = self.cfg.halo_width
          if var_name == types.RHO_E:
            domain_expected = expected_rho_e[
                hw : -(hw - 1), hw : -(hw - 1), hw : -(hw - 1)
            ]
          else:
            domain_expected = np.zeros([11, 11, 11])
          domain_results = np.array(val)[
              hw : -(hw - 1), hw : -(hw - 1), hw : -(hw - 1)
          ]
          # The roundoff error in the diffusive flux calculation can be fairly
          # significant, thus the somewhat elevated rtol here.
          self.assertAllClose(
              domain_expected, domain_results, atol=5.0e-5, rtol=5.0e-5
          )

  @mock.patch('swirl_c.physics.diffusion.shear_stress', _mock_shear_stress)
  def test_intercell_diffusive_flux_returns_correct_for_shear(self):
    """Checks if the diffusive flux is correct for shearing isothermal flow."""
    dim = 'x'
    nx = 16
    size = [nx, nx, nx]
    h = 1.0
    self.cfg.dx = h
    self.cfg.dy = h
    self.cfg.dz = h
    # Note that the shear stress on the faces is specified through the mock
    # function, but the velocities are still used when the viscous dissipation
    # is computed from the dot product of the shear stress and the velocity
    # vector.
    u = 1.0 * np.ones(nx)
    v = 2.0 * np.ones(nx)
    w = 3.0 * np.ones(nx)
    p = 101325.0 * np.ones(nx)
    t = 300.0 * np.ones(nx)
    rho = p / (constant.R * t)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    y = np.ones(nx) * 0.5

    primitive = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.E: testing_utils.to_3d_tensor(e, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        'y': testing_utils.to_3d_tensor(y, dim, size),
    }
    results = self.evaluate(
        compressible_navier_stokes._intercell_diffusive_flux(
            primitive,
            self.cfg,
            self.physics_models,
        )
    )

    for flux_dim in types.DIMS:
      with self.subTest(name=f'flux direction:{flux_dim}, keys'):
        self.assertSequenceEqual(
            [types.RHO] + list(types.MOMENTUM) + [types.RHO_E, 'rho_y'],
            list(results[flux_dim].keys()),
        )

      match flux_dim:
        case 'x':
          u_fac = 1.0
          v_fac = 4.0
          w_fac = 7.0
        case 'y':
          u_fac = 2.0
          v_fac = 5.0
          w_fac = 8.0
        case _:  # 'z'
          u_fac = 3.0
          v_fac = 6.0
          w_fac = 9.0
      e_fac = 1.0 * u_fac + 2.0 * v_fac + 3.0 * w_fac
      expected = {
          types.RHO: testing_utils.to_3d_tensor(
              np.zeros(nx), dim, size, as_tf_tensor=False
          ),
          types.RHO_U: testing_utils.to_3d_tensor(
              -u_fac * np.ones(nx), dim, size, as_tf_tensor=False
          ),
          types.RHO_V: testing_utils.to_3d_tensor(
              -v_fac * np.ones(nx), dim, size, as_tf_tensor=False
          ),
          types.RHO_W: testing_utils.to_3d_tensor(
              -w_fac * np.ones(nx), dim, size, as_tf_tensor=False
          ),
          types.RHO_E: testing_utils.to_3d_tensor(
              -e_fac * np.ones(nx), dim, size, as_tf_tensor=False
          ),
          'rho_y': testing_utils.to_3d_tensor(
              np.zeros(nx), dim, size, as_tf_tensor=False
          ),
      }

      for var_name, field in expected.items():
        with self.subTest(name=f'flux direction:{flux_dim}, {var_name}'):
          hw = self.cfg.halo_width
          domain_expected = field[
              hw : -(hw - 1), hw : -(hw - 1), hw : -(hw - 1)
          ]
          domain_results = np.array(results[flux_dim][var_name])[
              hw : -(hw - 1), hw : -(hw - 1), hw : -(hw - 1)
          ]
          self.assertAllClose(domain_expected, domain_results)

  @mock.patch(
      'swirl_c.equations.compressible_navier_stokes._intercell_convective_flux',
      _mock_intercell_flux_function_return_zeros,
  )
  @mock.patch(
      'swirl_c.equations.compressible_navier_stokes._intercell_diffusive_flux',
      _mock_intercell_diffusive_flux_function_returns_sin_profiles,
  )
  @mock.patch(
      'swirl_c.boundary.boundary.update_conservative_cell_averages',
      _mock_boundary_function,
  )
  def test_rhs_specified_intercell_diffusive_flux_computes_correctly(self):
    """Confirm RHS calculation for arbitrary imposed intercell flux."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    dim = 'x'
    nx = 16
    size = [nx, nx, nx]
    self.cfg.bc = {}
    dx = 1.0
    self.cfg.dx = dx
    self.cfg.dy = dx
    self.cfg.dz = dx
    u = np.ones(nx) * 10.0
    v = np.ones(nx) * -25.0
    w = np.ones(nx) * 15.0
    p = 101325.0 * np.ones(nx)
    t = np.ones(nx) * 300.0
    rho = p / (constant.R * t)
    e = constant.CV * t + 0.5 * (u**2 + v**2 + w**2)
    y = np.linspace(0.0, 1.0, nx)
    conservative = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
        'rho_y': testing_utils.to_3d_tensor(rho * y, dim, size),
    }
    self.cfg.include_diffusion = True

    rhs_fn = compressible_navier_stokes.get_rhs_fn(
        replicas, self.cfg, self.physics_models
    )

    results = self.evaluate(rhs_fn(replica_id, conservative, {}))

    # Now compute the expected RHS from mock fluxes.
    def compute_forward_diff(flux_vec):
      rhs_vec = np.zeros(nx)
      rhs_vec[:-1] = -np.diff(flux_vec)
      return rhs_vec

    xx = np.linspace(0.0, 1.0, 16) * 2.0 * np.pi
    flux_x_vec = np.sin(xx)
    rhs_x_vec = compute_forward_diff(flux_x_vec)

    yy = np.linspace(0.0, 1.0, 16) * 2.0 * np.pi + np.pi / 3.0
    flux_y_vec = np.sin(yy)
    rhs_y_vec = compute_forward_diff(flux_y_vec)

    zz = np.linspace(0.0, 1.0, 16)
    flux_z_vec = np.sin(zz**2.0)
    rhs_z_vec = compute_forward_diff(flux_z_vec)

    rhs_x = testing_utils.to_3d_tensor(rhs_x_vec, 'x', size, as_tf_tensor=False)
    rhs_y = testing_utils.to_3d_tensor(rhs_y_vec, 'y', size, as_tf_tensor=False)
    rhs_z = testing_utils.to_3d_tensor(rhs_z_vec, 'z', size, as_tf_tensor=False)
    rhs_matrix = (rhs_x + rhs_y + rhs_z) / dx

    expected = {
        types.RHO: rhs_matrix,
        types.RHO_U: rhs_matrix * 2.0,
        types.RHO_V: rhs_matrix * 3.0,
        types.RHO_W: rhs_matrix * 4.0,
        types.RHO_E: rhs_matrix * 5.0,
        'rho_y': rhs_matrix * 6.0,
    }

    with self.subTest(name='keys'):
      self.assertSequenceEqual(conservative.keys(), results.keys())

    for var_name, val in expected.items():
      hw = self.cfg.halo_width
      with self.subTest(name=var_name):
        self.assertAllClose(
            val[hw:-hw, hw:-hw, hw:-hw],
            np.array(results[var_name])[hw:-hw, hw:-hw, hw:-hw],
            rtol=2.0 * types.SMALL,
        )


if __name__ == '__main__':
  tf.test.main()
