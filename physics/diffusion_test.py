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
"""Tests for diffusion."""

from absl.testing import parameterized
import numpy as np
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.numerics import kernel_op_types
from swirl_c.physics import constant
from swirl_c.physics import diffusion
from swirl_c.physics import physics_models as physics_models_lib
import tensorflow as tf


class DiffusionTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes the generic thermodynamics library object."""
    super().setUp()
    self.cfg = parameter.SwirlCParameters()

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

    # Rearange the coordinates from x-y-z to z-x-y
    u, v, w, rho = [np.transpose(f, (2, 0, 1)) for f in [u, v, w, rho]]

    states = {
        'u': tf.unstack(tf.convert_to_tensor(u[1::2, 1::2, 1::2])),
        'v': tf.unstack(tf.convert_to_tensor(v[1::2, 1::2, 1::2])),
        'w': tf.unstack(tf.convert_to_tensor(w[1::2, 1::2, 1::2])),
        'rho': tf.unstack(tf.convert_to_tensor(rho[1::2, 1::2, 1::2])),
    }

    states_fx = {
        'rho': tf.unstack(tf.convert_to_tensor(rho[1::2, ::2, 1::2])),
    }

    states_fy = {
        'rho': tf.unstack(tf.convert_to_tensor(rho[1::2, 1::2, ::2])),
    }

    states_fz = {
        'rho': tf.unstack(tf.convert_to_tensor(rho[::2, 1::2, 1::2])),
    }

    cfg = {f'd{dim}': 2.0 * h_i for dim, h_i in zip(('x', 'y', 'z'), h)}
    cfg.update({'nu': 1e-1})

    # Compute the strain rate analytically. Note that in this configuration:
    # dudx = dvdy = dwdz, dudy = dvdx, dudz = dwdx, dvdz = dwdy.
    s_kk = np.zeros(n_tot, dtype=np.float32)
    s_12 = np.cos(xx) * np.cos(yy) * np.sin(zz)
    s_13 = np.cos(xx) * np.sin(yy) * np.cos(zz)
    s_23 = np.sin(xx) * np.cos(yy) * np.cos(zz)

    def get_face_val(s_full, dim):
      """Get values on the specific face in `dim` with node values on `::2`."""
      # Note that all face values at i - 1/2 are saved at index i.
      if dim == 'x':
        s_face = s_full[::2, 1::2, 1::2]
      elif dim == 'y':
        s_face = s_full[1::2, ::2, 1::2]
      elif dim == 'z':
        s_face = s_full[1::2, 1::2, ::2]
      else:
        raise ValueError(
            f'"{dim}" is not a valid dimension. Should be one of "x", "y", and'
            ' "z".'
        )
      return s_face.transpose((2, 0, 1))

    tau = [
        [
            0.2 * rho[1::2, ::2, 1::2] * get_face_val(s_kk, 'x'),
            0.2 * rho[1::2, 1::2, ::2] * get_face_val(s_12, 'y'),
            0.2 * rho[::2, 1::2, 1::2] * get_face_val(s_13, 'z'),
        ],
        [
            0.2 * rho[1::2, ::2, 1::2] * get_face_val(s_12, 'x'),
            0.2 * rho[1::2, 1::2, ::2] * get_face_val(s_kk, 'y'),
            0.2 * rho[::2, 1::2, 1::2] * get_face_val(s_23, 'z'),
        ],
        [
            0.2 * rho[1::2, ::2, 1::2] * get_face_val(s_13, 'x'),
            0.2 * rho[1::2, 1::2, ::2] * get_face_val(s_23, 'y'),
            0.2 * rho[::2, 1::2, 1::2] * get_face_val(s_kk, 'z'),
        ],
    ]

    return states, states_fx, states_fy, states_fz, tau, cfg

  _DIMS = ('x', 'y', 'z')

  def test_shear_stress_computed_correctly(self):
    """Checks if the shear stress is computed correctly."""
    states, states_fx, states_fy, states_fz, expected, cfg = (
        self.gen_taylor_green_vortex()
    )
    cfg.update(
        {'kernel_op_type': kernel_op_types.KernelOpType.KERNEL_OP_SLICE.value}
    )
    cfg['transport_parameters'] = {'nu': cfg['nu'], 'pr': 0.7}
    cfg = parameter.SwirlCParameters(cfg)
    physics_models = physics_models_lib.PhysicsModels(cfg)
    tau = self.evaluate(
        diffusion.shear_stress(
            states,
            states_fx,
            states_fy,
            states_fz,
            cfg,
            physics_models,
        )
    )

    for i in range(3):
      for j in range(3):
        with self.subTest(name=f'tau_{i}{j}'):
          tau_ij = np.stack(tau[i][j])
          self.assertAllClose(
              expected[i][j][1:-1, 1:-1, 1:-1],
              tau_ij[1:-1, 1:-1, 1:-1],
              atol=1e-2,  # O(h^3)
          )

  def test_single_component_heat_flux_computes_correctly(self):
    """Checks if the computed heat flux matches expected."""
    size = [16, 18, 12]
    t_np = np.random.uniform(size=size, low=300.0, high=500.0)
    rho_np = np.random.uniform(size=size, low=0.5, high=1.5)
    rho_tf = tf.unstack(tf.convert_to_tensor(rho_np, dtype=types.DTYPE))
    p_np = rho_np * constant.R * t_np
    states = {
        'rho': tf.unstack(rho_tf),
        'p': tf.unstack(tf.convert_to_tensor(p_np, dtype=types.DTYPE)),
    }
    self.cfg.transport_parameters = {'nu': 2.0e-4, 'pr': 0.7}
    physics_models = physics_models_lib.PhysicsModels(self.cfg)
    self.cfg.dx = 1.0
    self.cfg.dy = 2.0
    self.cfg.dz = 3.0
    results = self.evaluate(
        diffusion.single_component_heat_flux(states, self.cfg, physics_models)
    )
    kappa_centers = (
        self.cfg.transport_parameters['nu']
        / self.cfg.transport_parameters['pr']
        * rho_np
        * constant.CP
    )
    kappa_faces = {dim: np.zeros(size) for dim in types.DIMS}
    kappa_faces['x'][:, 1:, :] = kappa_centers[:, :-1, :] + 0.5 * np.diff(
        kappa_centers, axis=1
    )
    kappa_faces['y'][:, :, 1:] = kappa_centers[:, :, :-1] + 0.5 * np.diff(
        kappa_centers, axis=2
    )
    kappa_faces['z'][1:, :, :] = kappa_centers[:-1, :, :] + 0.5 * np.diff(
        kappa_centers, axis=0
    )
    dtdh = {dim: np.zeros(size) for dim in types.DIMS}
    dtdh['x'][:, 1:, :] = np.diff(t_np, axis=1) / self.cfg.dx
    dtdh['y'][:, :, 1:] = np.diff(t_np, axis=2) / self.cfg.dy
    dtdh['z'][1:, :, :] = np.diff(t_np, axis=0) / self.cfg.dz
    expected = {dim: -kappa_faces[dim] * dtdh[dim] for dim in types.DIMS}
    for key, val in expected.items():
      with self.subTest(name=f'flux_dim: {key}'):
        # The conversion from temperature to pressure and back to temperature,
        # then taking the derivative, seems to cause a fairly large round-off
        # error here. Thus, we increase `atol`` and `rtol` for this test.
        self.assertAllClose(
            val[1:, 1:, 1:],
            np.array(results[key])[1:, 1:, 1:],
            atol=5.0e-5,
            rtol=5.0e-5,
        )

if __name__ == '__main__':
  tf.test.main()
