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
"""Tests for boundary."""

import copy
import itertools

from absl.testing import parameterized
import numpy as np
from swirl_c.boundary import bc_types
from swirl_c.boundary import boundary
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.physics import constant
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework import tpu_runner

_REPLICAS = [
    np.array([[[0, 1]]]),
    np.array([[[0], [1]]]),
    np.array([[[0]], [[1]]]),
]

_DIMS = ('x', 'y', 'z')
_SIDES = (0, 1)
_FLUX_TYPES = (
    bc_types.BoundaryFluxType.CONVECTIVE,
    bc_types.BoundaryFluxType.DIFFUSIVE,
    bc_types.BoundaryFluxType.TOTAL,
)


class BoundaryTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes the cfg object."""
    super().setUp()
    self.cfg = parameter.SwirlCParameters()

  def run_tpu_test(self, replicas, device_fn, inputs):
    """Runs `device` function on TPU."""
    device_inputs = [list(x) for x in zip(*inputs)]
    computation_shape = replicas.shape
    runner = tpu_runner.TpuRunner(computation_shape=computation_shape)
    return runner.run(device_fn, *device_inputs)

  @parameterized.parameters(*zip(_REPLICAS))
  def test_update_boundary_provides_correct_boundary_conditions(self, replicas):
    """Checks if boundary conditions are updated correctly."""
    nx = 16
    ny = 16
    nz = 16

    # Define the boundary conditions.
    u_bc_z0 = [
        tf.ones((16, 16), dtype=tf.float32),
        2.0 * tf.ones((16, 16), dtype=tf.float32),
        3.0 * tf.ones((16, 16), dtype=tf.float32),
    ]
    p_bc_x1 = [
        [
            60.0 * tf.ones((1, ny)),
        ]
        * nz,
        [
            80.0 * tf.ones((1, ny)),
        ]
        * nz,
        [
            120.0 * tf.ones((1, ny)),
        ]
        * nz,
    ]
    bc = {
        'u': {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 8.0),
                1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.DIRICHLET, u_bc_z0),
                1: (bc_types.BoundaryCondition.NEUMANN, 1.0),
            },
        },
        'p': {
            'x': {
                0: (bc_types.BoundaryCondition.NEUMANN, 8.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, p_bc_x1),
            },
            'y': {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
            'z': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 100.0),
                1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
            },
        },
    }
    self.cfg.halo_width = 3
    self.cfg.bc = bc

    # Define input variables.
    u_0 = np.zeros((nz, nx, ny), dtype=np.float32)
    u_0[3:-3, 3:-3, 3:-3] = 6.0

    p_0 = np.zeros((nz, nx, ny), dtype=np.float32)
    p_0[3:-3, 3:-3, 3:-3] = 200.0

    input_0 = [
        tf.constant(0),
        {
            'u': tf.unstack(tf.convert_to_tensor(u_0)),
            'p': tf.unstack(tf.convert_to_tensor(p_0)),
        },
    ]

    u_1 = np.zeros((nz, nx, ny), dtype=np.float32)
    u_1[3:-3, 3:-3, 3:-3] = 16.0

    p_1 = np.zeros((nz, nx, ny), dtype=np.float32)
    p_1[3:-3, 3:-3, 3:-3] = 300.0

    input_1 = [
        tf.constant(1),
        {
            'u': tf.unstack(tf.convert_to_tensor(u_1)),
            'p': tf.unstack(tf.convert_to_tensor(p_1)),
        },
    ]

    inputs = [input_0, input_1]

    # Define the device function.
    def device_fn(replica_id, states):
      """Wraps the boundary_update function to be executed on TPU."""
      return boundary.update_boundary(
          replica_id,
          replicas,
          states,
          self.cfg,
      )

    output = self.run_tpu_test(replicas, device_fn, inputs)

    output_u_0 = np.stack(output[0]['u'])
    output_u_1 = np.stack(output[1]['u'])
    output_p_0 = np.stack(output[0]['p'])
    output_p_1 = np.stack(output[1]['p'])

    # Generate the expected output.
    computation_shape = replicas.shape

    with self.subTest(name='u0Interior'):
      expected = 6.0 * np.ones((nz - 6, nx - 6, ny - 6), dtype=np.float32)
      self.assertAllEqual(expected, output_u_0[3:-3, 3:-3, 3:-3])

    with self.subTest(name='u0DimXFace0'):
      expected = 8.0 * np.ones((nz - 6, 3, ny - 6), dtype=np.float32)
      self.assertAllEqual(expected, output_u_0[3:-3, :3, 3:-3])

    with self.subTest(name='u0DimXFace1'):
      if computation_shape[0] == 1:
        expected = 6.0 * np.ones((nz - 6, 3, ny - 6), dtype=np.float32)
      else:
        expected = 16.0 * np.ones((nz - 6, 3, ny - 6), dtype=np.float32)
      self.assertAllEqual(expected, output_u_0[3:-3, -3:, 3:-3])

    with self.subTest(name='u0DimYFace0'):
      if computation_shape[1] == 1:
        expected = 6.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      else:
        expected = 16.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      self.assertAllEqual(expected, output_u_0[3:-3, 3:-3, :3])

    with self.subTest(name='u0DimYFace1'):
      if computation_shape[1] == 1:
        expected = 6.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      else:
        expected = 16.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      self.assertAllEqual(expected, output_u_0[3:-3, 3:-3, -3:])

    with self.subTest(name='u0DimZFace0'):
      expected = np.zeros((3, nx - 6, ny - 6), dtype=np.float32)
      expected[0, ...] = 1.0
      expected[1, ...] = 2.0
      expected[2, ...] = 3.0
      self.assertAllEqual(expected, output_u_0[:3, 3:-3, 3:-3])

    with self.subTest(name='u0DimZFace1'):
      expected = np.zeros((3, nx - 6, ny - 6), dtype=np.float32)
      expected[-3, ...] = 7.0 if computation_shape[2] == 1 else 16.0
      expected[-2, ...] = 8.0 if computation_shape[2] == 1 else 16.0
      expected[-1, ...] = 9.0 if computation_shape[2] == 1 else 16.0
      self.assertAllEqual(expected, output_u_0[-3:, 3:-3, 3:-3])

    with self.subTest(name='p0Interior'):
      expected = 200.0 * np.ones((nz - 6, nx - 6, ny - 6), dtype=np.float32)
      self.assertAllEqual(expected, output_p_0[3:-3, 3:-3, 3:-3])

    with self.subTest(name='p0DimXFace0'):
      expected = np.zeros((nz - 6, 3, ny - 6), dtype=np.float32)
      expected[:, 0, :] = 176.0
      expected[:, 1, :] = 184.0
      expected[:, 2, :] = 192.0
      self.assertAllEqual(expected, output_p_0[3:-3, :3, 3:-3])

    with self.subTest(name='p0DimXFace1'):
      expected = np.zeros((nz - 6, 3, ny - 6), dtype=np.float32)
      expected[:, -3, :] = 60.0 if computation_shape[0] == 1 else 300.0
      expected[:, -2, :] = 80.0 if computation_shape[0] == 1 else 300.0
      expected[:, -1, :] = 120.0 if computation_shape[0] == 1 else 300.0
      self.assertAllEqual(expected, output_p_0[3:-3, -3:, 3:-3])

    with self.subTest(name='p0DimYFace0'):
      if computation_shape[1] == 1:
        expected = 200.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      else:
        expected = 300.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      self.assertAllEqual(expected, output_p_0[3:-3, 3:-3, :3])

    with self.subTest(name='p0DimYFace1'):
      if computation_shape[1] == 1:
        expected = 200.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      else:
        expected = 300.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      self.assertAllEqual(expected, output_p_0[3:-3, 3:-3, -3:])

    with self.subTest(name='p0DimZFace0'):
      expected = 100.0 * np.ones((3, nx - 6, ny - 6), dtype=np.float32)
      self.assertAllEqual(expected, output_p_0[:3, 3:-3, 3:-3])

    with self.subTest(name='p0DimZFace1'):
      if computation_shape[2] == 1:
        expected = 200.0 * np.ones((3, nx - 6, ny - 6), dtype=np.float32)
      else:
        expected = 300.0 * np.ones((3, nx - 6, ny - 6), dtype=np.float32)
      self.assertAllEqual(expected, output_p_0[-3:, 3:-3, 3:-3])

    with self.subTest(name='u1Interior'):
      expected = 16.0 * np.ones((nz - 6, nx - 6, ny - 6), dtype=np.float32)
      self.assertAllEqual(expected, output_u_1[3:-3, 3:-3, 3:-3])

    with self.subTest(name='u1DimXFace0'):
      if computation_shape[0] == 1:
        expected = 8.0 * np.ones((nz - 6, 3, ny - 6), dtype=np.float32)
      else:
        expected = 6.0 * np.ones((nz - 6, 3, ny - 6), dtype=np.float32)
      self.assertAllEqual(expected, output_u_1[3:-3, :3, 3:-3])

    with self.subTest(name='u1DimXFace1'):
      expected = 16.0 * np.ones((nz - 6, 3, ny - 6), dtype=np.float32)
      self.assertAllEqual(expected, output_u_1[3:-3, -3:, 3:-3])

    with self.subTest(name='u1DimYFace0'):
      if computation_shape[1] == 1:
        expected = 16.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      else:
        expected = 6.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      self.assertAllEqual(expected, output_u_1[3:-3, 3:-3, :3])

    with self.subTest(name='u1DimYFace1'):
      if computation_shape[1] == 1:
        expected = 16.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      else:
        expected = 6.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      self.assertAllEqual(expected, output_u_1[3:-3, 3:-3, -3:])

    with self.subTest(name='u1DimZFace0'):
      expected = np.zeros((3, nx - 6, ny - 6), dtype=np.float32)
      expected[0, ...] = 1.0 if computation_shape[2] == 1 else 6.0
      expected[1, ...] = 2.0 if computation_shape[2] == 1 else 6.0
      expected[2, ...] = 3.0 if computation_shape[2] == 1 else 6.0
      self.assertAllEqual(expected, output_u_1[:3, 3:-3, 3:-3])

    with self.subTest(name='u1DimZFace1'):
      expected = np.zeros((3, nx - 6, ny - 6), dtype=np.float32)
      expected[-3, ...] = 17.0
      expected[-2, ...] = 18.0
      expected[-1, ...] = 19.0
      self.assertAllEqual(expected, output_u_1[-3:, 3:-3, 3:-3])

    with self.subTest(name='p1Interior'):
      expected = 300.0 * np.ones((nz - 6, nx - 6, ny - 6), dtype=np.float32)
      self.assertAllEqual(expected, output_p_1[3:-3, 3:-3, 3:-3])

    with self.subTest(name='p1DimXFace0'):
      expected = np.zeros((nz - 6, 3, ny - 6), dtype=np.float32)
      expected[:, 0, :] = 276.0 if computation_shape[0] == 1 else 200.0
      expected[:, 1, :] = 284.0 if computation_shape[0] == 1 else 200.0
      expected[:, 2, :] = 292.0 if computation_shape[0] == 1 else 200.0
      self.assertAllEqual(expected, output_p_1[3:-3, :3, 3:-3])

    with self.subTest(name='p1DimXFace1'):
      expected = np.zeros((nz - 6, 3, ny - 6), dtype=np.float32)
      expected[:, -3, :] = 60.0
      expected[:, -2, :] = 80.0
      expected[:, -1, :] = 120.0
      self.assertAllEqual(expected, output_p_1[3:-3, -3:, 3:-3])

    with self.subTest(name='p1DimYFace0'):
      if computation_shape[1] == 1:
        expected = 300.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      else:
        expected = 200.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      self.assertAllEqual(expected, output_p_1[3:-3, 3:-3, :3])

    with self.subTest(name='p0DimYFace1'):
      if computation_shape[1] == 1:
        expected = 300.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      else:
        expected = 200.0 * np.ones((nz - 6, nx - 6, 3), dtype=np.float32)
      self.assertAllEqual(expected, output_p_1[3:-3, 3:-3, -3:])

    with self.subTest(name='p0DimZFace0'):
      if computation_shape[2] == 1:
        expected = 100.0 * np.ones((3, nx - 6, ny - 6), dtype=np.float32)
      else:
        expected = 200.0 * np.ones((3, nx - 6, ny - 6), dtype=np.float32)
      self.assertAllEqual(expected, output_p_1[:3, 3:-3, 3:-3])

    with self.subTest(name='p0DimZFace1'):
      expected = 300.0 * np.ones((3, nx - 6, ny - 6), dtype=np.float32)
      self.assertAllEqual(expected, output_p_1[-3:, 3:-3, 3:-3])

  _BC_DIM_ORDER = [['x', 'y', 'z'], ['x', 'z', 'y'],]

  @parameterized.parameters(*itertools.product(_REPLICAS, _BC_DIM_ORDER))
  def test_conservative_cell_averages_variable_inflow(
      self, replicas, bc_dim_order
  ):
    """Checks updated conservative cell values for an inflow condition."""
    self.cfg.conservative_variable_names = (
        [
            types.RHO,
        ]
        + list(types.MOMENTUM)
        + [
            types.RHO_E,
        ]
    )
    size = [16, 16, 16]
    dim = ('x', 'z', 'y')[replicas.shape.index(2)]

    bc_base = {}
    for bc_dim in bc_dim_order:
      if dim != bc_dim:
        bc_base.update({
            bc_dim: {
                0: (bc_types.BoundaryCondition.PERIODIC,),
                1: (bc_types.BoundaryCondition.PERIODIC,),
            },
        })
      else:
        bc_base.update({bc_dim: {}})

    var_names = [types.RHO, types.RHO_E]
    for idx in range(3):
      if dim == types.DIMS[idx]:
        var_names.append(types.VELOCITY[idx])
      else:
        var_names.append(types.MOMENTUM[idx])

    def make_bc(val):
      if dim == 'z':
        return val * tf.ones((16, 16), dtype=tf.float32)
      elif dim == 'x':
        return [val * tf.ones((1, 16))] * 16
      else:  # dim == 'y'
        return [val * tf.ones((16, 1))] * 16

    rho_f0 = [make_bc(1.0), make_bc(2.0), make_bc(3.0)]
    u_f0 = [
        [make_bc(11.0), make_bc(21.0), make_bc(31.0)],
        [make_bc(-12.0), make_bc(-22.0), make_bc(-32.0)],
        [make_bc(13.0), make_bc(23.0), make_bc(33.0)],
    ]
    rhoe_f0 = [make_bc(15.0), make_bc(25.0), make_bc(35.0)]

    bc = {var_name: bc_base.copy() for var_name in var_names}
    bc[types.RHO][dim] = {
        0: (bc_types.BoundaryCondition.DIRICHLET, rho_f0),
        1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
    }

    bc[types.RHO_E][dim] = {
        0: (bc_types.BoundaryCondition.DIRICHLET, rhoe_f0),
        1: (bc_types.BoundaryCondition.NEUMANN, 1.0),
    }

    for idx in range(3):
      bc[var_names[idx + 2]][dim] = {
          0: (bc_types.BoundaryCondition.DIRICHLET, u_f0[idx]),
          1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
      }

    self.cfg.bc = {'cell_averages': bc}

    conservative_0 = {
        types.RHO: testing_utils.to_3d_tensor(1.0 * np.ones(16), dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(2.0 * np.ones(16), dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(3.0 * np.ones(16), dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(4.0 * np.ones(16), dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(5.0 * np.ones(16), dim, size),
    }

    conservative_1 = {
        types.RHO: testing_utils.to_3d_tensor(1.5 * np.ones(16), dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(2.5 * np.ones(16), dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(3.5 * np.ones(16), dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(4.5 * np.ones(16), dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(5.5 * np.ones(16), dim, size),
    }

    input_0 = [tf.constant(0), conservative_0]
    input_1 = [tf.constant(1), conservative_1]
    inputs = [input_0, input_1]

    # Define the device function.
    def device_fn(replica_id, conservative):
      """Wraps the boundary_update function to be executed on TPU."""
      return boundary.update_conservative_cell_averages(
          replica_id, replicas, conservative, self.cfg
      )

    output = self.run_tpu_test(replicas, device_fn, inputs)

    def make_expected(val_0, val_1, b_n, b_p):
      if replicas.shape[0] == 2:
        expected = np.concatenate(
            (val_0 * np.ones([10, 16, 16]), val_1 * np.ones([10, 16, 16])),
            axis=0,
        )
        expected = np.concatenate(
            (expected[-3:, :, :], expected, expected[:3, :, :]),
            axis=0,
        )
      elif replicas.shape[1] == 2:
        expected = np.concatenate(
            (val_0 * np.ones([16, 10, 16]), val_1 * np.ones([16, 10, 16])),
            axis=1,
        )
        expected = np.concatenate(
            (expected[:, -3:, :], expected, expected[:, :3, :]),
            axis=1,
        )
      else:  # replicas.shape[2] == 2
        expected = np.concatenate(
            (val_0 * np.ones([16, 16, 10]), val_1 * np.ones([16, 16, 10])),
            axis=2,
        )
        expected = np.concatenate(
            (expected[:, :, -3:], expected, expected[:, :, :3]),
            axis=2,
        )
      for i in range(3):
        if dim == 'x':
          expected[i, :, :] = b_n[i]
          expected[i - 3, :, :] = expected[i - 4, :, :] + b_p
        elif dim == 'y':
          expected[:, i, :] = b_n[i]
          expected[:, i - 3, :] = expected[:, i - 4, :] + b_p
        else:  # dim == 'z'
          expected[:, :, i] = b_n[i]
          expected[:, :, i - 3] = expected[:, :, i - 4] + b_p

      if replicas.shape[0] == 2:
        expected = [expected[:16, :, :], expected[10:, :, :]]
      elif replicas.shape[1] == 2:
        expected = [expected[:, :16, :], expected[:, 10:, :]]
      else:  # replicas.shape[2] == 2:
        expected = [expected[:, :, :16], expected[:, :, 10:]]
      expected[0] = np.transpose(expected[0], axes=(2, 0, 1))
      expected[1] = np.transpose(expected[1], axes=(2, 0, 1))
      return expected

    with self.subTest(name='replica 0, keys'):
      self.assertSequenceEqual(conservative_0.keys(), output[0].keys())
    with self.subTest(name='replica 1, keys'):
      self.assertSequenceEqual(conservative_1.keys(), output[1].keys())

    expected = make_expected(1.0, 1.5, [1.0, 2.0, 3.0], 0.0)
    for replica_id in range(2):
      with self.subTest(name=f'replica {replica_id}, rho'):
        self.assertAllClose(
            expected[replica_id], np.stack(output[replica_id]['rho'])
        )

    if dim == 'x':
      expected = make_expected(2.0, 2.5, [11.0, 42.0, 93.0], 0.0)
    else:
      expected = make_expected(2.0, 2.5, [11.0, 21.0, 31.0], 0.0)
    for replica_id in range(2):
      with self.subTest(name=f'replica {replica_id}, rho_u'):
        self.assertAllClose(
            expected[replica_id], np.stack(output[replica_id]['rho_u'])
        )

    if dim == 'y':
      expected = make_expected(3.0, 3.5, [-12.0, -44.0, -96.0], 0.0)
    else:
      expected = make_expected(3.0, 3.5, [-12.0, -22.0, -32.0], 0.0)
    for replica_id in range(2):
      with self.subTest(name=f'replica {replica_id}, rho_v'):
        self.assertAllClose(
            expected[replica_id], np.stack(output[replica_id]['rho_v'])
        )

    if dim == 'z':
      expected = make_expected(4.0, 4.5, [13.0, 46.0, 99.0], 0.0)
    else:
      expected = make_expected(4.0, 4.5, [13.0, 23.0, 33.0], 0.0)
    for replica_id in range(2):
      with self.subTest(name=f'replica {replica_id}, rho_w'):
        self.assertAllClose(
            expected[replica_id], np.stack(output[replica_id]['rho_w'])
        )

    expected = make_expected(5.0, 5.5, [15.0, 25.0, 35.0], 1.0)
    for replica_id in range(2):
      with self.subTest(name=f'replica {replica_id}, rho_e'):
        self.assertAllClose(
            expected[replica_id], np.stack(output[replica_id]['rho_e'])
        )

  @parameterized.parameters(*_DIMS)
  def test_get_bc_dict_along_dim_returns_correctly(self, dim):
    """Tests that the BC dictionary is parsed along a given dim correctly."""

    bc = {
        types.RHO: {
            'x': {
                0: (None, None),
                1: (None, None),
            },
            'y': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'z': {
                0: (None, None),
                1: (None, None),
            },
        },
        types.RHO_U: {
            'x': {
                0: (
                    bc_types.BoundaryCondition.DIRICHLET,
                    [2.0 * tf.ones((1, 16))] * 16,
                ),
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
        types.E: {
            'x': {
                0: (None, None),
                1: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
            },
            'y': {
                0: (None, None),
                1: (None, None),
            },
            'z': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 3.0),
                1: (None, None),
            },
        },
    }
    match dim:
      case 'x':
        expected_0 = {
            types.RHO_U: [2.0 * np.ones((1, 16))] * 16,
        }
        expected_1 = {
            types.E: 2.0,
        }
      case 'y':
        expected_0 = {
            types.RHO: 1.0,
        }
        expected_1 = {
            types.RHO: 2.0,
        }
      case _:  # 'z'
        expected_0 = {
            types.E: 3.0,
        }
        expected_1 = {}
    results_0 = boundary._get_bc_dict_along_dim(bc, dim, 0)
    results_1 = boundary._get_bc_dict_along_dim(bc, dim, 1)

    if dim == 'x':
      with self.subTest(name='x, 0'):
        self.assertSequenceEqual(expected_0.keys(), results_0.keys())
        self.assertAllClose(
            expected_0[types.RHO_U], self.evaluate(results_0[types.RHO_U])
        )
      with self.subTest(name='x, 1'):
        self.assertDictEqual(expected_1, results_1)
    elif dim == 'y':
      with self.subTest(name='y, 0'):
        self.assertDictEqual(expected_0, results_0)
      with self.subTest(name='y, 1'):
        self.assertDictEqual(expected_1, results_1)
    else:  # dim == 'z'
      with self.subTest(name='z, 0'):
        self.assertDictEqual(expected_0, results_0)
      with self.subTest(name='z, 1'):
        self.assertEqual(expected_1, results_1)

  def test_get_bc_dict_along_dim_raises_on_neumann(self):
    """Tests that Neumann face BC raises an error."""
    bc = {
        types.RHO: {
            'x': {
                0: (bc_types.BoundaryCondition.NEUMANN, 1.0),
                1: (bc_types.BoundaryCondition.NEUMANN, 1.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.NEUMANN, 1.0),
                1: (bc_types.BoundaryCondition.NEUMANN, 1.0),
            },
            'z': {
                0: (bc_types.BoundaryCondition.NEUMANN, 1.0),
                1: (bc_types.BoundaryCondition.NEUMANN, 1.0),
            },
        },
    }
    msg = r'^(Unsupported interface)'
    with self.assertRaisesRegex(AssertionError, msg):
      boundary._get_bc_dict_along_dim(bc, 'x', 0)
    with self.assertRaisesRegex(AssertionError, msg):
      boundary._get_bc_dict_along_dim(bc, 'y', 0)
    with self.assertRaisesRegex(AssertionError, msg):
      boundary._get_bc_dict_along_dim(bc, 'z', 1)

  def test_face_boundary_update_1d_returns_unchanged_for_none(self):
    """Confirms that conservative and primitive are unchanged for none BC."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    size = [16, 16, 16]
    rho = np.linspace(1.0, 2.0, 16)
    u = np.linspace(-1.0, 2.0, 16)
    v = np.linspace(0.0, 5.0, 16)
    w = -np.linspace(0.0, 6.0, 16)
    t = np.linspace(250.0, 500.0, 16)
    e = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)

    conservative = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'x', size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, 'x', size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, 'x', size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, 'x', size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, 'x', size),
    }
    primitive = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'x', size),
        types.U: testing_utils.to_3d_tensor(u, 'x', size),
        types.V: testing_utils.to_3d_tensor(v, 'x', size),
        types.W: testing_utils.to_3d_tensor(w, 'x', size),
        types.E: testing_utils.to_3d_tensor(e, 'x', size),
    }
    bc_dict_1d = None
    dim = 'x'
    loc = 3
    results = self.evaluate(
        boundary._face_boundary_update_1d(
            replica_id,
            replicas,
            conservative,
            primitive,
            bc_dict_1d,
            dim,
            0,
            loc,
            self.cfg,
        )
    )
    with self.subTest(name='conservative'):
      expected = {
          types.RHO: testing_utils.to_3d_tensor(
              rho, 'x', size, as_tf_tensor=False
          ),
          types.RHO_U: testing_utils.to_3d_tensor(
              rho * u, 'x', size, as_tf_tensor=False
          ),
          types.RHO_V: testing_utils.to_3d_tensor(
              rho * v, 'x', size, as_tf_tensor=False
          ),
          types.RHO_W: testing_utils.to_3d_tensor(
              rho * w, 'x', size, as_tf_tensor=False
          ),
          types.RHO_E: testing_utils.to_3d_tensor(
              rho * e, 'x', size, as_tf_tensor=False
          ),
      }
      self.assertDictEqual(expected, results[0])
    with self.subTest(name='primitive'):
      expected = {
          types.RHO: testing_utils.to_3d_tensor(
              rho, 'x', size, as_tf_tensor=False
          ),
          types.U: testing_utils.to_3d_tensor(u, 'x', size, as_tf_tensor=False),
          types.V: testing_utils.to_3d_tensor(v, 'x', size, as_tf_tensor=False),
          types.W: testing_utils.to_3d_tensor(w, 'x', size, as_tf_tensor=False),
          types.E: testing_utils.to_3d_tensor(e, 'x', size, as_tf_tensor=False),
      }
      self.assertDictEqual(expected, results[1])

  @parameterized.product(dim=_DIMS, side=_SIDES)
  def test_face_boundary_update_1d_updates_correctly(self, dim, side):
    """Confirms that conservative and primitive planes properly update."""
    self.cfg.conservative_variable_names = (
        [
            types.RHO,
        ]
        + list(types.MOMENTUM)
        + [
            types.RHO_E,
        ]
    )

    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    size = [16, 16, 16]

    rho = np.linspace(1.0, 2.0, 16)
    u = np.linspace(-1.0, 2.0, 16)
    v = np.linspace(0.0, 5.0, 16)
    w = -np.linspace(0.0, 6.0, 16)
    t = np.linspace(250.0, 500.0, 16)
    e = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    conservative = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'x', size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, 'x', size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, 'x', size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, 'x', size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, 'x', size),
    }
    primitive = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'x', size),
        types.U: testing_utils.to_3d_tensor(u, 'x', size),
        types.V: testing_utils.to_3d_tensor(v, 'x', size),
        types.W: testing_utils.to_3d_tensor(w, 'x', size),
        types.E: testing_utils.to_3d_tensor(e, 'x', size),
    }

    def make_bc(val):
      if dim == 'z':
        return val * tf.ones((16, 16), dtype=tf.float32)
      elif dim == 'x':
        return [val * tf.ones((1, 16))] * 16
      else:  # dim == 'y'
        return [val * tf.ones((16, 1))] * 16

    bc = {
        types.RHO: 1.5,
        types.U: 2.0,
        types.RHO_E: 12.0,
        types.RHO_V: 6.0,
        types.RHO_W: make_bc(9.0),
    }

    if side == 0:
      loc = 3
    else:  # side == 1
      loc = 13

    results = self.evaluate(
        boundary._face_boundary_update_1d(
            replica_id,
            replicas,
            conservative,
            primitive,
            bc,
            dim,
            0,
            loc,
            self.cfg,
        )
    )

    conservative_expected = {
        types.RHO: testing_utils.to_3d_tensor(
            rho, 'x', size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            rho * u, 'x', size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rho * v, 'x', size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rho * w, 'x', size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            rho * e, 'x', size, as_tf_tensor=False
        ),
    }
    primitive_expected = {
        types.RHO: testing_utils.to_3d_tensor(
            rho, 'x', size, as_tf_tensor=False
        ),
        types.U: testing_utils.to_3d_tensor(u, 'x', size, as_tf_tensor=False),
        types.V: testing_utils.to_3d_tensor(v, 'x', size, as_tf_tensor=False),
        types.W: testing_utils.to_3d_tensor(w, 'x', size, as_tf_tensor=False),
        types.E: testing_utils.to_3d_tensor(e, 'x', size, as_tf_tensor=False),
    }
    if side == 0:
      match dim:
        case 'x':
          conservative_expected['rho'][:, 3, :] = 1.5
          conservative_expected['rho_u'][:, 3, :] = 3.0
          conservative_expected['rho_v'][:, 3, :] = 6.0
          conservative_expected['rho_w'][:, 3, :] = 9.0
          conservative_expected['rho_e'][:, 3, :] = 12.0
          primitive_expected['rho'][:, 3, :] = 1.5
          primitive_expected['u'][:, 3, :] = 2.0
          primitive_expected['v'][:, 3, :] = 4.0
          primitive_expected['w'][:, 3, :] = 6.0
          primitive_expected['e'][:, 3, :] = 8.0
        case 'y':
          conservative_expected['rho'][:, :, 3] = 1.5
          conservative_expected['rho_u'][:, :, 3] = 3.0
          conservative_expected['rho_v'][:, :, 3] = 6.0
          conservative_expected['rho_w'][:, :, 3] = 9.0
          conservative_expected['rho_e'][:, :, 3] = 12.0
          primitive_expected['rho'][:, :, 3] = 1.5
          primitive_expected['u'][:, :, 3] = 2.0
          primitive_expected['v'][:, :, 3] = 4.0
          primitive_expected['w'][:, :, 3] = 6.0
          primitive_expected['e'][:, :, 3] = 8.0
        case _:  # case 'z'
          conservative_expected['rho'][3, :, :] = 1.5
          conservative_expected['rho_u'][3, :, :] = 3.0
          conservative_expected['rho_v'][3, :, :] = 6.0
          conservative_expected['rho_w'][3, :, :] = 9.0
          conservative_expected['rho_e'][3, :, :] = 12.0
          primitive_expected['rho'][3, :, :] = 1.5
          primitive_expected['u'][3, :, :] = 2.0
          primitive_expected['v'][3, :, :] = 4.0
          primitive_expected['w'][3, :, :] = 6.0
          primitive_expected['e'][3, :, :] = 8.0
    else:  # side == 1
      match dim:
        case 'x':
          conservative_expected['rho'][:, 13, :] = 1.5
          conservative_expected['rho_u'][:, 13, :] = 3.0
          conservative_expected['rho_v'][:, 13, :] = 6.0
          conservative_expected['rho_w'][:, 13, :] = 9.0
          conservative_expected['rho_e'][:, 13, :] = 12.0
          primitive_expected['rho'][:, 13, :] = 1.5
          primitive_expected['u'][:, 13, :] = 2.0
          primitive_expected['v'][:, 13, :] = 4.0
          primitive_expected['w'][:, 13, :] = 6.0
          primitive_expected['e'][:, 13, :] = 8.0
        case 'y':
          conservative_expected['rho'][:, :, 13] = 1.5
          conservative_expected['rho_u'][:, :, 13] = 3.0
          conservative_expected['rho_v'][:, :, 13] = 6.0
          conservative_expected['rho_w'][:, :, 13] = 9.0
          conservative_expected['rho_e'][:, :, 13] = 12.0
          primitive_expected['rho'][:, :, 13] = 1.5
          primitive_expected['u'][:, :, 13] = 2.0
          primitive_expected['v'][:, :, 13] = 4.0
          primitive_expected['w'][:, :, 13] = 6.0
          primitive_expected['e'][:, :, 13] = 8.0
        case _:  # case 'z'
          conservative_expected['rho'][13, :, :] = 1.5
          conservative_expected['rho_u'][13, :, :] = 3.0
          conservative_expected['rho_v'][13, :, :] = 6.0
          conservative_expected['rho_w'][13, :, :] = 9.0
          conservative_expected['rho_e'][13, :, :] = 12.0
          primitive_expected['rho'][13, :, :] = 1.5
          primitive_expected['u'][13, :, :] = 2.0
          primitive_expected['v'][13, :, :] = 4.0
          primitive_expected['w'][13, :, :] = 6.0
          primitive_expected['e'][13, :, :] = 8.0

    with self.subTest(name='conservative keys'):
      self.assertSequenceEqual(conservative.keys(), results[0].keys())
    with self.subTest(name='primitive keys'):
      self.assertSequenceEqual(primitive.keys(), results[1].keys())
    with self.subTest(name='conservative values'):
      self.assertDictEqual(conservative_expected, results[0])
    with self.subTest(name='primitive values'):
      self.assertDictEqual(primitive_expected, results[1])

  def test_update_faces_returns_unchanged_no_bc_update(self):
    """Confirms that the variables are unchanged if no BC specified."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    size = [16, 16, 16]
    self.cfg.bc = {}

    self.cfg.nx = 10
    self.cfg.ny = 10
    self.cfg.nz = 10

    rho = np.linspace(1.0, 2.0, 16)
    u = np.linspace(-1.0, 2.0, 16)
    v = np.linspace(0.0, 5.0, 16)
    w = -np.linspace(0.0, 6.0, 16)
    t = np.linspace(250.0, 500.0, 16)
    e = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    conservative_neg_face = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'x', size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, 'x', size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, 'x', size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, 'x', size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, 'x', size),
    }
    conservative_pos_face = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'y', size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, 'y', size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, 'y', size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, 'y', size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, 'y', size),
    }
    primitive_neg_face = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'x', size),
        types.U: testing_utils.to_3d_tensor(u, 'x', size),
        types.V: testing_utils.to_3d_tensor(v, 'x', size),
        types.W: testing_utils.to_3d_tensor(w, 'x', size),
        types.E: testing_utils.to_3d_tensor(e, 'x', size),
    }
    primitive_pos_face = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'y', size),
        types.U: testing_utils.to_3d_tensor(u, 'y', size),
        types.V: testing_utils.to_3d_tensor(v, 'y', size),
        types.W: testing_utils.to_3d_tensor(w, 'y', size),
        types.E: testing_utils.to_3d_tensor(e, 'y', size),
    }

    results = self.evaluate(
        boundary.update_faces(
            replica_id,
            replicas,
            {
                'x': conservative_neg_face,
                'y': conservative_neg_face,
                'z': conservative_neg_face,
            },
            {
                'x': conservative_pos_face,
                'y': conservative_pos_face,
                'z': conservative_pos_face,
            },
            {
                'x': primitive_neg_face,
                'y': primitive_neg_face,
                'z': primitive_neg_face,
            },
            {
                'x': primitive_pos_face,
                'y': primitive_pos_face,
                'z': primitive_pos_face,
            },
            self.cfg,
        )
    )

    expected_conservative_neg_face = {
        types.RHO: testing_utils.to_3d_tensor(
            rho, 'x', size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            rho * u, 'x', size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rho * v, 'x', size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rho * w, 'x', size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            rho * e, 'x', size, as_tf_tensor=False
        ),
    }
    expected_conservative_pos_face = {
        types.RHO: testing_utils.to_3d_tensor(
            rho, 'y', size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            rho * u, 'y', size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rho * v, 'y', size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rho * w, 'y', size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            rho * e, 'y', size, as_tf_tensor=False
        ),
    }
    expected_primitive_neg_face = {
        types.RHO: testing_utils.to_3d_tensor(
            rho, 'x', size, as_tf_tensor=False
        ),
        types.U: testing_utils.to_3d_tensor(u, 'x', size, as_tf_tensor=False),
        types.V: testing_utils.to_3d_tensor(v, 'x', size, as_tf_tensor=False),
        types.W: testing_utils.to_3d_tensor(w, 'x', size, as_tf_tensor=False),
        types.E: testing_utils.to_3d_tensor(e, 'x', size, as_tf_tensor=False),
    }
    expected_primitive_pos_face = {
        types.RHO: testing_utils.to_3d_tensor(
            rho, 'y', size, as_tf_tensor=False
        ),
        types.U: testing_utils.to_3d_tensor(u, 'y', size, as_tf_tensor=False),
        types.V: testing_utils.to_3d_tensor(v, 'y', size, as_tf_tensor=False),
        types.W: testing_utils.to_3d_tensor(w, 'y', size, as_tf_tensor=False),
        types.E: testing_utils.to_3d_tensor(e, 'y', size, as_tf_tensor=False),
    }

    for dim in types.DIMS:
      with self.subTest(name=f'conservative_neg_f{dim}'):
        self.assertDictEqual(expected_conservative_neg_face, results[0][dim])

      with self.subTest(name=f'conservative_pos_f{dim}'):
        self.assertDictEqual(expected_conservative_pos_face, results[1][dim])

      with self.subTest(name=f'primitive_neg_f{dim}'):
        self.assertDictEqual(expected_primitive_neg_face, results[2][dim])

      with self.subTest(name=f'primitive_pos_f{dim}'):
        self.assertDictEqual(expected_primitive_pos_face, results[3][dim])

  def test_update_faces_returns_correctly_1_replica(self):
    """Confirms that the variables updated correctly on 1 replica."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    size = [16, 16, 16]
    self.cfg.bc = {}
    self.cfg.conservative_variable_names = (
        [
            types.RHO,
        ]
        + list(types.MOMENTUM)
        + [
            types.RHO_E,
        ]
    )
    self.cfg.nx = 10
    self.cfg.ny = 10
    self.cfg.nz = 10

    rho = np.linspace(1.0, 2.0, 16)
    u = np.linspace(-1.0, 2.0, 16)
    v = np.linspace(0.0, 5.0, 16)
    w = -np.linspace(0.0, 6.0, 16)
    t = np.linspace(250.0, 500.0, 16)
    e = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    conservative_neg_face = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'x', size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, 'x', size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, 'x', size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, 'x', size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, 'x', size),
    }
    conservative_pos_face = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'y', size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, 'y', size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, 'y', size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, 'y', size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, 'y', size),
    }
    primitive_neg_face = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'x', size),
        types.U: testing_utils.to_3d_tensor(u, 'x', size),
        types.V: testing_utils.to_3d_tensor(v, 'x', size),
        types.W: testing_utils.to_3d_tensor(w, 'x', size),
        types.E: testing_utils.to_3d_tensor(e, 'x', size),
    }
    primitive_pos_face = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'y', size),
        types.U: testing_utils.to_3d_tensor(u, 'y', size),
        types.V: testing_utils.to_3d_tensor(v, 'y', size),
        types.W: testing_utils.to_3d_tensor(w, 'y', size),
        types.E: testing_utils.to_3d_tensor(e, 'y', size),
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
                0: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
            },
        },
        types.U: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
            },
            'z': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
            },
        },
        types.RHO_V: {
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
        types.RHO_W: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 4.0),
                1: (None, None),
            },
            'y': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 6.0),
                1: (None, None),
            },
            'z': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 8.0),
                1: (None, None),
            },
        },
        types.RHO_E: {
            'x': {
                0: (None, None),
                1: (bc_types.BoundaryCondition.DIRICHLET, 3.0),
            },
            'y': {
                0: (None, None),
                1: (bc_types.BoundaryCondition.DIRICHLET, 5.0),
            },
            'z': {
                0: (None, None),
                1: (bc_types.BoundaryCondition.DIRICHLET, 7.0),
            },
        },
    }

    results = self.evaluate(
        boundary.update_faces(
            replica_id,
            replicas,
            {
                'x': copy.deepcopy(conservative_neg_face),
                'y': copy.deepcopy(conservative_neg_face),
                'z': copy.deepcopy(conservative_neg_face),
            },
            {
                'x': copy.deepcopy(conservative_pos_face),
                'y': copy.deepcopy(conservative_pos_face),
                'z': copy.deepcopy(conservative_pos_face),
            },
            {
                'x': copy.deepcopy(primitive_neg_face),
                'y': copy.deepcopy(primitive_neg_face),
                'z': copy.deepcopy(primitive_neg_face),
            },
            {
                'x': copy.deepcopy(primitive_pos_face),
                'y': copy.deepcopy(primitive_pos_face),
                'z': copy.deepcopy(primitive_pos_face),
            },
            self.cfg,
        )
    )

    expected_conservative_neg_face = {
        types.RHO: testing_utils.to_3d_tensor(
            rho, 'x', size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            rho * u, 'x', size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rho * v, 'x', size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rho * w, 'x', size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            rho * e, 'x', size, as_tf_tensor=False
        ),
    }
    expected_conservative_pos_face = {
        types.RHO: testing_utils.to_3d_tensor(
            rho, 'y', size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            rho * u, 'y', size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rho * v, 'y', size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rho * w, 'y', size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            rho * e, 'y', size, as_tf_tensor=False
        ),
    }

    for dim in types.DIMS:
      with self.subTest(name=f'conservative_neg_f{dim}'):
        match dim:
          case 'x':
            expected = copy.deepcopy(expected_conservative_neg_face)
            expected['rho'][:, 3, :] = 1.0
            expected['rho'][:, 13, :] = 2.0
            expected['rho_u'][:, 3, :] = 0.0
            expected['rho_u'][:, 13, :] = 0.0
            expected['rho_w'][:, 3, :] = 4.0
            expected['rho_e'][:, 13, :] = 3.0
          case 'y':
            expected = copy.deepcopy(expected_conservative_neg_face)
            expected['rho_u'][:, :, 3] = 0.0
            expected['rho_u'][:, :, 13] = 0.0
            expected['rho_w'][:, :, 3] = 6.0
            expected['rho_e'][:, :, 13] = 5.0
          case _:  # 'z'
            expected = copy.deepcopy(expected_conservative_neg_face)
            expected['rho'][3, :, :] = 2.0
            expected['rho'][13, :, :] = 1.0
            expected['rho_u'][3, :, :] = 0.0
            expected['rho_u'][13, :, :] = 0.0
            expected['rho_w'][3, :, :] = 8.0
            expected['rho_e'][13, :, :] = 7.0
        self.assertDictEqual(expected, results[0][dim])

      with self.subTest(name=f'primitive_neg_f{dim}'):
        expected_primitive = {
            types.RHO: expected['rho'],
            types.U: expected['rho_u'] / expected['rho'],
            types.V: expected['rho_v'] / expected['rho'],
            types.W: expected['rho_w'] / expected['rho'],
            types.E: expected['rho_e'] / expected['rho'],
        }
        self.assertDictEqual(expected_primitive, results[2][dim])

      with self.subTest(name=f'conservative_pos_f{dim}'):
        match dim:
          case 'x':
            expected = copy.deepcopy(expected_conservative_pos_face)
            expected['rho'][:, 3, :] = 1.0
            expected['rho'][:, 13, :] = 2.0
            expected['rho_u'][:, 3, :] = 0.0
            expected['rho_u'][:, 13, :] = 0.0
            expected['rho_w'][:, 3, :] = 4.0
            expected['rho_e'][:, 13, :] = 3.0
          case 'y':
            expected = copy.deepcopy(expected_conservative_pos_face)
            expected['rho_u'][:, :, 3] = 0.0
            expected['rho_u'][:, :, 13] = 0.0
            expected['rho_w'][:, :, 3] = 6.0
            expected['rho_e'][:, :, 13] = 5.0
          case _:  # 'z'
            expected = copy.deepcopy(expected_conservative_pos_face)
            expected['rho'][3, :, :] = 2.0
            expected['rho'][13, :, :] = 1.0
            expected['rho_u'][3, :, :] = 0.0
            expected['rho_u'][13, :, :] = 0.0
            expected['rho_w'][3, :, :] = 8.0
            expected['rho_e'][13, :, :] = 7.0
        self.assertDictEqual(expected, results[1][dim])

      with self.subTest(name=f'primitive_pos_f{dim}'):
        expected_primitive = {
            types.RHO: expected['rho'],
            types.U: expected['rho_u'] / expected['rho'],
            types.V: expected['rho_v'] / expected['rho'],
            types.W: expected['rho_w'] / expected['rho'],
            types.E: expected['rho_e'] / expected['rho'],
        }
        self.assertDictEqual(expected_primitive, results[3][dim])

  @parameterized.parameters(*zip(_REPLICAS))
  def test_update_faces_returns_correctly_2_replicas(self, replicas):
    """Confirms that the variables updated correctly on 2 replicas."""
    size = [16, 16, 16]
    self.cfg.bc = {}
    self.cfg.conservative_variable_names = (
        [
            types.RHO,
        ]
        + list(types.MOMENTUM)
        + [
            types.RHO_E,
        ]
    )
    self.cfg.core_nx = 10
    self.cfg.core_ny = 10
    self.cfg.core_nz = 10

    rho = np.linspace(1.0, 2.0, 16)
    u = np.linspace(-1.0, 2.0, 16)
    v = np.linspace(0.0, 5.0, 16)
    w = -np.linspace(0.0, 6.0, 16)
    t = np.linspace(250.0, 500.0, 16)
    e = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    conservative_neg_face = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'x', size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, 'x', size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, 'x', size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, 'x', size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, 'x', size),
    }
    conservative_pos_face = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'y', size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, 'y', size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, 'y', size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, 'y', size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, 'y', size),
    }
    primitive_neg_face = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'x', size),
        types.U: testing_utils.to_3d_tensor(u, 'x', size),
        types.V: testing_utils.to_3d_tensor(v, 'x', size),
        types.W: testing_utils.to_3d_tensor(w, 'x', size),
        types.E: testing_utils.to_3d_tensor(e, 'x', size),
    }
    primitive_pos_face = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'y', size),
        types.U: testing_utils.to_3d_tensor(u, 'y', size),
        types.V: testing_utils.to_3d_tensor(v, 'y', size),
        types.W: testing_utils.to_3d_tensor(w, 'y', size),
        types.E: testing_utils.to_3d_tensor(e, 'y', size),
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
                0: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
            },
        },
        types.U: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
            },
            'z': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
            },
        },
        types.RHO_V: {
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
        types.RHO_W: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 4.0),
                1: (None, None),
            },
            'y': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 6.0),
                1: (None, None),
            },
            'z': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 8.0),
                1: (None, None),
            },
        },
        types.RHO_E: {
            'x': {
                0: (None, None),
                1: (bc_types.BoundaryCondition.DIRICHLET, 3.0),
            },
            'y': {
                0: (None, None),
                1: (bc_types.BoundaryCondition.DIRICHLET, 5.0),
            },
            'z': {
                0: (None, None),
                1: (bc_types.BoundaryCondition.DIRICHLET, 7.0),
            },
        },
    }

    conservative_neg = {
        'x': conservative_neg_face,
        'y': conservative_neg_face,
        'z': conservative_neg_face,
    }
    conservative_pos = {
        'x': conservative_pos_face,
        'y': conservative_pos_face,
        'z': conservative_pos_face,
    }
    primitive_neg = {
        'x': primitive_neg_face,
        'y': primitive_neg_face,
        'z': primitive_neg_face,
    }
    primitive_pos = {
        'x': primitive_pos_face,
        'y': primitive_pos_face,
        'z': primitive_pos_face,
    }
    input_0 = [
        tf.constant(0),
        conservative_neg,
        conservative_pos,
        primitive_neg,
        primitive_pos,
    ]
    input_1 = [
        tf.constant(1),
        conservative_neg,
        conservative_pos,
        primitive_neg,
        primitive_pos,
    ]
    inputs = [input_0, input_1]

    # Define the device function.
    def device_fn(
        replica_id,
        conservative_neg,
        conservative_pos,
        primitive_neg,
        primitive_pos,
    ):
      """Wraps the boundary_update function to be executed on TPU."""
      return boundary.update_faces(
          replica_id,
          replicas,
          conservative_neg,
          conservative_pos,
          primitive_neg,
          primitive_pos,
          self.cfg,
      )

    output = self.run_tpu_test(replicas, device_fn, inputs)

    expected_conservative_neg_face = {
        types.RHO: testing_utils.to_3d_tensor(
            rho, 'x', size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            rho * u, 'x', size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rho * v, 'x', size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rho * w, 'x', size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            rho * e, 'x', size, as_tf_tensor=False
        ),
    }
    expected_conservative_pos_face = {
        types.RHO: testing_utils.to_3d_tensor(
            rho, 'y', size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            rho * u, 'y', size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rho * v, 'y', size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rho * w, 'y', size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            rho * e, 'y', size, as_tf_tensor=False
        ),
    }

    for face_idx, face_str, face_val in zip(
        (0, 1),
        ('neg', 'pos'),
        (expected_conservative_neg_face, expected_conservative_pos_face),
    ):
      for dim in types.DIMS:
        expected = [
            copy.deepcopy(face_val),
            copy.deepcopy(face_val),
        ]
        match dim:
          case 'x':
            if replicas.shape[0] == 1:
              expected[0]['rho'][:, 3, :] = 1.0
              expected[0]['rho'][:, 13, :] = 2.0
              expected[0]['rho_u'][:, 3, :] = 0.0
              expected[0]['rho_u'][:, 13, :] = 0.0
              expected[0]['rho_w'][:, 3, :] = 4.0
              expected[0]['rho_e'][:, 13, :] = 3.0
              expected[1] = expected[0]
            else:
              expected[0]['rho'][:, 3, :] = 1.0
              expected[1]['rho'][:, 13, :] = 2.0
              expected[0]['rho_u'][:, 3, :] = 0.0
              expected[1]['rho_u'][:, 13, :] = 0.0
              expected[0]['rho_w'][:, 3, :] = 4.0
              expected[1]['rho_e'][:, 13, :] = 3.0
          case 'y':
            if replicas.shape[1] == 1:
              expected[0]['rho_u'][:, :, 3] = 0.0
              expected[0]['rho_u'][:, :, 13] = 0.0
              expected[0]['rho_w'][:, :, 3] = 6.0
              expected[0]['rho_e'][:, :, 13] = 5.0
              expected[1] = expected[0]
            else:
              expected[0]['rho_u'][:, :, 3] = 0.0
              expected[1]['rho_u'][:, :, 13] = 0.0
              expected[0]['rho_w'][:, :, 3] = 6.0
              expected[1]['rho_e'][:, :, 13] = 5.0
          case _:  # 'z'
            if replicas.shape[2] == 1:
              expected[0]['rho'][3, :, :] = 2.0
              expected[0]['rho'][13, :, :] = 1.0
              expected[0]['rho_u'][3, :, :] = 0.0
              expected[0]['rho_u'][13, :, :] = 0.0
              expected[0]['rho_w'][3, :, :] = 8.0
              expected[0]['rho_e'][13, :, :] = 7.0
              expected[1] = expected[0]
            else:
              expected[0]['rho'][3, :, :] = 2.0
              expected[1]['rho'][13, :, :] = 1.0
              expected[0]['rho_u'][3, :, :] = 0.0
              expected[1]['rho_u'][13, :, :] = 0.0
              expected[0]['rho_w'][3, :, :] = 8.0
              expected[1]['rho_e'][13, :, :] = 7.0

        for replica_id in range(2):
          expected_primitive = {
              types.RHO: expected[replica_id]['rho'],
              types.U: (
                  expected[replica_id]['rho_u'] / expected[replica_id]['rho']
              ),
              types.V: (
                  expected[replica_id]['rho_v'] / expected[replica_id]['rho']
              ),
              types.W: (
                  expected[replica_id]['rho_w'] / expected[replica_id]['rho']
              ),
              types.E: (
                  expected[replica_id]['rho_e'] / expected[replica_id]['rho']
              ),
          }

          with self.subTest(
              name=f'conservative_{face_str}_f{dim}, replica_{replica_id}'
          ):
            self.assertDictEqual(
                expected[replica_id],
                output[replica_id][face_idx][dim],
            )

          with self.subTest(
              name=f'primitive_{face_str}_f{dim}, replica_{replica_id}'
          ):
            self.assertDictEqual(
                expected_primitive, output[replica_id][face_idx + 2][dim]
            )

  def test_update_fluxes_returns_unchanged_no_bc_update_convective(self):
    """Confirms that the variables are unchanged if no BC specified."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    size = [16, 16, 16]
    self.cfg.bc = {}

    self.cfg.nx = 10
    self.cfg.ny = 10
    self.cfg.nz = 10

    rho = np.linspace(1.0, 2.0, 16)
    u = np.linspace(-1.0, 2.0, 16)
    v = np.linspace(0.0, 5.0, 16)
    w = -np.linspace(0.0, 6.0, 16)
    t = np.linspace(250.0, 500.0, 16)
    e = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    flux_1d = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'x', size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, 'x', size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, 'x', size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, 'x', size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, 'x', size),
    }
    flux_3d = {
        'x': flux_1d,
        'y': flux_1d,
        'z': flux_1d,
    }

    results = self.evaluate(
        boundary.update_fluxes(
            replica_id,
            replicas,
            flux_3d,
            self.cfg,
            bc_types.BoundaryFluxType.CONVECTIVE,
        )
    )

    expected_1d = {
        types.RHO: testing_utils.to_3d_tensor(
            rho, 'x', size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            rho * u, 'x', size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rho * v, 'x', size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rho * w, 'x', size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            rho * e, 'x', size, as_tf_tensor=False
        ),
    }
    for dim in types.DIMS:
      with self.subTest(name=dim):
        self.assertDictEqual(expected_1d, results[dim])

  def test_update_fluxes_raises_assertion_error_for_primitive(self):
    """Tests that specifying a primitive flux raises assertion error."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    size = [16, 16, 16]
    self.cfg.bc = {}
    self.cfg.conservative_variable_names = (
        [
            types.RHO,
        ]
        + list(types.MOMENTUM)
        + [
            types.RHO_E,
        ]
    )
    self.cfg.nx = 10
    self.cfg.ny = 10
    self.cfg.nz = 10

    rho = np.linspace(1.0, 2.0, 16)
    u = np.linspace(-1.0, 2.0, 16)
    v = np.linspace(0.0, 5.0, 16)
    w = -np.linspace(0.0, 6.0, 16)
    t = np.linspace(250.0, 500.0, 16)
    e = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    flux_1d = {
        types.RHO: testing_utils.to_3d_tensor(rho, 'x', size),
        types.RHO_U: testing_utils.to_3d_tensor(rho * u, 'x', size),
        types.RHO_V: testing_utils.to_3d_tensor(rho * v, 'x', size),
        types.RHO_W: testing_utils.to_3d_tensor(rho * w, 'x', size),
        types.RHO_E: testing_utils.to_3d_tensor(rho * e, 'x', size),
    }
    flux_3d = {
        'x': flux_1d,
        'y': flux_1d,
        'z': flux_1d,
    }
    self.cfg.bc = {'intercell_fluxes': {}}
    self.cfg.bc['intercell_fluxes']['convective'] = {
        types.RHO: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
            },
            'z': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
            },
        },
        types.U: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
            },
            'z': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
            },
        },
    }

    msg = r"^({'u'} cannot be specified by a flux boundary condition)"
    with self.assertRaisesRegex(AssertionError, msg):
      boundary.update_fluxes(
          replica_id,
          replicas,
          flux_3d,
          self.cfg,
          bc_types.BoundaryFluxType.CONVECTIVE,
      )

  @parameterized.parameters(*_FLUX_TYPES)
  def test_update_fluxes_returns_correctly_all_types_1_replica(self, flux_type):
    """Confirms that the fluxes are updated correctly on 1 replica."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    size = [16, 16, 16]
    self.cfg.bc = {'intercell_fluxes': {}}
    self.cfg.conservative_variable_names = (
        [
            types.RHO,
        ]
        + list(types.MOMENTUM)
        + [
            types.RHO_E,
        ]
    )
    self.cfg.nx = 10
    self.cfg.ny = 10
    self.cfg.nz = 10

    rho = np.linspace(1.0, 2.0, 16)
    u = np.linspace(-1.0, 2.0, 16)
    v = np.linspace(0.0, 5.0, 16)
    w = -np.linspace(0.0, 6.0, 16)
    t = np.linspace(250.0, 500.0, 16)
    e = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    fluxes = {}
    for dim in types.DIMS:
      fluxes[dim] = {
          types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
          types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
          types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
          types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
          types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
      }
    self.cfg.bc['intercell_fluxes'][flux_type.value] = {
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
                0: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
            },
        },
        types.RHO_U: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
            },
            'z': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
            },
        },
        types.RHO_V: {
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
        types.RHO_W: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 4.0),
                1: (None, None),
            },
            'y': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 6.0),
                1: (None, None),
            },
            'z': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 8.0),
                1: (None, None),
            },
        },
        types.RHO_E: {
            'x': {
                0: (None, None),
                1: (bc_types.BoundaryCondition.DIRICHLET, 3.0),
            },
            'y': {
                0: (None, None),
                1: (bc_types.BoundaryCondition.DIRICHLET, 5.0),
            },
            'z': {
                0: (None, None),
                1: (bc_types.BoundaryCondition.DIRICHLET, 7.0),
            },
        },
    }

    results = self.evaluate(
        boundary.update_fluxes(
            replica_id,
            replicas,
            fluxes,
            self.cfg,
            flux_type,
        )
    )

    for dim in types.DIMS:
      expected = {
          types.RHO: testing_utils.to_3d_tensor(
              rho, dim, size, as_tf_tensor=False
          ),
          types.RHO_U: testing_utils.to_3d_tensor(
              rho * u, dim, size, as_tf_tensor=False
          ),
          types.RHO_V: testing_utils.to_3d_tensor(
              rho * v, dim, size, as_tf_tensor=False
          ),
          types.RHO_W: testing_utils.to_3d_tensor(
              rho * w, dim, size, as_tf_tensor=False
          ),
          types.RHO_E: testing_utils.to_3d_tensor(
              rho * e, dim, size, as_tf_tensor=False
          ),
      }
      match dim:
        case 'x':
          expected['rho'][:, 3, :] = 1.0
          expected['rho'][:, 13, :] = 2.0
          expected['rho_u'][:, 3, :] = 0.0
          expected['rho_u'][:, 13, :] = 0.0
          expected['rho_w'][:, 3, :] = 4.0
          expected['rho_e'][:, 13, :] = 3.0
        case 'y':
          expected['rho_u'][:, :, 3] = 0.0
          expected['rho_u'][:, :, 13] = 0.0
          expected['rho_w'][:, :, 3] = 6.0
          expected['rho_e'][:, :, 13] = 5.0
        case _:  # 'z'
          expected['rho'][3, :, :] = 2.0
          expected['rho'][13, :, :] = 1.0
          expected['rho_u'][3, :, :] = 0.0
          expected['rho_u'][13, :, :] = 0.0
          expected['rho_w'][3, :, :] = 8.0
          expected['rho_e'][13, :, :] = 7.0

      with self.subTest(name=dim):
        self.assertDictEqual(expected, results[dim])

  @parameterized.parameters(*zip(_REPLICAS))
  def test_update_fluxes_returns_correctly_convection_2_replicas(
      self, replicas
  ):
    """Confirms that the fluxes are updated correctly on 2 replicas."""
    size = [16, 16, 16]
    self.cfg.bc = {'intercell_fluxes': {}}
    self.cfg.conservative_variable_names = (
        [
            types.RHO,
        ]
        + list(types.MOMENTUM)
        + [
            types.RHO_E,
        ]
    )
    self.cfg.core_nx = 10
    self.cfg.core_ny = 10
    self.cfg.core_nz = 10

    rho = np.linspace(1.0, 2.0, 16)
    u = np.linspace(-1.0, 2.0, 16)
    v = np.linspace(0.0, 5.0, 16)
    w = -np.linspace(0.0, 6.0, 16)
    t = np.linspace(250.0, 500.0, 16)
    e = constant.CP * t + 0.5 * (u**2 + v**2 + w**2)
    fluxes = {}
    for dim in types.DIMS:
      fluxes[dim] = {
          types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
          types.RHO_U: testing_utils.to_3d_tensor(rho * u, dim, size),
          types.RHO_V: testing_utils.to_3d_tensor(rho * v, dim, size),
          types.RHO_W: testing_utils.to_3d_tensor(rho * w, dim, size),
          types.RHO_E: testing_utils.to_3d_tensor(rho * e, dim, size),
      }
    self.cfg.bc['intercell_fluxes']['convective'] = {
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
                0: (bc_types.BoundaryCondition.DIRICHLET, 2.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
            },
        },
        types.RHO_U: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
            },
            'y': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
            },
            'z': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
            },
        },
        types.RHO_V: {
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
        types.RHO_W: {
            'x': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 4.0),
                1: (None, None),
            },
            'y': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 6.0),
                1: (None, None),
            },
            'z': {
                0: (bc_types.BoundaryCondition.DIRICHLET, 8.0),
                1: (None, None),
            },
        },
        types.RHO_E: {
            'x': {
                0: (None, None),
                1: (bc_types.BoundaryCondition.DIRICHLET, 3.0),
            },
            'y': {
                0: (None, None),
                1: (bc_types.BoundaryCondition.DIRICHLET, 5.0),
            },
            'z': {
                0: (None, None),
                1: (bc_types.BoundaryCondition.DIRICHLET, 7.0),
            },
        },
    }

    input_0 = [tf.constant(0), fluxes]
    input_1 = [tf.constant(1), fluxes]
    inputs = [input_0, input_1]

    # Define the device function.
    def device_fn(replica_id, fluxes):
      """Wraps the boundary_update function to be executed on TPU."""
      return boundary.update_fluxes(
          replica_id,
          replicas,
          fluxes,
          self.cfg,
          bc_types.BoundaryFluxType.CONVECTIVE,
      )

    output = self.run_tpu_test(replicas, device_fn, inputs)

    for dim in types.DIMS:
      expected_base = {
          types.RHO: testing_utils.to_3d_tensor(
              rho, dim, size, as_tf_tensor=False
          ),
          types.RHO_U: testing_utils.to_3d_tensor(
              rho * u, dim, size, as_tf_tensor=False
          ),
          types.RHO_V: testing_utils.to_3d_tensor(
              rho * v, dim, size, as_tf_tensor=False
          ),
          types.RHO_W: testing_utils.to_3d_tensor(
              rho * w, dim, size, as_tf_tensor=False
          ),
          types.RHO_E: testing_utils.to_3d_tensor(
              rho * e, dim, size, as_tf_tensor=False
          ),
      }
      expected = [copy.deepcopy(expected_base), copy.deepcopy(expected_base)]
      match dim:
        case 'x':
          if replicas.shape[0] == 1:
            expected[0]['rho'][:, 3, :] = 1.0
            expected[0]['rho'][:, 13, :] = 2.0
            expected[0]['rho_u'][:, 3, :] = 0.0
            expected[0]['rho_u'][:, 13, :] = 0.0
            expected[0]['rho_w'][:, 3, :] = 4.0
            expected[0]['rho_e'][:, 13, :] = 3.0
            expected[1] = expected[0]
          else:
            expected[0]['rho'][:, 3, :] = 1.0
            expected[1]['rho'][:, 13, :] = 2.0
            expected[0]['rho_u'][:, 3, :] = 0.0
            expected[1]['rho_u'][:, 13, :] = 0.0
            expected[0]['rho_w'][:, 3, :] = 4.0
            expected[1]['rho_e'][:, 13, :] = 3.0
        case 'y':
          if replicas.shape[1] == 1:
            expected[0]['rho_u'][:, :, 3] = 0.0
            expected[0]['rho_u'][:, :, 13] = 0.0
            expected[0]['rho_w'][:, :, 3] = 6.0
            expected[0]['rho_e'][:, :, 13] = 5.0
            expected[1] = expected[0]
          else:
            expected[0]['rho_u'][:, :, 3] = 0.0
            expected[1]['rho_u'][:, :, 13] = 0.0
            expected[0]['rho_w'][:, :, 3] = 6.0
            expected[1]['rho_e'][:, :, 13] = 5.0
        case _:  # 'z'
          if replicas.shape[2] == 1:
            expected[0]['rho'][3, :, :] = 2.0
            expected[0]['rho'][13, :, :] = 1.0
            expected[0]['rho_u'][3, :, :] = 0.0
            expected[0]['rho_u'][13, :, :] = 0.0
            expected[0]['rho_w'][3, :, :] = 8.0
            expected[0]['rho_e'][13, :, :] = 7.0
            expected[1] = expected[0]
          else:
            expected[0]['rho'][3, :, :] = 2.0
            expected[1]['rho'][13, :, :] = 1.0
            expected[0]['rho_u'][3, :, :] = 0.0
            expected[1]['rho_u'][13, :, :] = 0.0
            expected[0]['rho_w'][3, :, :] = 8.0
            expected[1]['rho_e'][13, :, :] = 7.0

      for replica_id in range(2):
        with self.subTest(name=f'{dim}, replica_id={replica_id}'):
          self.assertDictEqual(expected[replica_id], output[replica_id][dim])


if __name__ == '__main__':
  tf.test.main()
