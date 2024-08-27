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
"""Tests for utils."""

from absl.testing import parameterized
import numpy as np
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.common import utils
from swirl_c.core import parameter
import tensorflow as tf


class UtilsTest(tf.test.TestCase, parameterized.TestCase):
  _REPLICA_ID = (0, 119, 37)

  def test_mesh_local_size_computes_correct_size(self):
    """Checks the computed local mesh size from replicas and global mesh."""
    user_cfg = {
        'x': np.linspace(0.0, 39.0, 32),
        'y': np.linspace(0.0, 49.0, 45),
        'z': np.linspace(0.0, 59.0, 60),
        'cx': 4,
        'cy': 5,
        'cz': 6,
    }
    cfg = parameter.SwirlCParameters(user_cfg)
    res = utils.mesh_local_size(cfg)
    expected = [8, 9, 10]
    self.assertListEqual(expected, res)

  @parameterized.named_parameters(
      (
          'Replica000',
          0,
          {
              'x': np.arange(-3, 13),
              'y': np.arange(-3, 13),
              'z': np.arange(-3, 13),
          },
      ),
      (
          'Replica345',
          119,
          {
              'x': np.arange(27, 43),
              'y': np.arange(37, 53),
              'z': np.arange(47, 63),
          },
      ),
      (
          'Replica111',
          37,
          {'x': np.arange(7, 23), 'y': np.arange(7, 23), 'z': np.arange(7, 23)},
      ),
  )
  def test_mesh_local_provides_correct_mesh_per_replica(
      self, replica_id, expected
  ):
    """Checks if the local mesh is retrieved correctly from a global mesh."""
    computation_shape = (4, 5, 6)
    user_cfg = {
        'x': np.linspace(0.0, 39.0, 40),
        'y': np.linspace(0.0, 49.0, 50),
        'z': np.linspace(0.0, 59.0, 60),
        'cx': 4,
        'cy': 5,
        'cz': 6,
    }
    cfg = parameter.SwirlCParameters(user_cfg)
    replicas = np.reshape(
        np.arange(np.prod(computation_shape)), computation_shape
    )
    replica_id = tf.constant(replica_id)

    res = self.evaluate(utils.mesh_local(replica_id, replicas, cfg))

    self.assertDictEqual(expected, res)

  @parameterized.named_parameters(
      (
          'Replica000',
          0,
          {
              'x': np.arange(-3, 13),
              'y': np.arange(-3, 13),
              'z': np.arange(-3, 13),
          },
      ),
      (
          'Replica345',
          119,
          {
              'x': np.arange(27, 43),
              'y': np.arange(37, 53),
              'z': np.arange(47, 63),
          },
      ),
      (
          'Replica111',
          37,
          {
              'x': np.arange(7, 23),
              'y': np.arange(7, 23),
              'z': np.arange(7, 23),
          },
      ),
  )
  def test_mesh_local_expanded_provides_correct_mesh_per_replica(
      self, replica_id, expected
  ):
    """Checks if the local mesh is expanded correctly from a global mesh."""
    computation_shape = (4, 5, 6)
    user_cfg = {
        'x': np.linspace(0.0, 39.0, 40),
        'y': np.linspace(0.0, 49.0, 50),
        'z': np.linspace(0.0, 59.0, 60),
        'cx': 4,
        'cy': 5,
        'cz': 6,
    }
    cfg = parameter.SwirlCParameters(user_cfg)
    replicas = np.reshape(
        np.arange(np.prod(computation_shape)), computation_shape
    )
    replica_id = tf.constant(replica_id)

    res = self.evaluate(utils.mesh_local_expanded(replica_id, replicas, cfg))

    expected['x'] = [
        np.expand_dims(expected['x'], 1),
    ] * 16
    expected['y'] = [
        np.expand_dims(expected['y'], 0),
    ] * 16
    expected['z'] = tf.unstack(expected['z'][:, np.newaxis, np.newaxis])

    self.assertDictEqual(expected, res)

  _FAKE_DIMS = ('x', 'y', 'z', 'w')

  @parameterized.parameters(*zip(_FAKE_DIMS))
  def test_gravity_direction_finds_the_correct_dimension(self, gravity_dim):
    """Checks if the dimension of the gravity is derived correctly."""
    dims = ('x', 'y', 'z')
    cfg = parameter.SwirlCParameters()
    if gravity_dim in dims:
      cfg.g[gravity_dim] = -1.0
      self.assertEqual(
          dims.index(gravity_dim),
          utils.gravity_direction(cfg),
      )
    else:
      self.assertEqual(-1, utils.gravity_direction(cfg))

  def test_gravity_direction_returns_correct_misaligned_gravity(self):
    """Checks the return is correct if gravity is not grid aligned."""
    cfg = parameter.SwirlCParameters()
    theta = np.pi / 3
    phi = -np.pi / 3
    cfg.g['x'] = np.sin(phi) * np.cos(theta)
    cfg.g['y'] = np.sin(phi) * np.sin(theta)
    cfg.g['z'] = np.cos(phi)
    self.assertEqual(3, utils.gravity_direction(cfg))

  _DIMS = ('x', 'y', 'z')

  @parameterized.parameters(*zip(_DIMS))
  def test_conversion_to_primitives(self, dim):
    """Checks the conversion from conservative to primitive variables."""
    size = [14, 12, 15]
    if dim == 'x':
      vector_length = size[1]
    elif dim == 'y':
      vector_length = size[2]
    else:
      vector_length = size[0]
    # Set flow field variables including density and velocity.
    rho = np.logspace(-1.0, 2.0, vector_length)
    u = np.linspace(-3.0, 12.0, vector_length)
    v = np.linspace(6.0, -24.0, vector_length)
    w = np.linspace(-9.0, 36.0, vector_length)
    # To mimic changes in temperature, we will assume an internal energy that
    # is proportional to the kinetic by the factor specified.
    proportional_factor = np.logspace(-1.0, 1.0, vector_length)
    # Compute kinetic and total energy for given flow.
    ke = 0.5 * (u**2 + v**2 + w**2)
    e = ke * (1 + proportional_factor)
    # We also define a pressure vector and state, which should be ignored by the
    # convesion.
    p = np.logspace(3.0, 6.0, vector_length)

    # Compute the conserved variables based on primitive variables.
    rhou = rho * u
    rhov = rho * v
    rhow = rho * w
    rhoe = rho * e

    states = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(rhou, dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(rhov, dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(rhow, dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(rhoe, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
    }

    expected = {
        types.RHO: testing_utils.to_3d_tensor(
            rho, dim, size, as_tf_tensor=False
        ),
        types.U: testing_utils.to_3d_tensor(u, dim, size, as_tf_tensor=False),
        types.V: testing_utils.to_3d_tensor(v, dim, size, as_tf_tensor=False),
        types.W: testing_utils.to_3d_tensor(w, dim, size, as_tf_tensor=False),
        types.E: testing_utils.to_3d_tensor(e, dim, size, as_tf_tensor=False),
    }

    result = self.evaluate(utils.conservative_to_primitive_variables(states))
    self.assertSequenceEqual(expected.keys(), result.keys())
    for var_name, expected_value in expected.items():
      with self.subTest(name=var_name):
        self.assertAllClose(expected_value, result[var_name])

  @parameterized.parameters(*zip(_DIMS))
  def test_conversion_to_conservatives(self, dim):
    """Checks the conversion from primitive to conservative variables."""
    size = [14, 12, 15]
    if dim == 'x':
      vector_length = size[1]
    elif dim == 'y':
      vector_length = size[2]
    else:
      vector_length = size[0]
    # Set flow field variables including density and velocity.
    rho = np.logspace(-1.0, 2.0, vector_length)
    u = np.linspace(-3.0, 12.0, vector_length)
    v = np.linspace(6.0, -24.0, vector_length)
    w = np.linspace(-9.0, 36.0, vector_length)
    # To mimic changes in temperature, we will assume an internal energy that
    # is proportional to the kinetic by the factor specified.
    proportional_factor = np.logspace(-1.0, 1.0, vector_length)
    # Compute kinetic and total energy for given flow.
    ke = 0.5 * (u**2 + v**2 + w**2)
    e = ke * (1 + proportional_factor)

    # Compute the conserved variables based on primitive variables.
    rhou = rho * u
    rhov = rho * v
    rhow = rho * w
    rhoe = rho * e

    states = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.E: testing_utils.to_3d_tensor(e, dim, size),
    }

    expected = {
        types.RHO: testing_utils.to_3d_tensor(
            rho, dim, size, as_tf_tensor=False
        ),
        types.RHO_U: testing_utils.to_3d_tensor(
            rhou, dim, size, as_tf_tensor=False
        ),
        types.RHO_V: testing_utils.to_3d_tensor(
            rhov, dim, size, as_tf_tensor=False
        ),
        types.RHO_W: testing_utils.to_3d_tensor(
            rhow, dim, size, as_tf_tensor=False
        ),
        types.RHO_E: testing_utils.to_3d_tensor(
            rhoe, dim, size, as_tf_tensor=False
        ),
    }

    result = self.evaluate(utils.primitive_to_conservative_variables(states))
    self.assertSequenceEqual(expected.keys(), result.keys())
    for var_name, expected_value in expected.items():
      with self.subTest(name=var_name):
        self.assertAllClose(expected_value, result[var_name])


if __name__ == '__main__':
  tf.test.main()
