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
"""Tests for generic."""

from absl.testing import parameterized
import numpy as np
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.physics import constant
from swirl_c.physics.thermodynamics import generic
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner


class GenericTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes the generic thermodynamics library object."""
    super().setUp()
    self.model = generic.ThermodynamicsGeneric()
    xx = np.linspace(0.0, 1800.0, 10)
    self.cfg = parameter.SwirlCParameters({'x': xx, 'y': xx, 'z': xx})

  def run_tpu_test(self, replicas, device_fn, inputs):
    """Runs `device` function on TPU."""
    device_inputs = [list(x) for x in zip(*inputs)]
    computation_shape = replicas.shape
    runner = TpuRunner(computation_shape=computation_shape)
    return runner.run(device_fn, *device_inputs)

  _DIMS = ('x', 'y', 'z')
  _GRAVITY_DIMS = ('x', 'y', 'z', 'w')
  _GRAVITY_THETA = tuple(np.pi * np.linspace(0.0, 1.0, 6)[:-1])
  _GRAVITY_PHI = tuple(np.pi * np.linspace(-1.0, 1.0, 8)[1:])

  @parameterized.parameters(*zip(_DIMS))
  def test_pressure_derived_from_total_energy_correctly(self, dim):
    """Checks if the pressure is computed correctly from the total energy."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    self.cfg.g[dim] = -1.0
    size = [16, 16, 16]

    # Set the expected pressure to be at constant potential temperature of 300K.
    z = np.linspace(-600.0, 2400.0, 16)
    theta = 300.0
    p = constant.P_0 * (1.0 - constant.G * z / constant.CP / theta) ** (
        1.0 / constant.KAPPA
    )
    t = theta * (p / constant.P_0) ** (constant.KAPPA)
    rho = p / constant.R / t

    u = np.linspace(-3.0, 12.0, 16)
    v = np.linspace(6.0, -24.0, 16)
    w = np.linspace(-9.0, 36.0, 16)
    e = 0.5 * (u**2 + v**2 + w**2) + constant.G * z + constant.CV * t

    states = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.E: testing_utils.to_3d_tensor(e, dim, size),
    }
    res = self.evaluate(
        self.model.pressure(replica_id, replicas, states, self.cfg)
    )
    expected = testing_utils.to_3d_tensor(p, dim, size, as_tf_tensor=False)
    self.assertAllClose(expected, res)

  @parameterized.parameters(*zip(_DIMS))
  def test_total_enthalpy_derived_from_total_energy_correctly(self, dim):
    """Checks if total enthalpy is computed correctly from the total energy."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    self.cfg.g[dim] = -1.0
    # Compute the size of the test domain, assuming the domain is cubic.
    # The size includes the halo width.
    nx = len(self.cfg.x) + 2 * self.cfg.halo_width
    size = [nx, nx, nx]
    # Set flow field variables includeing density, velocity, temperature, and
    # geopotential height.
    rho = np.logspace(-1.0, 2.0, nx)
    u = np.linspace(-3.0, 12.0, nx)
    v = np.linspace(6.0, -24.0, nx)
    w = np.linspace(-9.0, 36.0, nx)
    t = np.linspace(250.0, 3000.0, nx)
    z = np.linspace(-600.0, 2400.0, nx)
    # Compute kinetic and total energy from flow field variables.
    ke = 0.5 * (u**2 + v**2 + w**2)
    e = ke + constant.G * z + constant.CV * t

    states = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.E: testing_utils.to_3d_tensor(e, dim, size),
    }
    res = self.evaluate(
        self.model.total_enthalpy(replica_id, replicas, states, self.cfg)
    )

    expected_vector = constant.CP * t + ke + constant.G * z
    expected = testing_utils.to_3d_tensor(
        expected_vector, dim, size, as_tf_tensor=False
    )

    self.assertAllClose(expected, res)

  @parameterized.parameters(*zip(_DIMS))
  def test_sound_speed_from_total_energy_correctly(self, dim):
    """Checks if the sound speed is computed correctly from the total energy."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    self.cfg.g[dim] = -1.0

    # Compute the size of the test domain, assuming the domain is cubic.
    # The size includes the halo width.
    nx = len(self.cfg.x) + 2 * self.cfg.halo_width
    size = [nx, nx, nx]
    # Set flow field variables includeing density, velocity, temperature, and
    # geopotential height.
    u = np.linspace(-3.0, 12.0, nx)
    v = np.linspace(6.0, -24.0, nx)
    w = np.linspace(-9.0, 36.0, nx)
    t = np.linspace(250.0, 3000.0, nx)
    z = np.linspace(-600.0, 2400.0, nx)
    # Compute kinetic and total energy from flow field variables.
    ke = 0.5 * (u**2 + v**2 + w**2)
    e = ke + constant.G * z + constant.CV * t

    states = {
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.E: testing_utils.to_3d_tensor(e, dim, size),
    }
    res = self.evaluate(
        self.model.sound_speed(replica_id, replicas, states, self.cfg)
    )
    expected_vector = np.sqrt(constant.GAMMA * constant.R * t)
    expected = testing_utils.to_3d_tensor(
        expected_vector, dim, size, as_tf_tensor=False
    )
    self.assertAllClose(expected, res)

  @parameterized.parameters(*zip(_DIMS))
  def test_sound_speed_from_total_enthalpy_correctly(self, dim):
    """Checks if the sound speed is correct from the total enthalpy."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    self.cfg.g[dim] = -1.0

    # Compute the size of the test domain, assuming the domain is cubic.
    # The size includes the halo width.
    nx = len(self.cfg.x) + 2 * self.cfg.halo_width
    size = [nx, nx, nx]
    # Set flow field variables includeing density, velocity, temperature, and
    # geopotential height.
    u = np.linspace(-3.0, 12.0, nx)
    v = np.linspace(6.0, -24.0, nx)
    w = np.linspace(-9.0, 36.0, nx)
    t = np.linspace(250.0, 3000.0, nx)
    z = np.linspace(-600.0, 2400.0, nx)
    # Compute kinetic and total energy from flow field variables.
    ke = 0.5 * (u**2 + v**2 + w**2)
    h = ke + constant.G * z + constant.CP * t

    states = {
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.H: testing_utils.to_3d_tensor(h, dim, size),
    }
    res = self.evaluate(
        self.model.sound_speed(
            replica_id, replicas, states, self.cfg, opt=types.H
        )
    )
    expected_vector = np.sqrt(constant.GAMMA * constant.R * t)
    expected = testing_utils.to_3d_tensor(
        expected_vector, dim, size, as_tf_tensor=False
    )
    self.assertAllClose(expected, res)

  @parameterized.parameters(*zip(_DIMS))
  def test_sound_speed_calculation_from_pressure_and_density_correct(self, dim):
    """Checks if the sound speed is correct from pressure and density."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    self.cfg.g[dim] = -1.0

    # Compute the size of the test domain, assuming the domain is cubic.
    # The size includes the halo width.
    nx = len(self.cfg.x) + 2 * self.cfg.halo_width
    size = [nx, nx, nx]
    p = np.logspace(3.0, 6.0, nx)
    t = np.linspace(250.0, 3000.0, nx)
    rho = p / (constant.R * t)
    states = {
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
        types.P: testing_utils.to_3d_tensor(p, dim, size),
    }

    expected_vector = np.sqrt(constant.GAMMA * constant.R * t)
    expected = testing_utils.to_3d_tensor(
        expected_vector, dim, size, as_tf_tensor=False
    )
    # Here we check that either order of the options is allowable.
    opt = 'p_rho'
    res = self.evaluate(
        self.model.sound_speed(replica_id, replicas, states, self.cfg, opt=opt)
    )
    self.assertAllClose(expected, res)

  def test_sound_speed_incorrect_option(self):
    """Checks for error if incorrect option is specified."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    states = {}
    msg = r'^("bad_option" is not a valid option for sound speed)'
    with self.assertRaisesRegex(ValueError, msg):
      self.model.sound_speed(
          replica_id, replicas, states, self.cfg, opt='bad_option'
      )

  @parameterized.parameters(*zip(_DIMS))
  def test_conversion_to_potential_temperature(self, dim):
    """Checks the conversion to potential temperature."""
    self.cfg.g[dim] = -1.0

    # Compute the size of the test domain, assuming the domain is cubic.
    # The size includes the halo width.
    nx = len(self.cfg.x) + 2 * self.cfg.halo_width
    size = [nx, nx, nx]

    # Set the flow field pressure and temperature.
    p = np.logspace(3.0, 6.0, nx)
    t = np.linspace(250.0, 3000.0, nx)

    states = {
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        types.T: testing_utils.to_3d_tensor(t, dim, size),
    }
    res = self.evaluate(self.model.potential_temperature(states, self.cfg))
    expected_vector = t * (p / self.cfg.p_0) ** -constant.KAPPA
    expected = testing_utils.to_3d_tensor(
        expected_vector, dim, size, as_tf_tensor=False
    )

    self.assertAllClose(expected, res)

  @parameterized.parameters(*zip(_DIMS))
  def test_temperature_from_potential_temperature(self, dim):
    """Checks conversion of potential temperature to temperature."""
    self.cfg.g[dim] = -1.0

    # Compute the size of the test domain, assuming the domain is cubic.
    # The size includes the halo width.
    nx = len(self.cfg.x) + 2 * self.cfg.halo_width
    size = [nx, nx, nx]

    # Set the flow field pressure and temperature.
    p = np.logspace(3.0, 6.0, nx)
    theta = np.linspace(250.0, 3000.0, nx)

    states = {
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        types.POTENTIAL_T: testing_utils.to_3d_tensor(theta, dim, size),
    }
    res = self.evaluate(self.model.temperature(states, self.cfg, 'theta'))

    expected_vector = theta * (p / self.cfg.p_0) ** constant.KAPPA
    expected = testing_utils.to_3d_tensor(
        expected_vector, dim, size, as_tf_tensor=False
    )

    self.assertAllClose(expected, res)

  @parameterized.parameters(*zip(_DIMS))
  def test_temperature_from_eos(self, dim):
    """Checks calculation of temperature from EOS."""
    self.cfg.g[dim] = -1.0

    # Compute the size of the test domain, assuming the domain is cubic.
    # The size includes the halo width.
    nx = len(self.cfg.x) + 2 * self.cfg.halo_width
    size = [nx, nx, nx]

    # Set the flow field pressure and density.
    p = np.logspace(3.0, 6.0, nx)
    rho = np.logspace(-1.0, 2.0, nx)

    states = {
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
    }
    res = self.evaluate(self.model.temperature(states, self.cfg, 'eos'))

    expected_vector = p / (rho * constant.R)
    expected = testing_utils.to_3d_tensor(
        expected_vector, dim, size, as_tf_tensor=False
    )

    self.assertAllClose(expected, res)

  @parameterized.parameters(*zip(_DIMS))
  def test_temperature_incorrect_option(self, dim):
    """Checks for error if incorrect option is specified."""
    self.cfg.g[dim] = -1.0

    # Compute the size of the test domain, assuming the domain is cubic.
    # The size includes the halo width.
    nx = len(self.cfg.x) + 2 * self.cfg.halo_width
    size = [nx, nx, nx]

    # Set the flow field pressure and density.
    p = np.logspace(3.0, 6.0, nx)
    rho = np.logspace(-1.0, 2.0, nx)

    states = {
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        types.RHO: testing_utils.to_3d_tensor(rho, dim, size),
    }
    msg = (
        r'"bad_option" is not a valid option for temperature. Available options'
        r' are "eos", "theta".'
    )
    with self.assertRaisesRegex(ValueError, msg):
      self.model.temperature(states, self.cfg, 'bad_option')

  @parameterized.parameters(*zip(_DIMS))
  def test_density_from_eos(self, dim):
    """Checks EOS calculation for density."""
    self.cfg.g[dim] = -1.0
    nx = 16
    size = [nx, nx, nx]
    # Set the flow field pressure and temperature.
    p = np.logspace(3.0, 6.0, nx)
    t = np.linspace(250.0, 3000.0, nx)
    states = {
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        types.T: testing_utils.to_3d_tensor(t, dim, size),
    }
    res = self.evaluate(self.model.density(states, self.cfg, 'eos'))

    expected_vector = p / (constant.R * t)
    expected = testing_utils.to_3d_tensor(
        expected_vector, dim, size, as_tf_tensor=False
    )

    self.assertAllClose(expected, res)

  @parameterized.parameters(*zip(_DIMS))
  def test_density_from_potential_temperature(self, dim):
    """Checks conversion of potential temperature to density."""
    self.cfg.g[dim] = -1.0

    # Compute the size of the test domain, assuming the domain is cubic.
    # The size includes the halo width.
    nx = len(self.cfg.x) + 2 * self.cfg.halo_width
    size = [nx, nx, nx]

    # Set the flow field pressure and temperature.
    p = np.logspace(3.0, 6.0, nx)
    theta = np.linspace(250.0, 3000.0, nx)

    states = {
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        types.POTENTIAL_T: testing_utils.to_3d_tensor(theta, dim, size),
    }
    res = self.evaluate(self.model.density(states, self.cfg, 'theta'))

    expected_t = theta * (p / self.cfg.p_0) ** constant.KAPPA
    expected_vector = p / (constant.R * expected_t)
    expected = testing_utils.to_3d_tensor(
        expected_vector, dim, size, as_tf_tensor=False
    )

    self.assertAllClose(expected, res)

  @parameterized.parameters(*zip(_DIMS))
  def test_density_incorrect_option(self, dim):
    """Checks for error if incorrect option is specified."""
    self.cfg.g[dim] = -1.0

    # Compute the size of the test domain, assuming the domain is cubic.
    # The size includes the halo width.
    nx = len(self.cfg.x) + 2 * self.cfg.halo_width
    size = [nx, nx, nx]

    # Set the flow field pressure and density.
    p = np.logspace(3.0, 6.0, nx)
    t = np.linspace(200.0, 3000.0, nx)

    states = {
        types.P: testing_utils.to_3d_tensor(p, dim, size),
        types.T: testing_utils.to_3d_tensor(t, dim, size),
    }
    msg = (
        r'"bad_option" is not a valid option for density. Available options'
        r' are "eos", "theta".'
    )
    with self.assertRaisesRegex(ValueError, msg):
      self.model.density(states, self.cfg, 'bad_option')

  @parameterized.parameters(*zip(_GRAVITY_DIMS))
  def test_computing_gravitational_potential_1replica(self, gravity_dim):
    """Checks the graviational potenital computation for single replica."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    size = [16, 16, 16]
    if gravity_dim in ['x', 'y', 'z']:
      self.cfg.g[gravity_dim] = -1.0

    h = np.linspace(-600.0, 2400.0, 16)
    g = constant.G
    dim = gravity_dim
    if gravity_dim not in ['x', 'y', 'z']:
      dim = 'z'
      g = 0.0
    expected = testing_utils.to_3d_tensor(h * g, dim, size, as_tf_tensor=False)
    # If gravity is not present, the returned gravitational potential will be
    # 3D tensor represented as a list of 2D tensors with shape (1, 1). In
    # subsequent operations tf automatically casts this to the appropriate
    # shape. However, for tests we need to force the expected to match the
    # shape.
    if gravity_dim == 'w':
      expected = expected[:, 0, 0]
      expected = expected[:, np.newaxis, np.newaxis]
    res = self.evaluate(
        self.model.gravitational_potential_energy(
            replica_id, replicas, self.cfg
        )
    )

    self.assertAllClose(expected, res)

  @parameterized.product(theta=_GRAVITY_THETA, phi=_GRAVITY_PHI)
  def test_gravitational_potential_1replica_arbitrary_angle(self, theta, phi):
    """Checks the graviational potential for arbitrary vertical direction."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    size = [16, 16, 16]
    self.cfg.g['x'] = np.sin(phi) * np.cos(theta)
    self.cfg.g['y'] = np.sin(phi) * np.sin(theta)
    self.cfg.g['z'] = np.cos(phi)

    xv = np.linspace(-600.0, 2400.0, 16)
    expected_x = testing_utils.to_3d_tensor(
        xv * self.cfg.g['x'], 'x', size, as_tf_tensor=False
    )
    expected_y = testing_utils.to_3d_tensor(
        xv * self.cfg.g['y'], 'y', size, as_tf_tensor=False
    )
    expected_z = testing_utils.to_3d_tensor(
        xv * self.cfg.g['z'], 'z', size, as_tf_tensor=False
    )
    expected = -constant.G * (expected_x + expected_y + expected_z)
    res = self.evaluate(
        self.model.gravitational_potential_energy(
            replica_id, replicas, self.cfg
        )
    )
    self.assertAllClose(expected, res, atol=10.0, rtol=types.SMALL)

  @parameterized.product(dim=_DIMS, gravity_dim=_GRAVITY_DIMS)
  def test_computing_gravitational_potential_2replicas(self, dim, gravity_dim):
    """Checks the graviational potenital computation for two replicas."""
    dims = ('x', 'y', 'z')
    computation_shape = [1, 1, 1]
    computation_shape[dims.index(dim)] = 2
    self.cfg.cx = computation_shape[0]
    self.cfg.cy = computation_shape[1]
    self.cfg.cz = computation_shape[2]
    replicas = np.reshape(np.arange(2), computation_shape)
    if gravity_dim in ['x', 'y', 'z']:
      self.cfg.g[gravity_dim] = -1.0
    inputs = [[tf.constant(0)], [tf.constant(1)]]

    def device_fn(replica_id):
      """Wraps the gravitational potential function."""
      return self.model.gravitational_potential_energy(
          replica_id, replicas, self.cfg
      )

    output = self.run_tpu_test(replicas, device_fn, inputs)

    # First, compute the gravitational potential energy for the full domain.
    size = [16, 16, 16]
    h = np.linspace(-600.0, 2400.0, 16)
    g = constant.G
    gravity_vector_tile_dim = gravity_dim
    if gravity_dim not in ['x', 'y', 'z']:
      g = 0.0
      gravity_vector_tile_dim = 'z'
    full_domain_expected = testing_utils.to_3d_tensor(
        h * g, gravity_vector_tile_dim, size, as_tf_tensor=False
    )

    # Second, slice full domain into expected for each replica.
    if dim == 'x':
      expected = np.zeros([16, 11, 16, 2])
      expected[:, :, :, 0] = full_domain_expected[:, :11, :]
      expected[:, :, :, 1] = full_domain_expected[:, 5:, :]
    elif dim == 'y':
      expected = np.zeros([16, 16, 11, 2])
      expected[:, :, :, 0] = full_domain_expected[:, :, :11]
      expected[:, :, :, 1] = full_domain_expected[:, :, 5:]
    else:  # dim == 'z' or 'w'
      expected = np.zeros([11, 16, 16, 2])
      expected[:, :, :, 0] = full_domain_expected[:11, :, :]
      expected[:, :, :, 1] = full_domain_expected[5:, :, :]
    # If gravity is not present, the returned gravitational potential will be
    # 3D tensor represented as a list of 2D tensors with shape(1,1). In
    # subsequent operations tf automatically casts this to the appropriate
    # shape. However, for tests we need to force the expected to match the
    # shape.
    if gravity_dim == 'w':
      expected = expected[:, 0, 0, :]
      expected = expected[:, np.newaxis, np.newaxis, :]

    # Now loop over replicas, and select subdomain to compare.
    for i in range(2):
      with self.subTest(name=f'Replica{i}'):
        self.assertAllClose(expected[:, :, :, i], np.stack(output[i]))

  @parameterized.parameters(*zip(_DIMS))
  def test_total_energy_from_eos_without_gravity(self, dim):
    """Checks total energy is calculated correctly without gravity."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])

    # Compute the size of the test domain, assuming the domain is cubic.
    # The size includes the halo width.
    nx = 16
    size = [nx, nx, nx]
    # Set flow field variables including velocity, temperature, and
    # geopotential height.
    u = np.linspace(-3.0, 12.0, nx)
    v = np.linspace(6.0, -24.0, nx)
    w = np.linspace(-9.0, 36.0, nx)
    t = np.linspace(250.0, 3000.0, nx)
    # Compute kinetic and total energy from flow field variables.
    ke = 0.5 * (u**2 + v**2 + w**2)
    e = ke + constant.CV * t

    states = {
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.T: testing_utils.to_3d_tensor(t, dim, size),
    }
    res = self.evaluate(
        self.model.total_energy(replica_id, replicas, states, self.cfg)
    )

    expected = testing_utils.to_3d_tensor(e, dim, size, as_tf_tensor=False)

    self.assertAllClose(expected, res)

  @parameterized.product(theta=_GRAVITY_THETA, phi=_GRAVITY_PHI, dim=_DIMS)
  def test_total_energy_from_eos_with_misaligned_gravity(self, theta, phi, dim):
    """Checks the total energy for arbitrary vertical direction for gravity."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    size = [16, 16, 16]
    self.cfg.g['x'] = np.sin(phi) * np.cos(theta)
    self.cfg.g['y'] = np.sin(phi) * np.sin(theta)
    self.cfg.g['z'] = np.cos(phi)

    xv = np.linspace(-600.0, 2400.0, 16)
    pe_x = testing_utils.to_3d_tensor(
        xv * self.cfg.g['x'], 'x', size, as_tf_tensor=False
    )
    pe_y = testing_utils.to_3d_tensor(
        xv * self.cfg.g['y'], 'y', size, as_tf_tensor=False
    )
    pe_z = testing_utils.to_3d_tensor(
        xv * self.cfg.g['z'], 'z', size, as_tf_tensor=False
    )
    pe_3d = -constant.G * (pe_x + pe_y + pe_z)

    # Set flow field variables including velocity and temperature.
    u = np.linspace(-3.0, 12.0, 16)
    v = np.linspace(6.0, -24.0, 16)
    w = np.linspace(-9.0, 36.0, 16)
    t = np.linspace(250.0, 3000.0, 16)
    # Compute kinetic and total energy from flow field variables.
    ke = 0.5 * (u**2 + v**2 + w**2)
    e = ke + constant.CV * t
    e_3d = testing_utils.to_3d_tensor(e, dim, size, as_tf_tensor=False)
    expected = pe_3d + e_3d

    states = {
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.T: testing_utils.to_3d_tensor(t, dim, size),
    }
    res = self.evaluate(
        self.model.total_energy(replica_id, replicas, states, self.cfg)
    )
    self.assertAllClose(expected, res, atol=10.0, rtol=types.SMALL)

  @parameterized.product(theta=_GRAVITY_THETA, phi=_GRAVITY_PHI, dim=_DIMS)
  def test_internal_energy_from_total_with_gravity(self, theta, phi, dim):
    """Checks the internal energy from total with misaligned gravity."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    size = [16, 16, 16]
    self.cfg.g['x'] = np.sin(phi) * np.cos(theta)
    self.cfg.g['y'] = np.sin(phi) * np.sin(theta)
    self.cfg.g['z'] = np.cos(phi)

    xv = np.linspace(-600.0, 2400.0, 16)
    pe_x = testing_utils.to_3d_tensor(
        xv * self.cfg.g['x'], 'x', size, as_tf_tensor=False
    )
    pe_y = testing_utils.to_3d_tensor(
        xv * self.cfg.g['y'], 'y', size, as_tf_tensor=False
    )
    pe_z = testing_utils.to_3d_tensor(
        xv * self.cfg.g['z'], 'z', size, as_tf_tensor=False
    )
    pe_3d = -constant.G * (pe_x + pe_y + pe_z)

    # Set flow field variables including velocity and temperature.
    u = np.linspace(-3.0, 12.0, 16)
    v = np.linspace(6.0, -24.0, 16)
    w = np.linspace(-9.0, 36.0, 16)
    t = np.linspace(250.0, 3000.0, 16)
    # Compute kinetic and total energy from flow field variables.
    ke = 0.5 * (u**2 + v**2 + w**2)
    e_int = constant.CV * t
    e_tot = ke + e_int
    e_tot_3d = testing_utils.to_3d_tensor(e_tot, dim, size, as_tf_tensor=False)
    e_tot_3d = pe_3d + e_tot_3d

    states = {
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.E: tf.unstack(tf.convert_to_tensor(e_tot_3d, dtype=tf.float32)),
    }
    expected = testing_utils.to_3d_tensor(e_int, dim, size, as_tf_tensor=False)
    res = self.evaluate(
        self.model.internal_energy(replica_id, replicas, states, self.cfg)
    )
    self.assertAllClose(expected, res, atol=10.0, rtol=types.SMALL)

  @parameterized.product(theta=_GRAVITY_THETA, phi=_GRAVITY_PHI, dim=_DIMS)
  def test_sensible_enthalpy_from_total_with_gravity(self, theta, phi, dim):
    """Checks the sensible enthalpy from total with misaligned gravity."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    size = [16, 16, 16]
    self.cfg.g['x'] = np.sin(phi) * np.cos(theta)
    self.cfg.g['y'] = np.sin(phi) * np.sin(theta)
    self.cfg.g['z'] = np.cos(phi)

    xv = np.linspace(-600.0, 2400.0, 16)
    pe_x = testing_utils.to_3d_tensor(
        xv * self.cfg.g['x'], 'x', size, as_tf_tensor=False
    )
    pe_y = testing_utils.to_3d_tensor(
        xv * self.cfg.g['y'], 'y', size, as_tf_tensor=False
    )
    pe_z = testing_utils.to_3d_tensor(
        xv * self.cfg.g['z'], 'z', size, as_tf_tensor=False
    )
    pe_3d = -constant.G * (pe_x + pe_y + pe_z)

    # Set flow field variables including velocity and temperature.
    u = np.linspace(-3.0, 12.0, 16)
    v = np.linspace(6.0, -24.0, 16)
    w = np.linspace(-9.0, 36.0, 16)
    t = np.linspace(250.0, 3000.0, 16)
    # Compute kinetic and total energy from flow field variables.
    ke = 0.5 * (u**2 + v**2 + w**2)
    h_sens = constant.CP * t
    h_tot = ke + h_sens
    h_tot_3d = testing_utils.to_3d_tensor(h_tot, dim, size, as_tf_tensor=False)
    h_tot_3d = pe_3d + h_tot_3d

    states = {
        types.U: testing_utils.to_3d_tensor(u, dim, size),
        types.V: testing_utils.to_3d_tensor(v, dim, size),
        types.W: testing_utils.to_3d_tensor(w, dim, size),
        types.H: tf.unstack(tf.convert_to_tensor(h_tot_3d, dtype=tf.float32)),
    }
    expected = testing_utils.to_3d_tensor(h_sens, dim, size, as_tf_tensor=False)
    res = self.evaluate(
        self.model.sensible_enthalpy(replica_id, replicas, states, self.cfg)
    )
    self.assertAllClose(expected, res, atol=10.0, rtol=types.SMALL)

  def test_cp_returns_expected(self):
    """Tests that the cp method returns `constant.CP`."""
    nx = 16
    size = [nx, nx, nx]
    states = {
        types.RHO: testing_utils.to_3d_tensor(
            np.linspace(1.0, 2.0, nx), 'x', size
        )
    }
    result = self.evaluate(
        self.model.cp(states)
    )
    expected = testing_utils.to_3d_tensor(
        constant.CP * np.ones(nx), 'x', size, as_tf_tensor=False
    )
    self.assertAllClose(expected, result)


if __name__ == '__main__':
  tf.test.main()
