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
"""Tests for gravity."""

from absl.testing import parameterized
import numpy as np
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.physics import constant
from swirl_c.physics.additional_models import gravity
import tensorflow as tf


class GravityTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes the default cfg object."""
    super().setUp()
    self.cfg = parameter.SwirlCParameters()

  _GRAVITY_THETA = tuple(np.pi * np.linspace(0.0, 1.0, 6))
  _GRAVITY_PHI = tuple(np.pi * np.linspace(-1.0, 1.0, 8))

  def test_init_raise_no_gravity_direction(self):
    """Tests error is raised for when gravity direction is not specified."""
    self.cfg.g['x'] = 0.0
    self.cfg.g['y'] = 0.0
    self.cfg.g['z'] = 0.0
    msg = (
        'Gravity direction must be specified if the gravity source function'
        ' is to be used.'
    )
    with self.assertRaisesRegex(ValueError, msg):
      gravity.Gravity(self.cfg)

  @parameterized.product(theta=_GRAVITY_THETA, phi=_GRAVITY_PHI)
  def test_mask_arbitrary_gravity_angle(self, theta, phi):
    """Checks the mask from the gravity model at arbitrary angles."""
    self.cfg.g['x'] = np.sin(phi) * np.cos(theta)
    self.cfg.g['y'] = np.sin(phi) * np.sin(theta)
    self.cfg.g['z'] = np.cos(phi)
    gravity_model = gravity.Gravity(self.cfg)
    results = gravity_model.mask()
    if abs(np.abs(np.sin(phi) * np.cos(theta)) - 1.0) < 2.0 * types.SMALL:
      expected = ('rho_u',)
    elif abs(np.abs(np.sin(phi) * np.sin(theta)) - 1.0) < 2.0 * types.SMALL:
      expected = ('rho_v',)
    elif abs(np.abs(np.cos(phi)) - 1.0) < 2.0 * types.SMALL:
      expected = ('rho_w',)
    else:
      expected = ('rho_u', 'rho_v', 'rho_w')
    self.assertTupleEqual(expected, results)

  @parameterized.product(theta=_GRAVITY_THETA, phi=_GRAVITY_PHI)
  def test_source_term_arbitrary_gravity_directions(self, theta, phi):
    """Checks the rhs from the gravity model at arbitrary angles."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    size = [16, 16, 16]
    self.cfg.g['x'] = np.sin(phi) * np.cos(theta)
    self.cfg.g['y'] = np.sin(phi) * np.sin(theta)
    self.cfg.g['z'] = np.cos(phi)
    rho_vec = np.logspace(-1.0, 2.0, 16)
    g = constant.G
    conservative = {types.RHO: testing_utils.to_3d_tensor(rho_vec, 'x', size)}
    gravity_model = gravity.Gravity(self.cfg)
    results = self.evaluate(
        gravity_model.source_term(replica_id, replicas, conservative, {})
    )

    expected = {
        'rho_u': testing_utils.to_3d_tensor(
            rho_vec * g * np.sin(phi) * np.cos(theta),
            'x',
            size,
            as_tf_tensor=False,
        ),
        'rho_v': testing_utils.to_3d_tensor(
            rho_vec * g * np.sin(phi) * np.sin(theta),
            'x',
            size,
            as_tf_tensor=False,
        ),
        'rho_w': testing_utils.to_3d_tensor(
            rho_vec * g * np.cos(phi), 'x', size, as_tf_tensor=False
        ),
    }

    self.assertDictEqual(expected, results)


if __name__ == '__main__':
  tf.test.main()
