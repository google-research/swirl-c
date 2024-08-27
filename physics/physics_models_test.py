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
"""Tests for physics_models.py."""

from unittest import mock
from absl import logging
from absl.testing import parameterized
import numpy as np
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.physics import physics_models
import tensorflow as tf


def _mock_mask_fn(self):
  """Mocks the mask returned by a source term model."""
  del self  # Unused.
  return (types.RHO, types.RHO_U, types.RHO_E, 'rho_y')


def _mock_source_term_fn(
    self,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    conservative: types.FlowFieldMap,
    helper_vars: types.FlowFieldMap,
) -> types.FlowFieldMap:
  """Mocks the RHS source term calculation by a source term model."""
  del self, replica_id, replicas, conservative, helper_vars  # Unused.
  nx = 16
  size = [nx, nx, nx]
  dim = 'y'
  return {
      types.RHO: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
      types.RHO_U: testing_utils.to_3d_tensor(np.ones(nx), dim, size),
      types.RHO_V: testing_utils.to_3d_tensor(np.ones(nx), dim, size),
      types.RHO_E: testing_utils.to_3d_tensor(2.0 * np.ones(nx), dim, size),
      'rho_y': testing_utils.to_3d_tensor(np.linspace(0.0, 1.0, nx), dim, size),
  }


class PhysicsModelsTest(tf.test.TestCase, parameterized.TestCase):

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

  _THERMODYNAMICS_MODELS = ('generic',)
  _TRANSPORT_MODELS = ('simple',)
  _SOURCE_FUNCTION_MODELS = ('none', 'gravity')

  @parameterized.parameters(*zip(_THERMODYNAMICS_MODELS))
  @mock.patch.object(logging, 'log', autospec=True)
  def test_init_thermodynamics_model_selected_correctly(
      self, thermodynamics_model, mock_log
  ):
    """Tests that the correct thermodynamics model is selected."""
    self.cfg.thermodynamics_model = thermodynamics_model
    physics_models.PhysicsModels(self.cfg)
    log_str = (
        f'"{self.cfg.thermodynamics_model}" thermodynamics model selected.'
    )
    mock_log.assert_any_call(logging.INFO, log_str)

  def test_init_raise_for_thermodynamics_model_not_implemented(self):
    """Tests error is raised for not implemented thermodynamics model name."""
    self.cfg.thermodynamics_model = 'bad_model_name'
    msg = r'^(Unsupported thermodynamics model: "bad_model_name")'
    with self.assertRaisesRegex(NotImplementedError, msg):
      physics_models.PhysicsModels(self.cfg)

  @parameterized.parameters(*zip(_TRANSPORT_MODELS))
  @mock.patch.object(logging, 'log', autospec=True)
  def test_init_transport_model_selected_correctly(
      self, transport_model, mock_log
  ):
    """Tests that the correct transport model is selected."""
    self.cfg.transport_model = transport_model
    physics_models.PhysicsModels(self.cfg)
    log_str = (
        f'"{self.cfg.transport_model}" transport model selected.'
    )
    mock_log.assert_any_call(logging.INFO, log_str)

  def test_init_raise_for_transport_model_not_implemented(self):
    """Tests error is raised for not implemented transport model name."""
    self.cfg.transport_model = 'bad_model_name'
    msg = r'^(Unsupported transport model: "bad_model_name")'
    with self.assertRaisesRegex(NotImplementedError, msg):
      physics_models.PhysicsModels(self.cfg)

  def test_init_raise_for_source_model_not_implemented(self):
    """Tests error is raised for not implemented source term model name."""
    self.cfg.thermodynamics_model = 'generic'
    self.cfg.source_functions = ['bad_model_name']
    msg = r'^(Unsupported source term model: "bad_model_name")'
    with self.assertRaisesRegex(NotImplementedError, msg):
      physics_models.PhysicsModels(self.cfg)

  @parameterized.parameters(*zip(_SOURCE_FUNCTION_MODELS))
  @mock.patch.object(logging, 'log', autospec=True)
  def test_init_source_model_selected_correctly(self, source_model, mock_log):
    """Tests that the correct source function model is selected."""
    self.cfg.thermodynamics_model = 'generic'
    self.cfg.source_functions = [source_model]
    match source_model:
      case 'gravity':
        log_str = '"gravity" source function selected.'
      case _:  # 'none'
        self.cfg.source_functions = []
        log_str = 'No source functions selected.'
    self.cfg.g['x'] = 1.0
    physics_models.PhysicsModels(self.cfg)
    mock_log.assert_any_call(logging.INFO, log_str)

  def test_source_function_returns_zeros_for_undefined_option(self):
    """Confirms that the source function returns zeros for `None` cfg option."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    dim = 'x'

    self.cfg.source_functions = None
    self.physics_models = physics_models.PhysicsModels(self.cfg)
    conservative = {
        types.RHO: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        'rho_y': testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
    }
    results = self.physics_models.source_function(
        replica_id, replicas, conservative, {}
    )
    self.assertSequenceEqual(conservative.keys(), results.keys())
    for var_name in conservative:
      self.assertAllClose(
          self.evaluate(results[var_name]),
          testing_utils.to_3d_tensor(
              np.zeros(nx), dim, size, as_tf_tensor=False
          ),
      )

  def test_source_function_returns_zeros_for_empty_option_list(self):
    """Confirms that the source function returns zeros for empty list."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    dim = 'x'
    self.cfg.source_functions = []
    self.physics_models = physics_models.PhysicsModels(self.cfg)
    conservative = {
        types.RHO: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        'rho_y': testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
    }
    results = self.physics_models.source_function(
        replica_id, replicas, conservative, {}
    )
    self.assertSequenceEqual(conservative.keys(), results.keys())
    for var_name in conservative:
      self.assertAllClose(
          self.evaluate(results[var_name]),
          testing_utils.to_3d_tensor(
              np.zeros(nx), dim, size, as_tf_tensor=False
          ),
      )

  @mock.patch(
      'swirl_c.physics.additional_models.gravity.Gravity.mask', _mock_mask_fn
  )
  @mock.patch(
      'swirl_c.physics.additional_models.gravity.Gravity.source_term',
      _mock_source_term_fn,
  )
  def test_source_function_returns_values_correctly_mock_function(self):
    """Confirms that the source function returns correct values."""
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    nx = 16
    size = [nx, nx, nx]
    dim = 'y'
    self.cfg.source_functions = ['gravity']
    self.cfg.g['x'] = 1.0
    self.physics_models = physics_models.PhysicsModels(self.cfg)
    conservative = {
        types.RHO: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        types.RHO_U: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        types.RHO_V: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        types.RHO_W: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        types.RHO_E: testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
        'rho_y': testing_utils.to_3d_tensor(np.zeros(nx), dim, size),
    }
    results = self.evaluate(
        self.physics_models.source_function(
            replica_id, replicas, conservative, {}
        )
    )
    self.assertSequenceEqual(conservative.keys(), results.keys())
    for var_name in conservative:
      match var_name:
        case 'rho_u':
          self.assertAllClose(
              testing_utils.to_3d_tensor(
                  np.ones(nx), dim, size, as_tf_tensor=False
              ),
              results[var_name],
          )
        case 'rho_e':
          self.assertAllClose(
              testing_utils.to_3d_tensor(
                  2.0 * np.ones(nx), dim, size, as_tf_tensor=False
              ),
              results[var_name],
          )
        case 'rho_y':
          self.assertAllClose(
              testing_utils.to_3d_tensor(
                  np.linspace(0.0, 1.0, nx), dim, size, as_tf_tensor=False
              ),
              results[var_name],
          )
        case _:
          self.assertAllClose(
              testing_utils.to_3d_tensor(
                  np.zeros(nx), dim, size, as_tf_tensor=False
              ),
              results[var_name],
          )


if __name__ == '__main__':
  tf.test.main()
