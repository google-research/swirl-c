"""Tests for simulation.py."""

from unittest import mock
import numpy as np
from swirl_c.common import testing_utils
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.core import simulation
from swirl_c.numerics.time_integration import time_integration
from swirl_c.physics import physics_models
import tensorflow as tf


class SimulationTest(tf.test.TestCase):

  def setUp(self):
    """Initializes the generic thermodynamics library object."""
    super().setUp()
    self.cfg = parameter.SwirlCParameters(
        {'solver': 'compressible_navier_stokes'}
    )
    self.kernel_op = self.cfg.kernel_op
    self.physics_models = physics_models.PhysicsModels(self.cfg)

  def test_init_raises_not_implemented_error(self):
    """Tests that the _init_ raises NotImplementedError for bad model name."""
    self.cfg.solver = 'bad_option'
    msg = r'^("bad_option" is not an implemented solver)'
    with self.assertRaisesRegex(NotImplementedError, msg):
      simulation.Simulation(self.kernel_op, self.cfg)

  @mock.patch.object(time_integration, 'time_integration', autospec=True)
  def test_step_returns_time_integration_correctly(self, mock_time_integration):
    """Tests that step returns the time integration method."""
    model = simulation.Simulation(self.kernel_op, self.cfg)
    replica_id = tf.constant(0)
    replicas = np.array([[[0]]])
    step_id = tf.constant(0)
    states = {
        types.RHO: testing_utils.to_3d_tensor(
            np.ones(16) * 0.5, 'x', [16, 16, 16]
        )
    }
    additional_states = {}
    model.step(replica_id, replicas, step_id, states, additional_states)
    mock_time_integration.assert_called_once_with(
        replica_id,
        replicas,
        states,
        additional_states,
        model.cfg,
        model.physics_models,
        model.rhs_fn,
    )


if __name__ == '__main__':
  tf.test.main()
