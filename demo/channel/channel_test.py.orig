"""Tests for channel flow."""

import os

from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
from swirl_c.common import types
from swirl_c.core import driver
from swirl_c.demo.channel import channel
from swirl_c.physics import constant
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework.post_processing import data_processing

_FIELDS = types.BASE_CONSERVATIVE
_FIGURE_FIELDS = types.BASE_CONSERVATIVE
_PREFIX = 'channel'

FLAGS = flags.FLAGS


class ChannelTest(tf.test.TestCase):

  def setUp(self):
    """Initializes the simulation."""
    super().setUp()
    self._write_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')

  def test_channel(self):
    """Tests the channel flow problem."""

    # Construct simulation objects.
    simulation = channel.Channel()
    cfg = simulation.cfg
    run_name = _PREFIX
    FLAGS.data_dump_prefix = os.path.join(self._write_dir, run_name)

    # Run the simulation.
    driver.solver(simulation.init_fn, cfg)

    # Post-process simulation results.
    logging.info('Simulation completed. Post-processing results.')
    x = np.linspace(0.0, cfg.lx, cfg.nx)
    y = np.linspace(0.0, cfg.ly, cfg.ny)
    for cycle in range(cfg.num_cycles + 1):
      step = cycle * cfg.num_steps
      actual_prefix = '{}/{}/{}'.format(self._write_dir, step, run_name)
      conservative = {}
      for field in cfg.conservative_variable_names:
        slice_data = data_processing.get_slice(
            actual_prefix,
            field,
            step,
            cfg.halo_width,
            'z',
            0.5 * cfg.lz,
            [
                cfg.core_nx + 2 * cfg.halo_width,
                cfg.ny + 2 * cfg.halo_width,
                cfg.nz + 2 * cfg.halo_width,
            ],
            [cfg.cx, cfg.cy, cfg.cz],
            [cfg.lx, cfg.ly, cfg.lz],
        )
        conservative[field] = slice_data
        fig, ax = plt.subplots(figsize=(4, 4.5))
        c = ax.pcolor(x, y, slice_data, cmap='jet')
        ax.set_aspect('equal')
        fig.colorbar(c, ax=ax)
        figure_file_name = '{}/solution-field-{}-step-{:04d}.png'.format(
            self._write_dir,
            field,
            cfg.num_cycles * cfg.num_steps,
        )
        with tf.io.gfile.GFile(figure_file_name, 'wb') as f:
          plt.savefig(f)
      # Compute and plot primitive variables.
      primitive = {
          types.conservative_to_primitive_name(var_name): (
              field / conservative[types.RHO]
              if var_name != types.RHO and types.is_conservative_name(var_name)
              else field
          )
          for var_name, field in conservative.items()
      }
      e_int = primitive[types.E] - 0.5 * (
          primitive[types.U] ** 2
          + primitive[types.V] ** 2
          + primitive[types.W] ** 2
      )
      primitive[types.T] = e_int / constant.CV
      primitive[types.P] = (
          primitive[types.RHO] * constant.R * primitive[types.T]
      )
      primitive[types.POTENTIAL_T] = (
          primitive[types.T] * (cfg.p_0 / primitive[types.P]) ** constant.KAPPA
      )
      for var_name, val in primitive.items():
        if var_name == types.RHO:
          continue
        fig, ax = plt.subplots(figsize=(4, 4.5))
        c = ax.pcolor(x, y, val, cmap='jet')
        ax.set_aspect('equal')
        fig.colorbar(c, ax=ax)
        figure_file_name = '{}/solution-field-{}-step-{:04d}.png'.format(
            self._write_dir,
            var_name,
            step,
        )
        with tf.io.gfile.GFile(figure_file_name, 'wb') as f:
          plt.savefig(f)


if __name__ == '__main__':
  tf.test.main()
