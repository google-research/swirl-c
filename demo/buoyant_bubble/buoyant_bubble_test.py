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
"""Tests for buoyant_bubble."""

import os

from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
from swirl_c.common import types
from swirl_c.core import driver
from swirl_c.demo.buoyant_bubble import buoyant_bubble
from swirl_c.physics import constant
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework.post_processing import data_processing

_FIELDS = types.BASE_CONSERVATIVE
_FIGURE_FIELDS = types.BASE_CONSERVATIVE
_PREFIX = 'buoyant_bubble'

FLAGS = flags.FLAGS


class BuoyantBubbleTest(tf.test.TestCase):

  def setUp(self):
    """Initializes the simulation."""
    super().setUp()
    self._write_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')

  def test_buoyant_bubble(self):
    """Tests the Buoyant Bubble problem."""

    # Construct simulation objects.
    simulation = buoyant_bubble.BuoyantBubbleBuilder()
    cfg = simulation.buoyant_bubble_cfg()
    init_fn = simulation.buoyant_bubble_init_fn()
    run_name = _PREFIX
    FLAGS.data_dump_prefix = os.path.join(self._write_dir, run_name)

    # Run the simulation.
    driver.solver(init_fn, cfg)

    # Post-process simulation results.
    logging.info('Simulation completed. Post-processing results.')
    x = cfg.x / 1e3 - 10
    y = cfg.y / 1e3
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
                cfg.core_ny + 2 * cfg.halo_width,
                cfg.core_nz + 2 * cfg.halo_width,
            ],
            [cfg.cx, cfg.cy, cfg.cz],
            [cfg.lx, cfg.ly, cfg.lz],
        )
        conservative[field] = slice_data
      primitive = {}
      for var_name, field in conservative.items():
        if var_name == types.RHO:
          primitive[types.conservative_to_primitive_name(var_name)] = field
        else:
          primitive[types.conservative_to_primitive_name(var_name)] = (
              field / conservative[types.RHO]
          )
      e_int = primitive[types.E] - 0.5 * (
          primitive[types.U] ** 2
          + primitive[types.V] ** 2
          + primitive[types.W] ** 2
      ) - constant.G * cfg.y[:, np.newaxis].numpy()
      primitive[types.T] = e_int / constant.CV
      primitive[types.P] = (constant.GAMMA - 1.0) * conservative['rho'] * e_int
      primitive[types.POTENTIAL_T] = (
          primitive[types.T] * (cfg.p_0 / primitive[types.P]) ** constant.KAPPA
      )

      theta = primitive[types.POTENTIAL_T]
      theta[-10:, :] = 300.0
      v = primitive[types.V]
      fig, ax = plt.subplots(figsize=(8, 2.5), ncols=2)
      ax[0].contour(
          x,
          y,
          theta,
          np.linspace(300.2, 302, 10),
          vmin=300,
          vmax=302,
          cmap='coolwarm',
          linewidths=0.5,
      )
      c_0 = plt.pcolor(theta, vmin=300, vmax=302, cmap='coolwarm')
      fig.colorbar(
          c_0, ax=ax[0], fraction=0.046, pad=0.04, ticks=[300, 301, 302]
      )
      ax[0].set_aspect('equal')
      ax[0].set_xlim([-6, 6])
      ax[0].set_ylim([0, 10])
      ax[0].set_xlabel('x [km]')
      ax[0].set_ylabel('y [km]')
      ax[0].set_xticks([-6, -4, -2, 0, 2, 4, 6])

      ax[1].contour(
          x,
          y,
          v,
          np.linspace(-10, 10, 11),
          vmin=-10,
          vmax=10,
          cmap='coolwarm',
          linewidths=0.5,
      )
      c_1 = plt.pcolor(v, vmin=-10, vmax=10, cmap='coolwarm')
      fig.colorbar(c_1, ax=ax[1], fraction=0.046, pad=0.04, ticks=[-10, 0, 10])
      ax[1].set_aspect('equal')
      ax[1].set_xlim([-6, 6])
      ax[1].set_ylim([0, 10])
      ax[1].set_xlabel('x [km]')
      ax[1].set_ylabel('y [km]')
      ax[1].set_xticks([-6, -4, -2, 0, 2, 4, 6])

      figure_file_name = '{}/solution-step-{:04d}.png'.format(
          self._write_dir, step
      )
      with tf.io.gfile.GFile(figure_file_name, 'wb') as f:
        plt.savefig(f)


if __name__ == '__main__':
  tf.test.main()
