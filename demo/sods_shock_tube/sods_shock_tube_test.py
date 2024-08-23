"""Tests for sods_shock_tube.py."""

import os
from typing import Dict

from absl import flags
from absl import logging
from absl.testing import parameterized
import matplotlib.pyplot as plt
import numpy as np
from swirl_c.common import types
from swirl_c.core import driver
from swirl_c.demo.sods_shock_tube import sods_shock_tube
from swirl_c.physics import constant
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework.post_processing import data_processing
from google3.third_party.tensorflow.core.framework import tensor_pb2  # pylint: disable=g-direct-tensorflow-import

_FIELDS = types.BASE_CONSERVATIVE
_FIGURE_FIELDS = types.BASE_CONSERVATIVE
_PREFIX = 'sods_shock_tube'

FLAGS = flags.FLAGS


def shock_tube_exact_solution(
    xx: np.ndarray,
    t: np.ndarray,
    state_l: Dict[str, np.ndarray],
    state_r: Dict[str, np.ndarray],
    gamma: np.ndarray,
) -> Dict[str, np.ndarray]:
  """Computes the exact solution for the shock tube problem.

  The solution assumes a calorically perfect, inert gas. Diffusion effects are
  ignored.

  Args:
    xx: The mesh of spatial coordinates where the exact solution values are
      computed and returned.
    t: The time to compute the exact solution for.
    state_l: A dictionary defining the initial state on the left side of the
      discontinuity. The state is defined by key, value pairs for density 'rho',
      gas velocity 'u', and pressure 'p'.
    state_r: A dictionary defining the initial state on the right side of the
      discontinuity, containing the same variables as `state_l`.
    gamma: The ratio of specific heats.

  Returns:
    A dictionary where the key, value pairs represent the exact solution to the
    shock tube problem on the specified mesh at time `t`. The mesh 'xx', gas
    density 'rho', gas velocity 'u', and pressure 'p' are returned in the
    dictionary.
  """

  def sound_speed(state):
    """Computes the sound speed for a calorically perfect gas."""
    return np.sqrt(gamma * state['p'] / state['rho'])

  def f_function(state, p_star):
    """Computes the function to iterate to find p_star and its derivative."""
    a = 2.0 / ((gamma + 1.0) * state['rho'])
    b = (gamma - 1.0) / (gamma + 1.0) * state['p']
    if p_star > state['p']:
      f = (p_star - state['p']) * np.sqrt(a / (p_star + b))
      fp = np.sqrt(a / (p_star + b)) * (
          1.0 - (p_star - state['p']) / (2.0 * (b + p_star))
      )
    else:
      f = (
          2.0
          * state['c']
          / (gamma - 1.0)
          * ((p_star / state['p']) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
      )
      fp = (
          1.0
          / (state['rho'] * state['c'])
          * (p_star / state['p']) ** -((gamma + 1.0) / (2.0 * gamma))
      )
    return f, fp

  def get_pstar(state_l, state_r):
    """Finds p_star from left and right initial states."""
    p_star_guess = 0.5 * (state_l['p'] + state_r['p']) - 0.125 * (
        state_r['u'] - state_l['u']
    ) * (state_l['rho'] + state_r['rho']) * (state_l['c'] + state_r['c'])

    p_old = p_star_guess
    thresh = 1.0
    cycles = 0
    while thresh > 1e-6:
      cycles += 1
      if cycles > 100:
        raise ValueError('Too many iterations.')
      f_l, fp_l = f_function(state_l, p_old)
      f_r, fp_r = f_function(state_r, p_old)
      du = state_r['u'] - state_l['u']
      f = f_l + f_r + du
      fp = fp_l + fp_r
      p_new = p_old - f / fp
      thresh = np.abs((p_new - p_old) / (p_new + p_old))
      p_old = p_new

    p_star = p_old
    logging.info(
        'Shock tube exact solution iteration completed. %d steps required to'
        ' converge to tolerance %.2e.',
        cycles,
        thresh,
    )
    if p_star > state_l['p']:
      logging.info('Solution has left running shock wave.')
    else:
      logging.info('Solution has left running rarefaction wave.')
    if p_star > state_r['p']:
      logging.info('Solution has right running shock wave.')
    else:
      logging.info('Solution has right running rarefaction wave.')
    return p_star

  def rho_shock(state, p_star):
    """Computes density behind a shock."""
    return state['rho'] * (
        (p_star / state['p'] + ((gamma - 1.0) / (gamma + 1.0)))
        / (((gamma - 1.0) / (gamma + 1.0)) * p_star / state['p'] + 1.0)
    )

  def rho_fan(state, p_star):
    """Computes density at the tail of a rarefaction fan."""
    return state['rho'] * (p_star / state['p']) ** (1.0 / gamma)

  def rarefaction_fan_state(state, x, t, left_running):
    """Computes the local state in a rarefaction fan."""
    sign_fac = 1.0 if left_running else -1.0
    rho = state['rho'] * (
        2.0 / (gamma + 1.0)
        + sign_fac
        * (gamma - 1.0)
        / ((gamma + 1.0) * state['c'])
        * (state['u'] - x / t)
    ) ** (2.0 / (gamma - 1.0))
    u = (
        2.0
        / (gamma + 1.0)
        * (sign_fac * state['c'] + (gamma - 1.0) / 2.0 * state['u'] + x / t)
    )
    p = state['p'] * (
        2.0 / (gamma + 1.0)
        + sign_fac
        * (gamma - 1.0)
        / ((gamma + 1.0) * state['c'])
        * (state['u'] - x / t)
    ) ** (2.0 * gamma / (gamma - 1.0))
    return rho, u, p, state['y']

  def get_wave_speeds(state_l, state_r, p_star, u_star):
    """Computes the shock, rarefaction head, and rarefaction tail speeds."""
    ss_l = state_l['u'] - state_l['c'] * np.sqrt(
        (gamma + 1.0) / (2.0 * gamma) * p_star / state_l['p']
        + (gamma - 1.0) / (2.0 * gamma)
    )
    sh_l = state_l['u'] - state_l['c']
    st_l = u_star - state_l['c'] * (p_star / state_l['p']) ** (
        (gamma - 1.0) / (2.0 * gamma)
    )

    ss_r = state_r['u'] + state_r['c'] * np.sqrt(
        (gamma + 1.0) / (2.0 * gamma) * p_star / state_r['p']
        + (gamma - 1.0) / (2.0 * gamma)
    )
    sh_r = state_r['u'] + state_r['c']
    st_r = u_star + state_r['c'] * (p_star / state_r['p']) ** (
        (gamma - 1.0) / (2.0 * gamma)
    )
    return ss_l, sh_l, st_l, ss_r, sh_r, st_r

  def get_local_state(
      x,
      t,
      state_l,
      state_r,
      p_star,
      u_star,
      ss_l,
      sh_l,
      st_l,
      ss_r,
      sh_r,
      st_r,
  ):
    """Computes the state at a given `x`, `t` pair."""
    if x / t <= u_star:
      if p_star > state_l['p']:
        if x / t <= ss_l:
          return state_l['rho'], state_l['u'], state_l['p'], state_l['y']
        else:
          return (
              rho_shock(state_l, p_star),
              u_star,
              p_star,
              state_l['y'],
          )
      else:
        if x / t <= sh_l:
          return state_l['rho'], state_l['u'], state_l['p'], state_l['y']
        elif x / t <= st_l:
          return rarefaction_fan_state(state_l, x, t, True)
        else:
          return (
              rho_fan(state_l, p_star),
              u_star,
              p_star,
              state_l['y'],
          )
    else:
      if p_star > state_r['p']:
        if x / t >= ss_r:
          return state_r['rho'], state_r['u'], state_r['p'], state_r['y']
        else:
          return (
              rho_shock(state_r, p_star),
              u_star,
              p_star,
              state_r['y'],
          )
      else:
        if x / t >= sh_r:
          return state_r['rho'], state_r['u'], state_r['p'], state_r['y']
        elif x / t >= st_r:
          return rarefaction_fan_state(state_r, x, t, False)
        else:
          return (
              rho_fan(state_r, p_star),
              u_star,
              p_star,
              state_r['y'],
          )

  state_l['c'] = sound_speed(state_l)
  state_r['c'] = sound_speed(state_r)
  p_star = get_pstar(state_l, state_r)
  u_star = 0.5 * (state_l['u'] + state_r['u']) + 0.5 * (
      f_function(state_r, p_star)[0] - f_function(state_l, p_star)[0]
  )
  ss_l, sh_l, st_l, ss_r, sh_r, st_r = get_wave_speeds(
      state_l, state_r, p_star, u_star
  )
  solution = {
      'x': xx,
      'rho': np.zeros_like(xx),
      'u': np.zeros_like(xx),
      'p': np.zeros_like(xx),
      'y': np.zeros_like(xx),
  }
  for ii, x in enumerate(xx):
    rho_buf, u_buf, p_buf, y_buf = get_local_state(
        x,
        t,
        state_l,
        state_r,
        p_star,
        u_star,
        ss_l,
        sh_l,
        st_l,
        ss_r,
        sh_r,
        st_r,
    )
    solution['rho'][ii] = rho_buf
    solution['u'][ii] = u_buf
    solution['p'][ii] = p_buf
    solution['y'][ii] = y_buf

  return solution


class SodsShockTubeTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Initializes the simulation."""
    super().setUp()
    self._write_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')
  @parameterized.parameters(*('x', 'y'))
  def test_sods_shock_tube_against_exact_(self, dim):
    """Tests Sod's shock tube problem in 1D and compares to exact."""
    # BEGIN GOOGLE-INTERNAL
    # Currently, we only test the shock tube problem in the x direction, however
    # the initializer and test should work in x, y, or z. For `dim='z'`, we
    # encounter and OOM error, likely arising because we are slicing along the
    # 'z' direction, incurring a significant overhead when 'nz' is large. Both
    # the 'x' and 'y' test can be run individually without an issue, but when
    # the tests are run in a single blaze test invocation (through a
    # absl.testing.parameterized or distinct test methods) and OOM error occurs.
    # See b/311465474 for additoinal details.
    # END GOOGLE-INTERnAL

    # Specify initial conditions for simulation.
    state_l = {
        types.RHO: 1.0,
        types.U: 0.0,
        types.V: 0.0,
        types.W: 0.0,
        types.P: 1.0,
        'y': 1.0,
    }
    state_r = {
        types.RHO: 0.125,
        types.U: 0.0,
        types.V: 0.0,
        types.W: 0.0,
        types.P: 0.1,
        'y': 0.0,
    }
    conservative_l = {
        types.RHO: state_l[types.RHO],
        types.RHO_U: 0.0,
        types.RHO_V: 0.0,
        types.RHO_W: 0.0,
        types.RHO_E: 1.0 / (constant.GAMMA - 1.0) * state_l[types.P],
        'rho_y': state_l[types.RHO] * state_l['y'],
    }
    conservative_r = {
        types.RHO: state_r[types.RHO],
        types.RHO_U: 0.0,
        types.RHO_V: 0.0,
        types.RHO_W: 0.0,
        types.RHO_E: 1.0 / (constant.GAMMA - 1.0) * state_r[types.P],
        'rho_y': state_r[types.RHO] * state_r['y'],
    }
    conservative_l[types.MOMENTUM[types.DIMS.index(dim)]] = (
        state_l[types.RHO] * state_l[types.U]
    )
    conservative_r[types.MOMENTUM[types.DIMS.index(dim)]] = (
        state_r[types.RHO] * state_r[types.U]
    )

    # Construct simulation objects.
    simulation = sods_shock_tube.SodsShockTubeBuilder(dim)
    cfg = simulation.sods_shock_tube_cfg()
    init_fn = simulation.sods_shock_tube_init_fn(conservative_l, conservative_r)
    run_name = f'{_PREFIX}_{dim}'
    FLAGS.data_dump_prefix = os.path.join(self._write_dir, run_name)

    # Run the simulation.
    driver.solver(init_fn, cfg)

    # Post-process simulation results.
    logging.info('Simulation completed. Post-processing results.')
    actual_prefix = '{}/{}/{}'.format(
        self._write_dir, cfg.num_cycles * cfg.num_steps, run_name
    )
    computation_shape = [cfg.cx, cfg.cy, cfg.cz]
    mesh_shape = [12, 12, 12]
    mesh_shape[types.DIMS.index(dim)] = 128
    results = {}
    for field in cfg.conservative_variable_names:
      line_out_buf = data_processing.get_1d_profiles_from_ser_file(
          actual_prefix,
          field,
          cfg.halo_width,
          cfg.num_cycles * cfg.num_steps,
          mesh_shape,
          computation_shape,
          types.DIMS.index(dim),
          [
              (0.5, 0.5),
          ],
      )
      if dim == 'x':
        results[field] = line_out_buf['y_3_z_3']
      elif dim == 'y':
        results[field] = line_out_buf['x_3_z_3']
      else:  # dim == 'z'
        results[field] = line_out_buf['x_3_y_3']
    time = cfg.num_cycles * cfg.num_steps * cfg.dt
    mesh = cfg.x if dim == 'x' else cfg.y
    l = cfg.lx if dim == 'x' else cfg.ly
    exact_solution = shock_tube_exact_solution(
        mesh - 0.5 * l,
        time,
        state_l,
        state_r,
        constant.GAMMA,
    )
    e_int = exact_solution[types.P] / (
        (constant.GAMMA - 1.0) * exact_solution[types.RHO]
    )
    e_tot = e_int + 0.5 * exact_solution[types.U] ** 2
    expected = {
        types.RHO: exact_solution[types.RHO],
        types.RHO_U: np.zeros_like(exact_solution[types.RHO]),
        types.RHO_V: np.zeros_like(exact_solution[types.RHO]),
        types.RHO_W: np.zeros_like(exact_solution[types.RHO]),
        types.RHO_E: exact_solution[types.RHO] * e_tot,
        'rho_y': exact_solution[types.RHO] * exact_solution['y'],
    }
    expected[types.MOMENTUM[types.DIMS.index(dim)]] = (
        exact_solution[types.RHO] * exact_solution[types.U]
    )

    with self.subTest(name=f'{dim}, keys'):
      self.assertSequenceEqual(expected.keys(), results.keys())

    for var_name, val in expected.items():
      plt.figure(figsize=(4, 4.5))
      plt.plot(
          mesh,
          results[var_name],
          '.',
          ms=4,
          mec='#1e90ff',
          mfc='#1e90ff',
          label='Swirl-C',
      )
      plt.plot(
          mesh,
          val,
          'k-',
          linewidth=0.75,
          label='exact',
      )
      plt.legend()
      plt.title(var_name)
      plt.xlim([-0.1, 1.1])

      figure_file_name = '{}/solution-dim-{}-field-{}-step-{:04d}.png'.format(
          self._write_dir,
          dim,
          var_name,
          cfg.num_cycles * cfg.num_steps,
      )

      with tf.io.gfile.GFile(figure_file_name, 'wb') as f:
        plt.savefig(f)
      with self.subTest(name=f'{dim}, {var_name}'):
        # Because numeric diffusion is present in the simulation, but not in the
        # exact solution, we expect that near discontinuities the solutions can
        # diverge by nearly the nominal value. Clearly, setting `rtol` and
        # `atol` high enough to ignore these is not possible. Instead, we check
        # that most of the solution points are close, which is suffecient to
        # confirm the wave speed and structure, but not so restrictive as to
        # fail due to numeric diffusion.
        rtol = 1.0e-2
        atol = 1.0e-2
        tol = atol + rtol * np.abs(results[var_name])
        close = np.abs(val - results[var_name]) <= tol
        num_close = np.float64(close.sum())
        close_percent = num_close / np.size(val)
        self.assertGreaterEqual(close_percent, 0.9)


if __name__ == '__main__':
  tf.test.main()
