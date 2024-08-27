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
"""A library for the buoyant bubble simulation [1, 2, 3].

The buoyant bubble simulation performed in a quasi-2D domain. The x and y
dimensions are associated with the horizontal and vertical directions,
respectively. The z dimension is dummy, where periodic boundary condition is
applied.

References:
1. Robert, André. 1993. “Bubble Convection Experiments with a Semi-Implicit
Formulation of the Euler Equations.” Journal of the Atmospheric Sciences 50
(13): 1865–73.
2. Bryan, George H., and J. Michael Fritsch. 2002. “A Benchmark Simulation for
Moist Nonhydrostatic Numerical Models.” Monthly Weather Review 130 (12):
2917–28.
3. Kurowski, Marcin J., Wojciech W. Grabowski, and Piotr K. Smolarkiewicz. 2014.
“Anelastic and Compressible Simulation of Moist Deep Convection.” Journal of the
Atmospheric Sciences 71 (10): 3767–87.
"""

from typing import Callable, Tuple, Union
import numpy as np
from swirl_c.boundary import bc_types
from swirl_c.common import types
from swirl_c.core import initializer
from swirl_c.core import parameter
from swirl_c.numerics import kernel_op_types
from swirl_c.physics import constant
import tensorflow as tf

# The type describing the problem specified initialization function for the
# flow field variables.
InitFn = Callable[
    [Union[int, tf.Tensor], Tuple[int, int, int], parameter.SwirlCParameters],
    types.FlowFieldMap,
]
TensorOrArray = Union[tf.Tensor, np.ndarray]
ThreeIntTuple = Union[np.ndarray, tf.Tensor, Tuple[int, int, int]]
ValueFunction = Callable[
    [
        TensorOrArray,
        TensorOrArray,
        TensorOrArray,
        float,
        float,
        float,
        ThreeIntTuple,
    ],
    tf.Tensor,
]

# Set simulation parameters.
_THETA_AMBIENT = 300.0
_THETA_PTURB = 2.0
_P_0 = constant.P_0
_BOUYANT_BUBBLE_X_C = 1.0e4
_BOUYANT_BUBBLE_X_R = 2.0e3
_BOUYANT_BUBBLE_Y_C = 2.0e3
_BOUYANT_BUBBLE_Y_R = 2.0e3
_G_DIM = 'y'
_CONSTANT_P = False
_ADD_REFERENCE_STATES = False


def _get_cell_center_coordinates(
    start: float, stop: float, n_cells: int
) -> np.ndarray:
  xx = np.linspace(start, stop, n_cells, dtype=types.NP_DTYPE)
  return xx


def _hydrostatic_pressure(
    z: Union[float, tf.Tensor]
) -> Union[float, tf.Tensor]:
  """Finds the hydrostatic pressure from the height and potential temperature.

  This method assumes a constant potential temperature specified by
  `_THETA_AMBIENT` and a pressure at `z=0` given by `_PRESSURE_0`.

  Args:
    z: The geopotential height as either a float or a tensor.

  Returns:
    The hydrostatic pressure.
  """
  pressure_fn = lambda z: _P_0 * (
      1.0 - (constant.G * constant.KAPPA * z) / (constant.R * _THETA_AMBIENT)
  ) ** (1.0 / constant.KAPPA)
  return pressure_fn(z)


class BuoyantBubbleBuilder:
  """Class that defines the `cfg` object and initial condition generator."""

  def __init__(self):
    use_3d_tf_tensor = True
    lx = 2.0e4
    ly = 1.0e4
    lz = 4.8e2
    cx = 4
    cy = 2
    cz = 1
    halo_width = 2
    nx_core = 128
    ny_core = 128
    nz_core = halo_width * 4
    nx = int((nx_core - halo_width * 2) * cx)
    ny = int((ny_core - halo_width * 2) * cy)
    nz = int((nz_core - halo_width * 2) * cz)
    xx = _get_cell_center_coordinates(0.0, lx, nx)
    yy = _get_cell_center_coordinates(0.0, ly, ny)
    zz = _get_cell_center_coordinates(0.0, lz, nz)

    # Set the simulation boundary conditions. Simulation is pseudo-2D, so `z` is
    # periodic. We do periodic in `x` as well, but this should have no impact
    # on the results. In `y`, we set cell averages through zero-gradient
    # condition. We will use the total flux boundary condition to specify
    # zero-penetration adiabatic upper and lower boundaries in `y`.
    if _G_DIM == 'y':
      h = yy
      ones = tf.ones((nx_core, 1), dtype=types.DTYPE)
      l_dim = 'x'
      h_dim = 'y'
    elif _G_DIM == 'x':
      h = xx
      ones = tf.ones((1, ny_core), dtype=types.DTYPE)
      l_dim = 'y'
      h_dim = 'x'
    elif _G_DIM is None:
      h = np.zeros_like(yy)
      ones = tf.ones((nx_core, 1), dtype=types.DTYPE)
      l_dim = 'x'
      h_dim = 'y'
    else:
      raise ValueError(
          f'{_G_DIM} is not a valid gravitation direction option. Available'
          ' options are "x", "y", and "z".'
      )

    dh = np.diff(h)[0]

    if not _CONSTANT_P:
      rho_hi = []
      rho_lo = []
      rhoe_hi = []
      rhoe_lo = []
      for i in range(halo_width):
        h_hi = h[-1] + (i + 1) * dh
        h_lo = h[0] - (halo_width - i) * dh
        p_hi = _hydrostatic_pressure(h_hi)
        p_low = _hydrostatic_pressure(h_lo)
        t_hi = _THETA_AMBIENT * (p_hi / _P_0) ** constant.KAPPA
        t_lo = _THETA_AMBIENT * (p_low / _P_0) ** constant.KAPPA
        rho_hi_val = p_hi / (constant.R * t_hi)
        rho_lo_val = p_low / (constant.R * t_lo)
        rho_hi.append([rho_hi_val * ones] * nz_core)
        rho_lo.append([rho_lo_val * ones] * nz_core)
        rhoe_hi.append(
            [rho_hi_val * (constant.CV * t_hi + constant.G * h_hi) * ones]
            * nz_core
        )
        rhoe_lo.append(
            [rho_lo_val * (constant.CV * t_lo + constant.G * h_lo) * ones]
            * nz_core
        )

      if use_3d_tf_tensor:
        rho_hi = [tf.stack(rho_hi_i) for rho_hi_i in rho_hi]
        rho_lo = [tf.stack(rho_lo_i) for rho_lo_i in rho_lo]
        rhoe_hi = [tf.stack(rhoe_hi_i) for rhoe_hi_i in rhoe_hi]
        rhoe_lo = [tf.stack(rhoe_lo_i) for rhoe_lo_i in rhoe_lo]
    else:
      rho_hi = rho_lo = _P_0 / (constant.R * _THETA_AMBIENT)
      rhoe_hi = rhoe_lo = rho_hi * constant  .CV * _THETA_AMBIENT

    self._bc = {
        'cell_averages': {
            types.RHO: {
                l_dim: {
                    0: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                    1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                },
                h_dim: {
                    0: (bc_types.BoundaryCondition.DIRICHLET, rho_lo),
                    1: (bc_types.BoundaryCondition.DIRICHLET, rho_hi),
                },
                'z': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
            },
            types.RHO_U: {
                'x': {
                    0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                    1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                },
                'y': {
                    0: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                    1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                },
                'z': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
            },
            types.RHO_V: {
                'x': {
                    0: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                    1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                },
                'y': {
                    0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                    1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
                },
                'z': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
            },
            types.RHO_W: {
                'x': {
                    0: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                    1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                },
                'y': {
                    0: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                    1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                },
                'z': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
            },
            types.RHO_E: {
                l_dim: {
                    0: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                    1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                },
                h_dim: {
                    0: (bc_types.BoundaryCondition.DIRICHLET, rhoe_lo),
                    1: (bc_types.BoundaryCondition.DIRICHLET, rhoe_hi),
                },
                'z': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
            },
        },
    }

    g = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    if _G_DIM is not None:
      g[_G_DIM] = -1.0

    self.user_cfg = {
        # Swirl-LM compatibility
        'apply_preprocess': False,
        'apply_postprocess': False,
        'states_to_file': [],
        'states_from_file': [],
        # Set processor topology.
        'cx': cx,
        'cy': cy,
        'cz': cz,
        # Set mesh properties.
        'use_3d_tf_tensor': use_3d_tf_tensor,
        'halo_width': halo_width,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'core_nx': int(nx / cx),
        'core_ny': int(ny / cy),
        'core_nz': int(nz / cz),
        'lx': lx,
        'ly': ly,
        'lz': lz,
        'x': tf.convert_to_tensor(xx, dtype=types.DTYPE),
        'dx': np.diff(xx)[0],
        'y': tf.convert_to_tensor(yy, dtype=types.DTYPE),
        'dy': np.diff(yy)[0],
        'z': tf.convert_to_tensor(zz, dtype=types.DTYPE),
        'dz': np.diff(zz)[0],
        'g': g,
        # Set timestepping details.
        'start_step': 0,
        'num_steps': 20000,
        'num_cycles': 1,
        'dt': 0.05,
        # Set selected numerics and physics models.
        'solver': 'compressible_navier_stokes',
        'interpolation_scheme': 'WENO_5' if halo_width == 3 else 'WENO_3',
        'numeric_flux_scheme': 'HLLC',
        'include_diffusion': True,
        'time_integration_scheme': 'rk3',
        'source_functions': ['gravity'] if _G_DIM is not None else None,
        'kernel_op_type': kernel_op_types.KernelOpType.KERNEL_OP_CONV.value,
        'kernel_size': 8,
        # Set themodynamics model details.
        'thermodynamics_model': 'generic',
        'p_0': _P_0,
        # Set transport model details.
        'transport_model': 'simple',
        'transport_parameters': {'nu': 1e-5, 'pr': 1.0},
        # Specify conservative and primitive variable names.
        'conservative_variable_names': types.BASE_CONSERVATIVE,
        'primitive_variable_names': types.BASE_PRIMITIVES,
        # Specify helper variable names.
        'additional_state_names': (
            ['p_ref', 'rho_ref'] if _ADD_REFERENCE_STATES else []
        ),
        # Specify boundary conditions.
        'bc': self._bc,
    }

  def buoyant_bubble_cfg(self) -> parameter.SwirlCParameters:
    """Returns the `cfg` object for the buoyant bubble problem."""
    return parameter.SwirlCParameters(self.user_cfg)

  def buoyant_bubble_init_fn(self) -> InitFn:
    """Returns the function to initialize the buoyant bubble problem.

    Returns:
      A callable `InitFn` which will be distributed to the TPUs to initialize
      the flow field on each local mesh.
    """

    def init_fn(replica_id, coordinates, cfg):
      del replica_id

      def constant_field(val):
        return initializer.partial_field_for_core(
            cfg, coordinates, initializer.constant_initial_state_fn(val)
        )

      def bubble_field(var_name):
        return initializer.partial_field_for_core(
            cfg, coordinates, buoyant_bubble_initial_state_fn(var_name)
        )

      states = {
          types.RHO: bubble_field(types.RHO),
          types.RHO_U: constant_field(0.0),
          types.RHO_V: constant_field(0.0),
          types.RHO_W: constant_field(0.0),
          types.RHO_E: bubble_field(types.RHO_E),
      }

      # Add helper variables.
      if _ADD_REFERENCE_STATES:
        states.update({
            helper_var: bubble_field(helper_var)
            for helper_var in ('p_ref', 'rho_ref')
        })

      return states

    return init_fn


def buoyant_bubble_initial_state_fn(var_name: str) -> ValueFunction:
  """Defines a function to generate the bubble initial condition.

  To generate the bubble initial conditions, first the hydrostatic pressure
  across the domain is calculated for a constant potential temperature
  `_THETA_AMBIENT`. Using this hydrostatic temperature, the potential
  temperature is perturbed by `_THETA_PTURB` to generate a region of warmer gas
  in the center of the domain. The temperature, density, and total energy are
  then computed from the hydrostatic pressure and perturbed potential
  temperature.

  Args:
    var_name: The name of the thermodynamic state variable to set with the
      returned `init_fn`. Must be either density `RHO` or total energy `RHO_E`.

  Returns:
    A value function which will generate the bubble initial conditions.
  """

  if var_name not in (types.RHO, types.RHO_E, 'p_ref', 'rho_ref'):
    raise ValueError(f'Unrecognized bubble variable name: {var_name}')

  def init_fn(xx, yy, zz, lx, ly, lz, coord):
    del zz, lx, ly, lz, coord  # Unused.
    if _G_DIM == 'x':
      h = xx
    elif _G_DIM == 'y':
      h = yy
    elif _G_DIM is None:
      h = tf.zeros_like(yy)
    else:
      raise ValueError(f'Unsupported G_DIM: {_G_DIM}')

    rx = tf.math.divide_no_nan((xx - _BOUYANT_BUBBLE_X_C), _BOUYANT_BUBBLE_X_R)
    ry = tf.math.divide_no_nan((yy - _BOUYANT_BUBBLE_Y_C), _BOUYANT_BUBBLE_Y_R)
    rr = tf.sqrt(rx**2 + ry**2)
    if not _CONSTANT_P:
      p = _hydrostatic_pressure(h)
      theta = _THETA_AMBIENT + tf.where(
          rr < 1.0,
          _THETA_PTURB * tf.cos(0.5 * np.pi * rr) ** 2,
          tf.zeros_like(rr),
      )
      t = theta * (p / _P_0) ** constant.KAPPA
      t_ref = _THETA_AMBIENT * (p / _P_0) ** constant.KAPPA
    else:
      p = _P_0 * tf.ones_like(yy)
      t = _THETA_AMBIENT + tf.where(
          rr < 1.0,
          _THETA_PTURB * tf.cos(0.5 * np.pi * rr) ** 2,
          tf.zeros_like(rr),
      )
      t_ref = _THETA_AMBIENT * tf.ones_like(yy)

    rho = p / constant.R / t
    rho_e = rho * (constant.CV * t + constant.G * h)
    if var_name == types.RHO:
      return rho
    elif var_name == 'p_ref':
      return p
    elif var_name == 'rho_ref':
      return p / constant.R / t_ref
    else:
      return rho_e

  return init_fn
