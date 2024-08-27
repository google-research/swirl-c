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
"""Simulation of Sod's shock tube problem."""

from typing import Callable, Dict, Tuple, Union
import numpy as np
from swirl_c.boundary import bc_types
from swirl_c.common import types
from swirl_c.core import initializer
from swirl_c.core import parameter
from swirl_c.numerics import kernel_op_types
import tensorflow as tf

# The type describing the problem specified initialization function for the
# flow field variables.
InitFn = Callable[
    [Union[int, tf.Tensor], Tuple[int, int, int], parameter.SwirlCParameters],
    types.FlowFieldMap,
]


def _get_cell_center_coordinates(
    start: float, stop: float, n_cells: int
) -> np.ndarray:
  xx = np.linspace(start, stop, n_cells + 1, dtype=types.NP_DTYPE)
  xx = xx[:-1] + 0.5 * np.diff(xx)[0]
  return xx


class SodsShockTubeBuilder:
  """Class that defines the `cfg` object and initial condition generator."""

  def __init__(self, dim: str):
    self._dim = dim
    lx = 1.0 if dim == 'x' else 0.0246
    ly = 1.0 if dim == 'y' else 0.0246
    lz = 1.0 if dim == 'z' else 0.0246
    # Here nx, ny, and nz are the total number of cells (excluding halos) along
    # the x, y, and z directions, respectively. This number is computed as:
    # nx = (nx_per_core - 2 * halo_width) * cx.
    # In this simulation, along the shock tube, the total number of mesh cells
    # in each core is 128 with ghost cells included. With 2 cores being
    # allocated in this direction, the total number of cells is 244.
    nx = 244 if dim == 'x' else 6
    ny = 244 if dim == 'y' else 6
    nz = 244 if dim == 'z' else 6
    cx = 2 if dim == 'x' else 1
    cy = 2 if dim == 'y' else 1
    cz = 2 if dim == 'z' else 1
    xx = _get_cell_center_coordinates(0.0, lx, nx)
    yy = _get_cell_center_coordinates(0.0, ly, ny)
    zz = _get_cell_center_coordinates(0.0, lz, nz)
    self._bc = {
        'cell_averages': {
            types.RHO: {
                'x': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
                'y': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
                'z': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
            },
            types.RHO_U: {
                'x': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
                'y': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
                'z': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
            },
            types.RHO_V: {
                'x': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
                'y': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
                'z': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
            },
            types.RHO_W: {
                'x': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
                'y': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
                'z': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
            },
            types.RHO_E: {
                'x': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
                'y': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
                'z': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
            },
            'rho_y': {
                'x': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
                'y': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
                'z': {
                    0: (bc_types.BoundaryCondition.PERIODIC,),
                    1: (bc_types.BoundaryCondition.PERIODIC,),
                },
            },
        }
    }

    self._bc['cell_averages'][types.RHO][dim] = {
        0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
        1: (bc_types.BoundaryCondition.DIRICHLET, 0.125),
    }
    for var_name in types.MOMENTUM:
      self._bc['cell_averages'][var_name][dim] = {
          0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
          1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
      }
    self._bc['cell_averages'][types.RHO_E][dim] = {
        0: (bc_types.BoundaryCondition.DIRICHLET, 2.5),
        1: (bc_types.BoundaryCondition.DIRICHLET, 0.25),
    }
    self._bc['cell_averages']['rho_y'][dim] = {
        0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
        1: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
    }

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
        'use_3d_tf_tensor': False,
        'halo_width': 3,
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
        'g': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        # Set timestepping details.
        'start_step': 0,
        'num_steps': 250,
        'num_cycles': 1,
        'dt': 1.0e-3,
        # Set reference quantities.
        'p_0': 101325.0,
        'nu': 1e-1,
        'nu_t': None,
        # Set selected numerics and physics models.
        'solver': 'compressible_navier_stokes',
        'interpolation_scheme': 'WENO_5',
        'numeric_flux_scheme': 'HLL',
        'include_diffusion': False,
        'time_integration_scheme': 'rk3',
        'thermodynamics_model': 'generic',
        'source_functions': None,
        'kernel_op_type': kernel_op_types.KernelOpType.KERNEL_OP_SLICE.value,
        # Specify conservative and primitive variable names.
        'conservative_variable_names': list(types.BASE_CONSERVATIVE) + [
            'rho_y',
        ],
        'primitive_variable_names': list(types.BASE_PRIMITIVES) + [
            'y',
        ],
        # Specify boundary conditions.
        'bc': self._bc,
    }

  def sods_shock_tube_cfg(self) -> parameter.SwirlCParameters:
    """Returns the `cfg` object for the Sod's shock tube problem."""
    return parameter.SwirlCParameters(self.user_cfg)

  def sods_shock_tube_init_fn(
      self,
      conservative_l: Dict[str, float],
      conservative_r: Dict[str, float],
  ) -> InitFn:
    """Returns the function to initialize the domain for the shock tube problem.

    Args:
      conservative_l: A dictionary which defines the state to the left of the
        discontinuity. Must include key, value pairs for all conservative
        variables in `cfg.conservative_variable_names`.
      conservative_r: A dictionary which defines the state to the right of the
        discontinuity. Must include key, value pairs for all conservative
        variables in `cfg.conservative_variable_names`.

    Returns:
      A callable `InitFn` which will be distributed to the TPUs to initialize
      the flow field on each local mesh.
    """

    def init_fn(replica_id, coordinates, cfg):
      del replica_id

      def step_field(left_val, right_val):
        return initializer.partial_field_for_core(
            cfg,
            coordinates,
            initializer.step_function_initial_state_fn(
                left_val, right_val, 0.5, self._dim
            ),
        )

      return {
          types.RHO: step_field(
              conservative_l[types.RHO], conservative_r[types.RHO]
          ),
          types.RHO_U: step_field(
              conservative_l[types.RHO_U], conservative_r[types.RHO_U]
          ),
          types.RHO_V: step_field(
              conservative_l[types.RHO_V], conservative_r[types.RHO_V]
          ),
          types.RHO_W: step_field(
              conservative_l[types.RHO_W], conservative_r[types.RHO_W]
          ),
          types.RHO_E: step_field(
              conservative_l[types.RHO_E], conservative_r[types.RHO_E]
          ),
          'rho_y': step_field(conservative_l['rho_y'], conservative_r['rho_y']),
      }

    return init_fn
