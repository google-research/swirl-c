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
"""Configurations for the channel flow simulation."""

from typing import Tuple, Union

import numpy as np
from swirl_c.boundary import bc_types
from swirl_c.common import types
from swirl_c.core import initializer
from swirl_c.core import parameter
from swirl_c.numerics import kernel_op_types
import tensorflow as tf


class Channel:
  """Defines parameters and initial conditions to configure a channel flow."""

  def __init__(self):
    cx = 2
    cy = 1
    cz = 1
    halo_width = 3
    nx_core = 128
    ny_core = 64
    nz_core = 12
    # Get the total number of points in the physical domain (excluding halos).
    nx, ny, nz = [
        int((n - 2 * halo_width) * c)
        for n, c in zip((nx_core, ny_core, nz_core), (cx, cy, cz))
    ]
    # Set the domain size so that the grid spacing in all directions is 1 m.
    lx, ly, lz = [float(n - 1) for n in (nx, ny, nz)]
    # Generate the mesh.
    x, y, z = [np.linspace(0, l, n) for l, n in zip((lx, ly, lz), (nx, ny, nz))]

    customized_cfg = {
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
        'x': tf.convert_to_tensor(x, dtype=types.DTYPE),
        'dx': np.diff(x)[0],
        'y': tf.convert_to_tensor(y, dtype=types.DTYPE),
        'dy': np.diff(y)[0],
        'z': tf.convert_to_tensor(z, dtype=types.DTYPE),
        'dz': np.diff(z)[0],
        'g': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        # Set timestepping details.
        'start_step': 0,
        'num_steps': 1000,
        'num_cycles': 1,
        'dt': 2.5e-2,
        # Set selected numerics and physics models.
        'solver': 'compressible_navier_stokes',
        'interpolation_scheme': 'WENO_5',
        'numeric_flux_scheme': 'HLL',
        'include_diffusion': True,
        'time_integration_scheme': 'rk3',
        'source_functions': None,
        'kernel_op_type': kernel_op_types.KernelOpType.KERNEL_OP_CONV.value,
        'kernel_size': 8,
        # Set thermodynamics model details.
        'thermodynamics_model': 'generic',
        'p_0': 100.0,
        # Set transport model details.
        'transport_model': 'simple',
        'transport_parameters': {'nu': 1e-3, 'pr': 1.0},
        # Specify conservative and primitive variable names.
        'conservative_variable_names': types.BASE_CONSERVATIVE,
        'primitive_variable_names': types.BASE_PRIMITIVES,
        # Specify boundary conditions.
        'bc': {
            'cell_averages': {
                types.RHO: {
                    'x': {
                        0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                        1: (bc_types.BoundaryCondition.NEUMANN, 0.0),
                    },
                    'y': {
                        0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                        1: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
                    },
                    'z': {
                        0: (bc_types.BoundaryCondition.PERIODIC,),
                        1: (bc_types.BoundaryCondition.PERIODIC,),
                    },
                },
                types.RHO_U: {
                    'x': {
                        0: (bc_types.BoundaryCondition.DIRICHLET, 1.0),
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
                types.RHO_V: {
                    'x': {
                        0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
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
                        0: (bc_types.BoundaryCondition.DIRICHLET, 0.0),
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
                types.RHO_E: {
                    'x': {
                        0: (bc_types.BoundaryCondition.DIRICHLET, 251.5),
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
            },
        },
    }

    self.cfg = parameter.SwirlCParameters(customized_cfg)

  def init_fn(
      self,
      replica_id: Union[int, tf.Tensor],
      coordinates: Tuple[int, int, int],
      cfg: parameter.SwirlCParameters,
  ) -> types.FlowFieldMap:
    """Initializes flow field variables."""
    del replica_id

    def constant_field(val):
      return initializer.partial_field_for_core(
          cfg, coordinates, initializer.constant_initial_state_fn(val)
      )

    return {
        types.RHO: constant_field(1.0),
        types.RHO_U: constant_field(1.0),
        types.RHO_V: constant_field(0.0),
        types.RHO_W: constant_field(0.0),
        types.RHO_E: constant_field(251.5),
    }
