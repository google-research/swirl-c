"""Configurations for the Kelvin-Helmholtz instability simulation."""

from typing import Tuple, Union

import numpy as np
from swirl_c.boundary import bc_types
from swirl_c.common import types
from swirl_c.core import initializer
from swirl_c.core import parameter
from swirl_c.numerics import kernel_op_types
from swirl_c.physics import constant
import tensorflow as tf


class KHInstability:
  """Defines parameters and initial conditions to configure a KH instability."""

  def __init__(self, del_rho: float = 1.0, re: float = 1e5):
    """Initializes the simulation parameters."""
    # The characteristic length of the domain.
    l = 1.0
    # The density jump represented as a ratio of the density difference between
    # the two fluids and the reference density.
    self.del_rho = del_rho
    # The horizontal flow velocity.
    self.u_flow = 1.0
    # The kinematic viscosity and diffusivity (for passive scalars in included).
    del_u = 2.0 * self.u_flow
    nu = l * del_u / re
    # The pressure of the flow field.
    p_0 = 10.0
    # The sound speed.
    s_max = np.sqrt(constant.GAMMA * p_0 / (1.0 - 0.5 * del_rho))

    cx = 2
    cy = 4
    cz = 1
    halo_width = 3
    nx_core = 256
    ny_core = 256
    nz_core = 12
    # Get the total number of points in the physical domain (excluding halos).
    nx, ny, nz = [
        int((n - 2 * halo_width) * c)
        for n, c in zip((nx_core, ny_core, nz_core), (cx, cy, cz))
    ]
    # Set the domain size so that the grid spacing in all directions is 1 m.
    lx, ly, lz = [2 * l / n * (n - 1) for n in (nx, ny, nz)]
    # Generate the mesh.
    x, y, z = [np.linspace(0, l, n) for l, n in zip((lx, ly, lz), (nx, ny, nz))]
    dx, dy, dz = [np.diff(coord)[0] for coord in (x, y, z)]

    # Parameters for the time integration.
    u_max = np.maximum(self.u_flow, s_max)
    cfl = 0.6
    dx_min = min(dx, dy, dz)
    dt = cfl * dx_min / u_max

    t_end = 4.0
    num_steps = int(round(t_end / dt))

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
        'use_3d_tf_tensor': True,
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
        'dx': dx,
        'y': tf.convert_to_tensor(y, dtype=types.DTYPE),
        'dy': dy,
        'z': tf.convert_to_tensor(z, dtype=types.DTYPE),
        'dz': dz,
        'g': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        # Set timestepping details.
        'start_step': 0,
        'num_steps': num_steps,
        'num_cycles': 1,
        'dt': dt,
        # Set selected numerics and physics models.
        'solver': 'compressible_navier_stokes',
        'interpolation_scheme': 'WENO_5',
        'numeric_flux_scheme': 'HLLC',
        'include_diffusion': True,
        'time_integration_scheme': 'rk2',
        'source_functions': None,
        'kernel_op_type': kernel_op_types.KernelOpType.KERNEL_OP_CONV.value,
        'kernel_size': 8,
        # Set thermodynamics model details.
        'thermodynamics_model': 'generic',
        'p_0': p_0,
        # Set transport model details.
        'transport_model': 'simple',
        'transport_parameters': {'nu': nu, 'pr': 1.0},
        # Specify conservative and primitive variable names.
        'conservative_variable_names': list(types.BASE_CONSERVATIVE) + [
            'rho_c'
        ],
        'primitive_variable_names': list(types.BASE_PRIMITIVES) + ['c'],
        # Specify boundary conditions.
        'bc': {
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
                'rho_c': {
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

    # Location of the first transition point in the vertical direction.
    y_1 = 0.5
    # Location of the second transition point in the vertical direction.
    y_2 = 1.5
    # Width factor in the tanh profile for the horizontal velocity
    # initialization.
    a = 0.05
    # Standard deviation in the Gaussian profile for the vertical velocity
    # initialization.
    sig = 0.2
    # The initial vertical velocity perturbation [m / s].
    u_pert = 0.01

    def gen_init_state(state_init_fn):
      return initializer.partial_field_for_core(cfg, coordinates, state_init_fn)

    def rho_init_fn(xx, yy, zz, lx, ly, lz, coord):
      """Initializes the density."""
      del xx, zz, lx, ly, lz, coord  # Unused.
      return tf.ones_like(yy) + self.del_rho * 0.5 * (
          tf.math.tanh((yy - y_1) / a) - tf.math.tanh((yy - y_2) / a)
      )

    def rho_u_init_fn(xx, yy, zz, lx, ly, lz, coord):
      """Initializes the density."""
      rho = rho_init_fn(xx, yy, zz, lx, ly, lz, coord)
      return (
          rho
          * self.u_flow
          * (tf.math.tanh((yy - y_1) / a) - tf.math.tanh((yy - y_2) / a)  - 1.0)
      )

    def rho_v_init_fn(xx, yy, zz, lx, ly, lz, coord):
      """Initializes the density."""
      rho = rho_init_fn(xx, yy, zz, lx, ly, lz, coord)
      return (
          rho
          * u_pert
          * tf.math.sin(2.0 * np.pi * xx)
          * (
              tf.math.exp(-((yy - y_1) ** 2) / sig**2)
              + tf.math.exp(-((yy - y_2) ** 2) / sig**2)
          )
      )

    def rho_c_init_fn(xx, yy, zz, lx, ly, lz, coord):
      """Initializes the passive scalar."""
      rho = rho_init_fn(xx, yy, zz, lx, ly, lz, coord)
      return (
          rho
          * 0.5
          * (tf.math.tanh((yy - y_2) / a) - tf.math.tanh((yy - y_1) / a) + 2.0)
      )

    return {
        types.RHO: gen_init_state(rho_init_fn),
        types.RHO_U: gen_init_state(rho_u_init_fn),
        types.RHO_V: gen_init_state(rho_v_init_fn),
        types.RHO_W: gen_init_state(initializer.constant_initial_state_fn(0.0)),
        types.RHO_E: gen_init_state(
            initializer.constant_initial_state_fn(
                self.cfg.p_0 / (constant.GAMMA - 1.0)
            )
        ),
        'rho_c': gen_init_state(rho_c_init_fn),
    }
