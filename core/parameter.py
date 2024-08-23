"""A template of the global parameter context to be used in simulations."""

import copy
import os
from typing import Any, Dict, Optional
import numpy as np
from swirl_c.boundary import bc_types
from swirl_c.common import types
from swirl_c.numerics import kernel_op_types
from swirl_lm.utility import get_kernel_fn
import tensorflow as tf
import yaml


def _get_cell_center_coordinates(
    start: float, stop: float, n_cells: int
) -> np.ndarray:
  xx = np.linspace(start, stop, n_cells + 1, dtype=types.NP_DTYPE)
  xx = xx[:-1] + 0.5 * np.diff(xx)[0]
  return xx


_XX = _get_cell_center_coordinates(0.0, 1.0, 10)
_DX = np.diff(_XX)[0]

# BEGIN GOOGLE-INTERNAL
# TODO: b/308713108 - Use of default values like this is not advised. Correct
# definition of required fields, and hard failures on undefined required
# parameters are required.
# END GOOGLE-INTERNAL
_DEFAULT_OPTIONS = {
    # Swirl-LM compatibility
    'apply_preprocess': False,
    'apply_postprocess': False,
    'states_to_file': None,
    'states_from_file': None,
    # Set processor topology.
    'cx': 1,
    'cy': 1,
    'cz': 1,
    # Set mesh properties.
    'use_3d_tf_tensor': False,
    'halo_width': 3,
    'nx': 10,
    'ny': 10,
    'nz': 10,
    'core_nx': 10,
    'core_ny': 10,
    'core_nz': 10,
    'lx': 1.0,
    'ly': 1.0,
    'lz': 1.0,
    'x': _XX,
    'dx': _DX,
    'y': _XX,
    'dy': _DX,
    'z': _XX,
    'dz': _DX,
    'g': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    # Set timestepping details.
    'start_step': 0,
    'num_steps': 50,
    'num_cycles': 5,
    'dt': 1.0e-3,
    # Set selected numerics and physics models.
    'solver': 'compressible_navier_stokes',
    'interpolation_scheme': 'WENO_5',
    'numeric_flux_scheme': 'HLL',
    'include_diffusion': False,
    'time_integration_scheme': 'rk3',
    'source_functions': None,
    'kernel_op_type': kernel_op_types.KernelOpType.KERNEL_OP_SLICE.value,
    'kernel_size': 8,
    # Set themodynamics model details.
    'thermodynamics_model': 'generic',
    'p_0': 101325.0,
    # Set transport model details.
    'transport_model': 'simple',
    'transport_parameters': {'nu': 1.0e-4, 'pr': 0.7},
    # BEGIN GOOGLE-INTERNAL
    # TODO: b/311341083 - The 'nu' and 'nu_t' parameters should be removed here,
    # and specified through the transport library.
    # END GOOGLE-INTERNAL
    'nu': 1e-1,  # Retained for compatibility with fluid.py
    'nu_t': None,  # Retained for compatibility with fluid.py
    # Specify conservative and primitive variable names.
    'conservative_variable_names': list(types.BASE_CONSERVATIVE),
    'primitive_variable_names': list(types.BASE_PRIMITIVES),
    # Specify names of helper variables that take the same shape and format as
    # flow field variables.
    'additional_state_names': [],
    # Specify names of helper variables that takes arbitrary shape and/or
    # format.
    'helper_variable_names': [],
    # Specify names of debug variables to save at the end of each cycle.
    'debug_variables': [],

    # Specify boundary conditions.
    'bc': {
        'cell_averages': {
            var_name: {
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
            }
            for var_name in types.BASE_CONSERVATIVE
        }
    },
}


class SwirlCParameters:
  """The class defining the simulation context object."""

  # BEGIN GOOGLE-INTERNAL
  # TODO(b/308713108): While this approach to defining the parameters is
  # suffecient, it is not particularly elegant, intuitive, or safe. Use of class
  # parameters and proper documentation is required.
  # END GOOGLE-INTERNAL

  # Swirl-LM compatibility
  apply_preprocess: bool
  apply_postprocess: bool
  states_to_file: list[str]
  states_from_file: list[str]
  # Swirl-LM compatibility
  cx: int
  cy: int
  cz: int
  # Set mesh properties.
  use_3d_tf_tensor: bool
  halo_width: int
  # BEGIN GOOGLE-INTERNAL
  # TODO(b/309863073): Note that Swirl-C and Swirl-LM define `nx`, `ny`, and
  # `nz` differently. In Swirl-C, `n_` is the global mesh size excluding halo
  # cells. Future cleanup will align these definitions.
  # END GOOGLE-INTERNAL
  # The total number of mesh points for the global computational domain
  # (excluding halos).
  nx: int
  ny: int
  nz: int
  # The number of mesh points in each core (excluding halos).
  core_nx: int
  core_ny: int
  core_nz: int
  lx: float
  ly: float
  lz: float
  x: tf.Tensor
  dx: float
  y: tf.Tensor
  dy: float
  z: tf.Tensor
  dz: float
  g: Dict[str, float]
  # Set timestepping details.
  start_step: int
  num_steps: int
  num_cycles: int
  dt: float
  # Set selected numerics and physics models.
  solver: str
  interpolation_scheme: str
  numeric_flux_scheme: str
  include_diffusion: bool
  time_integration_scheme: str
  source_functions: Optional[list[str]]
  kernel_op_type: kernel_op_types.KernelOpType
  kernel_size: int
  kernel_op: get_kernel_fn.ApplyKernelOp
  # Specify thermodynamics model and parameters.
  thermodynamics_model: str
  p_0: float
  # Specify transport model and parameters.
  transport_model: str
  transport_parameters: Dict[str, Any]
  nu: float
  nu_t: Optional[float]
  # Specify conservative and primitive variable names.
  conservative_variable_names: list[str]
  primitive_variable_names: list[str]
  # Specify  names of helper variables that takes the same shape and format as
  # flow field variables.
  additional_state_names: list[str]
  # Specify  names of helper variables that takes arbitrary shape and/or format.
  helper_variable_names: list[str]
  # Specify boundary conditions.
  bc: Dict[str, Any]

  def __init__(self, user_cfg: Optional[Dict[str, Any]] = None):
    """Method to initialize the simulation context object `CFG`.

    Args:
      user_cfg: A dictionary of key/value pairs where keys refer to parameters
        of `CFG` and values are user specified values. If a key does not exist
        for a parameter, the default value is used.

    Raises:
      ValueError: If the specified kernel operator is not recognized.
    """
    # While only one instance of `SwirlCParameters` exists for a given
    # simulation, during testing multiple instances of `SwirlCParameters` are
    # constructed. Thus it is necessary to copy rather than assign to avoid
    # changing `_DEFAULT_OPTIONS` during testing. A deepcopy is required because
    # we use nested dictionaries.
    self._simulation_cfg = copy.deepcopy(_DEFAULT_OPTIONS)
    if user_cfg:
      for key, val in user_cfg.items():
        self._simulation_cfg[key] = val
    for key, val in self._simulation_cfg.items():
      setattr(self, key, val)
    if self.kernel_op_type == kernel_op_types.KernelOpType.KERNEL_OP_CONV.value:
      self.kernel_op = get_kernel_fn.ApplyKernelConvOp(self.kernel_size)
    elif (
        self.kernel_op_type
        == kernel_op_types.KernelOpType.KERNEL_OP_SLICE.value
    ):
      self.kernel_op = get_kernel_fn.ApplyKernelSliceOp()
    elif (
        self.kernel_op_type
        == kernel_op_types.KernelOpType.KERNEL_OP_MATMUL.value
    ):
      self.kernel_op = get_kernel_fn.ApplyKernelMulOp(self.nx, self.ny)
    else:
      raise ValueError('Unknown kernel operator {}'.format(self.kernel_op_type))

  def save_to_file(self, prefix: str) -> None:
    """Saves configuration dictionary as YAML."""
    output_dir, _ = os.path.split(prefix)
    tf.io.gfile.makedirs(output_dir)
    with tf.io.gfile.GFile(f'{prefix}_swirl_c_cfg.yml', 'w') as f:
      yaml.dump(self._simulation_cfg, f, default_flow_style=False)
