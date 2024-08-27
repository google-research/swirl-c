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
"""A library for commonly used types."""

import re
from typing import Dict, List, Sequence

import numpy as np
import tensorflow as tf


# Precision.
DTYPE = tf.float32
NP_DTYPE = np.float32

# Define a small number to be used in comparison to find zero values.
SMALL = np.finfo(np.float32).resolution

# Basic data types.
FlowFieldVar = Sequence[tf.Tensor] | tf.Tensor
MutableFlowFieldVar = List[tf.Tensor]
FlowFieldMap = Dict[str, FlowFieldVar]

# Define the dimensions of a 3D tensor.
DIMS = ('x', 'y', 'z')

# Here we define the minimum neccessary primitive and conservative variables for
# the simulation, the dictionary keys used identify the flow field variable, and
# utilities to convert between primitive and conservative dictionary keys.

# Define density, which is a special case that is carried in both the PRIMITIVE
# and CONSERVATIVE variable tuples.
RHO = 'rho'

# Define three components of velocity as primitives.
U = 'u'
V = 'v'
W = 'w'
VELOCITY = (U, V, W)

# Define the specific total energy as the final primitive variable.
E = 'e'

# Define the tuple which lists the minimum necessary primitive variables.
BASE_PRIMITIVES = (RHO, U, V, W, E)

# Define the string that will prepend the primitive variable names to give the
# conservative variable names.
CONS_VAR_PREFIX = 'rho_'

# Define linear momentum terms.
RHO_U = CONS_VAR_PREFIX + U
RHO_V = CONS_VAR_PREFIX + V
RHO_W = CONS_VAR_PREFIX + W
MOMENTUM = (RHO_U, RHO_V, RHO_W)

# Define the volumetric total energy as final conservative variable.
RHO_E = CONS_VAR_PREFIX + E

# Define the tuple which lists the minimum necessary conservative variables.
BASE_CONSERVATIVE = (RHO, RHO_U, RHO_V, RHO_W, RHO_E)

# Define additional intensive thermodynamic properties: pressure,
# temperature, potential temperature, and total enthalpy.
P = 'p'
T = 't'
POTENTIAL_T = 'theta'
H = 'h'


def primitive_to_conservative_name(primitive: str) -> str:
  """Converts a primitive variable name to a conservative variable name.

  Args:
    primitive: A string containing the name of the primitive variable.

  Returns:
    The associated conservative variable name as a string.

  Raises:
    ValueError if the input string is empty.
  """
  if not primitive:
    raise ValueError('Primitive variable name must not be empty.')
  if primitive == RHO:
    return RHO
  else:
    return CONS_VAR_PREFIX + primitive


def conservative_to_primitive_name(conservative: str) -> str:
  """Converts a conservative variable name to a primitive variable name.

  The conservative variable name must start with the `CONS_VAR_PREFIX` and must
    have at least 1 character following the `CONS_VAR_PREFIX`.

  Args:
    conservative: A string containing the name of the conserved variable.

  Returns:
    The associated primitive variable name as a string.

  Raises:
    ValueError if the input string does not start with `CONS_VAR_PREFIX` or is
    `CONS_VAR_PREFIX` only.
  """
  if conservative == RHO:
    return RHO
  elif is_conservative_name(conservative):
    return conservative[len(CONS_VAR_PREFIX) :]
  else:
    raise ValueError(
        f'Invalid conserved variable name: {conservative}. Must start with'
        f' "{CONS_VAR_PREFIX}".'
    )


def is_conservative_name(varname: str) -> bool:
  """Helper function to check if `varname` matches expected format.

  The conservative variable name must start with the `CONS_VAR_PREFIX` and must
  have at least 1 character following the `CONS_VAR_PREFIX`.

  Args:
    varname: The variable name to be compared against the expected format.

  Returns:
    `True` if `varname` matches expected conservative variable naming format, or
    `False` if the name does not match the expected format.
  """
  return bool(re.match(f'^({CONS_VAR_PREFIX}).', varname))
