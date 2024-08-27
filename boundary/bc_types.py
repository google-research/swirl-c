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
"""Defines boundary condition related types."""

import enum
from typing import Dict, Sequence, Tuple, Union

import tensorflow as tf


class BoundaryCondition(enum.Enum):
  """Defines the types of the boundary condition."""
  PERIODIC = 0
  DIRICHLET = 1
  NEUMANN = 2
  REFLECTIVE = 3


class BoundaryFluxType(enum.Enum):
  """Defines the different boundary fluxes which can be specified."""
  CONVECTIVE = "convective"
  DIFFUSIVE = "diffusive"
  TOTAL = "total"


class BoundaryDictionaryType(enum.Enum):
  """Defines the location and type of the boundary dictionary."""
  CELL_AVERAGES = "cell_averages"
  CELL_FACES = "cell_faces"
  INTERCELL_FLUXES = "intercell_fluxes"


# The type of boundary condition to be specified in the configuration.
BCDict = Dict[
    str, Dict[int, Tuple[BoundaryCondition, Union[float, Sequence[tf.Tensor]]]]
]
