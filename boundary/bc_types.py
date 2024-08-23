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
