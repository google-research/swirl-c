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
"""Defines the class for the gravity physics model."""

from typing import Tuple

import numpy as np
from swirl_c.common import types
from swirl_c.common import utils
from swirl_c.core import parameter
from swirl_c.physics import constant
import tensorflow as tf


class Gravity:
  """Defines the function contracts for the gravity physics model."""

  def __init__(self, cfg: parameter.SwirlCParameters):
    """Constructs the gravity physics model.

    For all source function models, the `cfg` context object is error checked
    during `__init__`, and the `cfg` is then stored internally as `self._cfg` to
    be used with the source function model.

    Args:
      cfg: The context object that stores parameters and information required by
        the simulation.

    Raises:
      `ValueError` if the gravity direction is not specified in `cfg`.
    """
    gravity_direction_index = utils.gravity_direction(cfg)
    if gravity_direction_index == -1:
      raise ValueError(
          'Gravity direction must be specified if the gravity source function'
          ' is to be used.'
      )
    self._cfg = cfg

  def mask(self) -> Tuple[str, ...]:
    """Generates mask listing conservative variables to be updated by the model.

    Returns:
      A tuple listing the conservative variable names associated with the
      conservative variables to be updated by the gravity model.
    """
    gravity_direction_index = utils.gravity_direction(self._cfg)
    if gravity_direction_index == 3:
      return types.MOMENTUM
    else:
      return (types.MOMENTUM[gravity_direction_index],)

  def source_term(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      conservative: types.FlowFieldMap,
      helper_vars: types.FlowFieldMap,
  ) -> types.FlowFieldMap:
    """Computes the body force contribution by gravity to the momentum equation.

    Note that unlike elsewhere in the code, the `cfg` context object is not an
    argument, and instead the internal `self._cfg` is used.

    Args:
      replica_id: The ID of the computatinal subdomain (replica).
      replicas: A numpy array that maps a replica's grid coordinate to its
        `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`.
      conservative: Dictionary of conservative scalar flow field variables,
        listed in `cfg.conservative_variable_names`.
      helper_vars: Helper variables used in the simulation. These variables are
        not updated by the Navier-Stokes equations.

    Returns:
      A dictionary of flow field variables where each dictionary item indicates
      the contribution of gravity as a body force to the momentum equation for
      each of the three dimensions.
    """
    del replica_id, replicas  # unused by the current source term.

    d_rho = helper_vars.get('d_rho', conservative[types.RHO])

    rhs = {
        types.MOMENTUM[types.DIMS.index(dim)]: tf.nest.map_structure(
            lambda d_r: d_r * constant.G * self._cfg.g[dim],  # pylint: disable=cell-var-from-loop
            d_rho,
        )
        for dim in types.DIMS
    }

    return rhs
