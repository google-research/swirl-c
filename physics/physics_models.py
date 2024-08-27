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
"""Defines a class which contains the physics models used in the simulation."""

import enum
from absl import logging
import numpy as np
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.physics.additional_models import gravity
from swirl_c.physics.thermodynamics import generic
from swirl_c.physics.transport import simple
import tensorflow as tf


class ThermodynamicsModels(enum.Enum):
  """Defines the available thermodynamics models."""
  GENERIC = 'generic'


class TransportModels(enum.Enum):
  """Defines the available transport models."""
  SIMPLE = 'simple'


class SourceTermModels(enum.Enum):
  """Defines the available source term models."""
  GRAVITY = 'gravity'


class PhysicsModels:
  """Defines physical models employed in the present simulation."""

  def __init__(self, cfg: parameter.SwirlCParameters):
    """Constructs the phyiscs model class as a container for simulation models.

    This class is used to collect all of the neccessary physical model classes
    required for the simulation.

    Args:
      cfg: The context object that stores parameters and information required by
        the simulation.

    Raises:
      `NotImplementedError` if a user specified model is not a valid model
      option.
    """
    if cfg.thermodynamics_model == ThermodynamicsModels.GENERIC.value:
      self.thermodynamics_model = generic.ThermodynamicsGeneric()
      logging.info('"generic" thermodynamics model selected.')
    else:
      raise NotImplementedError(
          'Unsupported thermodynamics model:'
          f' "{cfg.thermodynamics_model}". Supported models are: '
          + str([model.value for model in ThermodynamicsModels])
      )

    if cfg.transport_model == TransportModels.SIMPLE.value:
      self.transport_model = simple.TransportSimple(cfg)
      logging.info('"simple" transport model selected.')
    else:
      raise NotImplementedError(
          'Unsupported transport model:'
          f' "{cfg.transport_model}". Supported models are: '
          + str([model.value for model in TransportModels])
      )

    self.source_functions = []
    if cfg.source_functions:
      for source_function_name in cfg.source_functions:
        if source_function_name == SourceTermModels.GRAVITY.value:
          self.source_functions.append(gravity.Gravity(cfg))
          logging.info('"gravity" source function selected.')
        else:
          raise NotImplementedError(
              f'Unsupported source term model: "{source_function_name}".'
              ' Supported models are: '
              + str([model.value for model in SourceTermModels])
          )
    if not self.source_functions:
      logging.info('No source functions selected.')

  def source_function(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      conservative: types.FlowFieldMap,
      helper_vars: types.FlowFieldMap,
  ) -> types.FlowFieldMap:
    """Computes the source term for conservative variables from user models.

    When the physics model is initialized, a list of classes `source_functions`
    is populated with the user selected source function model classes. Each
    class must have a method `mask` and a method `source_term`. The `mask`
    method lists the conservative variable names of the conservative variables
    to be updated by that particular model. Only the variables returned by
    `mask` have their RHS updated, the rest remain unchanged by the model. The
    method `source_term` computes the contribution to the RHS of the governing
    equation by the specified model.

    Note that unlike other functions, the source function model uses an internal
    `cfg` that is defined during initialization, and does not use `cfg` as an
    argument to `mask` or `source_term`.

    Args:
      replica_id: The ID of the computatinal subdomain (replica).
      replicas: A numpy array that maps a replica's grid coordinate to its
        `replica_id`, e.g. `replicas[0, 0, 0] = 0`, `replicas[0, 0, 1] = 1`.
      conservative: Dictionary of conservative scalar flow field variables,
        listed in `cfg.conservative_variable_names`.
      helper_vars: Helper variables used in the simulation. These variables are
        not updated by the Navier-Stokes equations.

    Returns:
      A dictionary of flow field variables representing the volumetric
      source terms of the conserved variables. Returns all zeros if no source
      term models are selected by the user.
    """

    source_terms = {
        var_name: tf.nest.map_structure(tf.zeros_like, val)
        for var_name, val in conservative.items()
    }

    for model in self.source_functions:
      rhs = model.source_term(replica_id, replicas, conservative, helper_vars)
      for var_name in model.mask():
        source_terms[var_name] = tf.nest.map_structure(
            tf.add, source_terms[var_name], rhs[var_name]
        )

    return source_terms
