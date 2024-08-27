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
"""Library for the Swirl-C simulation driver."""

# BEGIN GOOGLE-INTERNAL
# TODO: b/310756130 - Currently, Swirl-C patches the Swirl-LM driver by
# overriding internal methods in swirl_lm.core.driver.py, then calling the
# Swirl-LM driver. Ideally, an abstract class would define the Swirl family
# (LM and C) driver, which each solver would then create an instance of and
# modify as necessary.
# END GOOGLE-INTERNAL


from typing import Callable, Tuple, Union
from absl import logging
from swirl_c.common import types
from swirl_c.core import initializer
from swirl_c.core import parameter
from swirl_c.core import simulation
from swirl_lm.base import driver
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types as swirl_lm_types
import tensorflow as tf


# The type describing the problem specified initialization function for the
# flow field variables.
InitFn = Callable[
    [Union[int, tf.Tensor], Tuple[int, int, int], parameter.SwirlCParameters],
    types.FlowFieldMap,
]


# BEGIN GOOGLE-INTERNAL
# This method is necessary to patch the Swirl-LM simulation driver
# `swirl_lm.base.driver.py` to be compatible with Swirl-C. Future updates to
# unify Swirl-LM and Swirl-C will remove the need for this patch. See
# b/310756130 for further details.
# END GOOGLE-INTERNAL
def _init_fn(
    cfg: parameter.SwirlCParameters,
    customized_init_fn: InitFn,
) -> Callable[[tf.Tensor, Tuple[int, int, int]], types.FlowFieldMap]:
  """Defines the method to generate initial conditions for the simulation.

  Args:
    cfg: The context object that stores parameters and information required by
      the simulation.
    customized_init_fn: A callable which takes `replica_id`, `coordinates`, and
      `cfg` as arguments, and returns the initial conditions for the simulation
      as a dictionary of `tf.tensors`, where each key, value pair refers to a
      single conservative variable. If `customized_init_fn = None`, the default
      initialization is used, setting all flow field variables to zero across
      the domain. This must be updated by either loading a saved file or
      checkpoint before time stepping begins to produce physically meaningful
      results.

  Returns:
    A callable to distribute the initial conditions across the multiple
    processors. The callable takes the processor id `replica_id`, and processor
    coordinate `coordinates`, and returns the essential simulation variables for
    the local processor as the dictionary `states`. `states` includes
    `replica_id` and the initial values for the conservative flow field
    variables on the local processor mesh.
  """

  if customized_init_fn is not None:
    logging.info(
        'Performing initialization from provided `customized_init_fn`.'
    )
  else:
    logging.warning(
        'Performing default initialization. Initial state must be overwritten'
        ' with simulation state from save file. Please ensure a save is loaded.'
    )

  def init_fn(
      replica_id: tf.Tensor,
      coordinates: swirl_lm_types.ReplicaCoordinates,
  ) -> types.FlowFieldMap:
    """Initializes all variables required in the simulation on the local TPU.

    Args:
      replica_id: The ID of the computatinal subdomain (replica).
      coordinates: A vector/sequence of integer with length 3 representing the
        logical coordinate of the core in the logical mesh [x, y, z].

    Returns:
      A dictionary of the essential simulation variables for the local
      processor. Includes the `replica_id` and all conservative flow field
      variables on the local mesh.
    """

    states = {'replica_id': replica_id}

    if customized_init_fn is not None:
      states.update(customized_init_fn(replica_id, coordinates, cfg))
    else:
      states.update(
          initializer.default_initializer(replica_id, coordinates, cfg)
      )
    return states

  return init_fn


# BEGIN GOOGLE-INTERNAL
# This method is necessary to patch the Swirl-LM simulation driver
# `swirl_lm.base.driver.py` to be compatible with Swirl-C. Future updates to
# unify Swirl-LM and Swirl-C will remove the need for this patch. See
# b/310756130 for further details.
# END GOOGLE-INTERNAL
def _get_state_keys(
    cfg: parameter.SwirlCParameters,
):
  """Returns the essential, additional, and helper variable keys.

  Args:
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns: Lists of the essential, additional, and helper variable keys for use
    with the `states` dictionary. For Swirl-C, only essential keys are required,
    and the essential keys are `cfg.conservative_variable_names`.
  """
  essential_keys = cfg.conservative_variable_names
  additional_keys = cfg.additional_state_names
  helper_var_keys = cfg.helper_variable_names
  return essential_keys, additional_keys, helper_var_keys


# BEGIN GOOGLE-INTERNAL
# This method is necessary to patch the Swirl-LM simulation driver
# `swirl_lm.base.driver.py` to be compatible with Swirl-C. Future updates to
# unify Swirl-LM and Swirl-C will remove the need for this patch. See
# b/310756130 for further details.
# END GOOGLE-INTERNAL
def _get_model(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    cfg: parameter.SwirlCParameters,
):
  """Returns the simulation model.

  Args:
    kernel_op: Unused argument provided for compatibility with Swirl-LM. In
      Swirl-C the `kernel_op` is contained in the `cfg` object and is specified
      when the `cfg` class instance is constructed.
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    An instance of the simulation model class.
  """
  return simulation.Simulation(kernel_op, cfg)


# BEGIN GOOGLE-INTERNAL
# This method is necessary to patch the Swirl-LM simulation driver
# `swirl_lm.base.driver.py` to be compatible with Swirl-C. Future updates to
# unify Swirl-LM and Swirl-C will remove the need for this patch. See
# b/310756130 for further details.
def _update_additional_states(
    essential_states, additional_states, step_id, **common_kwargs
):
  """Updates the additional states based on the essential states.

  In Swirl-C, only essential keys are required. For compatibility with Swirl-LM,
  this method is retained and returns the empty additional state dictionary
  unmodified.


  Args:
    essential_states: A dictionary of flow field variables containing the states
      essential for the simulation. In Swirl-C, these are the states given by
      `cfg.conservative_variable_names`.
    additional_states: A dictionary of nonessential flow field variables used
      during simulation. For Swirl-C, `additional_states` are unused and the
      dictionary is empty.
    step_id: ID of the current simulation step.
    **common_kwargs: Commonly used arguments `kernel_op`, `replica_id`,
      `replicas`, and `cfg`.

  Returns:
    The updated additional states dictionary. For Swirl-C, additional states is
    passed through umodified.
  """
  del essential_states, step_id, common_kwargs  # Unused.
  return additional_states


def solver(
    customized_init_fn: InitFn,
    cfg: parameter.SwirlCParameters,
):
  """Runs the Navier-Stokes Solver with TF2 Distribution strategy.

  Args:
    customized_init_fn: The function that initializes the flow field. The
      function needs to be replica dependent.
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    A tuple of the final state on each replica.
  """
  # First, patch Swirl-LM driver for compatibility issues.
  driver._init_fn = _init_fn  # pylint: disable=protected-access
  driver._get_model = _get_model  # pylint: disable=protected-access
  driver._update_additional_states = (  # pylint: disable=protected-access
      _update_additional_states
  )
  driver._get_state_keys = _get_state_keys  # pylint: disable=protected-access

  # Run the patched Swirl-LM driver.
  return driver.solver(customized_init_fn, cfg)
