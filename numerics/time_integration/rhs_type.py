"""Library which defines the type for the RHS function for integration."""

from typing import Callable
from swirl_c.common import types
import tensorflow as tf

# Define the type describing the function which evaluates the right hand side
# of the governing equations given an initial state.
RHS = Callable[
    [
        tf.Tensor,  # replica_id: The ID of the computatinal subdomain.
        types.FlowFieldMap,  # state_0: The state which the RHS is computed for.
        types.FlowFieldMap,  # helper_vars: Helper variables.
    ],
    types.FlowFieldMap,
]
