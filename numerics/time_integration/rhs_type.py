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
