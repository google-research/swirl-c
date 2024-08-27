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
"""Specifies the available kernel operators for Swirl-C."""

import enum


class KernelOpType(enum.Enum):
  """Defines the available kernel operators."""
  # BEGIN GOOGLE-INTERNAL
  # TODO: b/310751202 - Here we use the same value convention as Swirl-LM for
  # KernelOpType, including the UNKNOWN type and integer values. An update to
  # across both codes to use descriptive string values is suggested. Further,
  # the UNKNOWN kernnel_op is not required.
  # END GOOGLE-INTERNAL
  KERNEL_OP_UNKNOWN = 0
  KERNEL_OP_CONV = 1
  KERNEL_OP_SLICE = 2
  KERNEL_OP_MATMUL = 3
