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
