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
"""A wrapper to link the selected Riemann solver to the flux calculation."""

import enum
from typing import Callable
from swirl_c.common import types
from swirl_c.core import parameter
from swirl_c.numerics.riemann_solver import hll
from swirl_c.numerics.riemann_solver import hllc


class NumericFluxScheme(enum.Enum):
  """Defines the available numeric flux models."""
  HLL = 'HLL'
  HLLC = 'HLLC'


def select_numeric_flux_fn(
    cfg: parameter.SwirlCParameters,
) -> Callable[..., types.FlowFieldMap]:
  """Function to select the specified Riemann solver to the flux calculation.

    This function returns the function to compute the numeric flux as specified
    by `cfg.numeric_flux_scheme`.

  Args:
    cfg: The context object that stores parameters and information required by
      the simulation.

  Returns:
    A function which computes the intercell flux based on the user specified
    numeric flux algorithm. The flux function must return the approximate
    intercell fluxes at the i - 1/2 face of the computational cell normal to the
    dimension specified.

  Raises:
    NotImplementedError if the specified `cfg.numeric_flux_scheme` is not
    implmented.
  """

  if cfg.numeric_flux_scheme == NumericFluxScheme.HLL.value:
    return hll.hll_convective_flux
  elif cfg.numeric_flux_scheme == NumericFluxScheme.HLLC.value:
    return hllc.hllc_convective_flux
  else:
    raise ValueError(
        f'"{cfg.numeric_flux_scheme}" is not implemented as a'
        ' "numeric_flux_scheme". Valid options are: '
        + str([scheme.value for scheme in NumericFluxScheme])
    )
