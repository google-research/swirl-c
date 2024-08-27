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
"""A library of physical constants."""

# The gravitational constant (m^2/s).
G = 9.81

# The atmospheric pressure at sea level.
P_0 = 1.01325e5

# The heat capacity ratio.
GAMMA = 1.4

# The molecular weight of the dry air (kg/mol).
W_AIR = 0.029

# The universal gas constant (J/mol/K).
R_U = 8.314

# The mass specific gas constant of dry air (J/kg/K).
R = R_U / W_AIR

# The isochoric specific heat of dry air (J/kg/K).
CV = R / (GAMMA - 1.0)

# The isobaric specific heat of dry air (J/kg/K).
CP = CV + R

# The adiabatic exponent of dry air.
KAPPA = R / CP
