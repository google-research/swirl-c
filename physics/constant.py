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
