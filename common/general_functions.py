import numpy as np

# Atomic-unit conversions
AU_ENERGY    = 27.211386245988        # eV
AU_TIME      = 24.188843265857        # as
AU_INTENSITY = 3.5094452e16           # W/cm^2


def E0_to_I(E):
    return AU_INTENSITY * np.abs(E)**2


def I_to_E0(I):
    return np.sqrt(I / AU_INTENSITY)


def eV(E):
    return E * AU_ENERGY


def au(E):
    return E / AU_ENERGY
