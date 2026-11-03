"""General physical constants and unit-conversion helpers."""

import numpy as np

# Atomic-unit conversions
AU_ENERGY = 27.211386245988       # eV
AU_TIME = 24.188843265857         # as
AU_INTENSITY = 3.5094452e16       # W/cm^2


def E0_to_I(E):
    """Convert electric-field amplitude from a.u. to intensity in W/cm^2."""
    return AU_INTENSITY * np.abs(E) ** 2


def I_to_E0(I):
    """Convert intensity in W/cm^2 to electric-field amplitude in a.u."""
    return np.sqrt(I / AU_INTENSITY)


def au_to_eV(E):
    """Convert energy from atomic units to eV."""
    return E * AU_ENERGY


def eV_to_au(E):
    """Convert energy from eV to atomic units."""
    return E / AU_ENERGY


# Backwards-compatible short aliases used in older notebooks.
def eV(E):
    return au_to_eV(E)


def au(E):
    return eV_to_au(E)
