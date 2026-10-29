"""Core functions for the multi-pulse ion-electron entanglement model.

The functions in this module avoid hidden global variables: all quantities needed
for a calculation are passed explicitly as arguments. This makes the code easier
to reuse, test, and version-control with Git.
"""

import numpy as np
from scipy import special


_VALID_PULSE_TYPES = {"ftop", "gaus"}


def _validate_pulses(t_Is, EI, tau_I, typ):
    """Convert pulse inputs to arrays and perform basic consistency checks."""
    t_Is = np.asarray(t_Is, dtype=float)
    EI = np.asarray(EI, dtype=float)
    tau_I = np.asarray(tau_I, dtype=float)

    if typ not in _VALID_PULSE_TYPES:
        raise ValueError("typ must be either 'ftop' or 'gaus'.")

    if not (t_Is.ndim == EI.ndim == tau_I.ndim == 1):
        raise ValueError("t_Is, EI, and tau_I must be one-dimensional arrays.")

    if not (len(t_Is) == len(EI) == len(tau_I)):
        raise ValueError("t_Is, EI, and tau_I must have the same length.")

    if len(t_Is) == 0:
        raise ValueError("At least one pulse must be supplied.")

    if np.any(tau_I <= 0):
        raise ValueError("All pulse widths tau_I must be positive.")

    return t_Is, EI, tau_I

def def_t_array(typ, t_Is, tau_I):

    if typ == "ftop":        
        t = np.linspace(-tau_I[0]*0.5 + t_Is[0], tau_I[-1]*0.5 + t_Is[-1], 500)

    else:  # typ == 'gaus'
        t = np.linspace(-tau_I[0]*2.5 + t_Is[0], tau_I[-1]*2.5 + t_Is[-1], 500)
        
    return t

def calc_field(t, t_Is, EI, tau_I, omega_0, typ):
    """Calculate the total electric field and pulse envelopes.

    Parameters
    ----------
    t : array_like
        Time array.
    t_Is : array_like
        Center time of each pulse.
    EI : array_like
        Field amplitude of each pulse.
    tau_I : array_like
        Width of each pulse.
    omega_0 : float
        Carrier angular frequency.
    typ : {'ftop', 'gaus'}
        Pulse-envelope type.

    Returns
    -------
    Et : ndarray
        Real total electric field E(t).
    E_Envelope : ndarray
        Sum of the field envelopes, including pulse amplitudes EI.
    Envelope_I : ndarray
        Individual pulse envelopes, shape (Nt, Npulse).
    E_pos : ndarray
        Positive-frequency component of the total field.
    """
    t_Is, EI, tau_I = _validate_pulses(t_Is, EI, tau_I, typ)
    
    if typ == "ftop":        
        dt_from_center = t[:, None] - t_Is[None, :]
        
        Envelope_I = (
            np.heaviside(dt_from_center + tau_I[None, :] / 2, 0.0)
            - np.heaviside(dt_from_center - tau_I[None, :] / 2, 0.0)
        )

    else:  # typ == 'gaus'
        dt_from_center = t[:, None] - t_Is[None, :]
    
        ell = np.log(4.0) / tau_I**2
        Envelope_I = np.exp(-ell[None, :] * dt_from_center**2)

    E_Envelope = np.sum(EI[None, :] * Envelope_I, axis=1)

    E_pos = np.sum(
        EI[None, :] / 2
        * Envelope_I
        * np.exp(-1j * omega_0 * dt_from_center),
        axis=1,
    )

    Et = E_pos + np.conj(E_pos)

    return Et, E_Envelope, Envelope_I, E_pos


def ar_f_tot(t_A, t_B, t_Is, EI, tau_I, z12):
    """Pulse areas between t_B and t_A for flat-top pulses.

    t_A and t_B may have equal lengths, or one may contain a single value that
    is broadcast against the other.
    """
    t_Is, EI, tau_I = _validate_pulses(t_Is, EI, tau_I, "ftop")

    t_A = np.atleast_1d(np.asarray(t_A, dtype=float))
    t_B = np.atleast_1d(np.asarray(t_B, dtype=float))
    t_A, t_B = np.broadcast_arrays(t_A, t_B)

    lower = t_Is - tau_I / 2
    upper = t_Is + tau_I / 2

    clipped_A = np.clip(t_A[:, None], lower[None, :], upper[None, :])
    clipped_B = np.clip(t_B[:, None], lower[None, :], upper[None, :])

    return z12 * EI[None, :] * (clipped_A - clipped_B)


def ar_g_tot(t_A, t_B, t_Is, EI, tau_I, z12):
    """Pulse areas between t_B and t_A for Gaussian pulses."""
    t_Is, EI, tau_I = _validate_pulses(t_Is, EI, tau_I, "gaus")

    t_A = np.atleast_1d(np.asarray(t_A, dtype=float))
    t_B = np.atleast_1d(np.asarray(t_B, dtype=float))
    t_A, t_B = np.broadcast_arrays(t_A, t_B)

    ell = np.log(4.0) / tau_I**2
    sqrt_ell = np.sqrt(ell)[None, :]

    return (
        z12
        * EI[None, :]
        / 2
        * np.sqrt(np.pi / ell)[None, :]
        * (
            special.erf(sqrt_ell * (t_A[:, None] - t_Is[None, :]))
            - special.erf(sqrt_ell * (t_B[:, None] - t_Is[None, :]))
        )
    )


def Rabi_amp_Npulse(t_A, t_B, t_Is, EI, tau_I, z12, omega_0, typ):
    """Calculate bound-state amplitudes a and b for a sequence of pulses.

    The system starts in a=1, b=0 at t_B, and the pulses are applied in the
    order supplied in t_Is/EI/tau_I.
    """
    t_Is, EI, tau_I = _validate_pulses(t_Is, EI, tau_I, typ)

    if typ == "ftop":
        A_arr = ar_f_tot(t_A, t_B, t_Is, EI, tau_I, z12)
    else:
        A_arr = ar_g_tot(t_A, t_B, t_Is, EI, tau_I, z12)

    phi = omega_0 * (t_Is - t_Is[0])

    a0 = np.ones(A_arr.shape[0], dtype=complex)
    b0 = np.zeros(A_arr.shape[0], dtype=complex)

    for i in range(A_arr.shape[1]):
        half_area = A_arr[:, i] / 2

        a1 = (
            np.cos(half_area) * a0
            - 1j * np.sin(half_area) * np.exp(-1j * phi[i]) * b0
        )
        b1 = (
            np.cos(half_area) * b0
            - 1j * np.sin(half_area) * np.exp(+1j * phi[i]) * a0
        )

        a0, b0 = a1, b1

    return a0, b0


def WF_amp_Npulse(
    t,
    eps,
    t_Is,
    EI,
    tau_I,
    omega_0,
    z12,
    z0,
    typ,
    depletion=True,
):
    """Calculate a(t), b(t), g(t), c_a(eps), and c_b(eps).

    Parameters
    ----------
    t : array_like
        Time array.
    eps : array_like
        Photoelectron-energy grid.
    t_Is, EI, tau_I : array_like
        Pulse centers, amplitudes, and widths.
    omega_0 : float
        Carrier angular frequency.
    z12 : float
        Bound-bound dipole matrix element.
    z0 : float
        Ionization dipole matrix element.
    typ : {'ftop', 'gaus'}
        Pulse-envelope type.
    depletion : bool, default=True
        Include ground-state depletion when True.
    """
    eps = np.asarray(eps, dtype=float)
    
    _, E_Envelope, _, E_pos = calc_field(
        t, t_Is, EI, tau_I, omega_0, typ
    )

    dt = np.diff(t)[0]
    
    a, b = Rabi_amp_Npulse(
        t,
        np.array([t[0]]),
        t_Is,
        EI,
        tau_I,
        z12,
        omega_0,
        typ,
    )

    if depletion:
        g = np.exp(-np.pi * z0**2 / 4 * np.cumsum(E_Envelope**2) * dt)
    else:
        g = np.ones_like(t)

    # Bound-state amplitudes from each possible ionization time to the final time.
    a_after, b_after = Rabi_amp_Npulse(
        np.array([t[-1]]),
        t,
        t_Is,
        EI,
        tau_I,
        z12,
        omega_0,
        typ,
    )

    continuum_phase = np.exp(1j * (eps[:, None] + omega_0) * t[None, :])
    common = g * E_pos * z0

    c_a = -1j * dt * np.sum(
        common[None, :] * a_after[None, :] * continuum_phase,
        axis=1,
    )
    c_b = -1j * dt * np.sum(
        common[None, :] * b_after[None, :] * continuum_phase,
        axis=1,
    )

    return a, b, g, c_a, c_b
