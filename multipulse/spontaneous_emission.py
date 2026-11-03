"""Single-photon spontaneous-emission amplitudes for the multipulse model."""

import numpy as np

from .energy_grids import photon_coupling, spontaneous_decay_rate
from .functions import Rabi_amp_Npulse, calc_field


def WF_amp_spontaneous_Npulse(
    tf,
    t,
    eps,
    eps_l,
    t_Is,
    EI,
    tau_I,
    omega_0,
    eps_ba,
    z12,
    z0,
    typ,
    depletion=True,
):
    """Calculate a,b,g and the continuum amplitudes c_a,c_b,c_c,c_d.

    This is the multipulse implementation of the one-photon spontaneous-
    emission model. All inputs are explicit; the function does not rely on
    notebook globals.
    """
    t = np.asarray(t, dtype=float)
    eps = np.asarray(eps, dtype=float)
    eps_l = np.asarray(eps_l, dtype=float)

    if len(t) < 2:
        raise ValueError("t must contain at least two points.")

    dt = t[1] - t[0]
    kappa = spontaneous_decay_rate(eps_ba, z12)
    V_fi_l = photon_coupling(eps_l, z12)

    _, E_envelope, _, E_pos = calc_field(t, t_Is, EI, tau_I, omega_0, typ)

    a, b = Rabi_amp_Npulse(
        t, np.array([t[0]]), t_Is, EI, tau_I, z12, omega_0, typ
    )

    if depletion:
        g = np.exp(-np.pi * z0**2 / 4.0 * np.cumsum(E_envelope**2) * dt)
    else:
        g = np.ones_like(t)

    # Propagation from each possible ionisation time to the end of the pulse.
    a_to_end, b_to_end = Rabi_amp_Npulse(
        np.array([t[-1]]), t, t_Is, EI, tau_I, z12, omega_0, typ
    )

    continuum_phase = np.exp(1j * (eps[:, None] + omega_0) * t[None, :])
    survival_during_pulse = np.exp(-kappa / 4.0 * (t[-1] - t))
    common = g * E_pos * z0 * survival_during_pulse

    c_a = -1j * dt * np.sum(
        common[None, :] * a_to_end[None, :] * continuum_phase,
        axis=1,
    )
    c_b = (
        -1j
        * dt
        * np.sum(
            common[None, :] * b_to_end[None, :] * continuum_phase,
            axis=1,
        )
        * np.exp(-kappa / 2.0 * (tf - t[-1]))
    )

    c_c = np.zeros((len(eps), len(eps_l)), dtype=complex)
    c_d = np.zeros((len(eps), len(eps_l)), dtype=complex)

    delta_l = eps_l[None, :] - omega_0
    emission_prefactor = np.sqrt(2.0 * np.pi * 4.0 / 3.0) * z0 * V_fi_l[None, :]

    for i in range(len(t)):
        b_from_i = Rabi_amp_Npulse(
            t[i:],
            np.array([t[i]]),
            t_Is,
            EI,
            tau_I,
            z12,
            omega_0,
            typ,
        )[1]

        photon_phase = np.exp(1j * delta_l * t[i:, None])
        survival = np.exp(-kappa / 4.0 * (t[i:, None] - t[i]))

        # Emission during the laser interaction.
        c_term_during = dt * np.sum(
            a_to_end[i:, None]
            * 1j
            * b_from_i[:, None]
            * photon_phase
            * survival,
            axis=0,
        )[None, :]

        # Emission after the pulse has ended.
        denominator = 1j * delta_l - kappa / 2.0
        c_term_after = (
            1j
            * b_to_end[i]
            * np.exp(-kappa / 2.0 * (-t[i]))
            * (
                np.exp((1j * delta_l - kappa / 2.0) * tf)
                - np.exp((1j * delta_l - kappa / 2.0) * t[-1])
            )
            / denominator
        )

        ionisation_factor = (
            emission_prefactor
            * dt
            * g[i]
            * E_pos[i]
            * np.exp(1j * (eps[:, None] + omega_0) * t[i])
        )

        c_c += ionisation_factor * (c_term_during + c_term_after)

        d_term_during = dt * np.sum(
            (-1j)
            * 1j
            * b_to_end[i:, None]
            * 1j
            * b_from_i[:, None]
            * photon_phase
            * survival,
            axis=0,
        )[None, :]

        c_d += (
            np.exp(-kappa / 2.0 * tf)
            * ionisation_factor
            * d_term_during
        )

    return a, b, g, c_a, c_b, c_c, c_d
