"""Plotting helpers for the multipulse entanglement notebooks."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def _finish_axis(ax, xlabel=None, ylabel=None, title=None, legend=True):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    if legend:
        ax.legend(frameon=False)


def plot_field(t, Et, E_envelope, time_scale=1.0, time_label="Time (a.u.)"):
    """Plot the real electric field and total envelope."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.asarray(t) * time_scale

    ax.plot(x, np.real(Et), label=r"$E(t)$")
    ax.plot(x, E_envelope, "--", label="Envelope")
    _finish_axis(
        ax,
        xlabel=time_label,
        ylabel="Electric field (a.u.)",
        title="Driving field",
    )
    fig.tight_layout()
    return fig, ax


def plot_rabi_populations(t, a, b, g, time_scale=1.0, time_label="Time (a.u.)"):
    """Plot bound-state populations and neutral survival probability."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.asarray(t) * time_scale

    ax.plot(x, np.abs(a) ** 2, label=r"$|a|^2$")
    ax.plot(x, np.abs(b) ** 2, "--", label=r"$|b|^2$")
    ax.plot(x, np.abs(g) ** 2, ":", label=r"$|g|^2$")
    ax.set_ylim(bottom=0)
    _finish_axis(
        ax,
        xlabel=time_label,
        ylabel="Population",
        title="Rabi dynamics and depletion",
    )
    fig.tight_layout()
    return fig, ax


def plot_photoelectron_spectra(eps, c_a, c_b, energy_scale=1.0, energy_label="Relative photoelectron energy (a.u.)"):
    """Plot normalized channel spectra and their relative phase."""
    eps = np.asarray(eps)
    p_a = np.abs(c_a) ** 2
    p_b = np.abs(c_b) ** 2
    norm = max(float(np.max(p_a)), float(np.max(p_b)), 1e-30)
    p_a = p_a / norm
    p_b = p_b / norm

    relative_phase = np.angle(c_b * np.conj(c_a)) / np.pi
    phase_norm = Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap("twilight")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = eps * energy_scale

    for x_j, pa_j, pb_j, phase_j in zip(x, p_a, p_b, relative_phase):
        ax.plot(
            [x_j, x_j],
            [0.0, min(pa_j, pb_j)],
            color=cmap(phase_norm(phase_j)),
            linewidth=1.4,
            alpha=0.75,
        )

    ax.plot(x, p_a, label=r"$|c_a|^2$")
    ax.plot(x, p_b, "--", label=r"$|c_b|^2$")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    _finish_axis(
        ax,
        xlabel=energy_label,
        ylabel="Normalized probability density",
        title="Photoelectron spectra",
    )

    sm = ScalarMappable(norm=phase_norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"Relative phase $(\arg c_b-\arg c_a)/\pi$")

    fig.tight_layout()
    return fig, ax


def plot_spontaneous_spectra(
    eps,
    eps_l,
    c_a,
    c_b,
    c_c,
    c_d,
    electron_scale=1.0,
    photon_scale=1.0,
    electron_label="Relative photoelectron energy (a.u.)",
    photon_label="Photon energy (a.u.)",
):
    """Plot electron and photon marginals for the four continuum channels."""
    p_c_eps = np.trapz(np.abs(c_c) ** 2, x=eps_l, axis=1)
    p_d_eps = np.trapz(np.abs(c_d) ** 2, x=eps_l, axis=1)
    p_c_l = np.trapz(np.abs(c_c) ** 2, x=eps, axis=0)
    p_d_l = np.trapz(np.abs(c_d) ** 2, x=eps, axis=0)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))

    axs[0].plot(eps * electron_scale, np.abs(c_a) ** 2, label=r"$|c_a|^2$")
    axs[0].plot(eps * electron_scale, np.abs(c_b) ** 2, "--", label=r"$|c_b|^2$")
    axs[0].plot(eps * electron_scale, p_c_eps, "-.", label=r"$\int |c_c|^2 d\epsilon_\ell$")
    axs[0].plot(eps * electron_scale, p_d_eps, ":", label=r"$\int |c_d|^2 d\epsilon_\ell$")
    _finish_axis(
        axs[0],
        xlabel=electron_label,
        ylabel="Probability density",
        title="Electron marginal",
    )

    axs[1].plot(eps_l * photon_scale, p_c_l, label=r"$c$ channel")
    axs[1].plot(eps_l * photon_scale, p_d_l, "--", label=r"$d$ channel")
    _finish_axis(
        axs[1],
        xlabel=photon_label,
        ylabel="Probability density",
        title="Photon spectrum",
    )

    positive = np.concatenate((p_c_l[p_c_l > 0], p_d_l[p_d_l > 0]))
    floor = max(np.min(positive) * 0.5, 1e-30) if len(positive) else 1e-30
    axs[2].semilogy(eps_l * photon_scale, np.maximum(p_c_l, floor), label=r"$c$ channel")
    axs[2].semilogy(eps_l * photon_scale, np.maximum(p_d_l, floor), "--", label=r"$d$ channel")
    _finish_axis(
        axs[2],
        xlabel=photon_label,
        ylabel="Probability density",
        title="Photon spectrum (log scale)",
    )

    fig.tight_layout()
    return fig, axs


def plot_scan_results(
    x,
    populations,
    entropies=None,
    x_label="Scan coordinate",
    log_x=False,
    title=None,
):
    """Plot populations and, optionally, entanglement measures for a scan.

    ``populations`` and ``entropies`` are dictionaries mapping legend labels to
    one-dimensional arrays.
    """
    nrows = 2 if entropies is not None else 1
    fig, axs = plt.subplots(nrows, 1, figsize=(8, 4.4 * nrows), sharex=True)
    axs = np.atleast_1d(axs)

    for label, values in populations.items():
        axs[0].plot(x, np.real(values), label=label)
    axs[0].set_ylim(bottom=0)
    _finish_axis(
        axs[0],
        ylabel="Population",
        title=title or "Population scan",
    )

    if entropies is not None:
        for label, values in entropies.items():
            axs[1].plot(x, np.real(values), label=label)
        _finish_axis(
            axs[1],
            xlabel=x_label,
            ylabel="Entropy (bits)",
            title="Entanglement",
        )
    else:
        axs[0].set_xlabel(x_label)

    if log_x:
        for ax in axs:
            ax.set_xscale("log")

    fig.tight_layout()
    return fig, axs
