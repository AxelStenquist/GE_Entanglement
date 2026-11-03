"""Adaptive electron- and photon-energy grids for the multipulse model."""

import numpy as np


_FINE_STRUCTURE = 137.0


def def_eps(
    tau_I,
    t_Is,
    eps_ba,
    typ,
    rabi_frequency,
    adaptive=True,
    manual_limit=0.007,
    Ne=101,
    points_per_feature=15,
):
    """Construct the relative photoelectron-energy grid.

    The adaptive range is set by the bandwidth of the shortest pulse and the
    Rabi scale. For multiple pulses the resolution also resolves the finest
    interference fringes associated with the largest pulse separation.
    """
    tau_I = np.asarray(tau_I, dtype=float)
    t_Is = np.asarray(t_Is, dtype=float)

    if not adaptive:
        return np.linspace(-manual_limit, manual_limit, int(Ne))

    if len(tau_I) == 0 or np.any(tau_I <= 0):
        raise ValueError("tau_I must contain positive pulse durations.")

    tau_min = np.min(tau_I)

    if typ == "gaus":
        bandwidth = 4.0 * np.log(2.0) / tau_min
    elif typ == "ftop":
        bandwidth = 5.57 / tau_min
    else:
        raise ValueError("typ must be either 'gaus' or 'ftop'.")

    half_width = min(max(4.0 * bandwidth, abs(rabi_frequency)), eps_ba)

    feature_width = bandwidth
    if len(t_Is) > 1:
        max_delay = np.max(t_Is) - np.min(t_Is)
        if max_delay > 0:
            fringe_spacing = 2.0 * np.pi / max_delay
            feature_width = min(feature_width, fringe_spacing)

    deps_target = feature_width / points_per_feature
    n_eps = max(int(np.ceil(2.0 * half_width / deps_target)) + 1, 3)

    return np.linspace(-half_width, half_width, n_eps)


def spontaneous_decay_rate(eps_ba, z12):
    """Return the resonant spontaneous-emission rate kappa in atomic units."""
    return 4.0 * eps_ba**3 / _FINE_STRUCTURE**3 * z12**2


def photon_coupling(eps_l, z12):
    """Continuum photon-coupling matrix element V_fi(eps_l)."""
    eps_l = np.asarray(eps_l, dtype=float)
    if np.any(eps_l <= 0):
        raise ValueError("Photon energies eps_l must be positive.")
    return (
        1.0
        / (2.0 * np.pi)
        * np.sqrt(3.0 * eps_l**3 / _FINE_STRUCTURE**3)
        * z12
    )


def _graded_side_grid(edge, outer, n_points, first_step):
    """Return points from ``edge`` to ``outer`` with smoothly growing spacing.

    The first gap is approximately ``first_step``. Subsequent gaps grow
    geometrically so that the final point lands exactly at ``outer``.
    The returned array excludes ``edge`` and includes ``outer``.
    """
    n_points = int(n_points)
    length = abs(float(outer - edge))
    if n_points <= 0 or length <= 0:
        return np.array([], dtype=float)

    first_step = min(abs(float(first_step)), length / n_points)
    if first_step <= 0:
        return np.linspace(edge, outer, n_points + 1)[1:]

    # If uniform spacing is already fine enough, no grading is needed.
    if length <= n_points * first_step * (1.0 + 1e-12):
        return np.linspace(edge, outer, n_points + 1)[1:]

    def series_sum(r):
        if abs(r - 1.0) < 1e-12:
            return first_step * n_points
        return first_step * (r**n_points - 1.0) / (r - 1.0)

    lo, hi = 1.0, 1.1
    while series_sum(hi) < length:
        hi *= 1.2

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if series_sum(mid) < length:
            lo = mid
        else:
            hi = mid

    r = 0.5 * (lo + hi)
    increments = first_step * r ** np.arange(n_points)
    increments *= length / np.sum(increments)
    offsets = np.cumsum(increments)
    direction = np.sign(outer - edge)
    return edge + direction * offsets


def def_eps_l(
    tf,
    t,
    EI,
    z12,
    eps_ba,
    Nl=111,
    adaptive=True,
    manual_limit=0.01,
    include_secondary=False,
    Nl2=100,
    time_width_factor=20.0 * np.pi,
    linewidth_factor=25.0,
    driven_width_factor=1.5,
):
    """Construct the photon-energy grid around the ionic transition.

    The dense central grid resolves the finite-observation-time linewidth and
    the natural linewidth. Optionally, a graded secondary grid extends the
    range to the driven sideband scale. The graded spacing avoids the abrupt
    dense-to-sparse jump that can give an artificially large trapezoidal area.
    """
    t = np.asarray(t, dtype=float)
    EI = np.asarray(EI, dtype=float)

    if len(t) < 2:
        raise ValueError("t must contain at least two points.")

    kappa = spontaneous_decay_rate(eps_ba, z12)
    driven_half_width = np.max(np.abs(EI)) * abs(z12) * driven_width_factor

    if adaptive:
        T = max(float(tf - t[-1]), 0.0)
        finite_time_width = time_width_factor / max(T, 1e-12)
        central_half_width = min(
            driven_half_width,
            max(finite_time_width, linewidth_factor * kappa),
        )
        # Avoid a zero-width grid if the driving scale itself vanishes.
        central_half_width = max(central_half_width, linewidth_factor * kappa)
    else:
        central_half_width = float(manual_limit)

    eps_l = np.linspace(
        eps_ba - central_half_width,
        eps_ba + central_half_width,
        int(Nl),
    )

    if include_secondary and central_half_width < driven_half_width:
        n_left = int(Nl2) // 2
        n_right = int(Nl2) - n_left
        dense_step = eps_l[1] - eps_l[0]

        left_from_edge = _graded_side_grid(
            eps_l[0],
            eps_ba - driven_half_width,
            n_left,
            dense_step,
        )
        right = _graded_side_grid(
            eps_l[-1],
            eps_ba + driven_half_width,
            n_right,
            dense_step,
        )
        left = np.sort(left_from_edge)

        eps_l = np.concatenate((left, eps_l, right))

    if np.any(eps_l <= 0):
        raise ValueError(
            "The photon-energy grid reached non-positive energies. "
            "Reduce its range or increase eps_ba."
        )

    return eps_l
