"""Entanglement measures used by the ion-electron(-photon) models."""

import numpy as np


def _von_neumann_entropy(rho, tol=1e-14):
    """Return the von Neumann entropy of ``rho`` in bits.

    The input may be unnormalised. Tiny numerical deviations from Hermiticity
    and positivity are removed before the entropy is evaluated.
    """
    rho = np.asarray(rho, dtype=complex)
    tr = float(np.real(np.trace(rho)))

    if tr < tol:
        return 0.0

    rho = rho / tr
    rho = 0.5 * (rho + np.conj(rho.T))

    eigvals = np.linalg.eigvalsh(rho).real
    eigvals = np.clip(eigvals, 0.0, None)
    eigvals = eigvals[eigvals > tol]

    return float(-np.sum(eigvals * np.log2(eigvals)))


def entanglement_ab(eps, c_a, c_b):
    """Ion-electron entanglement entropy for the two ionic channels ``a,b``.

    Parameters
    ----------
    eps : array_like
        Photoelectron-energy grid. The grid may be nonuniform.
    c_a, c_b : array_like
        Continuum amplitudes on ``eps``.
    """
    eps = np.asarray(eps, dtype=float)
    c_a = np.asarray(c_a, dtype=complex)
    c_b = np.asarray(c_b, dtype=complex)

    if eps.ndim != 1 or c_a.ndim != 1 or c_b.ndim != 1:
        raise ValueError("eps, c_a, and c_b must be one-dimensional arrays.")
    if not (len(eps) == len(c_a) == len(c_b)):
        raise ValueError("eps, c_a, and c_b must have the same length.")
    if len(eps) < 2:
        raise ValueError("eps must contain at least two grid points.")

    aa = np.trapezoid(np.conj(c_a) * c_a, x=eps)
    ab = np.trapezoid(np.conj(c_a) * c_b, x=eps)
    ba = np.trapezoid(np.conj(c_b) * c_a, x=eps)
    bb = np.trapezoid(np.conj(c_b) * c_b, x=eps)

    rho_ab = np.array([[aa, ab], [ba, bb]], dtype=complex)
    return _von_neumann_entropy(rho_ab)


def entanglement_abcd(eps, eps_l, c_a, c_b, c_c, c_d):
    """Entanglement entropies for the ion-electron-photon amplitudes.

    ``c_a`` and ``c_b`` have shape ``(N_eps,)`` while ``c_c`` and ``c_d``
    have shape ``(N_eps, N_eps_l)``. Both energy grids may be nonuniform.

    Returns
    -------
    ent_ab, ent_ac, ent_abcd, ent_abc, ent_ll : float
        Entropies corresponding to the same reduced spaces as in the original
        notebook implementation.
    """
    eps = np.asarray(eps, dtype=float)
    eps_l = np.asarray(eps_l, dtype=float)
    c_a = np.asarray(c_a, dtype=complex)
    c_b = np.asarray(c_b, dtype=complex)
    c_c = np.asarray(c_c, dtype=complex)
    c_d = np.asarray(c_d, dtype=complex)

    if c_a.shape != (len(eps),) or c_b.shape != (len(eps),):
        raise ValueError("c_a and c_b must have shape (len(eps),).")
    expected_2d = (len(eps), len(eps_l))
    if c_c.shape != expected_2d or c_d.shape != expected_2d:
        raise ValueError(
            "c_c and c_d must have shape (len(eps), len(eps_l))."
        )

    def int_eps(values):
        return np.trapezoid(values, x=eps, axis=0)

    def int_eps_epsl(values):
        photon_integral = np.trapezoid(values, x=eps_l, axis=1)
        return np.trapezoid(photon_integral, x=eps)

    # a-b subspace
    aa = int_eps(np.conj(c_a) * c_a)
    ab = int_eps(np.conj(c_a) * c_b)
    ba = int_eps(np.conj(c_b) * c_a)
    bb = int_eps(np.conj(c_b) * c_b)

    rho_ab = np.array([[aa, ab], [ba, bb]], dtype=complex)
    ent_ab = _von_neumann_entropy(rho_ab)

    # Matrix elements involving the one-photon channels c,d
    ac = int_eps_epsl(np.conj(c_a[:, None]) * c_c)
    ca = int_eps_epsl(np.conj(c_c) * c_a[:, None])
    cc = int_eps_epsl(np.conj(c_c) * c_c)

    bc = int_eps_epsl(np.conj(c_b[:, None]) * c_c)
    cb = int_eps_epsl(np.conj(c_c) * c_b[:, None])

    ad = int_eps_epsl(np.conj(c_a[:, None]) * c_d)
    da = int_eps_epsl(np.conj(c_d) * c_a[:, None])

    bd = int_eps_epsl(np.conj(c_b[:, None]) * c_d)
    db = int_eps_epsl(np.conj(c_d) * c_b[:, None])

    cd = int_eps_epsl(np.conj(c_c) * c_d)
    dc = int_eps_epsl(np.conj(c_d) * c_c)
    dd = int_eps_epsl(np.conj(c_d) * c_d)

    rho_ac = np.array([[aa, ac], [ca, cc]], dtype=complex)
    ent_ac = _von_neumann_entropy(rho_ac)

    rho_abcd = np.array(
        [
            [aa, ab, ac, ad],
            [ba, bb, bc, bd],
            [ca, cb, cc, cd],
            [da, db, dc, dd],
        ],
        dtype=complex,
    )
    ent_abcd = _von_neumann_entropy(rho_abcd)

    rho_abc = np.array(
        [
            [aa, ab, ac],
            [ba, bb, bc],
            [ca, cb, cc],
        ],
        dtype=complex,
    )
    ent_abc = _von_neumann_entropy(rho_abc)

    # Photon reduced density matrix rho(l,l'). Integration is over electron
    # energy; l and l' remain explicit matrix indices.
    rho_llp_integrand = (
        np.conj(c_c)[:, :, None] * c_c[:, None, :]
        + np.conj(c_d)[:, :, None] * c_d[:, None, :]
    )
    rho_llp = np.trapezoid(rho_llp_integrand, x=eps, axis=0)
    ent_ll = _von_neumann_entropy(rho_llp)

    return ent_ab, ent_ac, ent_abcd, ent_abc, ent_ll
