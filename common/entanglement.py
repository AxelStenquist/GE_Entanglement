import numpy as np

def entanglement_ab(eps, c_a, c_b):
    """Von Neumann entropy of the reduced ionic density matrix in bits.

    This is the ion-electron entanglement entropy for a pure state with
    continuum amplitudes c_a(eps) and c_b(eps).
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

    deps_values = np.diff(eps)
    if not np.allclose(deps_values, deps_values[0]):
        raise ValueError("entanglement_ab currently assumes a uniform energy grid.")
    deps = deps_values[0]

    aa = np.vdot(c_a, c_a) * deps
    ab = np.vdot(c_a, c_b) * deps
    ba = np.vdot(c_b, c_a) * deps
    bb = np.vdot(c_b, c_b) * deps

    rho = np.array([[aa, ab], 
                    [ba, bb]], dtype=complex)
    trace = np.trace(rho).real

    if trace< 1e-14:
        return 0.0
    
    rho /= trace

    # rho is Hermitian, so eigvalsh is more stable than matrix logarithms.
    eigenvalues = np.linalg.eigvalsh(rho).real
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
    nonzero = eigenvalues > 1e-15

    entropy = -np.sum(eigenvalues[nonzero] * np.log2(eigenvalues[nonzero]))

    return float(entropy)
