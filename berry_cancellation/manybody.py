r"""A non-trivial many-qubit loop: Heisenberg chain in a spiral rotating field.

    H(s) = -sum_i B_i(s) . S_i  +  J sum_<ij> S_i . S_j,    S = sigma/2,

with a site-dependent azimuthal offset so the field is a spiral on the cone:

    B_i(s) = B0 (sin th cos(2 pi s + phi_i), sin th sin(2 pi s + phi_i), cos th),
    phi_i = 2 pi i / N.

The per-site offsets break the total-spin symmetry, so the ground state is
genuinely entangled and the virtual-excitation term A^(2) (the n,k != 0 sum in the
phase-error expansion) is nonzero -- unlike the two-level / spin-S / uniform-field
cases. It is therefore a real many-body test of the adiabatic error cancellation.

Because every site's azimuth advances by the same 2 pi s, the loop is the rigid
global rotation R(s) = exp(-i 2 pi s S^z_tot), i.e. H(s) = R(s) H(0) R(s)^dagger.
This has two convenient consequences used below:

  * the Berry phase is exactly theta_B = 2 pi <S^z_tot>_0 (ground-state
    magnetization), and
  * in the co-rotating frame the generator is time-independent, so the loop
    propagator is a single matrix exponential (no time-ordering / integrator):
        U_T(1)      = R(1) exp(-i (T H(0) - 2 pi S^z_tot)),
        U_hat_T(1)  = R(1) exp(-i (-T H(0) - 2 pi S^z_tot)).

The model exposes ``amplitudes(T)``; the generic estimators pick it up via
``loop_amplitudes`` (see evolution.py), so single / forward-reverse / Richardson /
randomized estimators all work unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, expm

from .reference import wrap_to_pi

_I = np.eye(2, dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _site(op: np.ndarray, i: int, N: int) -> np.ndarray:
    """Operator ``op`` acting on site ``i`` of an ``N``-qubit register."""
    m = _I if i != 0 else op
    for k in range(1, N):
        m = np.kron(m, op if k == i else _I)
    return m


class SpiralHeisenbergChain:
    """Heisenberg chain in a spiral rotating field (see module docstring)."""

    def __init__(self, N=4, J=1.0, B0=1.0, theta=0.4 * np.pi, periodic=False):
        self.N, self.J, self.B0, self.theta = N, J, B0, theta
        dim = 2 ** N
        Sx = [_site(_X, i, N) / 2 for i in range(N)]
        Sy = [_site(_Y, i, N) / 2 for i in range(N)]
        Sz = [_site(_Z, i, N) / 2 for i in range(N)]
        self.Sz_tot = sum(Sz)

        bonds = [(i, i + 1) for i in range(N - 1)]
        if periodic and N > 2:
            bonds.append((N - 1, 0))
        Hint = np.zeros((dim, dim), dtype=complex)
        for i, j in bonds:
            Hint += J * (Sx[i] @ Sx[j] + Sy[i] @ Sy[j] + Sz[i] @ Sz[j])

        st, ct = np.sin(theta), np.cos(theta)
        Hf = np.zeros((dim, dim), dtype=complex)
        for i in range(N):
            phi = 2.0 * np.pi * i / N
            bx, by, bz = B0 * st * np.cos(phi), B0 * st * np.sin(phi), B0 * ct
            Hf += -(bx * Sx[i] + by * Sy[i] + bz * Sz[i])

        self.H0 = Hf + Hint
        E, V = eigh(self.H0)
        self.E = E
        self.E0 = float(E[0])
        self.gap = float(E[1] - E[0])
        # Fastest oscillation in the loop amplitude is set by the full spectral
        # width; the randomization quadrature uses this to choose its node count.
        self.osc_freq = float(E[-1] - E[0])
        self.psi0 = V[:, 0]
        # Global factor R(1) = exp(-i 2 pi S^z_tot) (= I for even N).
        self._R1 = expm(-1j * 2.0 * np.pi * self.Sz_tot)

    @property
    def berry_phase(self) -> float:
        """Analytic Berry phase ``theta_B = 2 pi <S^z_tot>_0`` (wrapped)."""
        m0 = np.real(self.psi0.conj() @ self.Sz_tot @ self.psi0)
        return float(wrap_to_pi(2.0 * np.pi * m0))

    def ground_energy(self, s: float) -> float:
        """Ground-state energy (constant along the isospectral loop)."""
        return self.E0

    def ground_state(self, s: float) -> np.ndarray:
        """Instantaneous ground state ``|psi(s)> = R(s)|psi(0)>`` (for Wilson loop)."""
        R = expm(-1j * 2.0 * np.pi * s * self.Sz_tot)
        return R @ self.psi0

    def amplitudes(self, T):
        r"""Survival amplitudes ``(z_fwd, z_rev)`` for forward (+H) and reverse (-H).

        Exact via the co-rotating frame: ``U_T(1) = R(1) exp(-i(T H0 - 2 pi Sz))``
        and ``U_hat_T(1) = R(1) exp(-i(-T H0 - 2 pi Sz))``. ``T`` may be scalar or
        array.
        """
        T = np.atleast_1d(np.asarray(T, float))
        p = self.psi0
        H0, Sz, R1 = self.H0, self.Sz_tot, self._R1
        twopiSz = 2.0 * np.pi * Sz
        z_fwd = np.empty(T.shape, dtype=complex)
        z_rev = np.empty(T.shape, dtype=complex)
        for k, t in enumerate(T):
            U_fwd = R1 @ expm(-1j * (t * H0 - twopiSz))
            U_rev = R1 @ expm(-1j * (-t * H0 - twopiSz))
            z_fwd[k] = p.conj() @ U_fwd @ p
            z_rev[k] = p.conj() @ U_rev @ p
        return z_fwd, z_rev
