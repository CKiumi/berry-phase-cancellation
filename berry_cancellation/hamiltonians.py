r"""Loop Hamiltonian families used to demonstrate adiabatic error cancellation.

The headline model is a spin-1/2 in a magnetic field that traces a closed loop
on the Bloch sphere,

    H(s) = -(1/2) B(s) . sigma,        s in [0, 1],   B(1) = B(0),

with the field swept around a cone of fixed polar angle ``theta0`` and azimuth
``phi = 2 pi s``.  The ground state is the spin aligned with ``+B``.  For a cone
loop the Berry phase of the ground state is known analytically,

    theta_B = -Omega / 2 = -pi (1 - cos theta0),

where ``Omega = 2 pi (1 - cos theta0)`` is the solid angle subtended by the loop.
This exact value is what the numerics in this repo are checked against.

Because ``|B|`` is constant the spectral gap is constant (Delta = |B|) and the
dynamical phase is simply ``theta_D = -T |B| / 2``, which keeps the model in a
clean, controllable adiabatic regime.  Both endpoint conditions assumed in the
paper hold for this loop: ``H(0) = H(1)`` and ``dH/ds(0) = dH/ds(1)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pauli matrices.
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


@dataclass(frozen=True)
class SpinHalfLoop:
    """Spin-1/2 in a field swept around a cone on the Bloch sphere.

    Parameters
    ----------
    theta0:
        Polar (cone) angle of the field in radians.  Must be in ``(0, pi)`` for a
        non-trivial, non-degenerate loop.
    field:
        Magnitude ``|B|`` of the magnetic field at the loop endpoints.
    gap_dip:
        Magnitude modulation amplitude ``a`` in ``[0, 1)``.  The field strength is
        scaled by ``g(s) = 1 - a sin^2(pi s)``, so the gap is ``|B|`` at the
        endpoints and dips to ``|B|(1-a)`` in the middle of the loop -- a
        *non-isospectral* loop.  Default ``0`` gives the constant-gap loop.  The
        Berry phase is unchanged (it depends only on the field direction).
    """

    theta0: float = 0.4 * np.pi
    field: float = 1.0
    gap_dip: float = 0.0

    def _g(self, s: float) -> float:
        """Field-magnitude modulation ``g(s) = 1 - a sin^2(pi s)``."""
        return 1.0 - self.gap_dip * np.sin(np.pi * s) ** 2

    def field_vector(self, s: float) -> np.ndarray:
        """Magnetic field vector ``B(s) = |B| g(s) n(s)`` on the cone loop."""
        phi = 2.0 * np.pi * s
        st, ct = np.sin(self.theta0), np.cos(self.theta0)
        return self.field * self._g(s) * np.array(
            [st * np.cos(phi), st * np.sin(phi), ct])

    def H(self, s: float) -> np.ndarray:
        """Hamiltonian ``H(s) = -(1/2) B(s) . sigma`` as a 2x2 complex array."""
        bx, by, bz = self.field_vector(s)
        return -0.5 * (bx * SIGMA_X + by * SIGMA_Y + bz * SIGMA_Z)

    def gap_at(self, s: float) -> float:
        """Instantaneous gap ``Delta(s) = |B| g(s)``."""
        return self.field * self._g(s)

    def ground_energy(self, s: float) -> float:
        """Instantaneous ground-state energy ``E_0(s) = -|B| g(s) / 2``."""
        return -0.5 * self.field * self._g(s)

    @property
    def gap(self) -> float:
        """Minimum gap along the loop, ``|B|(1 - a)`` (in the middle)."""
        return self.field * (1.0 - self.gap_dip)

    @property
    def osc_freq(self) -> float:
        """Fastest oscillation frequency = max gap ``|B|`` (sets quadrature nodes)."""
        return self.field

    def ground_state(self, s: float) -> np.ndarray:
        """Instantaneous ground state ``|+n(s)>`` (spin aligned with ``+B(s)``)."""
        phi = 2.0 * np.pi * s
        half = 0.5 * self.theta0
        return np.array([np.cos(half), np.exp(1j * phi) * np.sin(half)], dtype=complex)

    @property
    def psi0(self) -> np.ndarray:
        """Initial state ``|psi(0)>`` = ground state at the start of the loop."""
        return self.ground_state(0.0)

    @property
    def solid_angle(self) -> float:
        """Solid angle ``Omega = 2 pi (1 - cos theta0)`` subtended by the loop."""
        return 2.0 * np.pi * (1.0 - np.cos(self.theta0))

    @property
    def berry_phase(self) -> float:
        """Analytic ground-state Berry phase ``theta_B = -Omega / 2``."""
        return -0.5 * self.solid_angle
