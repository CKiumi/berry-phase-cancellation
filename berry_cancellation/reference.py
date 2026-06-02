r"""Reference (exact) quantities the estimators are compared against.

* ``berry_phase_wilson`` computes the ground-state Berry phase of *any* loop
  model by a gauge-invariant Wilson loop (discrete Bargmann product).  This is
  the ground truth used for general models and as a cross-check of the analytic
  spin-1/2 value.

* ``dynamical_phase`` computes ``theta_D = T \int_0^1 E_0(s) ds`` by quadrature.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def berry_phase_wilson(model, n_points: int = 4000) -> float:
    r"""Ground-state Berry phase via the Wilson loop.

    Discretising the loop and using ``<psi(s)|psi(s+ds)> ~ exp(-i a(s) ds)`` with
    Berry connection ``a = i <psi|psi'>``, the product of consecutive overlaps
    around the closed loop equals ``exp(-i theta_B)``.  The arbitrary per-point
    phases of the eigenvectors cancel telescopically, so the result is gauge
    invariant.  Returned wrapped to ``(-pi, pi]``.
    """
    s = np.linspace(0.0, 1.0, n_points, endpoint=False)
    states = [model.ground_state(si) for si in s]
    product = 1.0 + 0.0j
    for k in range(n_points):
        # np.vdot(a, b) = a.conj() . b, i.e. <psi_k | psi_{k+1}>.
        product *= np.vdot(states[k], states[(k + 1) % n_points])
    return wrap_to_pi(-np.angle(product))


def dynamical_phase(model, T: float) -> float:
    r"""Dynamical phase ``theta_D = T \int_0^1 E_0(s) ds`` (not wrapped)."""
    integral, _ = quad(model.ground_energy, 0.0, 1.0, limit=200)
    return T * integral


def wrap_to_pi(x):
    """Wrap angle(s) to ``(-pi, pi]``."""
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_to_half_pi(x):
    """Wrap angle(s) to ``(-pi/2, pi/2]`` (the mod-pi sector of theta_B)."""
    return (np.asarray(x) + 0.5 * np.pi) % np.pi - 0.5 * np.pi
