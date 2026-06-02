r"""Time-ordered evolution of the loop Hamiltonian.

We need the propagator ``U_T(1) = T exp(-i T \int_0^1 H(s) ds)`` (and its reverse
counterpart under ``-H``) accurately enough that the *integration* error sits far
below the *adiabatic* error we are trying to measure -- which decays as fast as
``T^{-3}`` for the randomized estimator.  A fourth-order Magnus integrator with
two Gauss-Legendre nodes per step delivers that with a modest step count.

Everything is specialised to 2x2 matrices and fully vectorised over a batch of
runtimes ``T``.  This is what makes the runtime-randomization sweeps (averaging
over hundreds of runtimes per base ``T``) tractable: a single sweep propagates
the whole batch of 2x2 propagators at once via broadcasting.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Two-point Gauss-Legendre nodes on [0, 1].
_C1 = 0.5 - np.sqrt(3.0) / 6.0
_C2 = 0.5 + np.sqrt(3.0) / 6.0
_SQRT3_OVER_12 = np.sqrt(3.0) / 12.0

_I2 = np.eye(2, dtype=complex)


def expm2x2(X: np.ndarray) -> np.ndarray:
    """Matrix exponential of a batch of 2x2 matrices, in closed form.

    For ``X`` of shape ``(..., 2, 2)`` returns ``exp(X)`` of the same shape.
    Writes ``X = t I + Y`` with ``t = tr(X)/2`` and traceless ``Y``; then
    ``Y^2 = mu^2 I`` with ``mu = sqrt(Y00^2 + Y01 Y10)`` and
    ``exp(X) = e^t (cosh(mu) I + sinc_h(mu) Y)``, where ``sinc_h(mu)=sinh(mu)/mu``.
    """
    X = np.asarray(X, dtype=complex)
    a, b = X[..., 0, 0], X[..., 0, 1]
    c, d = X[..., 1, 0], X[..., 1, 1]

    t = 0.5 * (a + d)
    # Traceless part Y = X - t I; its (0,0) entry is (a-d)/2.
    y00 = 0.5 * (a - d)
    mu = np.sqrt(y00 * y00 + b * c)

    cosh = np.cosh(mu)
    # sinh(mu)/mu, regularised near mu = 0 by the leading Taylor terms.
    small = np.abs(mu) < 1e-8
    mu_safe = np.where(small, 1.0, mu)
    sinch = np.where(small, 1.0 + mu * mu / 6.0, np.sinh(mu_safe) / mu_safe)

    et = np.exp(t)
    out = np.empty(X.shape, dtype=complex)
    out[..., 0, 0] = et * (cosh + sinch * y00)
    out[..., 0, 1] = et * (sinch * b)
    out[..., 1, 0] = et * (sinch * c)
    out[..., 1, 1] = et * (cosh - sinch * y00)
    return out


def propagator(
    H: Callable[[float], np.ndarray],
    T,
    steps: int,
    sign: float = 1.0,
) -> np.ndarray:
    """Propagator ``U(1)`` for ``i d|psi>/ds = sign * T * H(s) |psi>``.

    Parameters
    ----------
    H:
        Callable ``s -> (2, 2)`` Hamiltonian.
    T:
        Runtime.  A scalar gives a single ``(2, 2)`` propagator; a 1-D array of
        runtimes is propagated as a batch, returning shape ``(len(T), 2, 2)``.
    steps:
        Number of Magnus steps on ``[0, 1]``.
    sign:
        ``+1`` for the forward evolution under ``H`` (``U_T``); ``-1`` for the
        reverse evolution under ``-H`` (``\hat U_T``).
    """
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    batch = T_arr.shape[0]
    # Per-runtime scalar prefactor -i * sign * T, broadcast over (batch, 1, 1).
    pref = (-1j * sign * T_arr)[:, None, None]

    U = np.broadcast_to(_I2, (batch, 2, 2)).copy()
    h = 1.0 / steps
    for k in range(steps):
        s0 = k * h
        H1 = H(s0 + _C1 * h)
        H2 = H(s0 + _C2 * h)
        A1 = pref * H1[None, :, :]
        A2 = pref * H2[None, :, :]
        comm = A1 @ A2 - A2 @ A1
        omega = 0.5 * h * (A1 + A2) - _SQRT3_OVER_12 * h * h * comm
        U = expm2x2(omega) @ U

    if np.isscalar(T) or np.ndim(T) == 0:
        return U[0]
    return U


def loop_amplitudes(model, T, steps: int):
    """Survival amplitudes ``<psi0| U |psi0>`` for forward and reverse evolution.

    Returns ``(z_forward, z_reverse)`` where

        z_forward = <psi0| U_T(1)      |psi0>  ~ exp(i(-theta_D + theta_B + phi)),
        z_reverse = <psi0| \hat U_T(1) |psi0>  ~ exp(i(+theta_D + theta_B + phihat)).

    ``T`` may be a scalar or a 1-D array (batched).

    If the model supplies its own ``amplitudes(T)`` (e.g. a many-body model with an
    exact closed-form propagator), it is used directly and ``steps`` is ignored.
    """
    if hasattr(model, "amplitudes"):
        return model.amplitudes(T)
    psi0 = model.psi0
    U_fwd = propagator(model.H, T, steps, sign=+1.0)
    U_rev = propagator(model.H, T, steps, sign=-1.0)
    z_fwd = np.einsum("i,...ij,j->...", psi0.conj(), U_fwd, psi0)
    z_rev = np.einsum("i,...ij,j->...", psi0.conj(), U_rev, psi0)
    return z_fwd, z_rev
