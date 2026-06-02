r"""Berry-phase estimators and their adiabatic errors.

Given the survival amplitudes of the forward (``H``) and reverse (``-H``)
evolutions around the loop, we build the four estimators studied in the paper and
return their errors against the exact Berry phase:

1. ``single_phase_error`` -- the single-evolution phase error ``phi``, the term
   the protocol is designed to remove.  Leading order ``O(T^{-1})``.
2. ``forward_reverse_error`` -- averaging the forward and reverse eigenphases
   cancels the dynamical phase and the leading ``O(T^{-1})`` term, leaving
   ``O(T^{-2})`` (with an oscillatory part).
3. ``richardson_error`` -- a two-runtime Richardson extrapolant removes the
   non-oscillatory ``T^{-2}`` part, leaving the oscillatory ``T^{-2}`` residual.
4. ``randomized_richardson_bias`` -- averaging the Richardson estimator over a
   smooth runtime distribution suppresses the oscillatory residual; for uniform
   randomization the deterministic bias drops by one further power to
   ``O(T^{-3})``.

All errors are returned as non-negative magnitudes.  Estimators that determine
``theta_B`` only modulo ``pi`` are compared using the mod-``pi`` sector.
"""

from __future__ import annotations

import numpy as np

from .evolution import loop_amplitudes
from .reference import dynamical_phase, wrap_to_half_pi, wrap_to_pi


def default_steps(T_max: float) -> int:
    """A Magnus step count that keeps integration error well below ``T^{-3}``."""
    return int(max(1500, np.ceil(80.0 * T_max)))


def single_phase_error(model, T, steps=None):
    r"""Single-evolution phase error ``|phi|``, expected ``~ O(T^{-1})``.

    The forward eigenphase is ``arg z_fwd = -theta_D + theta_B + phi``.  Granting
    exact knowledge of ``theta_D`` and ``theta_B``, the residual is the phase
    error ``phi`` itself.
    """
    T = np.atleast_1d(np.asarray(T, float))
    steps = steps or default_steps(T.max())
    z_fwd, _ = loop_amplitudes(model, T, steps)
    theta_D = np.array([dynamical_phase(model, t) for t in T])
    phi = wrap_to_pi(np.angle(z_fwd) - (-theta_D + model.berry_phase))
    return np.abs(phi)


def _theta_B_forward_reverse(model, T, steps):
    """Forward--reverse Berry-phase estimate ``(arg z_fwd + arg z_rev) / 2``.

    The dynamical phase cancels in the sum; the result is defined only modulo
    ``pi``.  ``T`` may be scalar or array.
    """
    z_fwd, z_rev = loop_amplitudes(model, T, steps)
    return 0.5 * (np.angle(z_fwd) + np.angle(z_rev))


def forward_reverse_error(model, T, steps=None):
    r"""Forward--reverse estimator error, expected ``~ O(T^{-2})``."""
    T = np.atleast_1d(np.asarray(T, float))
    steps = steps or default_steps(T.max())
    theta_est = _theta_B_forward_reverse(model, T, steps)
    return np.abs(wrap_to_half_pi(theta_est - model.berry_phase))


def richardson_error(model, T, alpha=2.0, steps=None):
    r"""Richardson-extrapolated forward--reverse error.

    Combines forward--reverse estimates at runtimes ``T`` and ``alpha T``:

        theta_R = (alpha^2 theta(alpha T) - theta(T)) / (alpha^2 - 1),

    cancelling the non-oscillatory ``T^{-2}`` term and leaving the oscillatory
    ``T^{-2}`` residual.
    """
    T = np.atleast_1d(np.asarray(T, float))
    steps = steps or default_steps(alpha * T.max())
    theta_T = _theta_B_forward_reverse(model, T, steps)
    theta_aT = _theta_B_forward_reverse(model, alpha * T, steps)
    # Resolve the per-runtime mod-pi ambiguity against the reference before
    # combining, so the extrapolation acts on the small errors, not on branches.
    err_T = wrap_to_half_pi(theta_T - model.berry_phase)
    err_aT = wrap_to_half_pi(theta_aT - model.berry_phase)
    err_R = (alpha**2 * err_aT - err_T) / (alpha**2 - 1.0)
    return np.abs(err_R)


def _recursive_richardson_error(model, r, alpha, levels, steps):
    r"""Error of the ``levels``-fold recursive Richardson forward--reverse estimate.

    ``r`` is an array of base runtimes.  Evaluates the forward--reverse error at
    ``r, alpha r, ..., alpha^levels r`` and recursively extrapolates: level ``j``
    cancels the non-oscillatory ``T^{-2j}`` term using weight ``alpha^{2j}``.
    Returns the extrapolated (signed, wrapped) error at base runtime ``r``.
    """
    r = np.asarray(r, float)
    errs = []
    for k in range(levels + 1):
        theta = _theta_B_forward_reverse(model, alpha**k * r, steps)
        errs.append(wrap_to_half_pi(theta - model.berry_phase))
    for j in range(1, levels + 1):
        a2j = alpha ** (2 * j)
        errs = [(a2j * errs[k + 1] - errs[k]) / (a2j - 1.0)
                for k in range(len(errs) - 1)]
    return errs[0]


def randomized_richardson_bias(
    model, T, alpha=2.0, lam=0.5, levels=1, dist="uniform", n_nodes=129, steps=None
):
    r"""Deterministic bias of the runtime-randomized recursive-Richardson estimator.

    For each base runtime ``T`` the runtime is randomized as ``T_j = T X_j`` and
    the ``levels``-fold recursive Richardson forward--reverse estimator is averaged
    over ``X``.  Returns ``|E_X[theta_R(T X)] - theta_B|`` (the infinite-shot bias),
    evaluated by Simpson quadrature over the distribution of ``X``.

    Two effects set the bias scaling:

    * recursive Richardson cancels the non-oscillatory terms up to ``T^{-2 levels}``
      (so the non-oscillatory residual is ``T^{-2(levels+1)}``);
    * averaging suppresses the residual oscillatory ``T^{-2}`` term by the decay of
      the distribution's characteristic function -- one power of ``1/T`` for the
      ``uniform`` distribution (CF ``~ k^{-1}``), two for the ``triangle`` (CF
      ``~ k^{-2}``).

    Hence ``levels=1, dist='uniform'`` gives ``O(T^{-3})`` and
    ``levels=2, dist='triangle'`` gives ``O(T^{-4})``.

    Parameters
    ----------
    levels:
        Number of recursive Richardson extrapolation levels (>= 1).
    dist:
        Runtime distribution on ``[1-lam, 1+lam]``: ``"uniform"`` or ``"triangle"``
        (the triangle peaks at ``X=1``).
    """
    from scipy.integrate import simpson

    T = np.atleast_1d(np.asarray(T, float))

    # Symmetric grid with the apex X=1 as a node; integrate each half separately
    # so the triangle's kink at X=1 does not spoil Simpson's accuracy.
    if dist not in ("uniform", "triangle"):
        raise ValueError(f"unknown dist {dist!r}")

    # The integrand oscillates in X at frequency ~ omega * alpha^levels * t (omega
    # is the integrated gap), so the number of quadrature nodes must grow with t to
    # keep enough samples per oscillation -- otherwise the small high-T bias is
    # swamped by Simpson error (a spurious upward spike).
    omega = float(getattr(model, "gap", 1.0))

    bias = np.empty(T.shape)
    for i, t in enumerate(T):
        phase_span = omega * (alpha**levels) * t * (2.0 * lam)
        n_target = max(n_nodes, int(np.ceil(phase_span / (2.0 * np.pi) * 16)))
        n_half = min(8192, max(2, n_target // 2)) // 2 * 2  # even -> odd half-grids
        xl = np.linspace(1.0 - lam, 1.0, n_half + 1)
        xr = np.linspace(1.0, 1.0 + lam, n_half + 1)
        x = np.concatenate([xl, xr[1:]])
        if dist == "uniform":
            dens = np.full_like(x, 1.0 / (2.0 * lam))
        else:
            dens = (1.0 - np.abs(x - 1.0) / lam) / lam

        r = t * x
        steps_i = steps or default_steps(alpha**levels * r.max())
        err_R = _recursive_richardson_error(model, r, alpha, levels, steps_i)
        integrand = err_R * dens
        val = (simpson(integrand[:n_half + 1], x=x[:n_half + 1])
               + simpson(integrand[n_half:], x=x[n_half:]))
        bias[i] = np.abs(val)
    return bias
