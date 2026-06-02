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


def _richardson_event_errors(model, t, x, alpha, steps):
    """Per-event Richardson estimator errors for runtimes ``t * x``.

    ``x`` is an array of runtime multipliers; returns ``theta_R(t x_j) - theta_B``
    for each ``j`` (signed, wrapped to the mod-pi sector).  Forward/reverse
    evolutions for both ``t x`` and ``alpha t x`` are batched into one sweep.
    """
    runtimes = t * np.asarray(x, float)
    all_rt = np.concatenate([runtimes, alpha * runtimes])
    theta = _theta_B_forward_reverse(model, all_rt, steps)
    n = runtimes.shape[0]
    err_T = wrap_to_half_pi(theta[:n] - model.berry_phase)
    err_aT = wrap_to_half_pi(theta[n:] - model.berry_phase)
    return (alpha**2 * err_aT - err_T) / (alpha**2 - 1.0)


def randomized_richardson_montecarlo(
    model, T, n_shots=10000, alpha=2.0, lam=0.5, seed=0, steps=None
):
    r"""Actual runtime randomization: sample-mean error over ``n_shots`` draws.

    For each base runtime ``T`` we draw ``n_shots`` multipliers ``X_j`` uniformly
    from ``[1 - lam, 1 + lam]``, evaluate the single-event Richardson estimator at
    ``T_j = T X_j`` exactly (each draw is the infinite-measurement value), and
    average.  Unlike :func:`randomized_richardson_bias` (which integrates the
    expectation by quadrature), this is a real Monte Carlo sample mean and so
    carries statistical fluctuation ``~ T^{-2} n_shots^{-1/2}``.

    Returns ``(mean_error, sem, event_std)``: the signed sample-mean error
    ``mean_j theta_R(T_j) - theta_B`` (wrapped), its standard error of the mean,
    and the per-event standard deviation (whose ``/sqrt(n_shots)`` is the floor).
    """
    T = np.atleast_1d(np.asarray(T, float))
    rng = np.random.default_rng(seed)
    mean_err = np.empty(T.shape)
    sem = np.empty(T.shape)
    event_std = np.empty(T.shape)
    for i, t in enumerate(T):
        x = rng.uniform(1.0 - lam, 1.0 + lam, n_shots)
        # Magnus is far more accurate than the ~T^-2 statistical floor we resolve,
        # so a light step count (verified << floor) keeps the 10k-shot sweep fast.
        steps_i = steps or int(max(1500, np.ceil(20.0 * alpha * t * (1.0 + lam))))
        err = _richardson_event_errors(model, t, x, alpha, steps_i)
        mean_err[i] = np.mean(err)
        event_std[i] = np.std(err, ddof=1)
        sem[i] = event_std[i] / np.sqrt(n_shots)
    return mean_err, sem, event_std


def randomized_richardson_bias(
    model, T, alpha=2.0, lam=0.5, n_nodes=257, steps=None
):
    r"""Deterministic bias of the runtime-randomized Richardson estimator.

    For each base runtime ``T`` the runtime is randomized as ``T_j = T X_j`` with
    ``X`` uniform on ``[1 - lam, 1 + lam]``.  The reported quantity is the bias of
    the *expected* estimator, ``|E_X[theta_R(T X)] - theta_B|``, evaluated by a
    deterministic Simpson quadrature over the ``X`` support (i.e. the
    infinite-shot limit, isolating the deterministic bias).  Uniform
    randomization removes the leading oscillatory ``T^{-2}`` term in expectation,
    giving ``O(T^{-3})``.

    The reported value is the *bias* only; an actual finite-shot run also carries
    a statistical floor ``~ T^{-2} N^{-1/2}`` from the residual oscillatory
    sector (see paper, Sec. on runtime randomization).
    """
    from scipy.integrate import simpson

    T = np.atleast_1d(np.asarray(T, float))
    x = np.linspace(1.0 - lam, 1.0 + lam, n_nodes)
    weight = 1.0 / (2.0 * lam)  # uniform density on [1-lam, 1+lam]

    bias = np.empty(T.shape)
    for i, t in enumerate(T):
        runtimes = t * x
        steps_i = steps or default_steps(alpha * runtimes.max())
        theta_T = _theta_B_forward_reverse(model, runtimes, steps_i)
        theta_aT = _theta_B_forward_reverse(model, alpha * runtimes, steps_i)
        err_T = wrap_to_half_pi(theta_T - model.berry_phase)
        err_aT = wrap_to_half_pi(theta_aT - model.berry_phase)
        err_R = (alpha**2 * err_aT - err_T) / (alpha**2 - 1.0)
        bias[i] = np.abs(simpson(err_R * weight, x=x))
    return bias
