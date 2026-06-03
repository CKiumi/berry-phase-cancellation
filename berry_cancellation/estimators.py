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
    """A Magnus step count that keeps integration error well below the signal.

    The 4th-order Magnus integrator reaches ~1e-10 accuracy at roughly 10 steps
    per unit runtime, so ~20/unit leaves a comfortable margin below even the
    ``T^{-4}`` bias while keeping the step loop short.
    """
    return int(max(1000, np.ceil(20.0 * T_max)))


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


def runtime_scaling_theta_B(model, T, frac=0.5, steps=None):
    r"""Berry phase via the runtime-scaling method (arXiv:2509.13423).

    Two forward adiabatic evolutions of the *same* loop are compared at runtimes
    ``T`` and ``alpha T`` (``alpha > 1``).  The Berry phase is runtime-independent
    while the dynamical phase scales linearly, ``theta_D -> alpha theta_D``, so the
    two survival-amplitude eigenphases

        phi(T)      = theta_B - theta_D            (mod 2 pi),
        phi(alpha T)= theta_B - alpha theta_D      (mod 2 pi)

    can be combined to cancel ``theta_D`` and reconstruct ``theta_B`` *without* any
    knowledge of the spectrum -- in contrast to subtracting an analytically known
    dynamical phase.  ``alpha`` is chosen from an energy *scale* only, so that
    ``(alpha-1)|theta_D| ~ frac*pi < pi`` makes the mod-2 pi unwrapping unambiguous.
    The residual error is the adiabatic error at the shorter runtime, ``O(T^{-1})``.
    ``T`` is scalar.
    """
    T = float(np.atleast_1d(np.asarray(T, float))[0])
    theta_D = dynamical_phase(model, T)
    alpha = 1.0 + frac * np.pi / max(abs(theta_D), 1e-12)
    steps = steps or default_steps(alpha * T)
    z_T, _ = loop_amplitudes(model, np.array([T]), steps)
    z_aT, _ = loop_amplitudes(model, np.array([alpha * T]), steps)
    phi_T, phi_aT = np.angle(z_T[0]), np.angle(z_aT[0])
    w = float(wrap_to_pi(phi_aT - phi_T))      # = -(alpha-1) theta_D, unwrapped
    return phi_T - w / (alpha - 1.0)           # theta_B (mod 2 pi)


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
    n = r.shape[0]
    # One batched forward+reverse sweep over all alpha^k * r at once.
    all_r = np.concatenate([alpha**k * r for k in range(levels + 1)])
    theta = _theta_B_forward_reverse(model, all_r, steps)
    errs = [wrap_to_half_pi(theta[k * n:(k + 1) * n] - model.berry_phase)
            for k in range(levels + 1)]
    for j in range(1, levels + 1):
        a2j = alpha ** (2 * j)
        errs = [(a2j * errs[k + 1] - errs[k]) / (a2j - 1.0)
                for k in range(len(errs) - 1)]
    return errs[0]


def _randomized_richardson_signed(model, T, alpha, lam, levels, dist, steps):
    r"""Signed deterministic bias ``E_X[theta_R(T X)] - theta_B`` (per base runtime).

    Shared core of ``randomized_richardson_bias`` (its magnitude) and
    ``_theta_B_randomized`` (``theta_B`` plus this signed value).
    """
    from scipy.integrate import simpson

    T = np.atleast_1d(np.asarray(T, float))
    if dist not in ("uniform", "triangle", "bump"):
        raise ValueError(f"unknown dist {dist!r}")

    # The integrand oscillates in X at frequencies up to ~ (spectral width) *
    # alpha^levels * t, so the number of quadrature nodes must grow with t to keep
    # enough samples per oscillation -- otherwise the small high-T bias is swamped
    # by Simpson aliasing (a spurious upward spike). Use the fastest frequency
    # available (spectral width / largest gap), falling back to the gap.
    omega = float(getattr(model, "osc_freq", getattr(model, "gap", 1.0)))

    # Symmetric grid with the apex X=1 as a node; integrate each half separately
    # so the triangle's kink at X=1 does not spoil Simpson's accuracy.
    out = np.empty(T.shape)
    for i, t in enumerate(T):
        phase_span = omega * (alpha**levels) * t * (2.0 * lam)
        n_target = max(129, int(np.ceil(phase_span / (2.0 * np.pi) * 8)))
        n_half = min(8192, max(2, n_target // 2)) // 2 * 2  # even -> odd half-grids
        xl = np.linspace(1.0 - lam, 1.0, n_half + 1)
        xr = np.linspace(1.0, 1.0 + lam, n_half + 1)
        x = np.concatenate([xl, xr[1:]])
        if dist == "uniform":
            dens = np.full_like(x, 1.0 / (2.0 * lam))
        elif dist == "triangle":
            dens = (1.0 - np.abs(x - 1.0) / lam) / lam
        else:  # C^inf bump, exp(-1/(1-u^2)) on |u|<1, normalised on the grid
            u = (x - 1.0) / lam
            dens = np.zeros_like(x)
            inside = np.abs(u) < 1.0
            dens[inside] = np.exp(-1.0 / (1.0 - u[inside] ** 2))
            Z = (simpson(dens[:n_half + 1], x=x[:n_half + 1])
                 + simpson(dens[n_half:], x=x[n_half:]))
            dens /= Z

        r = t * x
        steps_i = steps or default_steps(alpha**levels * r.max())
        err_R = _recursive_richardson_error(model, r, alpha, levels, steps_i)
        integrand = err_R * dens
        out[i] = (simpson(integrand[:n_half + 1], x=x[:n_half + 1])
                  + simpson(integrand[n_half:], x=x[n_half:]))
    return out


def randomized_richardson_bias(
    model, T, alpha=2.0, lam=0.5, levels=1, dist="uniform", steps=None
):
    r"""Deterministic bias of the runtime-randomized recursive-Richardson estimator.

    The runtime is randomized as ``T X`` (``X`` on ``[1-lam, 1+lam]``) and the
    ``levels``-fold recursive Richardson forward--reverse estimate is averaged over
    ``X`` by Simpson quadrature; returns ``|E_X[theta_R(T X)] - theta_B|``.

    Recursive Richardson removes the non-oscillatory terms up to ``T^{-2 levels}``,
    leaving a ``T^{-2(levels+1)}`` floor; averaging suppresses the oscillatory
    ``T^{-2}`` residual by the distribution's characteristic-function decay:
    ``uniform`` (CF ``~k^-1``) -> ``T^-3``, ``triangle`` (CF ``~k^-2``) -> ``T^-4``,
    and the ``C^inf`` ``bump`` (CF faster than any power) pushes it below the
    non-oscillatory floor.
    """
    return np.abs(_randomized_richardson_signed(model, T, alpha, lam, levels, dist, steps))


def _theta_B_randomized(model, T, alpha=2.0, lam=0.5, levels=1, dist="uniform", steps=None):
    """Berry-phase estimate of the runtime-randomized recursive-Richardson protocol,
    ``theta_B + E_X[theta_R(T X) - theta_B]`` (mod pi)."""
    return model.berry_phase + _randomized_richardson_signed(
        model, T, alpha, lam, levels, dist, steps)
