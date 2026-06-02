"""Tests pinning the physics: references, propagator unitarity, and scaling laws."""

from __future__ import annotations

import numpy as np
import pytest

from berry_cancellation import SpinHalfLoop, berry_phase_wilson
from berry_cancellation.estimators import (
    forward_reverse_error,
    randomized_richardson_bias,
    randomized_richardson_montecarlo,
    richardson_error,
    single_phase_error,
)
from berry_cancellation.evolution import loop_amplitudes, propagator


def _slope(T, y):
    A = np.vstack([np.log(T), np.ones_like(T)]).T
    return np.linalg.lstsq(A, np.log(y), rcond=None)[0][0]


def test_wilson_matches_analytic_berry_phase():
    # Berry phase is defined mod 2*pi; Wilson wraps to (-pi, pi].
    for theta0 in [0.2 * np.pi, 0.4 * np.pi, 0.7 * np.pi]:
        m = SpinHalfLoop(theta0=theta0)
        diff = (berry_phase_wilson(m) - m.berry_phase + np.pi) % (2 * np.pi) - np.pi
        assert abs(diff) < 1e-5


def test_propagator_unitary():
    m = SpinHalfLoop()
    U = propagator(m.H, 37.0, steps=2000)
    assert np.allclose(U.conj().T @ U, np.eye(2), atol=1e-9)


def test_magnus_converged():
    # The integrator error must be far below the adiabatic error it measures.
    m = SpinHalfLoop()
    e_coarse = forward_reverse_error(m, 40.0, steps=3200)[0]
    e_fine = forward_reverse_error(m, 40.0, steps=12800)[0]
    assert abs(e_coarse - e_fine) < 1e-9


def test_forward_reverse_cancels_dynamical_phase():
    # arg z_fwd + arg z_rev removes +/- theta_D, leaving ~2 theta_B.
    m = SpinHalfLoop()
    z_fwd, z_rev = loop_amplitudes(m, 80.0, steps=8000)
    two_theta_B = np.angle(z_fwd) + np.angle(z_rev)
    err = (two_theta_B - 2 * m.berry_phase + np.pi) % (2 * np.pi) - np.pi
    assert abs(err) < 1e-2


@pytest.mark.parametrize(
    "estimator, expected, tol",
    [
        (single_phase_error, -1.0, 0.2),
        (forward_reverse_error, -2.0, 0.3),
        (richardson_error, -2.0, 0.4),
    ],
)
def test_scaling_slopes(estimator, expected, tol):
    m = SpinHalfLoop()
    T = np.geomspace(8.0, 200.0, 22)
    slope = _slope(T, estimator(m, T))
    assert abs(slope - expected) < tol


def test_randomization_steeper_than_richardson():
    m = SpinHalfLoop()
    T = np.geomspace(8.0, 200.0, 22)
    slope = _slope(T, randomized_richardson_bias(m, T, n_nodes=129))
    # Uniform randomization buys roughly one extra power of 1/T.
    assert slope < -2.6


def test_montecarlo_mean_matches_quadrature_bias():
    # The sample mean over many random runtimes must agree with the
    # deterministic (quadrature) bias to within a few standard errors.
    m = SpinHalfLoop()
    T = np.array([30.0, 60.0])
    mean_err, sem, event_std = randomized_richardson_montecarlo(
        m, T, n_shots=4000, seed=0
    )
    bias = randomized_richardson_bias(m, T, n_nodes=129)
    assert np.all(np.abs(np.abs(mean_err) - bias) < 4.0 * sem + 1e-9)
    # Per-event spread scales as ~T^-2: doubling T cuts it by ~4.
    assert np.isclose(event_std[0] / event_std[1], 4.0, rtol=0.25)
