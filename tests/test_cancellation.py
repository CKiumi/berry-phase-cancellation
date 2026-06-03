"""Tests pinning the physics: references, propagator unitarity, and scaling laws."""

from __future__ import annotations

import numpy as np
import pytest

from berry_cancellation import SpinHalfLoop, berry_phase_wilson
from berry_cancellation.estimators import (
    forward_reverse_error,
    randomized_richardson_bias,
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
    slope = _slope(T, randomized_richardson_bias(m, T))
    # Uniform randomization buys roughly one extra power of 1/T (-> ~T^-3).
    assert slope < -2.6


def test_triangle_randomization_reaches_T4():
    m = SpinHalfLoop()
    T = np.geomspace(8.0, 120.0, 16)
    # The triangle distribution (CF ~ k^-2) buys a further power -> ~T^-4.
    slope = _slope(T, randomized_richardson_bias(
        m, T, levels=1, dist="triangle"))
    assert slope < -3.6


def test_phi1_leading_coefficient():
    # The single-evolution error converges to phi_1/T, with the loop invariant
    # phi_1 = int |<1|Hdot|0>|^2 / Delta^3 ds computed from the model (no fit).
    from scipy.linalg import eigh
    m = SpinHalfLoop()
    h, s = 1e-6, np.linspace(0.0, 1.0, 2000, endpoint=False)
    tot = 0.0
    for si in s:
        Hd = (m.H(si + h) - m.H(si - h)) / (2 * h)
        E, V = eigh(m.H(si))
        tot += abs(V[:, 1].conj() @ Hd @ V[:, 0]) ** 2 / (E[1] - E[0]) ** 3
    phi1 = tot / len(s)
    sT = single_phase_error(m, 400.0)[0] * 400.0
    assert abs(sT - phi1) / phi1 < 0.03


def test_gap_dip_nonisospectral():
    # Field-magnitude modulation makes a non-isospectral loop: gap dips in the
    # middle, large at the endpoints, with the Berry phase unchanged.
    m = SpinHalfLoop(gap_dip=0.8)
    assert np.isclose(m.gap_at(0.0), 1.0)
    assert np.isclose(m.gap_at(0.5), 0.2)
    assert np.isclose(m.gap, 0.2)  # minimum over the loop
    assert np.isclose(m.berry_phase, SpinHalfLoop().berry_phase)
    # The cancellation still works: forward-reverse steepens past single.
    T = np.geomspace(20.0, 200.0, 16)
    assert _slope(T, single_phase_error(m, T)) > -1.3
    assert _slope(T, forward_reverse_error(m, T)) < -1.5


def test_manybody_cancellation():
    # The cancellation must also work in a non-trivial entangled many-qubit model.
    from berry_cancellation.manybody import SpiralHeisenbergChain
    m = SpiralHeisenbergChain(N=4, J=1.0, B0=1.0, theta=0.4 * np.pi)
    # Non-degenerate ground state, and a non-trivial (not 0/pi) Berry phase that
    # matches the gauge-invariant Wilson loop.
    assert m.gap > 1e-3
    diff = (berry_phase_wilson(m, n_points=2000) - m.berry_phase + np.pi) \
        % (2 * np.pi) - np.pi
    assert abs(diff) < 1e-4
    # Forward-reverse cancels the leading O(T^-1) term: slope steepens past single.
    T = np.geomspace(8.0, 80.0, 14)
    assert _slope(T, single_phase_error(m, T)) > -1.4
    assert _slope(T, forward_reverse_error(m, T)) < -1.6
