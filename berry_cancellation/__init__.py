"""Numerics for adiabatic error cancellation in Berry phase estimation.

See the module docstrings for the physics.  Typical entry points:

    from berry_cancellation import SpinHalfLoop
    from berry_cancellation.estimators import (
        single_phase_error,
        forward_reverse_error,
        richardson_error,
        randomized_richardson_bias,
    )
"""

from __future__ import annotations

from .hamiltonians import SpinHalfLoop
from .reference import berry_phase_wilson, dynamical_phase

__all__ = [
    "SpinHalfLoop",
    "berry_phase_wilson",
    "dynamical_phase",
]
