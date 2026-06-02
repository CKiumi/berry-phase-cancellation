r"""Sanity check: numerical Berry phase vs the analytic half-solid-angle.

For the spin-1/2 cone loop the ground-state Berry phase is exactly
``theta_B = -Omega/2 = -pi (1 - cos theta0)``.  This script sweeps the cone angle
``theta0`` and overlays:

  * the analytic value ``-pi (1 - cos theta0)``,
  * the gauge-invariant Wilson-loop value,
  * the forward--reverse estimate at a large but finite runtime ``T``.

Run with:  uv run python experiments/fig_spin_half_check.py
Writes:    figures/spin_half_check.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from berry_cancellation import SpinHalfLoop, berry_phase_wilson
from berry_cancellation.estimators import _theta_B_forward_reverse, default_steps
from berry_cancellation.reference import wrap_to_half_pi

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def main() -> None:
    thetas = np.linspace(0.15 * np.pi, 0.85 * np.pi, 18)
    T = 120.0

    analytic, wilson, estimate = [], [], []
    for th in thetas:
        model = SpinHalfLoop(theta0=float(th))
        ref = model.berry_phase
        analytic.append(ref)
        # Berry phase is defined mod 2*pi; put Wilson on the analytic branch.
        wil = berry_phase_wilson(model)
        wilson.append(ref + (wil - ref + np.pi) % (2 * np.pi) - np.pi)
        # The forward--reverse estimate is defined mod pi; align it too.
        est = _theta_B_forward_reverse(model, T, default_steps(T))
        estimate.append(ref + float(wrap_to_half_pi(est - ref)))

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(thetas, analytic, "-", lw=2, label=r"analytic $-\pi(1-\cos\theta_0)$")
    ax.plot(thetas, wilson, "o", ms=5, label="Wilson loop")
    ax.plot(thetas, estimate, "x", ms=7, label=rf"forward--reverse ($T={T:.0f}$)")
    ax.set_xlabel(r"cone angle $\theta_0$")
    ax.set_ylabel(r"Berry phase $\theta_B$")
    ax.set_title("Spin-1/2 cone loop: Berry phase vs analytic half-solid-angle")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "spin_half_check.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"max |analytic - wilson|   = {np.max(np.abs(np.array(analytic) - wilson)):.2e}")
    print(f"max |analytic - estimate| = {np.max(np.abs(np.array(analytic) - estimate)):.2e}")


if __name__ == "__main__":
    main()
