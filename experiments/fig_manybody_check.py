r"""Berry-phase check for the many-qubit model (analog of fig_spin_half_check.py).

Sweeps the cone angle ``theta`` of the spiral Heisenberg chain and overlays three
independent determinations of the ground-state Berry phase:

  * analytic   theta_B = 2 pi <S^z_tot>_0   (from the rotation reduction H(s)=R H0 R^dag),
  * the gauge-invariant Wilson loop,
  * the forward--reverse estimate from a finite-runtime evolution.

Agreement of all three validates both the analytic formula and the estimator on a
genuine entangled many-body state.

Run with:  uv run python experiments/fig_manybody_check.py
Writes:    figures/manybody_check.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from berry_cancellation.manybody import SpiralHeisenbergChain
from berry_cancellation.reference import (
    berry_phase_wilson,
    wrap_to_half_pi,
    wrap_to_pi,
)
from berry_cancellation.estimators import _theta_B_forward_reverse

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

N = 4
J = 1.0
B0 = 1.0
T_CHECK = 150.0


def main() -> None:
    thetas = np.linspace(0.15 * np.pi, 0.85 * np.pi, 18)

    analytic, wilson, estimate, gaps = [], [], [], []
    for th in thetas:
        m = SpiralHeisenbergChain(N=N, J=J, B0=B0, theta=float(th))
        # raw (unwrapped) analytic Berry phase 2 pi <S^z_tot>_0.
        raw = 2.0 * np.pi * float(np.real(m.psi0.conj() @ m.Sz_tot @ m.psi0))
        analytic.append(raw)
        gaps.append(m.gap)
        # Wilson loop, placed on the analytic branch (mod 2 pi).
        wilson.append(raw + float(wrap_to_pi(berry_phase_wilson(m, 2000) - raw)))
        # Forward--reverse estimate (mod pi), placed on the analytic branch.
        est = float(np.ravel(_theta_B_forward_reverse(m, T_CHECK, steps=1))[0])
        estimate.append(raw + float(wrap_to_half_pi(est - raw)))

    analytic = np.array(analytic)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(thetas, analytic, "-", lw=2, color="C0",
            label=r"analytic  $2\pi\langle S^z_{tot}\rangle_0$")
    ax.plot(thetas, wilson, "o", ms=6, color="C1", label="Wilson loop")
    ax.plot(thetas, estimate, "x", ms=8, color="C3",
            label=rf"forward--reverse ($T={T_CHECK:.0f}$)")
    ax.set_xlabel(r"cone angle $\theta$")
    ax.set_ylabel(r"Berry phase $\theta_B$  (rad)")
    ax.set_title(rf"Berry-phase check, spiral Heisenberg chain "
                 rf"($N={N}$, $J={J:g}$, $B_0={B0:g}$)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "manybody_check.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"gap range over sweep: [{min(gaps):.3f}, {max(gaps):.3f}]")
    print(f"max |analytic - wilson|   = {np.max(np.abs(analytic - wilson)):.2e}")
    print(f"max |analytic - estimate| = {np.max(np.abs(analytic - estimate)):.2e}")


if __name__ == "__main__":
    main()
