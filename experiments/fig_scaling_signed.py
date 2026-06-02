r"""Signed version of the scaling plot (no absolute value), on a symlog axis.

Same estimators and settings as ``fig_scaling.py``, but the *signed* error is
plotted on a symmetric-log y-axis (linear near 0, log far out). This shows that
the "spikes" in the |error| plot are really smooth sign changes: the oscillatory
residual passes through zero rather than the estimator failing.

Run with:  uv run python experiments/fig_scaling_signed.py
Writes:    figures/scaling_signed.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from berry_cancellation import SpinHalfLoop
from berry_cancellation.estimators import (
    forward_reverse_error,
    randomized_richardson_bias,
    single_phase_error,
)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

LAM = 0.7
ALPHA = 2.0
T_MAX = 100.0
N_POINTS = 20
LINTHRESH = 1e-9  # symlog linear band; below the smallest meaningful |error|


def main() -> None:
    model = SpinHalfLoop()
    print(f"model: spin-1/2 cone loop, theta_B={model.berry_phase:.6f}")

    T = np.geomspace(8.0, T_MAX, N_POINTS)

    print("computing signed errors ...")
    e_single = single_phase_error(model, T, signed=True)
    e_fr = forward_reverse_error(model, T, signed=True)
    e_bump1 = randomized_richardson_bias(model, T, alpha=ALPHA, lam=LAM,
                                         levels=1, dist="bump", signed=True)
    e_bump2 = randomized_richardson_bias(model, T, alpha=ALPHA, lam=LAM,
                                         levels=2, dist="bump", signed=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(T, e_single, "o-", ms=5, color="C0",
            label=r"single evolution  $\varphi$")
    ax.plot(T, e_fr, "s-", ms=5, color="C1", label=r"forward--reverse")
    ax.plot(T, e_bump1, "^-", ms=5, color="C2", label=r"1 Richardson + bump")
    ax.plot(T, e_bump2, "D-", ms=5, color="C3", label=r"2 Richardson + bump")
    ax.axhline(0.0, color="0.6", lw=0.8)

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=LINTHRESH)
    ax.set_xlim(7.0, T_MAX * 1.15)
    ax.set_xlabel("runtime $T$")
    ax.set_ylabel(r"signed error  $\tilde\theta_B - \theta_B$  (rad)")
    ax.set_title("Signed adiabatic error (symlog) -- spikes are sign changes\n"
                 "(spin-1/2 cone loop)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "scaling_signed.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
