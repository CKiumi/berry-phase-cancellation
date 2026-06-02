r"""Main figure: adiabatic error vs runtime ``T`` for the four estimators.

Reproduces the central claim of the paper on the spin-1/2 cone loop:

    single evolution         ~ O(T^-1)
    forward--reverse         ~ O(T^-2)   (oscillatory)
    + Richardson             ~ O(T^-2)   (oscillatory, smaller constant)
    + runtime randomization  ~ O(T^-3)   (deterministic bias)

Run with:  uv run python experiments/fig_scaling.py
Writes:    figures/scaling.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from berry_cancellation import SpinHalfLoop
from berry_cancellation.estimators import (
    forward_reverse_error,
    randomized_richardson_montecarlo,
    richardson_error,
    single_phase_error,
)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

N_SHOTS = 1000


def main() -> None:
    model = SpinHalfLoop()
    print(f"model: spin-1/2 cone loop, theta0={model.theta0:.4f}, "
          f"gap={model.gap}, theta_B={model.berry_phase:.6f}")

    T = np.geomspace(8.0, 200.0, 40)
    alpha = 2.0

    print("computing single-evolution phase error ...")
    e_single = single_phase_error(model, T)
    print("computing forward--reverse error ...")
    e_fr = forward_reverse_error(model, T)
    print("computing Richardson error ...")
    e_rich = richardson_error(model, T, alpha=alpha)
    # Real runtime randomization (Monte Carlo) on a coarser grid -- N=10000 shots
    # per base T is the expensive curve, so it uses fewer T points.
    T_rand = np.geomspace(8.0, 150.0, 16)
    print(f"running real runtime randomization (N={N_SHOTS} shots) ...")
    # The randomized estimator's error is statistical, not a noisy bias point:
    # report its standard deviation sigma_N = std/sqrt(N) (positive-definite,
    # clean ~T^-2 on log-log), returned directly as the SEM.
    _, e_rand_std, _ = randomized_richardson_montecarlo(
        model, T_rand, n_shots=N_SHOTS, alpha=alpha, lam=0.5, seed=0
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.loglog(T, e_single, "o-", ms=4, label=r"single evolution  $|\varphi|$")
    ax.loglog(T, e_fr, "s-", ms=4, label=r"forward--reverse")
    ax.loglog(T, e_rich, "^-", ms=4, label=r"+ Richardson ($\alpha=2$)")
    ax.loglog(T_rand, e_rand_std, "d-", ms=5, color="C3",
              label=rf"+ randomization: std ($N={N_SHOTS}$)")

    # Reference power-law guides, anchored at the asymptotic (large-T) end, since
    # the small-T points are outside the regime where the 1/T expansion holds.
    Tref = T[-1]
    for power, eref, style in [
        (-1, e_single[-1], ":"),
        (-2, e_fr[-1], "--"),
    ]:
        ax.loglog(T, eref * (T / Tref) ** power, style, color="0.5", lw=1.0,
                  label=rf"$\propto T^{{{power}}}$")

    ax.set_xlim(7.0, 230.0)
    ax.set_xlabel("runtime $T$")
    ax.set_ylabel(r"phase error / std  (rad)")
    ax.set_title("Adiabatic error cancellation in Berry phase estimation\n"
                 "(spin-1/2 cone loop)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "scaling.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
