r"""Main figure: adiabatic error vs runtime ``T`` on the spin-1/2 cone loop.

    single evolution              ~ O(T^-1)
    forward--reverse              ~ O(T^-2)   (oscillatory)
    1 Richardson + bump random.   ~ O(T^-4)
    2 Richardson + bump random.   ~ O(T^-6)   (large-T envelope; oscillatory)

Each Richardson level cancels the next non-oscillatory term (T^-2, then T^-4); the
C^inf bump distribution suppresses the oscillatory residual super-polynomially.
With 1 level the non-oscillatory T^-4 floor dominates (smooth); with 2 levels the
floor drops to T^-6, which falls below the residual oscillation, so that curve is
oscillation-dominated (sign changes -> dips).

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
    randomized_richardson_bias,
    single_phase_error,
)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

LAM = 0.7  # randomization half-width (smallest-worst-runtime smooth config)
ALPHA = 1.75  # Richardson step ratio (>1). Smaller -> shorter worst runtime but the
              # oscillatory residual amplitude ~1/(alpha^2-1) grows, so curves roughen.
              # alpha=1.75 with lam=0.7 is about the smallest worst runtime (~298)
              # that still gives a smooth T^-4 plot.
T_MAX = 100.0
N_POINTS = 20


def main() -> None:
    model = SpinHalfLoop()
    print(f"model: spin-1/2 cone loop, theta0={model.theta0:.4f}, "
          f"gap={model.gap}, theta_B={model.berry_phase:.6f}")

    T = np.geomspace(8.0, T_MAX, N_POINTS)
    alpha = ALPHA

    print("computing single-evolution phase error ...")
    e_single = single_phase_error(model, T)
    print("computing forward--reverse error ...")
    e_fr = forward_reverse_error(model, T)
    print("computing 1 Richardson + bump randomization ...")
    e_bump1 = randomized_richardson_bias(model, T, alpha=alpha, lam=LAM,
                                         levels=1, dist="bump")
    print("computing 2 Richardson + bump randomization ...")
    e_bump2 = randomized_richardson_bias(model, T, alpha=alpha, lam=LAM,
                                         levels=2, dist="bump")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.loglog(T, e_single, "o-", ms=5, color="C0",
              label=r"single evolution  $|\varphi|$")
    ax.loglog(T, e_fr, "s-", ms=5, color="C1", label=r"forward--reverse")
    ax.loglog(T, e_bump1, "^-", ms=5, color="C2",
              label=r"1 Richardson + bump")
    ax.loglog(T, e_bump2, "D-", ms=5, color="C3",
              label=r"2 Richardson + bump")

    # Reference slopes, anchored through the median of each curve over the upper
    # (more asymptotic) half so the guide sits on the data.
    tail = T >= np.sqrt(8.0 * T_MAX)
    for power, e, style in [
        (-1, e_single, ":"),
        (-2, e_fr, "--"),
        (-4, e_bump1, (0, (5, 1))),
        (-6, e_bump2, "-."),
    ]:
        c = np.median(e[tail] / T[tail] ** power)
        ax.loglog(T, c * T ** power, color="0.55", lw=1.0, linestyle=style,
                  label=rf"$\propto T^{{{power}}}$")

    # Worst-case single evolution: T_max*(1+lam)*alpha^levels (1 vs 2 Richardson).
    worst1 = T_MAX * (1.0 + LAM) * alpha
    worst2 = T_MAX * (1.0 + LAM) * alpha**2
    params = (rf"$\alpha={alpha:g}$,  $\lambda={LAM:g}$,  $T\in[{8:g},{T_MAX:g}]$,  "
              rf"worst runtime $T(1{{+}}\lambda)\alpha^{{\rm lvl}}$: "
              rf"{worst1:.0f} (1R), {worst2:.0f} (2R)")

    ax.set_xlim(7.0, T_MAX * 1.15)
    ax.set_xlabel("runtime $T$")
    ax.set_ylabel(r"phase error  $|\tilde\theta_B - \theta_B|$  (rad)")
    ax.set_title("Adiabatic error cancellation in Berry phase estimation "
                 "(spin-1/2 cone loop)\n" + params, fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "scaling.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
