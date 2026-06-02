r"""Main figure: adiabatic error vs runtime ``T`` on the spin-1/2 cone loop.

    single evolution                 ~ O(T^-1)
    forward--reverse                 ~ O(T^-2)   (oscillatory)
    1 Richardson + triangle random.  ~ O(T^-4)
    1 Richardson + bump  random.     ~ O(T^-4)   (smoother)

One Richardson level cancels the non-oscillatory T^-2 term, leaving a
non-oscillatory T^-4 floor; a smooth runtime distribution (triangle: CF ~ k^-2;
C^inf bump: faster than any power) suppresses the oscillatory residual below that
floor, so T^-4 becomes the leading, *observable*, non-oscillatory error. The bump
suppresses the oscillation harder, so its curve is smoother. (A second Richardson
level would lower the floor to T^-6, but that sits below the residual oscillation
and machine precision, so it is not observable -- see fig_distributions.py.)

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

LAM = 0.4  # randomization half-width
ALPHA = 1.5  # Richardson step ratio (>1). Smaller -> shorter worst runtime but the
             # oscillatory residual amplitude ~1/(alpha^2-1) grows, so curves roughen.
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
    print("computing 1 Richardson + triangle randomization ...")
    e_tri = randomized_richardson_bias(model, T, alpha=alpha, lam=LAM,
                                       levels=1, dist="triangle", n_nodes=129)
    print("computing 1 Richardson + bump randomization ...")
    e_bump = randomized_richardson_bias(model, T, alpha=alpha, lam=LAM,
                                        levels=1, dist="bump", n_nodes=129)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.loglog(T, e_single, "o-", ms=5, color="C0",
              label=r"single evolution  $|\varphi|$")
    ax.loglog(T, e_fr, "s-", ms=5, color="C1", label=r"forward--reverse")
    ax.loglog(T, e_tri, "v-", ms=5, color="C4",
              label=r"1 Richardson + triangle")
    ax.loglog(T, e_bump, "^-", ms=5, color="C2",
              label=r"1 Richardson + bump")

    # Reference slopes, anchored through the median of each curve over the upper
    # (more asymptotic) half so the guide sits on the data.
    tail = T >= np.sqrt(8.0 * T_MAX)
    for power, e, style in [
        (-1, e_single, ":"),
        (-2, e_fr, "--"),
        (-4, e_bump, (0, (5, 1))),
    ]:
        c = np.median(e[tail] / T[tail] ** power)
        ax.loglog(T, c * T ** power, color="0.55", lw=1.0, linestyle=style,
                  label=rf"$\propto T^{{{power}}}$")

    # Worst-case single evolution for the randomized (1-Richardson) curves.
    worst = T_MAX * (1.0 + LAM) * alpha
    params = (rf"$\alpha={alpha:g}$,  $\lambda={LAM:g}$,  levels$=1$,  "
              rf"$T\in[{8:g},{T_MAX:g}]$,  "
              rf"worst runtime $T(1{{+}}\lambda)\alpha={worst:g}$")

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
