r"""Adiabatic error cancellation in a non-trivial many-qubit model.

Same figure as fig_scaling.py, but on an entangled several-qubit loop -- a
Heisenberg chain in a spiral rotating field (see berry_cancellation/manybody.py),
which has a nonzero virtual-excitation term A^(2). The same estimators are applied
unchanged (they pick up the model's exact amplitudes via loop_amplitudes):

    single evolution              ~ O(T^-1)
    forward--reverse              ~ O(T^-2)   (oscillatory)
    1 Richardson + bump random.   ~ O(T^-4)
    2 Richardson + bump random.   ~ O(T^-6)   (large-T envelope; oscillatory)

Run with:  uv run python experiments/fig_manybody.py
Writes:    figures/manybody.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from berry_cancellation.manybody import SpiralHeisenbergChain
from berry_cancellation.reference import berry_phase_wilson
from berry_cancellation.estimators import (
    forward_reverse_error,
    randomized_richardson_bias,
    single_phase_error,
)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

N = 4
J = 1.0
B0 = 1.0
THETA = 0.4 * np.pi
LAM = 0.8
ALPHA = 1.75
# The many-body gap (~0.37) is far smaller than the spin-1/2 gap (1.0), so the
# pre-asymptotic constants are larger and the deepest (2 Richardson) curve needs a
# longer runtime than scaling.png to settle onto its T^-6 asymptote.
T_MIN = 20.0
T_MAX = 200.0
N_POINTS = 22


def main() -> None:
    model = SpiralHeisenbergChain(N=N, J=J, B0=B0, theta=THETA)
    wl = berry_phase_wilson(model, n_points=2000)
    print(f"N={N} spiral Heisenberg chain: gap={model.gap:.4f}, "
          f"theta_B={model.berry_phase:.6f} (Wilson {wl:.6f})")

    T = np.geomspace(T_MIN, T_MAX, N_POINTS)
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
    tail = T >= np.sqrt(T_MIN * T_MAX)
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
    params = (rf"$N={N}$, $J={J:g}$, $B_0={B0:g}$, gap$={model.gap:.2f}$, "
              rf"$\alpha={alpha:g}$, $\lambda={LAM:g}$,  "
              rf"worst runtime $T(1{{+}}\lambda)\alpha^{{\rm lvl}}$: "
              rf"{worst1:.0f} (1R), {worst2:.0f} (2R)")

    ax.set_xlim(T_MIN * 0.9, T_MAX * 1.15)
    ax.set_xlabel("runtime $T$")
    ax.set_ylabel(r"phase error  $|\tilde\theta_B - \theta_B|$  (rad)")
    ax.set_title("Adiabatic error cancellation, many-qubit model "
                 "(spiral Heisenberg chain)\n" + params, fontsize=9)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "manybody.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
