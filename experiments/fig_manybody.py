r"""Adiabatic error cancellation in a non-trivial many-qubit model.

Heisenberg chain in a spiral rotating field (see berry_cancellation/manybody.py):
an entangled, several-qubit loop with a nonzero virtual-excitation term A^(2). The
same estimators as the spin-1/2 figure are applied unchanged (they pick up the
model's exact amplitudes via loop_amplitudes), demonstrating that the cancellation
cascade survives in a genuine many-body setting:

    single evolution             ~ O(T^-1)
    forward--reverse             ~ O(T^-2)
    + Richardson                 ~ O(T^-2)
    + bump randomization (1 lvl) ~ O(T^-4)

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
    richardson_error,
    single_phase_error,
)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

N = 4
J = 1.0
B0 = 1.0
THETA = 0.4 * np.pi
ALPHA = 2.0
LAM = 0.5


def main() -> None:
    model = SpiralHeisenbergChain(N=N, J=J, B0=B0, theta=THETA)
    wl = berry_phase_wilson(model, n_points=2000)
    print(f"N={N} spiral Heisenberg chain: gap={model.gap:.4f}, "
          f"theta_B={model.berry_phase:.6f} (Wilson {wl:.6f})")

    T = np.geomspace(8.0, 100.0, 20)

    print("single ..."); e_single = single_phase_error(model, T)
    print("forward-reverse ..."); e_fr = forward_reverse_error(model, T)
    print("Richardson ..."); e_rich = richardson_error(model, T, alpha=ALPHA)
    print("bump randomization ...")
    e_bump = randomized_richardson_bias(model, T, alpha=ALPHA, lam=LAM,
                                        levels=1, dist="bump", n_nodes=129)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.loglog(T, e_single, "o-", ms=5, color="C0",
              label=r"single evolution  $|\varphi|$")
    ax.loglog(T, e_fr, "s-", ms=5, color="C1", label=r"forward--reverse")
    ax.loglog(T, e_rich, "^-", ms=5, color="C2", label=r"+ Richardson ($\alpha=2$)")
    ax.loglog(T, e_bump, "D-", ms=5, color="C3", label=r"+ bump randomization")

    tail = T >= np.sqrt(8.0 * 100.0)
    for power, e, style in [
        (-1, e_single, ":"), (-2, e_fr, "--"), (-4, e_bump, "-."),
    ]:
        c = np.median(e[tail] / T[tail] ** power)
        ax.loglog(T, c * T ** power, color="0.55", lw=1.0, linestyle=style,
                  label=rf"$\propto T^{{{power}}}$")

    ax.set_xlim(7.0, 115.0)
    ax.set_xlabel("runtime $T$")
    ax.set_ylabel(r"phase error  $|\tilde\theta_B - \theta_B|$  (rad)")
    ax.set_title(rf"Many-qubit cancellation: spiral Heisenberg chain, $N={N}$ qubits"
                 "\n"
                 rf"$J={J:g}$, $B_0={B0:g}$, $\theta={THETA/np.pi:.1f}\pi$, "
                 rf"gap$={model.gap:.2f}$, $\theta_B={model.berry_phase:.3f}$, "
                 rf"$\alpha={ALPHA:g}$, $\lambda={LAM:g}$", fontsize=9)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "manybody.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
