r"""Adiabatic error cancellation on a non-isospectral loop (a dipping gap).

The field magnitude is modulated, |B(s)| = |B|(1 - a sin^2(pi s)), so the gap is
|B| at the endpoints and dips to Delta_min = |B|(1-a) in the middle -- a genuinely
non-isospectral loop -- while the Berry phase (set by the field direction) is
unchanged. The same estimators as fig_scaling.py reproduce the cancellation
cascade, showing it does not rely on a constant gap:

    single evolution             ~ O(T^-1)   (controlled by Delta_min, interior)
    forward--reverse             ~ O(T^-2)
    1 Richardson + bump random.  ~ O(T^-4)   (endpoint-controlled residual)

Left panel:  Delta(s) along the loop (constant-gap reference vs the dip).
Right panel: the error cascade vs runtime T.

Run with:  uv run python experiments/fig_gap_dip.py
Writes:    figures/gap_dip.png
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

A = 0.4            # gap-dip amplitude (Delta_min = |B|(1-a) = 0.6)
LAM = 0.7
ALPHA = 1.75
T_MIN = 15.0
T_MAX = 200.0
N_POINTS = 20


def main() -> None:
    model = SpinHalfLoop(gap_dip=A)
    print(f"gap_dip a={A}: gap(endpoints)={model.field:.2f}, "
          f"Delta_min={model.gap:.2f}, theta_B={model.berry_phase:.4f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # --- Left: Delta(s) along the loop --------------------------------------
    s = np.linspace(0.0, 1.0, 200)
    axL.plot(s, np.ones_like(s) * model.field, "--", color="0.6",
             label=r"$a=0$ (constant gap)")
    axL.plot(s, [model.gap_at(si) for si in s], "-", color="C3", lw=2,
             label=rf"$a={A}$")
    axL.axhline(model.gap, color="C3", ls=":", lw=1)
    axL.annotate(rf"$\Delta_{{\min}}=|B|(1-a)={model.gap:.1f}$",
                 xy=(0.5, model.gap), xytext=(0.5, model.gap - 0.13),
                 ha="center", fontsize=9, color="C3")
    axL.set_xlabel("loop parameter $s$")
    axL.set_ylabel(r"gap $\Delta(s)$")
    axL.set_title("Gap along the loop (non-isospectral)")
    axL.set_ylim(0, 1.15)
    axL.legend(loc="lower center")
    axL.grid(True, alpha=0.2)

    # --- Right: error cascade -----------------------------------------------
    T = np.geomspace(T_MIN, T_MAX, N_POINTS)
    print("single ..."); e_single = single_phase_error(model, T)
    print("forward-reverse ..."); e_fr = forward_reverse_error(model, T)
    print("1 Richardson + bump ...")
    e_bump = randomized_richardson_bias(model, T, alpha=ALPHA, lam=LAM,
                                        levels=1, dist="bump")

    axR.loglog(T, e_single, "o-", ms=5, color="C0",
               label=r"single evolution  $|\varphi|$")
    axR.loglog(T, e_fr, "s-", ms=5, color="C1", label=r"forward--reverse")
    axR.loglog(T, e_bump, "^-", ms=5, color="C2", label=r"1 Richardson + bump")

    tail = T >= np.sqrt(T_MIN * T_MAX)
    for power, e, style in [
        (-1, e_single, ":"), (-2, e_fr, "--"), (-4, e_bump, (0, (5, 1))),
    ]:
        c = np.median(e[tail] / T[tail] ** power)
        axR.loglog(T, c * T ** power, color="0.55", lw=1.0, linestyle=style,
                   label=rf"$\propto T^{{{power}}}$")

    axR.set_xlim(T_MIN * 0.9, T_MAX * 1.12)
    axR.set_xlabel("runtime $T$")
    axR.set_ylabel(r"phase error  $|\tilde\theta_B - \theta_B|$  (rad)")
    axR.set_title(rf"Cancellation with a dipping gap "
                  rf"($a={A}$, $\Delta_{{\min}}={model.gap:.1f}$, "
                  rf"$\alpha={ALPHA:g}$, $\lambda={LAM:g}$)", fontsize=10)
    axR.legend(fontsize=8, loc="lower left")
    axR.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "gap_dip.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
