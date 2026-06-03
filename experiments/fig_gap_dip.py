r"""Adiabatic error cancellation on a non-isospectral loop (dipping gap).

Same cascade as fig_scaling.py (single / forward-reverse / 1 Richardson + bump /
2 Richardson + bump, same alpha, lambda, C^inf bump), but on a loop whose gap
varies: modulating the field magnitude, |B(s)| = |B|(1 - a sin^2(pi s)), makes the
gap |B| at the endpoints and Delta_min = |B|(1-a) in the middle, without changing
the Berry phase. This shows the cancellation survives when the loop is genuinely
non-isospectral.

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

A = 0.8            # gap-dip amplitude
LAM = 0.7          # same as fig_scaling.py
ALPHA = 1.75
T_MIN = 20.0       # shifted up: the small mid-loop gap delays the adiabatic regime
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
    axL.annotate(rf"$\Delta_{{\min}}={model.gap:.1f}$", xy=(0.5, model.gap),
                 xytext=(0.5, model.gap + 0.16), ha="center", fontsize=9, color="C3")
    axL.set_xlabel("loop parameter $s$")
    axL.set_ylabel(r"gap $\Delta(s)$")
    axL.set_title("Gap along the loop (non-isospectral)")
    axL.set_ylim(0, 1.15)
    axL.legend()
    axL.grid(True, alpha=0.2)

    # --- Right: error cascade (same curves as scaling.png) ------------------
    T = np.geomspace(T_MIN, T_MAX, N_POINTS)
    alpha = ALPHA
    print("single ..."); e_single = single_phase_error(model, T)
    print("forward-reverse ..."); e_fr = forward_reverse_error(model, T)
    print("1 Richardson + bump ...")
    e_bump1 = randomized_richardson_bias(model, T, alpha=alpha, lam=LAM,
                                         levels=1, dist="bump", n_nodes=129)
    print("2 Richardson + bump ...")
    e_bump2 = randomized_richardson_bias(model, T, alpha=alpha, lam=LAM,
                                         levels=2, dist="bump", n_nodes=129)

    axR.loglog(T, e_single, "o-", ms=5, color="C0",
               label=r"single evolution  $|\varphi|$")
    axR.loglog(T, e_fr, "s-", ms=5, color="C1", label=r"forward--reverse")
    axR.loglog(T, e_bump1, "^-", ms=5, color="C2", label=r"1 Richardson + bump")
    axR.loglog(T, e_bump2, "D-", ms=5, color="C3", label=r"2 Richardson + bump")

    tail = T >= np.sqrt(T_MIN * T_MAX)
    for power, e, style in [
        (-1, e_single, ":"), (-2, e_fr, "--"),
        (-4, e_bump1, (0, (5, 1))), (-6, e_bump2, "-."),
    ]:
        c = np.median(e[tail] / T[tail] ** power)
        axR.loglog(T, c * T ** power, color="0.55", lw=1.0, linestyle=style,
                   label=rf"$\propto T^{{{power}}}$")

    worst1 = T_MAX * (1.0 + LAM) * alpha
    worst2 = T_MAX * (1.0 + LAM) * alpha**2
    axR.set_xlim(T_MIN * 0.9, T_MAX * 1.15)
    axR.set_xlabel("runtime $T$")
    axR.set_ylabel(r"phase error  $|\tilde\theta_B - \theta_B|$  (rad)")
    axR.set_title(rf"Cancellation, dipping gap ($a={A}$, $\Delta_{{\min}}={model.gap:.1f}$, "
                  rf"$\alpha={alpha:g}$, $\lambda={LAM:g}$); "
                  rf"worst rt {worst1:.0f}/{worst2:.0f}", fontsize=9)
    axR.legend(fontsize=8, ncol=2, loc="lower left")
    axR.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "gap_dip.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
