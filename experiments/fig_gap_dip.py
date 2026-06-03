r"""Non-isospectral loop: a gap that dips in the middle.

Modulating the field magnitude, B(s) = |B| g(s) n(s) with g(s) = 1 - a sin^2(pi s),
makes the gap vary along the loop -- large at the endpoints (|B|), small in the
middle (|B|(1-a)) -- without changing the Berry phase (which depends only on the
field direction). This is a genuinely non-isospectral loop, unlike the rigid
rotations elsewhere in this repo.

Left panel:  Delta(s) along the loop for a = 0 and a = 0.8.
Right panel: the error cascade (single / forward-reverse / Richardson) for a = 0.8;
             the cancellation still works, with errors set by the smaller mid-loop
             gap Delta_min = |B|(1-a).

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
    richardson_error,
    single_phase_error,
)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

A = 0.8  # gap-dip amplitude


def main() -> None:
    model = SpinHalfLoop(gap_dip=A)
    print(f"gap_dip a={A}: gap(endpoints)={model.field:.2f}, "
          f"gap(min)={model.gap:.2f}, theta_B={model.berry_phase:.4f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # --- Left: Delta(s) along the loop --------------------------------------
    s = np.linspace(0.0, 1.0, 200)
    axL.plot(s, [SpinHalfLoop(gap_dip=0.0).gap_at(si) for si in s], "--",
             color="0.6", label=r"$a=0$ (constant gap)")
    axL.plot(s, [model.gap_at(si) for si in s], "-", color="C3", lw=2,
             label=rf"$a={A}$")
    axL.axhline(model.gap, color="C3", ls=":", lw=1)
    axL.annotate(rf"$\Delta_{{\min}}=|B|(1-a)={model.gap:.1f}$",
                 xy=(0.5, model.gap), xytext=(0.5, model.gap + 0.18),
                 ha="center", fontsize=9, color="C3")
    axL.set_xlabel("loop parameter $s$")
    axL.set_ylabel(r"gap $\Delta(s)$")
    axL.set_title("Gap along the loop (non-isospectral)")
    axL.set_ylim(0, 1.15)
    axL.legend()
    axL.grid(True, alpha=0.2)

    # --- Right: error cascade at a = 0.8 ------------------------------------
    T = np.geomspace(20.0, 300.0, 24)
    e_single = single_phase_error(model, T)
    e_fr = forward_reverse_error(model, T)
    e_rich = richardson_error(model, T)
    axR.loglog(T, e_single, "o-", ms=4, color="C0",
               label=r"single evolution  $|\varphi|$")
    axR.loglog(T, e_fr, "s-", ms=4, color="C1", label=r"forward--reverse")
    axR.loglog(T, e_rich, "^-", ms=4, color="C2", label=r"+ Richardson ($\alpha=2$)")
    tail = T >= np.sqrt(20.0 * 300.0)
    for power, e, style in [(-1, e_single, ":"), (-2, e_fr, "--")]:
        c = np.median(e[tail] / T[tail] ** power)
        axR.loglog(T, c * T ** power, color="0.55", lw=1.0, linestyle=style,
                   label=rf"$\propto T^{{{power}}}$")
    axR.set_xlabel("runtime $T$")
    axR.set_ylabel(r"phase error  $|\tilde\theta_B - \theta_B|$  (rad)")
    axR.set_title(rf"Cancellation with a dipping gap ($a={A}$, "
                  rf"$\Delta_{{\min}}={model.gap:.1f}$)")
    axR.legend(fontsize=8)
    axR.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "gap_dip.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
