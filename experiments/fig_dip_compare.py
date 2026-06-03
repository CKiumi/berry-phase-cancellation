r"""Dipping gap vs constant gap: how the dip affects each estimator.

Compares single evolution, forward--reverse, and one Richardson level (no
randomization) on a constant-gap loop (a=0) and a non-isospectral loop with a
dipping gap (a=0.8, Delta_min=0.2). The single and forward--reverse errors are
controlled by the interior minimum gap Delta_min, so they rise with the dip; the
Richardson residual is endpoint-controlled (~1/Delta(0)^4), so it is far less
sensitive to the interior dip.

Run with:  uv run python experiments/fig_dip_compare.py
Writes:    figures/dip_compare.png
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

A = 0.8
ALPHA = 2.0
T_MIN = 20.0
T_MAX = 200.0
N_POINTS = 22


def main() -> None:
    m_flat = SpinHalfLoop(gap_dip=0.0)
    m_dip = SpinHalfLoop(gap_dip=A)
    print(f"no dip: Delta=1.0;  dip a={A}: Delta(0)=1.0, Delta_min={m_dip.gap:.2f}")

    T = np.geomspace(T_MIN, T_MAX, N_POINTS)
    estimators = [
        ("single evolution", single_phase_error, "C0", "o"),
        ("forward--reverse", forward_reverse_error, "C1", "s"),
        ("1 Richardson", lambda m, T: richardson_error(m, T, alpha=ALPHA), "C2", "^"),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for label, fn, color, mk in estimators:
        e_flat = fn(m_flat, T)
        e_dip = fn(m_dip, T)
        ax.loglog(T, e_flat, ls="--", marker=mk, ms=4, mfc="none", color=color,
                  lw=1.0, label=f"{label} — no dip")
        ax.loglog(T, e_dip, ls="-", marker=mk, ms=5, color=color,
                  label=f"{label} — dip ($a={A}$)")

    ax.set_xlim(T_MIN * 0.9, T_MAX * 1.12)
    ax.set_xlabel("runtime $T$")
    ax.set_ylabel(r"phase error  $|\tilde\theta_B - \theta_B|$  (rad)")
    ax.set_title("Effect of a dipping gap on each estimator\n"
                 rf"(spin-1/2; dip $a={A}$, $\Delta_{{\min}}={m_dip.gap:.1f}$ vs "
                 rf"constant gap; 1 Richardson $\alpha={ALPHA:g}$, no randomization)",
                 fontsize=9)
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "dip_compare.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
