r"""Quantitative validation of the leading-order adiabatic phase error.

The theory predicts the single-evolution phase error converges to phi_1 / T with
the loop invariant

    phi_1 = \int_0^1 |<1|Hdot|0>|^2 / Delta(s)^3 ds,

i.e. an integral over BOTH the drive |Hdot| and the gap Delta -- not the gap
alone. This figure overlays the numerical single-evolution error on the reference
line phi_1 / T (with phi_1 computed from the model, no fitting), for a constant-gap
loop and a dipping-gap loop. The numerics fall on the lines, and phi_1 correctly
captures the dip (it rises because the Hdot^2/Delta^3 integrand grows where the
gap dips -- partly offset by the smaller drive there).

Run with:  uv run python experiments/fig_phi1_validation.py
Writes:    figures/phi1_validation.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from berry_cancellation import SpinHalfLoop
from berry_cancellation.estimators import single_phase_error

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def phi1(model, n=4000, h=1e-6):
    """Leading phase-error coefficient phi_1 = mean_s |<1|Hdot|0>|^2 / Delta^3."""
    total = 0.0
    for si in np.linspace(0.0, 1.0, n, endpoint=False):
        Hd = (model.H(si + h) - model.H(si - h)) / (2.0 * h)
        E, V = eigh(model.H(si))
        M = V[:, 1].conj() @ Hd @ V[:, 0]
        total += abs(M) ** 2 / (E[1] - E[0]) ** 3
    return total / n


def main() -> None:
    T = np.geomspace(20.0, 400.0, 18)
    cases = [
        ("no dip", SpinHalfLoop(gap_dip=0.0), "C0", "o"),
        ("dip ($a=0.4$)", SpinHalfLoop(gap_dip=0.4), "C3", "s"),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for label, model, color, mk in cases:
        p1 = phi1(model)
        e = single_phase_error(model, T)
        ax.loglog(T, e, ls="none", marker=mk, ms=6, color=color,
                  label=rf"{label}: numerics")
        ax.loglog(T, p1 / T, "-", color=color, lw=1.5,
                  label=rf"{label}: theory $\varphi_1/T$ ($\varphi_1={p1:.2f}$)")
        print(f"{label}: phi1={p1:.3f}, single*T(T=400)={e[-1]*T[-1]:.3f}")

    ax.set_xlabel("runtime $T$")
    ax.set_ylabel(r"single-evolution phase error  $|\varphi|$  (rad)")
    ax.set_title(r"Leading-order validation: $|\varphi|\to\varphi_1/T$,  "
                 r"$\varphi_1=\int_0^1 |\langle1|\dot H|0\rangle|^2/\Delta(s)^3\,ds$"
                 "\n(reference lines are theory, not fits)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "phi1_validation.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
