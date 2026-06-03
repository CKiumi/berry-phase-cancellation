r"""Theory bounds vs numerics on a non-isospectral (dipping-gap) loop.

Checks the paper's worst-case (max-min) bounds against the numerics, and shows the
qualitative difference between the single evolution and one Richardson level:

  * single evolution:    |phi| ~ Hdot_max^2 / (Delta_min^3 T)   (interior / Delta_min)
  * 1 Richardson:        residual ~ Hdot(0)^2 / (Delta(0)^4 T^2) (ENDPOINT-controlled)

On a loop whose gap dips in the middle (a=0.4), Hdot_max and Delta_min do not occur
at the same s, so the max-min bound is conservative. The single error sits a few x
below its max-min bound; the Richardson residual sits FAR (~20x) below its max-min
bound and instead tracks the endpoint scale Hdot(0)^2/Delta(0)^4 -- i.e. Richardson
trades the Delta_min dependence for an endpoint Delta(0) dependence.

Run with:  uv run python experiments/fig_theory_validation.py
Writes:    figures/theory_validation.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from berry_cancellation import SpinHalfLoop
from berry_cancellation.estimators import richardson_error, single_phase_error

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

A = 0.4
T_MIN, T_MAX, N_POINTS = 20.0, 400.0, 20


def hdot_norm(model, s, h=1e-6):
    return np.linalg.norm(model.H(s + h) - model.H(s - h), 2) / (2 * h)


def main() -> None:
    model = SpinHalfLoop(gap_dip=A)
    s = np.linspace(0.0, 1.0, 400, endpoint=False)
    Hmax = max(hdot_norm(model, si) for si in s)
    H0 = hdot_norm(model, 0.0)
    Dmin, D0 = model.gap, model.gap_at(0.0)
    print(f"Hdot_max={Hmax:.3f}, Hdot(0)={H0:.3f}, Delta_min={Dmin:.2f}, Delta(0)={D0:.2f}")

    c_single = Hmax**2 / Dmin**3              # single max-min bound coeff
    c_rich_mm = Hmax**2 / Dmin**4             # Richardson max-min bound coeff
    c_rich_ep = H0**2 / D0**4                 # Richardson endpoint coeff

    T = np.geomspace(T_MIN, T_MAX, N_POINTS)
    e_single = single_phase_error(model, T)
    e_rich = richardson_error(model, T)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    # single
    ax.loglog(T, e_single, "o", ms=6, color="C0", label="single (numerics)")
    ax.loglog(T, c_single / T, ":", color="C0", lw=1.5,
              label=r"single bound $\dot H_{\max}^2/(\Delta_{\min}^3 T)$")
    # 1 Richardson
    ax.loglog(T, e_rich, "^", ms=6, color="C2", label="1 Richardson (numerics)")
    ax.loglog(T, c_rich_mm / T**2, ":", color="C2", lw=1.5,
              label=r"Rich max-min $\dot H_{\max}^2/(\Delta_{\min}^4 T^2)$")
    ax.loglog(T, c_rich_ep / T**2, "-", color="C3", lw=1.5,
              label=r"Rich endpoint $\dot H(0)^2/(\Delta(0)^4 T^2)$")

    ax.set_xlabel("runtime $T$")
    ax.set_ylabel(r"phase error  (rad)")
    ax.set_title(rf"Theory bounds vs numerics, dipping-gap loop "
                 rf"($a={A}$, $\Delta_{{\min}}={Dmin:.1f}$, $\Delta(0)=1$)"
                 "\n"
                 r"single follows $\Delta_{\min}$; Richardson is endpoint-controlled "
                 r"($\ll$ its max-min bound)", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "theory_validation.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
