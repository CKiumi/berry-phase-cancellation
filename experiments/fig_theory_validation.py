r"""Theory bounds vs numerics: dipped vs non-dipped loop, single and 1 Richardson.

Worst-case (max-min) bounds from the paper:
  * single evolution:  |phi| <= Hdot_max^2 / (Delta_min^3 T)
  * 1 Richardson:      residual ~ Hdot(0)^2 / (Delta(0)^4 T^2)  (ENDPOINT-controlled)

The dip (a=0.4, |B(s)|=|B|(1-a sin^2 pi s)) lowers the interior gap to Delta_min=0.6
while leaving the endpoints (Delta(0)=1, Hdot(0)) untouched. Left panel: the single
error tracks Delta_min, so the dip raises both the numerics and (more) the max-min
bound. Right panel: the Richardson residual is endpoint-controlled -- dipped and
non-dipped numerics both track the SAME endpoint line Hdot(0)^2/(Delta(0)^4 T^2),
far below the dip's (inflated) max-min bound.

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
T_MIN, T_MAX, N_POINTS = 20.0, 400.0, 18


def hdot_norm(model, s, h=1e-6):
    return np.linalg.norm(model.H(s + h) - model.H(s - h), 2) / (2 * h)


def info(model):
    s = np.linspace(0.0, 1.0, 400, endpoint=False)
    Hmax = max(hdot_norm(model, si) for si in s)
    return Hmax, model.gap, hdot_norm(model, 0.0), model.gap_at(0.0)


def main() -> None:
    T = np.geomspace(T_MIN, T_MAX, N_POINTS)
    flat = SpinHalfLoop(gap_dip=0.0)
    dip = SpinHalfLoop(gap_dip=A)
    Hmf, Dmf, H0f, D0f = info(flat)
    Hmd, Dmd, H0d, D0d = info(dip)
    print(f"no dip: Delta_min={Dmf:.2f};  dip a={A}: Delta_min={Dmd:.2f}, Delta(0)={D0d:.2f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 5.4), sharey=True)

    # --- Left: single evolution, dip vs no dip ------------------------------
    axL.loglog(T, single_phase_error(flat, T), "o", ms=6, mfc="none", color="C0",
               label="no dip: numerics")
    axL.loglog(T, Hmf**2 / Dmf**3 / T, ":", color="C0", lw=1.5,
               label=r"no dip: $\dot H_{\max}^2/(\Delta_{\min}^3 T)$")
    axL.loglog(T, single_phase_error(dip, T), "s", ms=6, color="C3",
               label=rf"dip ($a={A}$): numerics")
    axL.loglog(T, Hmd**2 / Dmd**3 / T, "--", color="C3", lw=1.5,
               label=r"dip: $\dot H_{\max}^2/(\Delta_{\min}^3 T)$")
    axL.set_xlabel("runtime $T$")
    axL.set_ylabel(r"phase error  (rad)")
    axL.set_title(r"single evolution — tracks $\Delta_{\min}$ (interior)")
    axL.legend(fontsize=8, loc="lower left")
    axL.grid(True, which="both", alpha=0.2)

    # --- Right: 1 Richardson, dip vs no dip ---------------------------------
    axR.loglog(T, richardson_error(flat, T), "o", ms=6, mfc="none", color="C0",
               label="no dip: numerics")
    axR.loglog(T, richardson_error(dip, T), "s", ms=6, color="C3",
               label=rf"dip ($a={A}$): numerics")
    axR.loglog(T, H0d**2 / D0d**4 / T**2, "-", color="k", lw=1.5,
               label=r"endpoint $\dot H(0)^2/(\Delta(0)^4 T^2)$ (both)")
    axR.loglog(T, Hmd**2 / Dmd**4 / T**2, "--", color="C3", lw=1.2,
               label=r"dip max-min $\dot H_{\max}^2/(\Delta_{\min}^4 T^2)$")
    axR.set_xlabel("runtime $T$")
    axR.set_title(r"1 Richardson — endpoint-controlled ($\Delta(0)$, dip-independent)")
    axR.legend(fontsize=8, loc="lower left")
    axR.grid(True, which="both", alpha=0.2)

    fig.suptitle("Theory bounds vs numerics: dipped vs non-dipped loop "
                 rf"($\Delta_{{\min}}$: 1.0 vs {Dmd:.1f}, $\Delta(0)=1$ both)",
                 fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "theory_validation.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
