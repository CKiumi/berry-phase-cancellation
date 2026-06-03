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
from berry_cancellation.estimators import (
    randomized_richardson_bias,
    richardson_error,
    single_phase_error,
)

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

    fig, (axL, axR, axC) = plt.subplots(1, 3, figsize=(17.4, 5.2))

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
    axR.set_xlabel("runtime $T$")
    axR.set_ylabel(r"phase error  (rad)")
    axR.set_title(r"1 Richardson — endpoint-controlled ($\Delta(0)$, dip-independent)")
    axR.legend(fontsize=8, loc="lower left")
    axR.grid(True, which="both", alpha=0.2)

    # --- Far right: 1 Richardson + bump randomization, dip vs no dip ---------
    # Theorem 3: with |chi_mu(xi)| <= C(1+|xi|)^-M, the randomized Richardson bias
    # is O( ||Hdot(0)||^2 / (Delta(0)^4 Delta_min^M T^(M+2)) ). For a single
    # Richardson level the observable C^inf-bump curve is limited to T^-4 (the
    # non-oscillatory floor), i.e. effective M=2, so bias ~ 1/(Delta(0)^4 Delta_min^2 T^4):
    # endpoint Delta(0)^-4 AND interior Delta_min^-2.
    bf = randomized_richardson_bias(flat, T, alpha=1.75, lam=0.7, levels=1, dist="bump")
    bd = randomized_richardson_bias(dip, T, alpha=1.75, lam=0.7, levels=1, dist="bump")
    # Upper bound of the Theorem-3 (M=2) form B(Delta) = C*||Hdot(0)||^2/(Delta(0)^4
    # Delta_min^2 T^4). One distribution-dependent constant C covers both cases.
    base_f = H0f**2 / D0f**4 / Dmf**2
    base_d = H0d**2 / D0d**4 / Dmd**2
    C = max(np.max(bf * T**4 / base_f), np.max(bd * T**4 / base_d))
    bound_f, bound_d = C * base_f / T**4, C * base_d / T**4
    print(f"upper-bound constant C={C:.1f}; tightness (median bound/num): "
          f"no-dip {np.median(bound_f / bf):.2f}, dip {np.median(bound_d / bd):.2f}")
    axC.loglog(T, bf, "o", ms=6, mfc="none", color="C0", label="no dip: numerics")
    axC.loglog(T, bound_f, ":", color="C0", lw=1.5,
               label=rf"upper bound (no dip), $C={C:.0f}$")
    axC.loglog(T, bd, "s", ms=6, color="C3", label=rf"dip ($a={A}$): numerics")
    axC.loglog(T, bound_d, "--", color="C3", lw=1.5,
               label=r"upper bound (dip), $\times\,\Delta_{\min}^{-2}$")
    axC.set_xlabel("runtime $T$")
    axC.set_ylabel(r"phase error  (rad)")
    axC.set_title(r"1 Richardson + bump — bound $C\,\dot H(0)^2/(\Delta(0)^4\Delta_{\min}^2 T^4)$")
    axC.legend(fontsize=8, loc="lower left")
    axC.grid(True, which="both", alpha=0.2)

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
