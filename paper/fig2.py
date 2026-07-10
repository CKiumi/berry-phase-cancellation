"""Figure 2: Theory bounds vs numerics, and which gap controls each estimator.

Non-isospectral spin-1/2 loop with field modulation |B(s)| = |B|(1 - a sin^2(pi s)).
(a) The dipping gap Delta(s) along the loop.
(b) Single evolution tracks the interior worst-case gap Delta_min.
(c) One-fold Richardson residual is controlled by the endpoint gap Delta(0).
(d) One Richardson + uniform randomization against the Theorem-3 M = 1 (T^-3)
bound with the exact all-xi a priori uniform constant (no fitted constant);
the randomized oscillatory bias itself falls as T^-3, tracking the bound.

The expensive evolutions are parallelized over all cores and cached in
cache/fig2.npz: rerunning with unchanged physics parameters (e.g. after a
styling change) replots instantly from the cache.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match the paper's (REVTeX / Computer Modern) fonts.
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "text.latex.preamble": r"\usepackage{amsmath}",
                     "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13})

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from berry_cancellation import SpinHalfLoop
from berry_cancellation.estimators import (
    single_phase_error, richardson_error, randomized_richardson_bias,
    default_steps,
)

A = 0.4
T_MIN, T_MAX, N_POINTS = 20.0, 400.0, 18
LAM, ALPHA = 0.7, 1.75
ALPHA_R = 2.0                      # richardson_error's default extrapolation ratio
PARAMS = dict(a=A, t_min=T_MIN, t_max=T_MAX, n=N_POINTS, lam=LAM, alpha=ALPHA,
              alpha_r=ALPHA_R, dist="uniform")
CACHE = HERE / "cache" / "fig2.npz"


def cached_compute(path, params, fn):
    "Load {arrays} from path if its stored params match, else compute and store."
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if z["__params__"].item() == params:
            print(f"loaded cache {path}")
            return {k: z[k] for k in z.files if k != "__params__"}
    data = fn()
    path.parent.mkdir(exist_ok=True)
    np.savez(path, __params__=np.array(params, dtype=object), **data)
    print(f"wrote cache {path}")
    return data


def compute():
    T = np.geomspace(T_MIN, T_MAX, N_POINTS)
    flat, dip = SpinHalfLoop(gap_dip=0.0), SpinHalfLoop(gap_dip=A)
    steps_s = default_steps(T.max())
    steps_r = default_steps(ALPHA_R * T.max())

    out = {}
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {}
        for name, m in [("f", flat), ("d", dip)]:
            futs["single_" + name] = [ex.submit(single_phase_error, m, np.array([t]), steps_s) for t in T]
            futs["rich_" + name] = [ex.submit(richardson_error, m, np.array([t]), ALPHA_R, steps_r) for t in T]
            futs["rand_" + name] = [ex.submit(randomized_richardson_bias, m, np.array([t]),
                                              alpha=ALPHA, lam=LAM, levels=1, dist="uniform") for t in T]
        for key, fl in futs.items():
            out[key] = np.concatenate([f.result() for f in fl])
    out["T"] = T
    return out


def hdot_norm(model, s, h=1e-6):
    return np.linalg.norm(model.H(s + h) - model.H(s - h), 2) / (2 * h)


def info(model):
    s = np.linspace(0.0, 1.0, 400, endpoint=False)
    Hmax = max(hdot_norm(model, si) for si in s)
    return Hmax, model.gap, hdot_norm(model, 0.0), model.gap_at(0.0)


def main():
    d = cached_compute(CACHE, PARAMS, compute)
    T = d["T"]
    flat, dip = SpinHalfLoop(gap_dip=0.0), SpinHalfLoop(gap_dip=A)
    Hmf, Dmf, H0f, D0f = info(flat)
    Hmd, Dmd, H0d, D0d = info(dip)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.6))
    (ax0, axL), (axR, axC) = axes

    # Panel 0: the dipping gap Delta(s) along the loop.
    s = np.linspace(0.0, 1.0, 200)
    ax0.plot(s, np.ones_like(s) * flat.field, "--", color="0.6", label=r"$a=0$ (constant gap)")
    ax0.plot(s, [dip.gap_at(si) for si in s], "-", color="C3", lw=2, label=rf"$a={A}$")
    ax0.plot([0.5], [dip.gap], "*", ms=14, color="C3", zorder=5)
    ax0.annotate(rf"$\Delta_{{\min}}={dip.gap:.1f}$", xy=(0.5, dip.gap),
                 xytext=(0.5, dip.gap - 0.05), ha="center", va="top",
                 fontsize=12, color="C3")
    ax0.set_xlabel("loop parameter $s$")
    ax0.set_ylabel(r"gap $\Delta(s)$")
    ax0.set_title(r"(a) gap $\Delta(s)$ along the loop")
    ax0.set_ylim(0.4, 1.1)
    ax0.legend(fontsize=12, loc="lower right", frameon=True, framealpha=0.9, edgecolor="0.8", fancybox=False)
    ax0.grid(True, alpha=0.2)

    # Panel 1: single evolution, dip vs no dip.
    axL.loglog(T, d["single_f"], "o", ms=6, mfc="none", color="C0", label=r"$a=0$")
    axL.loglog(T, Hmf**2 / Dmf**3 / T, ":", color="C0", lw=1.5, label=r"$\dot H_{\max}^2/(\Delta_{\min}^3 T)$ ($a=0$)")
    axL.loglog(T, d["single_d"], "s", ms=6, color="C3", label=rf"$a={A}$")
    axL.loglog(T, Hmd**2 / Dmd**3 / T, "--", color="C3", lw=1.5, label=rf"$\dot H_{{\max}}^2/(\Delta_{{\min}}^3 T)$ ($a={A}$)")
    axL.set_xlabel("runtime $T$")
    axL.set_ylabel(r"phase error  (rad)")
    axL.set_title(r"(b) single evolution")
    axL.legend(fontsize=12, loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.8", fancybox=False)
    axL.grid(True, which="both", alpha=0.2)

    # Panel 2: 1 Richardson, dip vs no dip.
    axR.loglog(T, d["rich_f"], "o", ms=6, mfc="none", color="C0", label=r"$a=0$")
    axR.loglog(T, d["rich_d"], "s", ms=6, color="C3", label=rf"$a={A}$")
    axR.loglog(T, H0d**2 / D0d**4 / T**2, "-", color="k", lw=1.5, label=r"endpoint $\dot H(0)^2/(\Delta(0)^4 T^2)$ (both)")
    axR.set_xlabel("runtime $T$")
    axR.set_ylabel(r"phase error  (rad)")
    axR.set_title(r"(c) 1 Richardson")
    axR.legend(fontsize=12, loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.8", fancybox=False)
    axR.grid(True, which="both", alpha=0.2)

    # Panel 3: 1 Richardson + uniform randomization, with the Theorem-3 M=1 (T^-3)
    # bound. The uniform characteristic function decays as xi^-1 (M=1), so the
    # randomized oscillatory bias falls as T^-3 and dominates the non-oscillatory
    # Richardson remainder O(T^-4) that one extrapolation level cannot remove: the
    # data itself exhibits the Theorem-3 oscillatory scaling, in both power and gap
    # dependence. The a priori constant is exact for the uniform density: one
    # integration by parts gives |chi_{mu,2}(xi)| <= C1/|xi| for all xi, with
    # C1 = (1-lam)^-2/lam (endpoint jumps plus the interior x^-3 term). The
    # Richardson-transformed k=2 modes oscillate at frequencies alpha^l omega_n, so
    # the M=1 decay contributes an extra alpha^-l per level: the factor is
    # L_{1,3}(alpha) = sum |w_l| alpha^-3l (Appendix D), tighter than the
    # frequency-agnostic L_{1,2} = 2/(alpha^2-1). No constant is fitted.
    bf, bd = d["rand_f"], d["rand_d"]
    C1 = (1.0 - LAM) ** -2 / LAM
    w0, w1 = 1.0 / (ALPHA**2 - 1.0), ALPHA**2 / (ALPHA**2 - 1.0)   # |Richardson weights|
    L13 = w0 + w1 * ALPHA**-3
    C = C1 * L13
    base_f = H0f**2 / D0f**4 / Dmf
    base_d = H0d**2 / D0d**4 / Dmd
    bound_f, bound_d = C * base_f / T**3, C * base_d / T**3
    print(f"uniform M=1 constant C1={C1:.2f}, L13(alpha)={L13:.3f}, a priori C={C:.2f}")
    print(f"max data/bound ratio: no dip {np.max(bf / bound_f):.2f}, dip {np.max(bd / bound_d):.2f}")
    axC.loglog(T, bf, "o", ms=6, mfc="none", color="C0", label=r"$a=0$")
    axC.loglog(T, bound_f, ":", color="C0", lw=1.5, label=r"$C\,\dot H(0)^2/(\Delta(0)^4\Delta_{\min} T^3)$ ($a=0$)")
    axC.loglog(T, bd, "s", ms=6, color="C3", label=rf"$a={A}$")
    axC.loglog(T, bound_d, "--", color="C3", lw=1.5, label=rf"same ($a={A}$)")
    axC.set_xlabel("runtime $T$")
    axC.set_title(r"(d) 1 Richardson + uniform randomization")
    axC.legend(fontsize=12, loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.8", fancybox=False)
    axC.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.20, hspace=0.32)
    path = HERE / "fig2.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
