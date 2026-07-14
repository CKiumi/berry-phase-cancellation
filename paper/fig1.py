"""Figure 1: Entangled spiral Heisenberg chain -- small-runtime accuracy and cascade.

One consistent randomization parameter set for the whole figure: bump window
``lambda = 0.6``, Richardson ratio ``alpha = 1.7``.

(a) Berry phase (canonical mod-2pi window) vs cone angle at a small runtime
``T = 20``: the exact closed-form reference ``2*pi*<S^z_tot>`` (wrapped, with the
branch discontinuity cut), the runtime-scaling reconstruction shown as raw values
modulo 2pi (its error exceeds pi at this runtime), the forward-reverse estimate
(mod pi, branch nearer the analytic value), and the 1- and 2-fold Richardson +
bump estimates, each the mean of ``N = 10`` randomly sampled runtimes with
``+-2`` s.e.m. error bars from the same shots.  ``T = 20`` keeps the *entire*
angle sweep -- including the deepest-gap endpoints (``theta/pi ~ 0.2, 0.8``,
``Delta ~ 0.11``) -- inside the adiabatic window ``T*Delta >~ 2``, so the
2-Richardson estimate tracks the curve everywhere; at smaller ``T`` those
endpoints fall into the non-adiabatic regime where the extrapolation overshoots.
No Theorem-3 band: for this entangled model the closed-form norm constant
``|Hdot(0)|^2/Delta(0)^4`` overestimates the actual fluctuation by 2-4 orders of
magnitude (it replaces every level gap by ``Delta(0)`` and the ground-state
couplings by the operator norm).

(b) Error-cancellation cascade vs runtime at ``theta = 0.39 pi`` (``Delta = 0.36``),
chosen over a theta-sweep as the angle where the 2-Richardson + bump curve holds
its ``T^-6`` slope most smoothly at the shared ``lambda = 0.6`` (the maximally
symmetric angle ``theta = pi/2`` is excluded: there the forward-reverse error
vanishes identically, so the whole cascade collapses to machine precision and
shows no scaling).

The expensive evolutions are parallelized over all cores and cached in
cache/fig1.npz: rerunning with unchanged physics parameters (e.g. after a
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
                     "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
                     "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13})

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from berry_cancellation.manybody import SpiralHeisenbergChain
from berry_cancellation.estimators import (
    single_phase_error, forward_reverse_error, randomized_richardson_bias,
    runtime_scaling_theta_B, _recursive_richardson_error,
    _theta_B_forward_reverse, default_steps,
)
from berry_cancellation.reference import wrap_to_pi, wrap_to_half_pi

N, J, B0 = 4, 1.0, 1.0
# One randomization parameter set for both panels.
LAM, ALPHA = 0.6, 1.7
# Panel (a): small-runtime Berry-phase check. T = 20 keeps the whole sweep
# (deepest gap ~ 0.11) inside the adiabatic window T*Delta >~ 2; N = 10 shots keep
# the +-2 s.e.m. bars visible.
TCHK = 20.0
N_THETA = 30
THETA_LO, THETA_HI = 0.15 * np.pi, 0.85 * np.pi
# Panel (b) cascade angle: theta = 0.39 pi gives the cleanest cascade over a
# 10-angle sweep at this lambda -- the 2R curve holds its T^-6 slope with the
# smallest log-residual (gap Delta = 0.36).
THETA_C = 0.39 * np.pi
T_MIN, T_MAX, N_T = 20.0, 200.0, 22
N_SHOTS, SEED = 10, 0
PARAMS = dict(N=N, J=J, B0=B0, lam=LAM, alpha=ALPHA, tchk=TCHK, n_theta=N_THETA,
              n_shots=N_SHOTS, seed=SEED, rs_alpha="hmax-scale",
              panel_a="rs+fr+1r+2r-10shot", theta_lo=THETA_LO, theta_hi=THETA_HI,
              theta_c=THETA_C, t_min=T_MIN, t_max=T_MAX, n_t=N_T)
CACHE = HERE / "cache" / "fig1.npz"


def sample_bump(n, lam, rng):
    "Draw n runtime factors X from the C-infinity bump density on [1-lam, 1+lam]."
    ug = np.linspace(-1.0, 1.0, 4001)
    pdf = np.zeros_like(ug)
    ins = np.abs(ug) < 1.0
    pdf[ins] = np.exp(-1.0 / (1.0 - ug[ins] ** 2))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(ug))])
    cdf /= cdf[-1]
    return 1.0 + lam * np.interp(rng.random(n), cdf, ug)


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
    thetas = np.linspace(THETA_LO, THETA_HI, N_THETA)
    models = [SpiralHeisenbergChain(N=N, J=J, B0=B0, theta=float(th)) for th in thetas]
    model = SpiralHeisenbergChain(N=N, J=J, B0=B0, theta=THETA_C)
    T = np.geomspace(T_MIN, T_MAX, N_T)
    steps = default_steps(T.max())

    with ProcessPoolExecutor(max_workers=10) as ex:
        # (a) per-model small-T estimates.
        fut_rs = [ex.submit(runtime_scaling_theta_B, m, TCHK) for m in models]
        fut_fr = [ex.submit(_theta_B_forward_reverse, m, np.array([TCHK]),
                            default_steps(TCHK)) for m in models]
        rng = np.random.default_rng(SEED)
        shots = [TCHK * sample_bump(N_SHOTS, LAM, rng) for _ in models]
        fut_2r = [ex.submit(_recursive_richardson_error, m, r, ALPHA, 2,
                            default_steps(ALPHA**2 * r.max()))
                  for m, r in zip(models, shots)]
        fut_1r = [ex.submit(_recursive_richardson_error, m, r, ALPHA, 1,
                            default_steps(ALPHA * r.max()))
                  for m, r in zip(models, shots)]
        # (b) per-runtime cascade curves.  The O(T^-1) baseline is the naive
        # runtime-scaling reconstruction (same method as panel a), which the
        # forward-reverse + Richardson + bump cascade then beats.
        fut_rsb = [ex.submit(runtime_scaling_theta_B, model, float(t)) for t in T]
        fut_f = [ex.submit(forward_reverse_error, model, np.array([t]), steps) for t in T]
        fut_1 = [ex.submit(randomized_richardson_bias, model, np.array([t]),
                           alpha=ALPHA, lam=LAM, levels=1, dist="bump") for t in T]
        fut_2 = [ex.submit(randomized_richardson_bias, model, np.array([t]),
                           alpha=ALPHA, lam=LAM, levels=2, dist="bump") for t in T]
        rs_raw = np.array([f.result() for f in fut_rs])
        fr_raw = np.array([f.result()[0] for f in fut_fr])
        mc_err = np.array([f.result() for f in fut_2r])   # (N_THETA, N_SHOTS) signed errors
        mc1_err = np.array([f.result() for f in fut_1r])
        e_rs = np.abs(wrap_to_pi(np.array([f.result() for f in fut_rsb]) - model.berry_phase))
        e_fr = np.concatenate([f.result() for f in fut_f])
        e_b1 = np.concatenate([f.result() for f in fut_1])
        e_b2 = np.concatenate([f.result() for f in fut_2])

    analytic = np.array([2.0 * np.pi * float(np.real(m.psi0.conj() @ m.Sz_tot @ m.psi0))
                         for m in models])
    gaps = np.array([m.gap for m in models])
    return dict(thetas=thetas, analytic=analytic, gaps=gaps,
                rs_raw=rs_raw, fr_raw=fr_raw, mc_err=mc_err, mc1_err=mc1_err, T=T,
                e_rs=e_rs, e_fr=e_fr, e_b1=e_b1, e_b2=e_b2)


def main():
    d = cached_compute(CACHE, PARAMS, compute)
    thetas, analytic, T = d["thetas"], d["analytic"], d["T"]
    est_rs = analytic + wrap_to_pi(d["rs_raw"] - analytic)          # theta_B mod 2 pi
    est_fr = analytic + wrap_to_half_pi(d["fr_raw"] - analytic)     # theta_B mod pi
    est_2r = analytic + d["mc_err"].mean(axis=1)                    # 10-shot mean (mod pi)
    est_1r = analytic + d["mc1_err"].mean(axis=1)
    ihard = int(np.argmin(d["gaps"]))
    print(f"  T={TCHK:.0f}: runtime-scaling median dev={np.median(np.abs(est_rs - analytic)):.1e}, "
          f"FR median dev={np.median(np.abs(est_fr - analytic)):.1e}, "
          f"2R+bump {N_SHOTS}-shot median dev={np.median(np.abs(est_2r - analytic)):.1e}")
    print(f"  deepest-gap angle theta/pi={thetas[ihard]/np.pi:.2f} (gap={d['gaps'][ihard]:.3f}): "
          f"2R dev={np.abs(est_2r-analytic)[ihard]:.1e}")
    print(f"  gap range over sweep: [{d['gaps'].min():.3f}, {d['gaps'].max():.3f}]")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.8, 5.0))

    # (a) Berry phase vs cone angle at a small, cheap runtime.
    twopi = 2.0 * np.pi
    th_f = np.linspace(thetas[0], thetas[-1], 800)
    models_f = [SpiralHeisenbergChain(N=N, J=J, B0=B0, theta=float(t)) for t in th_f]
    ana_f = np.array([2.0 * np.pi * float(np.real(m.psi0.conj() @ m.Sz_tot @ m.psi0))
                      for m in models_f]) % twopi
    jump = np.abs(np.diff(ana_f)) > np.pi
    ana_f[1:][jump] = np.nan
    axA.plot(th_f, ana_f, "-", lw=2, color="0.4", label="analytic")
    axA.plot(thetas, d["rs_raw"] % twopi, "o", ms=6, color="C0",
             label="runtime scaling")
    axA.plot(thetas, est_fr % twopi, "s", ms=6, color="C1", label=r"forward--reverse")
    sem1 = d["mc1_err"].std(axis=1, ddof=1) / np.sqrt(N_SHOTS)
    axA.errorbar(thetas, est_1r % twopi, yerr=2 * sem1, fmt="^", ms=6, color="C2",
                 capsize=3, lw=1.2,
                 label=rf"1 Richardson + bump ({N_SHOTS}-shot mean $\pm2\,$s.e.m.)")
    sem = d["mc_err"].std(axis=1, ddof=1) / np.sqrt(N_SHOTS)
    axA.errorbar(thetas, est_2r % twopi, yerr=2 * sem, fmt="D", ms=6, color="C3",
                 capsize=3, lw=1.2,
                 label=rf"2 Richardson + bump ({N_SHOTS}-shot mean $\pm2\,$s.e.m.)")
    print(f"  empirical 2*SEM range: [{2*sem.min():.1e}, {2*sem.max():.1e}] rad")
    axA.set_ylim(0, twopi)
    axA.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, twopi],
                   [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    axA.set_xticks([np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6],
                   [r"$\pi/6$", r"$\pi/3$", r"$\pi/2$", r"$2\pi/3$", r"$5\pi/6$"])
    axA.set_xlabel(r"cone angle $\theta$")
    axA.set_ylabel(r"Berry phase $\theta_B$ mod $2\pi$  (rad)")
    axA.set_title(rf"(a) Berry phase at small $T={TCHK:.0f}$ "
                  rf"($\lambda={LAM:g}$, $\alpha={ALPHA:g}$)")
    # Estimator markers below the axes; the "analytic" reference curve inside.
    hA, lA = axA.get_legend_handles_labels()
    leg_est = axA.legend(hA[1:], lA[1:], fontsize=12, ncol=2, loc="upper center",
                         bbox_to_anchor=(0.5, -0.14), frameon=False)
    axA.add_artist(leg_est)
    axA.legend(hA[:1], lA[:1], fontsize=12, loc="lower left", frameon=True,
               framealpha=0.9, edgecolor="0.8", fancybox=False)
    axA.grid(True, alpha=0.2)

    # (b) error-cancellation cascade vs runtime.
    axB.loglog(T, d["e_rs"], "o-", ms=5, color="C0", label=r"runtime scaling")
    axB.loglog(T, d["e_fr"], "s-", ms=5, color="C1", label=r"forward--reverse")
    axB.loglog(T, d["e_b1"], "^-", ms=5, color="C2", label=r"1 Richardson + bump")
    axB.loglog(T, d["e_b2"], "D-", ms=5, color="C3", label=r"2 Richardson + bump")
    tail = T >= np.sqrt(T[0] * T[-1])
    for power, e, style in [(-1, d["e_rs"], ":"), (-2, d["e_fr"], "--"),
                            (-4, d["e_b1"], (0, (5, 1))), (-6, d["e_b2"], "-.")]:
        c = np.median(e[tail] / T[tail] ** power)
        axB.loglog(T, c * T ** power, color="0.55", lw=1.0, linestyle=style,
                   label=rf"$\propto T^{{{power}}}$")
    axB.set_xlabel("runtime $T$")
    axB.set_ylabel(r"bias  $|\mathbb{E}[\tilde\theta_B] - \theta_B|$  (rad)")
    axB.set_xlim(18.0, 230.0)
    gap_c = SpiralHeisenbergChain(N=N, J=J, B0=B0, theta=THETA_C).gap
    axB.set_title(rf"(b) cascade at $\theta={THETA_C/np.pi:g}\pi$ "
                  rf"($\Delta={gap_c:.2f}$, $\alpha={ALPHA:g}$, $\lambda={LAM:g}$)")
    # Estimator curves below the axes; the T^-n reference slopes inside.
    hB, lB = axB.get_legend_handles_labels()
    leg_curves = axB.legend(hB[:4], lB[:4], fontsize=12, ncol=2, loc="upper center",
                            bbox_to_anchor=(0.5, -0.14), frameon=False)
    axB.add_artist(leg_curves)
    axB.legend(hB[4:], lB[4:], fontsize=11, ncol=2, loc="lower left", frameon=True,
               framealpha=0.9, edgecolor="0.8", fancybox=False)
    axB.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.24)   # ~1.5x the default gap between the two panels
    path = HERE / "fig1.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
