"""Figure 1: Spin-1/2 cone loop -- small-runtime accuracy and the cancellation cascade.

(a) Berry phase (canonical mod-2pi window) vs cone half-angle at a small, cheap
runtime (T = 8): the analytic curve, the runtime-scaling reconstruction shown
as raw values modulo 2pi (no branch input), the forward-reverse estimate (mod
pi, shown on the branch nearer the analytic value), and a SINGLE 2-fold
Richardson + bump estimate from one randomly drawn runtime. The shaded band is
the fully a priori Theorem-3 prediction E[theta_hat] +- 2 x std bound,
2 L_2(alpha) sqrt(E[X^-4]) |Hdot(0)|^2 / (Delta(0)^4 T^2), centered on the
deterministic mean (bias included).
(b) Error-cancellation cascade vs runtime for the four estimators.

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

from berry_cancellation import SpinHalfLoop
from berry_cancellation.estimators import (
    single_phase_error, forward_reverse_error, randomized_richardson_bias,
    runtime_scaling_theta_B, _recursive_richardson_error,
    _theta_B_forward_reverse, _theta_B_randomized, default_steps,
)
from berry_cancellation.reference import wrap_to_pi, wrap_to_half_pi

LAM, ALPHA = 0.7, 1.75
TCHK = 8.0
N_THETA = 18
N_SHOTS, SEED = 1, 0
T_MIN, T_MAX, N_T = 8.0, 100.0, 20
PARAMS = dict(lam=LAM, alpha=ALPHA, tchk=TCHK, n_theta=N_THETA,
              n_shots=N_SHOTS, seed=SEED, rs_alpha="hmax-scale", panel_a="rs+fr+2r-1shot+2sigma",
              t_min=T_MIN, t_max=T_MAX, n_t=N_T)
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


def richardson_weights(m, alpha):
    "Recursive-Richardson weights w_{m,l} (paper Eq. (19))."
    x = alpha ** (-2.0 * np.arange(m + 1))
    w = np.empty(m + 1)
    for l in range(m + 1):
        others = np.delete(x, l)
        w[l] = np.prod(-others / (x[l] - others))
    return w


def hdot_norm(model, s, h=1e-6):
    return np.linalg.norm(model.H(s + h) - model.H(s - h), 2) / (2 * h)


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
    thetas = np.linspace(0.15 * np.pi, 0.85 * np.pi, N_THETA)
    models = [SpinHalfLoop(theta0=float(th)) for th in thetas]
    model = SpinHalfLoop()
    T = np.geomspace(T_MIN, T_MAX, N_T)
    steps = default_steps(T.max())

    with ProcessPoolExecutor(max_workers=10) as ex:
        # (a) per-model small-T estimates.
        fut_rs = [ex.submit(runtime_scaling_theta_B, m, TCHK) for m in models]
        fut_fr = [ex.submit(_theta_B_forward_reverse, m, np.array([TCHK]),
                            default_steps(TCHK)) for m in models]
        # 10-shot Monte-Carlo 2-fold Richardson estimate: per model, N_SHOTS
        # runtimes sampled from the bump density.
        rng = np.random.default_rng(SEED)
        shots = [TCHK * sample_bump(N_SHOTS, LAM, rng) for _ in models]
        fut_2r = [ex.submit(_recursive_richardson_error, m, r, ALPHA, 2,
                            default_steps(ALPHA**2 * r.max()))
                  for m, r in zip(models, shots)]
        # Deterministic (infinite-shot) randomized value: the estimator's mean
        # E[theta_hat] = theta_B + bias, used as the center of the Theorem-3
        # std band.
        fut_det = [ex.submit(_theta_B_randomized, m, TCHK, alpha=ALPHA, lam=LAM,
                             levels=2, dist="bump") for m in models]
        # (b) per-runtime cascade curves (explicit steps keep the split
        # identical to the batched evaluation).
        fut_s = [ex.submit(single_phase_error, model, np.array([t]), steps) for t in T]
        fut_f = [ex.submit(forward_reverse_error, model, np.array([t]), steps) for t in T]
        fut_1 = [ex.submit(randomized_richardson_bias, model, np.array([t]),
                           alpha=ALPHA, lam=LAM, levels=1, dist="bump") for t in T]
        fut_2 = [ex.submit(randomized_richardson_bias, model, np.array([t]),
                           alpha=ALPHA, lam=LAM, levels=2, dist="bump") for t in T]
        rs_raw = np.array([f.result() for f in fut_rs])
        fr_raw = np.array([f.result()[0] for f in fut_fr])
        mc_err = np.array([f.result() for f in fut_2r])   # (N_THETA, N_SHOTS) signed errors
        det_raw = np.array([f.result()[0] for f in fut_det])
        e_single = np.concatenate([f.result() for f in fut_s])
        e_fr = np.concatenate([f.result() for f in fut_f])
        e_b1 = np.concatenate([f.result() for f in fut_1])
        e_b2 = np.concatenate([f.result() for f in fut_2])

    return dict(thetas=thetas, analytic=np.array([m.berry_phase for m in models]),
                rs_raw=rs_raw, fr_raw=fr_raw, mc_err=mc_err, det_raw=det_raw,
                hdot0=np.array([hdot_norm(m, 0.0) for m in models]),
                gap0=np.array([m.gap_at(0.0) for m in models]), T=T,
                e_single=e_single, e_fr=e_fr, e_b1=e_b1, e_b2=e_b2)


def main():
    d = cached_compute(CACHE, PARAMS, compute)
    thetas, analytic, T = d["thetas"], d["analytic"], d["T"]
    est_rs = analytic + wrap_to_pi(d["rs_raw"] - analytic)          # theta_B mod 2 pi
    est_fr = analytic + wrap_to_half_pi(d["fr_raw"] - analytic)     # theta_B mod pi
    bias_det = wrap_to_half_pi(d["det_raw"] - analytic)              # E[theta_hat] - theta_B
    est_1r = analytic + d["mc_err"].mean(axis=1)                     # 10-shot mean (mod pi)
    # Theorem-3 standard-deviation bound for the randomized m=2 Richardson
    # estimator: L_m(alpha) sqrt(E[X^-4]) |Hdot(0)||Hdot(1)| / (Delta(0)^4 T^2 sqrt(N)),
    # with Hdot(0) = Hdot(1) for the cone loop. Fully a priori, per cone angle;
    # E[X^-4] is evaluated for the bump density by quadrature.
    w2 = richardson_weights(2, ALPHA)
    L2 = np.sum(np.abs(w2) * ALPHA ** (-2.0 * np.arange(3)))
    u = np.linspace(-1.0 + 1e-12, 1.0 - 1e-12, 400001)
    rho_u = np.exp(-1.0 / (1.0 - u**2))
    E4 = np.trapezoid(rho_u * (1.0 + LAM * u) ** -4, u) / np.trapezoid(rho_u, u)
    std_bound = (L2 * np.sqrt(E4) * d["hdot0"] ** 2
                 / d["gap0"] ** 4 / TCHK**2 / np.sqrt(N_SHOTS))   # N-shot mean
    print(f"  L_2(alpha)={L2:.3f}, sqrt(E[X^-4])={np.sqrt(E4):.3f}, "
          f"std-bound range [{std_bound.min():.4f}, {std_bound.max():.4f}] rad")
    print(f"  T={TCHK:.0f}: runtime-scaling median dev={np.median(np.abs(est_rs - analytic)):.1e}, "
          f"2R+bump {N_SHOTS}-shot median dev={np.median(np.abs(est_1r - analytic)):.1e}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.6, 5.2))

    # (a) Berry phase vs cone angle at a small, cheap runtime. The Theorem-3
    # std bound is drawn as a continuous band around the analytic curve, on a
    # fine angle grid (all quantities are analytic: Hdot(0) = pi sin(theta),
    # Delta(0) = 1).
    th_f = np.linspace(thetas[0], thetas[-1], 400)
    models_f = [SpinHalfLoop(theta0=float(th)) for th in th_f]
    analytic_f = np.array([m.berry_phase for m in models_f])
    hdot0_f = np.array([hdot_norm(m, 0.0) for m in models_f])
    gap0_f = np.array([m.gap_at(0.0) for m in models_f])
    band_f = (L2 * np.sqrt(E4) * hdot0_f ** 2
              / gap0_f ** 4 / TCHK**2 / np.sqrt(N_SHOTS))
    # Center the band on the estimator's true mean E[theta_hat] = theta_B +
    # bias (deterministic randomized value, interpolated across the sweep),
    # not on theta_B itself: Theorem 3 bounds the fluctuation about the mean.
    # Everything displayed in the canonical mod-2pi window [0, 2pi): the
    # runtime-scaling points are raw values modulo 2pi (no branch input); the
    # mod-pi estimators (FR, 2R) are shown on the branch nearer the analytic
    # value. For this loop the wrapped analytic curve has no discontinuity
    # inside the sweep.
    twopi = 2.0 * np.pi
    center_f = analytic_f + np.interp(th_f, thetas, bias_det)
    axA.fill_between(th_f, (center_f % twopi) - 2 * band_f,
                     (center_f % twopi) + 2 * band_f,
                     color="C3", alpha=0.18, lw=0,
                     label=r"theoretical $\pm2\sigma$ bound (single shot)")
    axA.plot(th_f, analytic_f % twopi, "-", lw=2, color="0.4", label="analytic")
    axA.plot(thetas, d["rs_raw"] % twopi, "o", ms=6, color="C0",
             label="runtime scaling")
    axA.plot(thetas, est_fr % twopi, "s", ms=6, color="C1", label=r"forward--reverse")
    axA.plot(thetas, est_1r % twopi, "D", ms=6, color="C3",
             label=r"2 Richardson + bump (1 shot)")
    axA.set_ylim(0, twopi)
    axA.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, twopi],
                   [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    axA.set_xticks([np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6],
                   [r"$\pi/6$", r"$\pi/3$", r"$\pi/2$", r"$2\pi/3$", r"$5\pi/6$"])
    axA.set_xlabel(r"cone angle $\theta_0$")
    axA.set_ylabel(r"Berry phase $\theta_B$ mod $2\pi$  (rad)")
    axA.set_title(rf"(a) Berry phase at small $T={TCHK:.0f}$")
    axA.legend(fontsize=12, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14),
               frameon=False)
    axA.grid(True, alpha=0.2)

    # (b) error-cancellation cascade vs runtime.
    axB.loglog(T, d["e_single"], "o-", ms=5, color="C0", label=r"single  $|\varphi|$")
    axB.loglog(T, d["e_fr"], "s-", ms=5, color="C1", label=r"forward--reverse")
    axB.loglog(T, d["e_b1"], "^-", ms=5, color="C2", label=r"1 Richardson + bump")
    axB.loglog(T, d["e_b2"], "D-", ms=5, color="C3", label=r"2 Richardson + bump")
    tail = T >= np.sqrt(T[0] * T[-1])
    for power, e, style in [(-1, d["e_single"], ":"), (-2, d["e_fr"], "--"),
                            (-4, d["e_b1"], (0, (5, 1))), (-6, d["e_b2"], "-.")]:
        c = np.median(e[tail] / T[tail] ** power)
        axB.loglog(T, c * T ** power, color="0.55", lw=1.0, linestyle=style,
                   label=rf"$\propto T^{{{power}}}$")
    axB.set_xlabel("runtime $T$")
    axB.set_ylabel(r"phase error  $|\tilde\theta_B - \theta_B|$  (rad)")
    axB.set_xlim(7.0, 115.0)
    axB.set_title(rf"(b) cancellation cascade  ($\alpha={ALPHA:g}$, $\lambda={LAM:g}$)")
    axB.legend(fontsize=12, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    axB.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    path = HERE / "fig1.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
