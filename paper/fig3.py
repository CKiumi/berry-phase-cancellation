"""Figure 3: End-to-end simulation of the Hadamard-test Berry-phase algorithm.

Runs on the *same problem as Fig. 1* -- the entangled spiral Heisenberg chain
(N = 4, J = 1, B0 = 1) with identical conditions (base runtime T = 20, bump
window lambda = 0.6, Richardson ratio alpha = 1.7, 30-angle sweep) -- but now
simulates the full sampling-based algorithm of Sec. VI.B, in the smoother variant
allowed by Remark 1 (two Richardson levels + C-infinity bump runtime
randomization), *including* projective measurement shot noise -- unlike Figs. 1
and 2, where the interference amplitudes are evaluated exactly.

Pipeline per estimate:
  1. Coarse branch resolution: the integer-ladder branch lift of Sec. VI.B,
     Theta_I = -2*phi(T1) + 5*phi(2 T1) - 2*phi(4 T1) (mod 2pi) = theta_B +
     O(T1^-2).  Integer coefficients cancel the dynamical phase (and the leading
     1/T adiabatic term) with no unwrapping and no T*H_max amplification.  The
     coarse base runtime is additionally bump-randomized (N_COARSE samples, each
     from M_COARSE Hadamard shots per ladder node): averaging the ladder output
     over runtimes suppresses its oscillatory residual, resolving every branch at
     a smaller T1 (= 15) than the fixed-runtime ladder (~20) on this many-body loop.
  2. For each trial j = 1..N: draw X_j from the C-infinity bump density on
     [1-lam, 1+lam] (inverse-CDF on the grid), T_j = T X_j.
  3. Single-shot Hadamard estimates z_hat = b_re + i b_im (b = +-1) for the
     six propagators U and U_hat at T_j, alpha T_j, alpha^2 T_j
     -- 12 measurement shots per trial.
  4. Forward-reverse interference signals at the three runtime scales:
     W_l = mean_j z_hat_f(alpha^l T_j) z_hat_r(alpha^l T_j) (the dynamical
     phase cancels sample-by-sample in the product, and the two measurements
     are independent, so W_l is unbiased for E_X[z_f z_r]);
     theta_tilde_l = arg(W_l)/2 (mod pi), lifted to the pi-interval centered
     at the coarse estimate.
  5. Two-fold Richardson: theta_hat = sum_l w_{2,l} theta_tilde_l^lift, mod 2pi.

Full pipeline vs cone angle in the canonical mod-2pi window: the plotted point is
the mean of R_A independent algorithm runs (N trials each), and the error bar is
the +-1 sigma spread of a single run.  The estimates lie on the analytic curve.

The exact amplitudes are precomputed once on a fine runtime grid (the Monte
Carlo then only draws grid indices and measurement bits), parallelized over
angles and cached in cache/fig3.npz.
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
from berry_cancellation.evolution import loop_amplitudes
from berry_cancellation.estimators import default_steps
from berry_cancellation.reference import wrap_to_pi

# Same problem as Fig. 1: the entangled spiral Heisenberg chain, same conditions.
N, J, B0 = 4, 1.0, 1.0
LAM, ALPHA = 0.6, 1.7
T = 20.0
N_THETA = 30
THETA_C = 0.4 * np.pi
K_GRID = 2048                    # runtime-grid points for the exact amplitudes
N_A = 200                        # randomized trials per algorithm run
R_A = 24                         # independent runs per angle (for the +-1 sigma bar)
# Coarse branch resolution via the integer-ladder of Sec. VI.B (Eq. (55) instance
# p = 1): Theta_I = -2*phi(T1) + 5*phi(2 T1) - 2*phi(4 T1)  (mod 2pi) = theta_B +
# O(T1^-2).  Integer coefficients cancel the dynamical phase and the leading 1/T
# adiabatic term with NO unwrapping and NO T*H_max amplification (noise multiplier
# sqrt(33)), so the branch is resolved reliably at a small T1 -- unlike the
# nearby-runtime runtime-scaling step, whose 1/(alpha-1) = 2 T1 H_max/pi blow-up
# fails at the deepest-gap angles of this many-body loop.
COARSE_RATIOS = np.array([1.0, 2.0, 4.0])
COARSE_COEFFS = np.array([-2.0, 5.0, -2.0])
# The coarse base runtime is itself bump-randomized (same window lambda): averaging
# the ladder output over runtimes suppresses its oscillatory O(T1^-2) residual,
# which lets T1 drop from ~20 (fixed) to 15 while still resolving every branch.
T1 = 15.0
N_COARSE, M_COARSE = 128, 200   # randomized coarse samples, Hadamard shots per sample-node
SEED = 0
PARAMS = dict(model="spiral-heisenberg", N=N, J=J, B0=B0,
              lam=LAM, alpha=ALPHA, t=T, n_theta=N_THETA,
              est="2R+bump", k_grid=K_GRID, n_a=N_A, r_a=R_A,
              coarse="integer-ladder-124-randomized", t1=T1,
              n_coarse=N_COARSE, m_coarse=M_COARSE, seed=SEED)
CACHE = HERE / "cache" / "fig3.npz"


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


def amplitude_grid(model):
    """Exact survival amplitudes on the runtime grid, plus the coarse-ladder nodes.

    Returns ``zs`` -- the (forward, reverse) amplitude arrays on the X grid at the
    three Richardson runtime scales -- and ``zc_f`` -- the forward amplitudes at
    the integer-ladder coarse runtimes ``T1, 2 T1, 4 T1``.
    """
    X = np.linspace(1.0 - LAM, 1.0 + LAM, K_GRID)
    runtimes = np.concatenate([T * X, ALPHA * T * X, ALPHA**2 * T * X])
    steps = default_steps(ALPHA**2 * (1.0 + LAM) * T)
    zf, zr = loop_amplitudes(model, runtimes, steps)
    coarse_rt = np.concatenate([q * T1 * X for q in COARSE_RATIOS])
    zc_f, _ = loop_amplitudes(model, coarse_rt,
                              default_steps(COARSE_RATIOS.max() * (1.0 + LAM) * T1))
    zs = [(zf[l * K_GRID:(l + 1) * K_GRID], zr[l * K_GRID:(l + 1) * K_GRID])
          for l in range(3)]
    zc = [zc_f[l * K_GRID:(l + 1) * K_GRID] for l in range(len(COARSE_RATIOS))]
    return (zs, zc)


def hadamard_bits(z, n, rng):
    "n single-shot Hadamard estimates b_re + i b_im (b = +-1) of amplitude(s) z."
    z = np.asarray(z)
    p_re = np.clip((1.0 + z.real) / 2.0, 0.0, 1.0)
    p_im = np.clip((1.0 + z.imag) / 2.0, 0.0, 1.0)
    b_re = 2.0 * (rng.random(n) < p_re) - 1.0
    b_im = 2.0 * (rng.random(n) < p_im) - 1.0
    return b_re + 1j * b_im


def hadamard_mean(z, m, rng):
    "Per-element mean of m single-shot Hadamard estimates; z shape (K,) -> (K,)."
    z = np.asarray(z)
    p_re = np.clip((1.0 + z.real) / 2.0, 0.0, 1.0)
    p_im = np.clip((1.0 + z.imag) / 2.0, 0.0, 1.0)
    b_re = 2.0 * (rng.random((m, z.size)) < p_re) - 1.0
    b_im = 2.0 * (rng.random((m, z.size)) < p_im) - 1.0
    return b_re.mean(0) + 1j * b_im.mean(0)


def coarse_estimate(zc, rng):
    "Randomized integer-ladder branch lift: mean over bump runtimes of the ladder."
    idx = bump_indices(N_COARSE, rng)                         # X_k ~ bump (grid-sampled)
    phi = [np.angle(hadamard_mean(z[idx], M_COARSE, rng)) for z in zc]
    theta_k = wrap_to_pi(sum(c * p for c, p in zip(COARSE_COEFFS, phi)))
    return float(np.angle(np.mean(np.exp(1j * theta_k))))     # circular mean, theta_B + O(T1^-2)


def lift(x_mod_pi, center):
    "Representative of x (defined mod pi) in the pi-interval centered at center."
    return x_mod_pi + np.pi * np.round((center - x_mod_pi) / np.pi)


def richardson_weights(m, alpha):
    "Recursive-Richardson weights w_{m,l} (paper Eq. (19))."
    x = alpha ** (-2.0 * np.arange(m + 1))
    w = np.empty(m + 1)
    for l in range(m + 1):
        others = np.delete(x, l)
        w[l] = np.prod(-others / (x[l] - others))
    return w


# Inverse CDF of the bump density on the X grid (for trial sampling).
_UG = np.linspace(-1.0 + 1e-12, 1.0 - 1e-12, K_GRID)
_PDF = np.exp(-1.0 / (1.0 - _UG**2))
_CDF = np.cumsum(_PDF)
_CDF /= _CDF[-1]


def bump_indices(n, rng):
    "n grid indices distributed as the C-infinity bump density."
    return np.searchsorted(_CDF, rng.random(n))


W2 = richardson_weights(2, ALPHA)


def run_pipeline(grid, n_trials, rng):
    "One full estimate: coarse + N randomized Hadamard trials + 2-fold Richardson."
    zs, zc = grid
    coarse = coarse_estimate(zc, rng)
    idx = bump_indices(n_trials, rng)                 # X_j ~ bump (grid-sampled)
    theta = 0.0
    for l, (zf_l, zr_l) in enumerate(zs):
        W_l = np.mean(hadamard_bits(zf_l[idx], n_trials, rng)
                      * hadamard_bits(zr_l[idx], n_trials, rng))
        theta += W2[l] * lift(np.angle(W_l) / 2.0, coarse)
    return theta                                      # compare mod 2 pi


def compute():
    thetas = np.linspace(0.15 * np.pi, 0.85 * np.pi, N_THETA)
    models = [SpiralHeisenbergChain(N=N, J=J, B0=B0, theta=float(th)) for th in thetas]

    with ProcessPoolExecutor(max_workers=10) as ex:
        grids = list(ex.map(amplitude_grid, models))

    rng = np.random.default_rng(SEED)
    analytic = np.array([m.berry_phase for m in models])

    # R_A independent full-pipeline runs per angle: the plotted point is their mean,
    # lifted onto the branch of the mean, and the +-1 sigma error bar is the spread
    # of a single run (std over the R_A runs).
    est_mean = np.empty(N_THETA)
    est_std = np.empty(N_THETA)
    for i, g in enumerate(grids):
        runs = np.array([run_pipeline(g, N_A, rng) for _ in range(R_A)])
        centered = analytic[i] + wrap_to_pi(runs - analytic[i])   # common branch
        est_mean[i] = centered.mean()
        est_std[i] = centered.std(ddof=1)
    return dict(thetas=thetas, analytic=analytic,
                est_mean=est_mean, est_std=est_std)


def main():
    d = cached_compute(CACHE, PARAMS, compute)
    thetas, analytic = d["thetas"], d["analytic"]
    twopi = 2.0 * np.pi

    dev = np.abs(wrap_to_pi(d["est_mean"] - analytic))
    print(f"  N={N_A} trials/run, {R_A} runs/angle: mean-dev median={np.median(dev):.2e}, "
          f"max={dev.max():.2e} rad; median 1-sigma={np.median(d['est_std']):.2e} rad")

    fig, (axA, axG) = plt.subplots(
        2, 1, figsize=(7.2, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08})

    # Full pipeline vs cone angle, canonical mod-2pi window, +-1 sigma error bars.
    th_f = np.linspace(thetas[0], thetas[-1], 400)
    models_f = [SpiralHeisenbergChain(N=N, J=J, B0=B0, theta=float(t)) for t in th_f]
    ana_f = np.array([m.berry_phase for m in models_f]) % twopi
    gap_f = np.array([m.gap for m in models_f])
    jump = np.abs(np.diff(ana_f)) > np.pi
    ana_f[1:][jump] = np.nan

    axA.plot(th_f, ana_f, "-", lw=2, color="0.4", label="analytic")
    axA.errorbar(thetas, d["est_mean"] % twopi, yerr=d["est_std"], fmt="D", ms=6,
                 color="C3", capsize=3, lw=1.2,
                 label="Hadamard-test algorithm\n"
                       rf"($N={N_A:d}$ trials, $\pm1\sigma$)")
    axA.set_ylim(0, twopi)
    axA.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, twopi],
                   [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    axA.set_ylabel(r"Berry phase $\theta_B$ mod $2\pi$  (rad)")
    axA.set_title(rf"Hadamard-test algorithm: spiral Heisenberg chain "
                  rf"($N={N}$, $T={T:.0f}$)")
    axA.legend(fontsize=11, loc="lower left", frameon=True, framealpha=0.9,
               edgecolor="0.8", fancybox=False)
    axA.grid(True, alpha=0.2)

    # Bottom strip: the instantaneous gap Delta(theta) -- context for the error-bar
    # size (the deepest-gap angles fluctuate most).
    axG.plot(th_f, gap_f, "-", lw=1.8, color="C0")
    axG.fill_between(th_f, 0.0, gap_f, color="C0", alpha=0.12)
    axG.set_ylim(0.0, 0.47)
    axG.set_yticks([0.1, 0.2, 0.3, 0.4])
    axG.set_ylabel(r"gap $\Delta(\theta)$")
    axG.set_xticks([np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6],
                   [r"$\pi/6$", r"$\pi/3$", r"$\pi/2$", r"$2\pi/3$", r"$5\pi/6$"])
    axG.set_xlabel(r"cone angle $\theta$")
    axG.grid(True, alpha=0.2)

    fig.tight_layout()
    path = HERE / "fig3.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
