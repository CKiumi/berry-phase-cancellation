r"""Actual runtime randomization by Monte Carlo (10,000 shots).

Unlike ``fig_scaling.py`` -- whose randomization curve is the deterministic bias
``E_X[theta_R] - theta_B`` evaluated by quadrature (the infinite-shot limit) --
this experiment draws real random runtimes ``T_j = T X_j``, evaluates the
single-event Richardson estimator at each, and averages.  It therefore exhibits
both effects the paper separates:

  * the deterministic bias ``~ O(T^-3)`` (what the sample mean converges to), and
  * the statistical floor ``~ T^-2 N^-1/2`` from the residual oscillatory sector.

Left panel  : sample-mean error vs runtime ``T`` at ``N = 10000`` shots, with
              +/- standard-error bars, against the quadrature bias and the floor.
Right panel : convergence of the running sample mean vs shot count at a fixed
              ``T``, showing the ``N^-1/2`` approach to the bias.

Run with:  uv run python experiments/fig_randomization_montecarlo.py
Writes:    figures/randomization_montecarlo.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from berry_cancellation import SpinHalfLoop
from berry_cancellation.estimators import (
    _richardson_event_errors,
    randomized_richardson_bias,
    randomized_richardson_montecarlo,
)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

N_SHOTS = 10000
ALPHA = 2.0
LAM = 0.5


def main() -> None:
    model = SpinHalfLoop()
    print(f"model: spin-1/2 cone loop, theta_B={model.berry_phase:.6f}")
    print(f"runtime randomization: uniform on [{1-LAM}, {1+LAM}], "
          f"N={N_SHOTS} shots, alpha={ALPHA}")

    # --- Left panel: error vs T at fixed N ----------------------------------
    T = np.geomspace(8.0, 150.0, 13)
    print("running Monte Carlo sweep over T ...")
    mean_err, sem, event_std = randomized_richardson_montecarlo(
        model, T, n_shots=N_SHOTS, alpha=ALPHA, lam=LAM, seed=0
    )
    print("computing deterministic bias (quadrature) for reference ...")
    bias = randomized_richardson_bias(model, T, alpha=ALPHA, lam=LAM, n_nodes=129)
    floor = event_std / np.sqrt(N_SHOTS)  # statistical floor ~ T^-2 N^-1/2

    # --- Right panel: convergence vs shot count at fixed T ------------------
    T_fix = 60.0
    rng = np.random.default_rng(7)
    x = rng.uniform(1.0 - LAM, 1.0 + LAM, N_SHOTS)
    steps = int(max(1500, np.ceil(20.0 * ALPHA * T_fix * (1.0 + LAM))))
    err_events = _richardson_event_errors(model, T_fix, x, ALPHA, steps)
    n = np.arange(1, N_SHOTS + 1)
    running_mean = np.cumsum(err_events) / n
    bias_fix = randomized_richardson_bias(
        model, np.array([T_fix]), alpha=ALPHA, lam=LAM, n_nodes=257
    )[0]
    sd_fix = np.std(err_events, ddof=1)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Left.
    axL.errorbar(T, np.abs(mean_err), yerr=sem, fmt="o", ms=5, capsize=3,
                 label=f"MC sample mean ($N={N_SHOTS}$)  $\\pm$ SEM")
    axL.loglog(T, bias, "-", color="C1",
               label=r"deterministic bias $|\mathbb{E}_X[\tilde\theta_{B,R}]-\theta_B|$")
    axL.loglog(T, floor, "--", color="C3",
               label=r"statistical floor $\propto T^{-2}N^{-1/2}$")
    T0 = T[0]
    axL.loglog(T, bias[0] * (T / T0) ** -3, ":", color="0.5", lw=1.0,
               label=r"$\propto T^{-3}$")
    axL.set_xlabel("runtime $T$")
    axL.set_ylabel(r"$|\langle\tilde\theta_{B,R}\rangle - \theta_B|$")
    axL.set_title(f"Runtime randomization: error vs $T$  ($N={N_SHOTS}$ shots)")
    axL.legend(fontsize=8)
    axL.grid(True, which="both", alpha=0.2)

    # Right.
    axR.loglog(n, np.abs(running_mean), color="C0", lw=1.0,
               label=f"running sample mean ($T={T_fix:.0f}$)")
    axR.axhline(abs(bias_fix), color="C1", ls="-",
                label=r"deterministic bias")
    axR.loglog(n, sd_fix / np.sqrt(n), "--", color="C3",
               label=r"$\propto N^{-1/2}$ (event std $/\sqrt{N}$)")
    axR.set_xlabel("number of shots $N$")
    axR.set_ylabel(r"$|\langle\tilde\theta_{B,R}\rangle_N - \theta_B|$")
    axR.set_title(f"Convergence to the bias vs shot count  ($T={T_fix:.0f}$)")
    axR.legend(fontsize=8)
    axR.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "randomization_montecarlo.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"  bias(T={T_fix:.0f}) = {abs(bias_fix):.3e}, "
          f"final running-mean error = {abs(running_mean[-1]):.3e}")


if __name__ == "__main__":
    main()
