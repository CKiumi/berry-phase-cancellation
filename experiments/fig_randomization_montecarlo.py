r"""Actual runtime randomization by Monte Carlo (10,000 shots), error vs T.

Draws real random runtimes ``T_j = T X_j`` (``X_j ~ Uniform[1-lam, 1+lam]``),
evaluates the single-event Richardson estimator at each (exact per draw), and
averages.  The sample mean exhibits both effects the paper separates:

  * the deterministic bias ``~ O(T^-3)`` (what the sample mean converges to), and
  * the statistical floor ``~ T^-2 N^-1/2`` from the residual oscillatory sector.

Run with:  uv run python experiments/fig_randomization_montecarlo.py
Writes:    figures/randomization_montecarlo.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from berry_cancellation import SpinHalfLoop
from berry_cancellation.estimators import (
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

    T = np.geomspace(8.0, 150.0, 13)
    print("running Monte Carlo sweep over T ...")
    mean_err, sem, event_std = randomized_richardson_montecarlo(
        model, T, n_shots=N_SHOTS, alpha=ALPHA, lam=LAM, seed=0
    )
    print("computing deterministic bias (quadrature) for reference ...")
    bias = randomized_richardson_bias(model, T, alpha=ALPHA, lam=LAM, n_nodes=129)
    floor = event_std / np.sqrt(N_SHOTS)  # statistical floor ~ T^-2 N^-1/2

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.errorbar(T, np.abs(mean_err), yerr=sem, fmt="o", ms=5, capsize=3,
                color="C0",
                label=f"MC sample mean ($N={N_SHOTS}$)  $\\pm$ SEM")
    ax.loglog(T, bias, "-", color="C1",
              label=r"deterministic bias $|\mathbb{E}_X[\tilde\theta_{B,R}]-\theta_B|$")
    ax.loglog(T, floor, "--", color="C3",
              label=r"statistical floor $\propto T^{-2}N^{-1/2}$")
    T0 = T[0]
    ax.loglog(T, bias[0] * (T / T0) ** -3, ":", color="0.5", lw=1.0,
              label=r"$\propto T^{-3}$")
    ax.set_xlabel("runtime $T$")
    ax.set_ylabel(r"$|\langle\tilde\theta_{B,R}\rangle - \theta_B|$")
    ax.set_title(f"Runtime randomization: error vs $T$  ($N={N_SHOTS}$ shots)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "randomization_montecarlo.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
