r"""Standard deviation of the runtime-randomized estimator vs T.

We randomize the runtime as ``T_j = T X_j`` (``X_j ~ Uniform[1-lam, 1+lam]``) and
average the single-event Richardson estimator over ``N`` shots.  Rather than the
(noisy, sign-changing) bias point estimate, we report the well-defined statistical
spread of the resulting estimate -- its standard deviation
``sigma_N = std / sqrt(N)`` -- which is positive-definite and follows a clean
``~T^-2`` power law on a log-log axis (so the variance scales as ``~T^-4``).

Run with:  uv run python experiments/fig_randomization_montecarlo.py
Writes:    figures/randomization_montecarlo.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from berry_cancellation import SpinHalfLoop
from berry_cancellation.estimators import randomized_richardson_montecarlo

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

N_SHOTS = 1000
ALPHA = 2.0
LAM = 0.5


def main() -> None:
    model = SpinHalfLoop()
    print(f"model: spin-1/2 cone loop, theta_B={model.berry_phase:.6f}")
    print(f"runtime randomization: uniform on [{1-LAM}, {1+LAM}], "
          f"N={N_SHOTS} shots, alpha={ALPHA}")

    T = np.geomspace(1.0, 150.0, 20)
    print("running Monte Carlo sweep over T ...")
    # sem == std of the N-shot randomized estimate (sigma_N).
    _, sigma_N, _ = randomized_richardson_montecarlo(
        model, T, n_shots=N_SHOTS, alpha=ALPHA, lam=LAM, seed=0
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.loglog(T, sigma_N, "d-", ms=5, color="C3",
              label=rf"std of randomized estimator ($N={N_SHOTS}$)")
    # T^-2 reference guide, anchored at the asymptotic (large-T) end.
    ax.loglog(T, sigma_N[-1] * (T / T[-1]) ** -2, "--", color="0.5", lw=1.0,
              label=r"$\propto T^{-2}$")

    ax.set_xlim(0.9, 170.0)
    ax.set_xlabel("runtime $T$")
    ax.set_ylabel(r"std of $\tilde\theta_{B,R}$  (rad)")
    ax.set_title(f"Runtime randomization: estimator standard deviation "
                 f"($N={N_SHOTS}$ shots)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "randomization_montecarlo.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
