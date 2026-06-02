r"""Compare runtime-randomization distributions for the deterministic bias.

Richardson removes the non-oscillatory ``T^-2`` term; runtime randomization then
suppresses the *oscillatory* residual by the decay of the distribution's
characteristic function (CF):

  * uniform  (CF ~ k^-1) -> oscillatory bias ~ T^-3 (still oscillatory)
  * triangle (CF ~ k^-2) -> oscillatory bias ~ T^-4
  * C^inf bump (CF decays faster than any power) -> oscillatory residual killed
    super-polynomially, so the bias is set by the *non-oscillatory* Richardson
    residual T^-2(levels+1): smooth, and improvable with more Richardson levels.

This is why a *second* Richardson level helps only for the bump: for uniform /
triangle the oscillatory residual, not the non-oscillatory one, sets the floor.

Run with:  uv run python experiments/fig_distributions.py
Writes:    figures/distributions.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from berry_cancellation import SpinHalfLoop
from berry_cancellation.estimators import randomized_richardson_bias

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def main() -> None:
    model = SpinHalfLoop()
    T = np.geomspace(8.0, 100.0, 36)
    alpha = 2.0

    curves = [
        ("uniform, 1 Richardson", dict(levels=1, dist="uniform"), "C3", "d"),
        ("triangle, 1 Richardson", dict(levels=1, dist="triangle"), "C4", "v"),
        ("bump, 1 Richardson", dict(levels=1, dist="bump"), "C0", "o"),
        ("bump, 2 Richardson", dict(levels=2, dist="bump"), "C1", "s"),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    results = {}
    for label, kw, color, mk in curves:
        print(f"computing {label} ...")
        e = randomized_richardson_bias(model, T, alpha=alpha, lam=0.5, **kw)
        results[label] = e
        ax.loglog(T, e, marker=mk, ms=4, color=color, label=label)

    # Reference slopes, each anchored to its matching curve over the large-T
    # (asymptotic) region so the guide sits on the data rather than floating.
    tail = T > 30.0
    refs = [
        (-3, "uniform, 1 Richardson", ":"),
        (-4, "triangle, 1 Richardson", "--"),
        (-6, "bump, 2 Richardson", "-."),
    ]
    for power, key, style in refs:
        c = np.median(results[key][tail] / T[tail] ** power)
        ax.loglog(T, c * T ** power, color="0.6", lw=1.0, linestyle=style,
                  label=rf"$\propto T^{{{power}}}$")

    ax.set_xlim(7.0, 115.0)
    ax.set_xlabel("runtime $T$")
    ax.set_ylabel(r"randomization bias  $|\mathbb{E}_X[\tilde\theta_{B,R}]-\theta_B|$  (rad)")
    ax.set_title("Runtime-randomization distributions: deterministic bias\n"
                 "(spin-1/2 cone loop)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "distributions.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
