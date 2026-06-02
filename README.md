# Berry phase cancellation — numerics

Numerical demonstration of the **adiabatic error-cancellation** mechanism for
Berry phase estimation studied in *"Adiabatic Error Cancellation in Berry Phase
Estimation"*. The code shows, on an exactly solvable spin-1/2 model, how the
finite-runtime adiabatic error of a Berry-phase estimate is suppressed step by
step:

| Estimator | Leading error | Mechanism |
|---|---|---|
| Single evolution | `O(T⁻¹)` | bare adiabatic phase error `φ` |
| Forward–reverse | `O(T⁻²)` | evolve under `+H` then `−H`; cancels `θ_D` **and** the `O(T⁻¹)` term |
| + Richardson (α=2) | `O(T⁻²)` | removes the non-oscillatory `T⁻²` part |
| + uniform randomization | `O(T⁻³)` | averages out the oscillatory `T⁻²` residual (CF `~ k⁻¹`) |
| + triangle randomization | `O(T⁻⁴)` | smoother distribution (CF `~ k⁻²`) suppresses it one power further |

Here `T` is the runtime (slower sweep = larger `T` = smaller error).

## The model

A spin-1/2 in a magnetic field swept around a cone on the Bloch sphere,

```
H(s) = -(1/2) B(s)·σ,   B(s) = |B| (sinθ₀ cos2πs, sinθ₀ sin2πs, cosθ₀),   s ∈ [0,1].
```

The loop is closed (`H(0)=H(1)`) and smooth at the seam (`Ḣ(0)=Ḣ(1)`), satisfying
the assumptions of the cancellation and Richardson theorems. The gap is constant
(`Δ = |B|`) and the ground-state Berry phase is known in closed form,

```
θ_B = -Ω/2 = -π(1 - cosθ₀),
```

where `Ω` is the solid angle subtended by the loop. This exact value is the
ground truth the estimators are compared against (and is independently
cross-checked by a gauge-invariant Wilson loop).

## How the estimators work

The forward and reverse loop evolutions give survival amplitudes

```
⟨ψ₀|U_T(1)|ψ₀⟩      ~ exp(i(−θ_D + θ_B + φ)),
⟨ψ₀|Û_T(1)|ψ₀⟩      ~ exp(i(+θ_D + θ_B + φ̂)).
```

Averaging the two eigenphases cancels the dynamical phase `±θ_D` and the leading
`O(T⁻¹)` adiabatic error, leaving `θ̃_B = θ_B + O(T⁻²)` (defined mod π). A
two-runtime Richardson extrapolant at `T` and `αT` removes the non-oscillatory
`T⁻²` term; averaging that extrapolant over a uniform runtime distribution
`T_j = T·X`, `X ∈ [1−λ, 1+λ]`, suppresses the remaining oscillatory `T⁻²` term
by one further power of `1/T`.

The reported randomization curves are the **deterministic bias**
`|E_X[θ̃_{B,R}] − θ_B|`, the infinite-shot limit evaluated by quadrature over the
runtime distribution. Richardson removes the non-oscillatory `T⁻²` term;
averaging then suppresses the residual oscillatory `T⁻²` term by the decay of the
distribution's characteristic function — one power of `1/T` for the **uniform**
distribution (CF `~ k⁻¹`, giving `T⁻³`), two for the **triangle** (CF `~ k⁻²`,
giving `T⁻⁴`). Note the extra power comes from the *smoother distribution*, not
from an additional Richardson level: a second extrapolation does not help here
because the oscillatory residual — not the non-oscillatory one — sets the floor.

## Layout

```
berry_cancellation/
  hamiltonians.py   spin-1/2 cone-loop model (H(s), ground state, analytic θ_B)
  evolution.py      batched 4th-order Magnus propagator for 2×2 systems
  reference.py      Wilson-loop Berry phase, dynamical phase, angle wrapping
  estimators.py     single / forward–reverse / Richardson / randomized errors
experiments/
  fig_scaling.py          main figure: error vs T for all four estimators
  fig_spin_half_check.py  Berry phase vs analytic half-solid-angle
  fig_distributions.py    randomization bias for uniform/triangle/bump + recursive
                          Richardson (shows when an extra Richardson level helps)
tests/
  test_cancellation.py    references, unitarity, integrator convergence, slopes
```

The propagator is a fully vectorised closed-form 2×2 Magnus integrator, so a
whole batch of runtimes is evolved at once — this is what makes the
runtime-randomization sweeps (hundreds of runtimes per base `T`) cheap. The
integrator error is verified to sit far below the adiabatic error it measures.

## Running

```bash
uv sync                                     # set up the environment
uv run python experiments/fig_scaling.py    # -> figures/scaling.png
uv run python experiments/fig_spin_half_check.py
uv run pytest                               # checks references + scaling slopes
```

`figures/scaling.png` is the headline plot: curves whose log–log envelopes follow
`T⁻¹` (single), `T⁻²` (forward–reverse, Richardson), and `T⁻⁴` (triangle
randomization). The forward–reverse and Richardson curves oscillate under their
`T⁻²` envelope because the residual oscillates at frequency `ω = ∫Δ ds`; the
triangle-distribution randomization averages this away to `T⁻⁴`. (Uniform
randomization gives `T⁻³` — still oscillatory, since averaging only suppresses
the oscillatory residual by one power; it is available via the `dist` argument
but omitted from the plot.)
