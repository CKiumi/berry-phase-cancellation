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
| 2 Richardson + bump randomization | `O(T⁻⁶)` | `C^∞` bump kills the oscillatory residual super-polynomially; the 2nd Richardson lowers the non-oscillatory floor to `T⁻⁶` |

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

## Many-qubit model

To show the cancellation isn't special to two levels, `manybody.py` provides a
genuinely **entangled `N`-qubit** loop — a Heisenberg chain in a *spiral* rotating
field,

```
H(s) = -Σᵢ Bᵢ(s)·Sᵢ + J Σ_⟨ij⟩ Sᵢ·Sⱼ,   Bᵢ(s) on the cone with offset φᵢ = 2πi/N.
```

The per-site offsets break the total-spin symmetry, so the ground state is
entangled and the virtual-excitation term `𝒜⁽²⁾` (the `n,k≠0` sum) is **nonzero** —
the piece a two-level or spin-`S` model can't exercise. Because every site's
azimuth advances by the same `2πs`, the loop is the rigid rotation
`H(s) = R(s)H(0)R(s)†`, which gives an exact closed-form propagator
`U_T(1) = R(1)e^{−i(T H(0) − 2π Sᶻ_tot)}` (no integrator) and an analytic Berry
phase `θ_B = 2π⟨Sᶻ_tot⟩₀` (cross-checked against the Wilson loop). `fig_manybody.py`
runs the *same* estimators on this model (`N=4`) and reproduces the
`T⁻¹ → T⁻² → T⁻⁴` cascade — confirming the cancellation in a real many-body state.

## Layout

```
berry_cancellation/
  hamiltonians.py   spin-1/2 cone-loop model (H(s), ground state, analytic θ_B)
  manybody.py       non-trivial N-qubit model: Heisenberg chain in a spiral field
  evolution.py      batched 4th-order Magnus propagator (+ dispatch to a model's
                    own exact amplitudes, used by the many-body model)
  reference.py      Wilson-loop Berry phase, dynamical phase, angle wrapping
  estimators.py     single / forward–reverse / Richardson / randomized errors
experiments/
  fig_scaling.py          main figure: error vs T for all four estimators
  fig_spin_half_check.py  Berry phase vs analytic half-solid-angle
  fig_manybody.py         same cancellation cascade on an entangled chain;
                          takes an optional qubit count: `fig_manybody.py 8`
  fig_manybody_check.py   many-body Berry phase vs cone angle (analytic/Wilson/FR)
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
uv run python experiments/fig_manybody.py   # -> figures/manybody.png (4-qubit model)
uv run python experiments/fig_manybody.py 8 # -> figures/manybody_N8.png (8 qubits, slow)
uv run python experiments/fig_manybody_check.py   # many-body Berry phase check
uv run pytest                               # checks references + scaling slopes
```

`figures/scaling.png` is the headline plot (`T ∈ [8, 100]`, 20 points, `λ = 0.7`,
chosen by a roughness scan as the smoothest while the `T⁻⁴` slope stays faithful):
four curves whose log–log envelopes follow `T⁻¹` (single), `T⁻²` (forward–reverse),
`T⁻⁴` (1 Richardson + `C^∞` bump), and `T⁻⁶` (2 Richardson + bump). Each Richardson
level cancels the next non-oscillatory term; the bump (CF decays faster than any
power) suppresses the oscillatory residual super-polynomially. With one level the
non-oscillatory `T⁻⁴` floor dominates → a smooth curve. With two levels the floor
drops to `T⁻⁶`, which falls *below* the residual oscillation, so that curve is
oscillation-dominated (the dips are sign changes). The worst-case runtime is
`T(1+λ)αˡᵉᵛᵉˡˢ` (≈298 for 1 level, ≈521 for 2).

