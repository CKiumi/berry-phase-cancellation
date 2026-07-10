# Adiabatic error cancellation in Berry phase estimation — paper figures

Each `figN.py` in this folder is a standalone script that produces the
corresponding figure of the paper (as `figN.pdf` / `figN.png` next to itself).
The reusable physics lives in the `berry_cancellation` package (loop models +
estimators); these scripts only drive it and plot. Run from the repo root, e.g.

```sh
uv run python paper/fig1.py
```

The cancellation cascade on a closed adiabatic loop:

| estimator | leading error |
|---|---|
| single evolution | $O(T^{-1})$ |
| forward&ndash;reverse (averages $\pm H$) | $O(T^{-2})$, oscillatory |
| + Richardson extrapolation | removes the non-oscillatory $T^{-2}$ |
| + runtime randomization ($C^\infty$ bump) | suppresses the oscillatory residual |

Each model figure combines, in one panel pair, the **Berry phase at a small
runtime** (left, in the canonical mod-$2\pi$ window: the analytic curve, the
runtime-scaling reconstruction of arXiv:2509.13423 shown as raw values modulo
$2\pi$, the forward&ndash;reverse estimate, and shot-based Richardson + bump
estimates) and the **error-cancellation cascade vs runtime** (right). Two
models: a spin-$1/2$ in a cone field (exactly solvable) and a spiral Heisenberg
chain (genuinely entangled many-qubit). A separate figure checks the analytic error
bounds and identifies which spectral gap controls each estimator.

## Figure 1 (`fig1.py`). Spin-1/2 cone loop: small-runtime accuracy and the cancellation cascade

As a controlled testbed we use a spin-$1/2$ in a field of fixed magnitude swept
around a cone of half-angle $\theta_0$ on the Bloch sphere,
$H(s)=-\tfrac{1}{2}\mathbf{B}(s)\!\cdot\!\boldsymbol{\sigma}$, whose ground-state
Berry phase is known exactly, $\theta_B=-\pi(1-\cos\theta_0)$. Panel (a) shows
everything in the canonical mod-$2\pi$ window $[0,2\pi)$: the estimators
determine $\theta_B$ only modulo $2\pi$, so the runtime-scaling points are raw
values modulo $2\pi$ with no branch input, while the mod-$\pi$ estimators
(forward&ndash;reverse and Richardson) are shown on the branch nearer the
analytic value. At the *small, cheap* runtime $T=8$ the runtime-scaling
reconstruction (arXiv:2509.13423; two forward evolutions, the second runtime
chosen adaptively from the dynamical-phase scale) carries the full
$O(1)$-radian adiabatic error and wanders off the curve; the forward&ndash;reverse estimate is
systematically off at the $10^{-1}$-rad level; a *single-shot*
two-Richardson + bump estimate &mdash; one randomly drawn runtime &mdash; lies
on the analytic curve. The shaded band is the fully a priori Theorem-3
prediction
$\mathbb{E}[\hat\theta] \pm 2L_2(\alpha)\sqrt{\mathbb{E}[X^{-4}]}\,\|\dot H(0)\|^2/(\Delta(0)^4T^2)$,
centered on the deterministic mean $\mathbb{E}[\hat\theta]$ (bias included);
here $\|\dot H(0)\|=\pi\sin\theta_0$, $\Delta(0)=1$, $L_2(1.75)\approx0.470$,
$\sqrt{\mathbb{E}[X^{-4}]}\approx1.63$. The single shot falls inside the band
at 17 of 18 angles &mdash; the $\sim$95% coverage expected of a two-sigma band
whose amplitude estimates hold with equality for the spin-$1/2$ loop. Panel (b)
then traces the full cancellation cascade versus runtime at $\theta_0=0.4\pi$
(gap $\Delta=1$): a single traversal errs at $O(T^{-1})$; averaging the $\pm H$
eigenphases cancels the dynamical phase and the leading term, leaving
$O(T^{-2})$; a Richardson extrapolation removes the non-oscillatory $T^{-2}$
part and runtime randomization over a $C^\infty$ bump suppresses the oscillatory
residual super-polynomially, improving the scaling one level at a time to $O(T^{-4})$
and $O(T^{-6})$.

**Caption.** Spin-$1/2$ cone loop. **(a)** Berry phase $\theta_B$ (canonical
window $[0,2\pi)$) versus cone half-angle $\theta_0$ at a fixed small runtime
$T=8$: the exact analytic value $-\pi(1-\cos\theta_0)$ mod $2\pi$ (line); the
runtime-scaling reconstruction from two forward evolutions (the second runtime
chosen adaptively from the dynamical-phase scale), shown as raw values modulo
$2\pi$ ($\circ$; arXiv:2509.13423 &mdash; its residual adiabatic error is
$\sim1$ rad at this runtime); the
forward&ndash;reverse estimate ($\square$; determined modulo $\pi$ and shown on
the branch nearer the analytic value); and a single-shot two-Richardson + bump
estimate from one randomly drawn runtime ($\diamond$). Shaded band: the a
priori Theorem-3 prediction
$\mathbb{E}[\hat\theta]\pm2L_2(\alpha)\sqrt{\mathbb{E}[X^{-4}]}\,\|\dot H(0)\|^2/(\Delta(0)^4T^2)$;
the single shot lies within the band at 17 of 18 angles.
**(b)** Error-cancellation cascade versus runtime ($\theta_0=0.4\pi$, $\Delta=1$):
single evolution ($\circ$, $\propto T^{-1}$), forward&ndash;reverse ($\square$,
$\propto T^{-2}$), one Richardson level + bump randomization ($\triangle$,
$\propto T^{-4}$) and two levels ($\diamond$, $\propto T^{-6}$); grey lines are
reference slopes, $\lambda=0.7$, $\alpha=1.75$. The two-level curve sits at its
$T^{-6}$ floor and is oscillation-limited (its dips are zero crossings of the signed
bias).

## Figure 2 (`fig2.py`). Theory bounds, and which gap controls each estimator (spin-1/2)

To check the analytic error bounds and, in particular, to identify *which* spectral
gap governs each stage of the cascade, we deform the loop into a non-isospectral one
by modulating the field magnitude, $|B(s)| = |B|\,(1 - a\sin^2\pi s)$ with $a=0.4$.
This lowers the gap in the *interior* of the loop to $\Delta_{\min}=0.6$ while
leaving the *endpoint* quantities &mdash; the gap $\Delta(0)=1$ and the rate
$\|\dot H(0)\|$ &mdash; unchanged, and it leaves the Berry phase invariant
(panel a). The single-evolution error is set by the interior worst-case gap:
$|\varphi|\lesssim \|\dot H\|_{\max}^2/(\Delta_{\min}^3\,T)$, so the dip raises both
the numerics and (more steeply) the bound (panel b). The Richardson residual, in
contrast, is an endpoint-controlled boundary term
$\sim\|\dot H(0)\|^2/(\Delta(0)^4 T^2)$:
the dipped and undipped data fall on the *same* line, independent of the
interior gap (panel c). Finally, the one-Richardson estimator with *uniform*
runtime randomization is compared against the Theorem-3 oscillatory estimate at
decay order $M=1$,
$C\,\|\dot H(0)\|^2/(\Delta(0)^4\,\Delta_{\min}\,T^3)$.
The uniform distribution is chosen deliberately: its characteristic function
decays only as $\xi^{-1}$ ($M=1$), so the randomized oscillatory bias falls as
$T^{-3}$ and *dominates* the non-oscillatory Richardson remainder $O(T^{-4})$
that a single extrapolation level cannot remove &mdash; the data itself
exhibits the Theorem-3 oscillatory scaling, in both the power of $T$ and the
gap dependence, rather than hiding it below the non-oscillatory floor (as a
smoother distribution would). The constant is fully *a priori*: one
integration by parts gives $|\chi_{\mu,2}(\xi)|\le C_1/|\xi|$ for all $\xi$,
with $C_1=(1-\lambda)^{-2}/\lambda\approx15.9$ at $\lambda=0.7$ (endpoint jumps
plus the interior $x^{-3}$ term), and $C=C_1L_{1,3}(\alpha)\approx12.1$ with
the Richardson factor
$L_{1,3}(\alpha)=\sum_\ell |w_{1,\ell}|\,\alpha^{-3\ell}\approx0.76$:
the extrapolated $k=2$ modes oscillate at frequencies $\alpha^\ell\omega_n$, so
the $M=1$ decay contributes an extra $\alpha^{-\ell}$ per level on top of the
$\alpha^{-2\ell}$ sector weight. No constant is fitted. Both loops respect the
bound over the whole range, with the oscillatory envelope reaching
$\approx0.8$ of it (panel d).

**Caption.** Numerics against the analytic bounds, and the controlling gap for each
estimator, on a non-isospectral spin-$1/2$ loop with field modulation
$|B(s)|=|B|(1-a\sin^2\pi s)$, $a=0.4$. (a) Instantaneous gap $\Delta(s)$: constant
($a=0$, dashed) versus dipped to $\Delta_{\min}=0.6$ in the interior (solid), with
the endpoints fixed at $\Delta(0)=1$. (b) Single-evolution error with the worst-case
bound $\|\dot H\|_{\max}^2/(\Delta_{\min}^3 T)$ for the dipped and undipped loops:
the error tracks the *interior* gap $\Delta_{\min}$. (c) One-Richardson residual:
dipped and undipped data collapse onto the same endpoint-controlled line
$\|\dot H(0)\|^2/(\Delta(0)^4 T^2)$, independent of $\Delta_{\min}$. (d) One
Richardson with uniform runtime randomization and the Theorem-3 $M=1$
oscillatory bound
$C\,\|\dot H(0)\|^2/(\Delta(0)^4\Delta_{\min} T^3)$
with the exact a priori constant $C=C_1L_{1,3}(\alpha)\approx12.1$,
$C_1=(1-\lambda)^{-2}/\lambda\approx15.9$ at $\lambda=0.7$,
$L_{1,3}(\alpha)=\sum_\ell|w_{1,\ell}|\alpha^{-3\ell}\approx0.76$; no constant
is fitted. The uniform distribution ($M=1$) makes the randomized oscillatory bias
$O(T^{-3})$ dominate the non-oscillatory Richardson remainder $O(T^{-4})$, so
the data directly track the bound's $T^{-3}$ power and its gap dependence.
$T\in[20,400]$.

## Figure 3 (`fig3.py`). Entangled many-qubit loop: small-runtime accuracy and the cancellation cascade

We repeat the programme on a genuinely interacting model: a four-site Heisenberg
chain
$H(s)=-\sum_i\mathbf{B}_i(s)\!\cdot\!\mathbf{S}_i+J\sum_{\langle ij\rangle}\mathbf{S}_i\!\cdot\!\mathbf{S}_j$
($J=1$) with a site-dependent azimuthal offset
$\phi_i=2\pi i/N$ (a "spiral"), which breaks the global spin-rotation symmetry so the
ground state is entangled and the virtual-excitation contribution to the phase error
is nonzero &mdash; a true many-body test. The rigid global rotation underlying the
loop gives an exact propagator and a closed-form Berry phase
$\theta_B=2\pi\langle S^z_{\rm tot}\rangle_0$ (taken as the reference). Panel (a),
in the same canonical mod-$2\pi$ window, plots the Berry phase versus cone
angle at a small runtime $T=10$ (the gap varies over $[0.11,0.42]$ across the
sweep). At this runtime the runtime-scaling error exceeds $\pi$, so its raw
mod-$2\pi$ points scatter across the window; the forward&ndash;reverse estimate
is visibly off; the 1- and 2-Richardson + bump estimates &mdash; each the mean
of $N=10$ randomly sampled runtimes, with $\pm2\,$s.e.m. error bars estimated
from the same shots &mdash; track both branches of the wrapped analytic curve
(the interference amplitudes at each runtime are evaluated exactly; measurement
shot noise is not simulated). The bars capture statistical uncertainty only:
near the gap minimum the deterministic adiabatic bias dominates and is visible
as a small offset from the curve. No Theorem-3 band is drawn here: for this
entangled model the closed-form constant $\|\dot H(0)\|^2/\Delta(0)^4$
overestimates the actual per-shot fluctuation by 2&ndash;4 orders of magnitude
(it replaces every excited-level gap $\Delta_n(0)$ by the smallest one and the
ground-state couplings by the operator norm), so the operationally accessible
bound lies far above the plot scale. Panel (b) applies the same four
estimators unchanged and reproduces the full hierarchy
$O(T^{-1})\to O(T^{-2})\to O(T^{-4})\to O(T^{-6})$
at $\theta=0.4\pi$ (gap $=0.37$); the smaller gap enlarges the
prefactors, so we take $T\in[20,200]$ and a wider window $\lambda=0.8$
($\lambda\Delta_{\min}T\gtrsim1$). The cancellation thus persists for a genuinely
entangled many-qubit adiabatic loop.

**Caption.** Entangled spiral Heisenberg chain ($N=4$, $J=1$, $B_0=1$). **(a)**
Berry phase $\theta_B$ (canonical window $[0,2\pi)$) versus cone angle $\theta$
at a fixed small runtime $T=10$: the exact analytic value
$2\pi\langle S^z_{\rm tot}\rangle_0$ (wrapped line, cut at the window
boundary); the runtime-scaling reconstruction shown as raw values modulo $2\pi$
($\circ$; arXiv:2509.13423 &mdash; its error exceeds $\pi$ at this runtime);
the forward&ndash;reverse estimate ($\square$; modulo $\pi$, branch nearer the
analytic value); and the 1- and 2-Richardson + bump estimates
($\triangle$/$\diamond$), each the mean of $N=10$ randomly sampled runtimes
with $\pm2\,$s.e.m. bars estimated from the same shots. The bars are
statistical only; near the gap minimum the deterministic bias is visible. The
closed-form Theorem-3 constant is 2&ndash;4 orders of magnitude conservative
for this entangled model, so no a priori band is drawn (cf. Fig. 1(a)).
**(b)** Error-cancellation cascade versus runtime ($\theta=0.4\pi$,
gap $=0.37$, $\theta_B=1.4530$) with the same four estimators as in Fig. 1(b)
($\circ\,T^{-1}$, $\square\,T^{-2}$, $\triangle\,T^{-4}$, $\diamond\,T^{-6}$; grey
reference slopes), $\alpha=1.75$, $\lambda=0.8$. The smaller gap pushes the clean
$T^{-6}$ regime to larger $T$.

---

## Numerical notes — full record of the 2026-07-02 experiments

### Infrastructure

- All heavy evolutions run through `ProcessPoolExecutor(max_workers=10)`, split
  per runtime / per model with explicit step counts so results are identical to
  the batched evaluation. Full recompute: fig1 ~10 s, fig2 ~40 s, fig3 ~6 s.
- Results are cached in `cache/figN.npz`, keyed by the `PARAMS` dict at the top
  of each script; any physics-parameter change triggers a recompute, while
  styling-only edits replot in ~1 s.
- Fonts: `text.usetex` (Computer Modern, matches REVTeX), base 15 pt,
  ticks 13 pt; legends outside (below the axes), 12 pt.
- Spin-$1/2$ loop: 4th-order Magnus integrator, 20 steps per unit runtime
  ($\sim10^{-10}$ accuracy). Heisenberg chain: exact rotating-frame propagator
  (single `expm`), so shots are cheap.

### Display conventions (both model figures)

- Everything is drawn in the canonical mod-$2\pi$ window $[0,2\pi)$; the
  analytic curve is wrapped (discontinuity cut with NaN).
- Runtime scaling determines $\theta_B$ mod $2\pi$: points are raw outputs
  modulo $2\pi$, no reference to the true value. Its raw output contains an
  arbitrary dynamical-phase winding ($\sim-30$ rad for the chain), so a
  mod-$2\pi$ reduction is unavoidable for display; when its error exceeds
  $\pi$ the displayed error is aliased (understated).
- The second runtime of runtime scaling is chosen *operationally*:
  $\alpha_{\rm rs}=1+\tfrac{\pi/2}{T\|H\|}$ using only the known energy
  scale ($|\theta_D|\le T\|H\|$); the earlier implementation used the exact
  $\theta_D$, which is unknown in an experiment, and flattered the method
  (fig1: identical since $\|H\|=|E_0|$ for spin-$1/2$; fig3: slightly worse).
- Forward--reverse and Richardson estimators are mod-$\pi$; they are shown on
  the branch nearer the analytic value (the algorithmic branch-resolution step
  is not simulated).
- A "shot" = one randomly drawn runtime $T_j=TX_j$ (bump density, fixed seed 0);
  the interference amplitudes at each runtime are evaluated *exactly* --
  projective-measurement shot noise is not simulated. One 2R shot costs six
  evolutions ($T_j,\alpha T_j,\alpha^2T_j$, forward and reverse).

### Fig. 1 (spin-1/2, $T=8$, $\lambda=0.7$, $\alpha=1.75$, 18 angles, seed 0)

- Red diamond = a *single* 2R+bump shot per angle (median dev $6.3\times10^{-3}$ rad).
- Band $=\mathbb{E}[\hat\theta]\pm2L_2(\alpha)\sqrt{\mathbb{E}[X^{-4}]}\,\|\dot H(0)\|^2/(\Delta(0)^4T^2)$,
  with $L_2(1.75)=0.470$ (recursive weights, Eq. (19)),
  $\sqrt{\mathbb{E}[X^{-4}]}=1.628$ (bump, $\lambda=0.7$; quadrature),
  $\|\dot H(0)\|=\pi\sin\theta_0$, $\Delta(0)=1$; band half-width
  0.05--0.24 rad. Centered on the deterministic mean (2R bias, median
  $6.4\times10^{-3}$) -- centering on $\theta_B$ would be wrong since the
  bound controls fluctuations about $\mathbb{E}[\hat\theta]$.
- Coverage: 17/18 angles inside the $2\sigma$ band (94%, textbook for a
  saturated bound: all amplitude estimates are equalities for spin-$1/2$).
- Runtime study (single-shot protocol vs $T$): $T=8$ clean everywhere;
  $T=5$ marginal (one edge angle off); $T=4$ edge angles break and FR branch
  wraps fail; $T=2$ non-adiabatic collapse. Rules of thumb: phase expansion is
  perturbative for $T\gtrsim\pi^2\sin^2\theta_0\approx10$; randomization
  samples down to $(1-\lambda)T$, so effectively $(1-\lambda)\Delta T\gtrsim2$;
  and the rotating-frame gap dips to $\Delta\sin\theta_0$ for shots near
  $T_j\approx2\pi|\cos\theta_0|/\Delta$, which is why sweep edges misbehave
  first (their per-shot std exceeds the leading $k=2$ bound by up to $\sim3\times$
  at $T=8$).

### Fig. 2 (non-isospectral spin-1/2, $a=0.4$, $T\in[20,400]$, 2x2 layout)

- Panel (d) uses *uniform* randomization ($\lambda=0.7$, $m=1$) deliberately.
  The bump's large-$T$ floor is the **non-oscillatory** Richardson remainder
  $E[X^{-4}]\,\Phi_4/(\alpha^2T^4)$, *not* the Theorem-3 oscillatory term:
  identified by distribution tests -- bias(bump $\lambda=0.35$)/bias(bump
  $\lambda=0.7$) $=0.46\approx E[X^{-4}]$-ratio $0.462$, triangle/bump
  $\approx1.2\approx$ its $E[X^{-4}]$ ratio, slope exactly $-4.00$ -- so a
  bump panel cannot test the oscillatory bound (its CF decay
  $\sim e^{-c\sqrt{\lambda\omega T}}$ kills the oscillatory sector by
  $T\sim100$).
- With uniform, the oscillatory $T^{-3}$ dominates the non-oscillatory
  $T^{-4}$, and the constant is exact and a priori:
  $|\chi_{\mu,2}(\xi)|\le C_1/|\xi|$ for all $\xi$ with
  $C_1=(1-\lambda)^{-2}/\lambda=15.87$ (endpoint jumps + interior $x^{-3}$
  term). The asymptotic endpoint prefactor
  $((1-\lambda)^{-2}+(1+\lambda)^{-2})/(2\lambda)=8.18$ is *not* an all-$\xi$
  bound: the data cross it at small $T$ (up to $1.5\times$) and graze it
  mid-range ($1.005$ at $T\approx130$; the comparison is saturated for this
  model), so the strict constant is used.
- Richardson factor: the extrapolated $k=2$ modes oscillate at
  $\alpha^\ell\omega_n$, giving $L_{1,3}(\alpha)=\sum_\ell|w_{1,\ell}|\alpha^{-3\ell}=0.762$
  (tighter than the frequency-agnostic $L_{1,2}=2/(\alpha^2-1)=0.970$).
  Total $C=C_1L_{1,3}=12.09$; max data/bound $=0.77$ ($a=0$), $0.58$ ($a=0.4$).
- $\Delta_{\min}$-scaling check (dips $a=0,0.4,0.6$): plateau constants under
  the $\Delta_{\min}^{-2}$ (bump/M=2) normalization were 38.7 / 33.8 / ~9
  (unconverged), i.e. the deep-dip behaviour is *weaker* than
  $\Delta_{\min}^{-2}$ in this $T$ range -- the norm-form gap dependence is
  safe (conservative) on the dipped side.

### Fig. 3 (Heisenberg chain, $T=10$, $\lambda=0.8$, $\alpha=1.75$, $N=10$ shots, seed 0)

- 1R and 2R points share the same sampled runtimes; error bars are
  $\pm2\,$s.e.m. with $s$ the sample std (ddof=1) of the shots (empirical,
  statistical only; nominal coverage $\approx93\%$ at $N=10$ via Student-$t$,
  $\approx95\%$ for $N\ge100$).
- Coverage was verified at $N=100$: $\mathbb{E}[\hat\theta]$ (deterministic
  value) lay inside mean$\pm2\,$s.e.m. at 17/18 angles. Bias/statistics
  crossover at $T=20$ sits near $N=100$ (median $|$bias$|\approx$ median
  $|$stat residual$|\approx6\times10^{-4}$); at the gap-minimum angle
  ($\theta/\pi\approx0.81$, $T=20$) the deterministic bias is $0.19$ rad and
  dominates -- the statistical bars cannot show bias.
- Why no a priori band: the closed-form Theorem-3 constant
  $\|\dot H(0)\|^2/\Delta(0)^4$ (values $5.3\times10^3$--$2.0\times10^5$
  across the sweep; $\|\dot H(0)\|=2\pi\|[S^z_{\rm tot},H_0]\|=5.7$--$12.5$,
  the Heisenberg term commutes with the rotation) overestimates the exact
  coefficient $\sum_n|\langle n|\dot H(0)|0\rangle|^2/\Delta_n(0)^4$
  (values 5--167) by a factor 450--13,300: it replaces every level gap by the
  smallest one and the ground-state couplings by the operator norm. The
  exact-$B_n$ band would be nearly saturated at the hardest angles
  (bound/measured std $=0.9$--$28$ with the $\sqrt{\mathbb{E}[X^{-4}]}$
  factor), but evaluating it requires diagonalization, which contradicts the
  premise that the model is classically hard -- hence rejected. Shot counts
  cannot fix this: the bound/actual ratio is $N$-independent, and putting the
  norm-form band on scale would need $N\sim10^5$--$10^7$.
- $\lambda$ is a weak lever for the variance band: it enters only through
  $\sqrt{\mathbb{E}[X^{-4}]}$ ($2.02$ at $\lambda=0.8$ $\to$ $1.08$ at
  $\lambda=0.3$, at most a factor 2), while the deterministic bias grows
  $\sim20\times$ (still negligible at large $T$); the strong levers are $T$
  ($T^{-2}$) and $N$ ($N^{-1/2}$).
