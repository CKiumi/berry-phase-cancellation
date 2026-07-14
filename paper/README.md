# Adiabatic error cancellation in Berry phase estimation — paper figures

Each `figN.py` in this folder is a standalone script that produces the
corresponding figure of the paper (as `figN.pdf` next to itself). The reusable
physics lives in the `berry_cancellation` package (loop models + estimators);
these scripts only drive it and plot. Run from the repo root, e.g.

```sh
uv run python paper/fig1.py
```

Each script caches its expensive evolutions in `cache/figN.npz`, keyed by the
`PARAMS` dict at the top: a styling-only edit replots in ~1 s, while any
physics-parameter change triggers a recompute.

## The cancellation cascade

On a closed adiabatic loop the Berry-phase estimators form a hierarchy:

| estimator | leading error |
|---|---|
| single evolution | $O(T^{-1})$ |
| forward&ndash;reverse (averages $\pm H$) | $O(T^{-2})$, oscillatory |
| + Richardson extrapolation | removes the non-oscillatory $T^{-2}$ |
| + runtime randomization ($C^\infty$ bump) | suppresses the oscillatory residual |

giving $O(T^{-4})$ after one Richardson level + bump and $O(T^{-6})$ after two.

## The figures

- **Fig. 1 (`fig1.py`)** &mdash; the method on a genuinely entangled many-qubit
  loop (spiral Heisenberg chain, $N=4$): small-runtime Berry-phase accuracy
  (left) and the full cascade versus runtime (right).
- **Fig. 2 (`fig2.py`)** &mdash; the analytic error bounds, and *which* spectral
  gap controls each stage, on a non-isospectral spin-$1/2$ loop.
- **Fig. 3 (`fig3.py`)** &mdash; the end-to-end Hadamard-test algorithm (with
  projective shot noise and the integer-ladder branch lift) on the same
  many-body loop as Fig. 1.

The exactly-solvable spin-$1/2$ cone-loop testbed (small-runtime accuracy +
cascade) was retired from the main sequence; its script is kept under
`old/fig1.py` and described at the end.

## Figure 1 (`fig1.py`). Entangled many-qubit loop: small-runtime accuracy and the cancellation cascade

We run the programme on a genuinely interacting model: a four-site Heisenberg
chain
$H(s)=-\sum_i\mathbf{B}_i(s)\!\cdot\!\mathbf{S}_i+J\sum_{\langle ij\rangle}\mathbf{S}_i\!\cdot\!\mathbf{S}_j$
($J=1$, $B_0=1$) with a site-dependent azimuthal offset $\phi_i=2\pi i/N$ (a
"spiral"), which breaks the global spin-rotation symmetry so the ground state is
entangled and the virtual-excitation contribution to the phase error is nonzero
&mdash; a true many-body test. The rigid global rotation underlying the loop
gives an exact propagator and a closed-form Berry phase
$\theta_B=2\pi\langle S^z_{\rm tot}\rangle_0$ (the reference). One consistent
randomization parameter set is used for the whole figure: bump window
$\lambda=0.6$, Richardson ratio $\alpha=1.7$.

Panel (a), in the canonical mod-$2\pi$ window $[0,2\pi)$, plots the Berry phase
versus cone angle over $\theta\in[0.15,0.85]\pi$ (30 angles; the gap varies over
$[0.11,0.42]$) at a small runtime $T=20$: the exact analytic value
$2\pi\langle S^z_{\rm tot}\rangle_0$ (wrapped, cut at the window boundary); the
runtime-scaling reconstruction shown as raw values modulo $2\pi$ ($\circ$;
arXiv:2509.13423 &mdash; its error exceeds $\pi$, so its points scatter across
the window); the forward&ndash;reverse estimate ($\square$; mod $\pi$); and the
1- and 2-Richardson + bump estimates ($\triangle$/$\diamond$), each the mean of
$N=10$ randomly sampled runtimes with $\pm2\,$s.e.m. bars from the same shots
(the interference amplitudes are evaluated exactly; shot noise is not
simulated). $T=20$ keeps the *entire* sweep &mdash; including the deepest-gap
endpoints ($\theta/\pi\approx0.2,0.8$, $\Delta\approx0.11$) &mdash; inside the
adiabatic window $T\Delta\gtrsim2$, so the two-Richardson estimate tracks the
curve everywhere; at smaller $T$ those endpoints fall into the non-adiabatic
regime where the extrapolation overshoots. No Theorem-3 band is drawn: for this
entangled model the closed-form norm constant $\|\dot H(0)\|^2/\Delta(0)^4$
overestimates the actual per-shot fluctuation by 2&ndash;4 orders of magnitude
(it replaces every excited-level gap by the smallest one and the ground-state
couplings by the operator norm), so the operationally accessible bound lies far
above the plot scale.

Panel (b) traces the full cancellation cascade versus runtime at $\theta=0.39\pi$
(gap $\Delta=0.36$) over $T\in[20,200]$: the naive runtime-scaling reconstruction
($\propto T^{-1}$, the same method as panel a),
forward&ndash;reverse ($\propto T^{-2}$), one Richardson level + bump
($\propto T^{-4}$) and two levels ($\propto T^{-6}$). The angle $\theta=0.39\pi$
was chosen over a 10-angle sweep as the one where the two-Richardson curve holds
its $T^{-6}$ slope most smoothly at the shared $\lambda=0.6$; the maximally
symmetric angle $\theta=\pi/2$ is excluded because there the forward&ndash;reverse
error vanishes identically and the whole cascade collapses to machine precision.

**Caption.** Entangled spiral Heisenberg chain ($N=4$, $J=1$, $B_0=1$;
$\lambda=0.6$, $\alpha=1.7$). **(a)** Berry phase $\theta_B$ (canonical window
$[0,2\pi)$) versus cone angle $\theta$ at a fixed small runtime $T=20$ (30
angles, gap $\in[0.11,0.42]$): the exact analytic value (wrapped line); the
runtime-scaling reconstruction ($\circ$; raw values mod $2\pi$, error $>\pi$);
the forward&ndash;reverse estimate ($\square$; mod $\pi$); and the 1- and
2-Richardson + bump estimates ($\triangle$/$\diamond$; 10-shot mean
$\pm2\,$s.e.m.). $T=20$ keeps the whole sweep adiabatic ($T\Delta\gtrsim2$), so
the two-Richardson estimate tracks the curve everywhere. **(b)**
Error-cancellation cascade versus runtime ($\theta=0.39\pi$, $\Delta=0.36$,
$T\in[20,200]$): runtime scaling ($\circ$, $\propto T^{-1}$), forward&ndash;reverse
($\square$, $\propto T^{-2}$), one Richardson + bump ($\triangle$,
$\propto T^{-4}$) and two ($\diamond$, $\propto T^{-6}$); grey lines are
reference slopes.

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

## Figure 3 (`fig3.py`). End-to-end Hadamard-test algorithm

We simulate the full sampling-based algorithm of Sec. VI.B on the *same problem
as Fig. 1* &mdash; the spiral Heisenberg chain ($N=4$, $T=20$, $\lambda=0.6$,
$\alpha=1.7$, 30 angles) &mdash; now *including* projective measurement shot
noise, unlike Figs. 1 and 2 where the interference amplitudes are evaluated
exactly. The smoother variant of Remark 1 is used (two Richardson levels +
$C^\infty$ bump randomization).

**Branch lifting.** The forward&ndash;reverse estimator determines $\theta_B$
only modulo $\pi$; the mod-$2\pi$ branch is fixed by the *integer-ladder* coarse
step,
$\Theta_I=-2\varphi(T_1)+5\varphi(2T_1)-2\varphi(4T_1)\pmod{2\pi}=\theta_B+O(T_1^{-2})$,
with the forward eigenphases $\varphi(qT_1)$ estimated from Hadamard shots.
Integer coefficients cancel the dynamical phase *and* the leading $1/T$ adiabatic
term with **no unwrapping** and **no $T\,H_{\max}$ amplification** (noise
multiplier $\sqrt{33}$). This is what makes a small coarse runtime possible: the
nearby-runtime runtime-scaling step of Ref. [1] instead amplifies by
$1/(\alpha-1)=2T_1 H_{\max}/\pi$, whose product with the $O(1/(T_1\Delta))$
adiabatic error is *independent of $T_1$*, so at the deepest-gap angles of this
loop it never falls below $\pi/2$ and the branch fails outright. The coarse base
runtime is additionally bump-randomized (128 samples, 200 Hadamard shots per
ladder node), suppressing the ladder's oscillatory residual and resolving every
branch at $T_1=15$ (vs $\sim20$ for the fixed-runtime ladder).

**Pipeline.** For each of $N$ randomized trials $T_j=TX_j$ ($X_j\sim$ bump),
single-shot Hadamard tests estimate the forward/reverse overlap amplitudes at
$T_j,\alpha T_j,\alpha^2T_j$; the forward&ndash;reverse products give unbiased
interference signals $W_l$, whose $\arg(W_l)/2$ (mod $\pi$) are lifted onto the
coarse branch and two-fold Richardson combined.

The plot shows $\theta_B$ mod $2\pi$ versus cone angle at $T=20$: each point is
the mean of $R=24$ independent runs ($N=200$ trials each), and the error bar is
the $\pm1\sigma$ spread of a *single* run (projective shot noise plus
runtime-randomization fluctuation). A strip below shows the instantaneous gap
$\Delta(\theta)$, which sets the fluctuation size &mdash; the deepest-gap angles
($\Delta\approx0.11$) carry the largest bars. Every branch is resolved and all
points lie on the analytic curve.

**Caption.** Full Hadamard-test Berry-phase algorithm on the spiral Heisenberg
chain ($N=4$, $T=20$; $\lambda=0.6$, $\alpha=1.7$), including projective shot
noise and the integer-ladder branch lift
($\Theta_I=-2\varphi(T_1)+5\varphi(2T_1)-2\varphi(4T_1)$, $T_1=15$, coarse
runtime bump-randomized). $\theta_B$ mod $2\pi$ versus cone angle: mean of 24
runs ($N=200$ trials each), $\pm1\sigma$ single-run spread. Lower strip:
instantaneous gap $\Delta(\theta)$, which controls the per-run fluctuation.

## Archived — Spin-1/2 cone loop (`old/fig1.py`): small-runtime accuracy and the cancellation cascade

As a controlled testbed we use a spin-$1/2$ in a field of fixed magnitude swept
around a cone of half-angle $\theta_0$ on the Bloch sphere,
$H(s)=-\tfrac{1}{2}\mathbf{B}(s)\!\cdot\!\boldsymbol{\sigma}$, whose ground-state
Berry phase is known exactly, $\theta_B=-\pi(1-\cos\theta_0)$. Panel (a) shows
everything in the canonical mod-$2\pi$ window $[0,2\pi)$: the runtime-scaling
points are raw values modulo $2\pi$ with no branch input, while the mod-$\pi$
estimators (forward&ndash;reverse and Richardson) are shown on the branch nearer
the analytic value. At the *small, cheap* runtime $T=8$ the runtime-scaling
reconstruction carries the full $O(1)$-radian adiabatic error and wanders off
the curve; the forward&ndash;reverse estimate is off at the $10^{-1}$-rad level;
a *single-shot* two-Richardson + bump estimate &mdash; one randomly drawn
runtime &mdash; lies on the analytic curve. The shaded band is the fully a priori
Theorem-3 prediction
$\mathbb{E}[\hat\theta] \pm 2L_2(\alpha)\sqrt{\mathbb{E}[X^{-4}]}\,\|\dot H(0)\|^2/(\Delta(0)^4T^2)$,
centered on the deterministic mean (bias included); the single shot falls inside
the band at 17 of 18 angles. Panel (b) traces the cascade at $\theta_0=0.4\pi$
(gap $\Delta=1$): $O(T^{-1})\to O(T^{-2})\to O(T^{-4})\to O(T^{-6})$. Because the
gap is constant this is the cleanest realization of the cascade, and it is the
only figure that shows the a priori Theorem-3 band (impossible on the many-body
loop, where the closed-form constant is orders of magnitude conservative) &mdash;
which is why it is retained here for reference.

## Implementation notes

- Heavy evolutions run through `ProcessPoolExecutor`; results are cached in
  `cache/figN.npz`, keyed by the `PARAMS` dict. Full recompute: fig1 (many-body)
  ~6 s, fig2 (theory) ~40 s; fig3 (algorithm) is a Monte-Carlo simulation
  (heavier). Styling-only edits replot from cache in ~1 s.
- **Models.** Spin-$1/2$ loops (fig2, `old/fig1`): 4th-order Magnus integrator,
  ~20 steps per unit runtime ($\sim10^{-10}$ accuracy). Heisenberg chain (fig1,
  fig3): exact rotating-frame propagator (a single `expm`), so amplitudes and
  shots are cheap and runtime-independent to evaluate classically.
- **Display conventions.** Everything is drawn in the canonical mod-$2\pi$ window
  $[0,2\pi)$; the analytic curve is wrapped (discontinuity cut with `NaN`).
  Runtime scaling shows raw outputs modulo $2\pi$ (no branch input); when its
  error exceeds $\pi$ the displayed value is aliased. Forward&ndash;reverse and
  Richardson are mod $\pi$, shown on the branch nearer the analytic value
  (figs 1&ndash;2) or lifted by the coarse integer-ladder step (fig 3).
- A "shot" / "trial" is one randomly drawn runtime $T_j=TX_j$ (bump density,
  fixed seed 0). In figs 1&ndash;2 the interference amplitudes are evaluated
  *exactly*; in fig 3 every measurement is a simulated $\pm1$ Hadamard bit, so
  its error bars carry genuine projective shot noise.
- Fonts match the paper (REVTeX / Computer Modern) via `text.usetex`.

[1] arXiv:2509.13423 (runtime-scaling method).
