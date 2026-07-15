# How the saturating law emerges from the effective-distance ↔ correlation-correction trade-off

$$\alpha^{*}(p) \;=\; \frac{L}{1+(p_{0}/p)^{m}}$$

**Summary.** The law is the minimal interpolation between two asymptotic regimes forced by two
competing effects that live at *different orders* in the physical error rate $p$. The plateau $L$ is
the internal optimum of the correlation-correction (benefit) sector; the tail exponent $m$ is set by
the geometry of the spurious detours that cost correlated matching one unit of fault-tolerant distance
(harm); and the crossover $p_0$ is where the two become comparable. Each fitted parameter maps onto a
mechanistic quantity, verified against the sweep CSVs and the exhaustive fault enumeration.

---

## 1. Bookkeeping: the two effects live at different fault orders

Expand the logical error rate as a fault-weight series at distance $d$, with $n=\lceil d/2\rceil$
($n=3$ for $d=5$):

$$\mathrm{LER}(\alpha,p)\;=\;\underbrace{[\,C-B(\alpha)\,]\,p^{\,n}}_{\text{benefit sector}}\;+\;\underbrace{H(\alpha,p)\,p^{\,n-1}}_{\text{harm sector}}$$

- **Benefit sector (correlation correction).** Genuine $Y$- and CNOT-correlated faults live in the
  *same* $p^{n}$ sector as MWPM's failures — correlated matching lowers the *coefficient* of $p^{n}$ by
  tipping near-degenerate matchings the right way. $B(\alpha)$ rises from 0 and has an internal maximum
  at $\alpha = A \approx 0.75 < 1$: even within the $p^n$ sector, full strength over-tips some
  decisions.
- **Harm sector (effective-distance loss).** The spurious-detour failures need one fault *fewer* — this
  is the enumeration result: **337 weight-2 chains defeat CM($\alpha{=}1$) at $d=5$**, whereas MWPM
  needs 3 faults to fail. So the harm enters one order down, at $p^{\,n-1}$.

The structural key is that **harm/benefit $\sim 1/p$**: the two sectors are not in fixed proportion —
the harm's relative weight grows without bound as $p\to0$. Were the overcorrections to cost at the
*same* fault order as the benefit, the trade-off would be $p$-independent and $\alpha^{*}$ would be a
constant. **The entire $p$-dependence of $\alpha^{*}$ exists because the harm lives one order below the
benefit.**

## 2. The $\alpha$-dependence of each sector

**Harm is a threshold staircase that slides with $p$.** A spurious detour built from $k$ boosted edges
(each costing $k(|\log\alpha|+c)$ after damping) beats an honest route of margin
$\Delta\approx a\log(1/p)$ iff

$$k\,(|\log\alpha|+c)\;<\;\Delta\approx a\log(1/p)\quad\Longleftrightarrow\quad \alpha \;>\; \alpha_{\mathrm{th}}\;\propto\;p^{\,a/k}.$$

So $H(\alpha,p)=\sum_{\text{configs}}(\text{prob. coeff.})\cdot\theta(\alpha>\alpha_{\mathrm{th}})$ — a
staircase in $\alpha$ whose steps were measured directly (failing pairs
$2\to6\to14\to54\to154\to225\to373$ as $\alpha$ goes $0.05\to1$ at $p=2\times10^{-4}$), and whose
thresholds all slide toward 0 as $p^{a/k}$. The measured shift is $\approx\sqrt{10}$ per decade, i.e.
**$a/k\approx\tfrac12$**.

**Why does $H$ depend on $p$ at all?** It looks like double counting — the occurrence probability of
a 2-fault configuration is already factored out as $p^{\,n-1}$ — but $p$ enters through a *second
door*: the decoder's **weight landscape**. Weights are $w=\log\frac{1-p_e}{p_e}\approx\log(1/p_e)$, so
whether the decoder *fails* on a given configuration (the $\theta$-functions above) is decided by
weight comparisons whose honest side scales as $\log(1/p)$. $H(\alpha,p)$ is therefore a slowly
sliding staircase in $\log p$ — logarithmic in origin, nothing like the power-law prefactor, and no
probability is counted twice (write it as $H(\alpha, L)$ with $L=\log(1/p)$ to make the two roles
explicit). Empirically: at $\alpha=1$ nearly all configurations are active, so $H$ is almost
$p$-independent (failing-pair counts $423\to268$, a factor 1.6, while $p^2$ varies by $1600$); at
intermediate $\alpha=0.3$ it varies by a factor $\sim23$ — the $p$-dependence lives exactly where the
threshold cloud sits. This dependence is *essential*: were $H$ $p$-independent, the optimum would
stall just below the smallest fixed threshold and $\alpha^*(p)$ would plateau at a small constant
instead of decaying — the sliding thresholds are what generate the $\sqrt{p}$ tail.

**Benefit is $p$-independent in $\alpha$.** $B(\alpha)$ is a property of the correlation structure
(which decisions need tipping, and by how much), not of the ambient weight scale, so its optimum $A$
does not move with $p$. Empirical check: the fitted plateau is $L=0.775\ (d=5)$ and $0.749\ (d=7)$ —
the *same* number at two distances, as required if it is the benefit sector's internal optimum rather
than anything tied to the harm.

## 3. First-order condition and the two regimes

Minimising LER over $\alpha$ gives

$$B'(\alpha^{*})\,p^{\,n}\;=\;\partial_{\alpha}H(\alpha^{*},p)\,p^{\,n-1}\quad\Longleftrightarrow\quad B'(\alpha^{*})\;=\;\frac{1}{p}\,\partial_{\alpha}H(\alpha^{*},p).$$

**Regime 1 — $p\gg p_{0}$ (near threshold).** The threshold cloud $\alpha_{\mathrm{th}}\propto\sqrt{p}$
sits *above* the benefit optimum $A$: at the $\alpha$ you would want for benefit reasons, no harmful
configurations are active ($\partial_\alpha H\approx0$). The condition reduces to $B'(\alpha^{*})=0$,
so

$$\alpha^{*}\;\to\;A\;\approx\;0.75\qquad\text{(plateau: the harm sector is simply out of reach).}$$

This is also why the literature, benchmarking near threshold, always found $\alpha\approx1$ correlated
matching to be fine.

**Regime 2 — $p\ll p_{0}$.** The cloud slides below $A$. Sitting at $A$ would now activate a large part
of the $p^{\,n-1}$-amplified staircase, and the $1/p$ factor makes $\partial_\alpha H$ overwhelming, so
the optimum is *pinned to the edge of the moving threshold cloud* (plus a fixed $\mathcal{O}(1)$ offset
into it, where the marginal benefit tail balances the first few staircase steps — which is why the
swept $\alpha^{*}=0.2$ sits slightly above the strict onset $\alpha_{\mathrm{th}}\approx0.05$–$0.1$).
Since every threshold scales as $p^{a/k}$,

$$\alpha^{*}\;\propto\;p^{\,a/k}\;\approx\;\sqrt{p}\qquad\text{(power-law tail).}$$

Check against data: the swept $\alpha^{*}$ falls $0.4\to0.1$ from $p=10^{-3}$ to $10^{-4}$, a factor
$\approx4$ per decade $\approx p^{0.6}$ — consistent with $\tfrac12$ within the grid resolution, and
independently confirmed by the $\alpha_{\mathrm{th}}$ scaling from the enumeration.

## 4. The saturating law is the minimal interpolation of these asymptotics

Any function with $\lim_{p\gg p_0}\alpha^{*}=A$ and
$\lim_{p\ll p_0}\alpha^{*}=A(p/p_0)^{m}$ is, at lowest order, exactly the saturating law, with every
parameter now identified:

| fit parameter | mechanistic meaning | check |
|---|---|---|
| plateau **$L=A\approx0.75$** | internal optimum of the benefit sector ($p$- and $d$-independent) | same value at $d=5$ and $d=7$ ✓ |
| tail **$m=a/k\approx\tfrac12$** | margin-to-boost-count geometry of the dominant spurious detours | $\alpha_{\mathrm{th}}$ map shifts $\sqrt{10}$ per decade ✓ |
| crossover **$p_0\approx7\times10^{-4}$** | where the distance-loss channel at $\alpha=A$ becomes comparable to the benefit-sector stakes | where CM(1)/MWPM peels away from best-$\alpha$/MWPM in the ratio table ✓ |

## 5. Two honest caveats

- The staircase is discrete and broad, and the optimum is *shallow* (the LER is flat to ~3% over
  $\alpha\in[0.15,0.4]$ at $p=4\times10^{-4}$): the "law" describes the bottom of a wide valley, not a
  sharp point. The least-squares $m$ from the fit (1.4 at $d=5$) is dragged upward by plateau-region
  points and should not be read as the true tail exponent — the tail itself says $\approx\tfrac12$.
- The exponent $a/k\approx\tfrac12$ is currently *measured*, not derived from geometry. Deriving it
  means characterising the 337 failing pairs (how many boosted edges $k$ the detour uses vs the honest
  margin in units of $\log(1/p)$); the enumeration map already contains the full threshold distribution
  $N(\alpha_{\mathrm{th}}<\alpha)$, so this last step is extractable.

## In one sentence

**The plateau is the correlation-correction optimum, the tail is the effective-distance-restoration
constraint, and the saturating law is nothing but the crossover between a $p$-independent benefit and a
harm whose relative weight grows as $1/p$ while its activation thresholds slide as $\sqrt{p}$.**
