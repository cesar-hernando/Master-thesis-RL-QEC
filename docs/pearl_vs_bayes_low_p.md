# Why Pearl's rule beats hard Bayesian conditioning ($\alpha=1$) only at low $p$

**Question.** In correlated matching (CM) we reweight a decoding edge $\mu$ using the
first‑pass selection of a correlated neighbour $\nu$. *Ordinary* CM applies **Bayes' rule under
hard evidence** — it treats "MWPM selected $\nu$" as a proof that $\nu$ fired and uses the conditional
$P(\mu\mid\nu)$. **Pearl's rule of virtual evidence** instead treats the selection as *uncertain*
evidence, mixing in the complementary hypothesis with a trust parameter $\alpha\in[0,1]$. Empirically,
Pearl ($\alpha<1$) gives a large logical‑error‑rate (LER) reduction at low physical error rate $p$ and
essentially none near threshold. This note explains, with mathematics, *why* the advantage is confined
to low $p$.

> **Terminology.** We use **Pearl's virtual (soft) evidence**: an uncertain observation about $\nu$ is
> encoded by a likelihood ratio, which updates the belief that $\nu$ fired to a value $\alpha = P(\nu\
> \text{fired}\mid \text{selection})$. Belief in any correlated $\mu$ is then obtained by ordinary
> propagation. This is *not* Jeffrey's rule (which would posit the new marginal directly); here $\alpha$
> is the **posterior belief produced by the soft evidence**, and $\alpha=1$ recovers hard Bayesian
> conditioning.

---

## 1. Setup and notation

The decoding graph has edges $e$, each an independent Bernoulli fault with probability $p_e$ taken from
the detector error model (DEM). Minimum‑weight matching (MWPM) uses additive edge **weights**

$$
w_e \;=\; \log\frac{1-p_e}{p_e}\;\approx\; -\log p_e \qquad (p_e\ll 1),
$$

and returns the minimum‑weight correction consistent with the syndrome.

Fix two correlated edges $\mu$ and $\nu$. Treat their fault indicators as Bernoulli random variables and
abbreviate

$$
q_\mu = P(\mu),\qquad q_\nu = P(\nu),\qquad J = P(\mu,\nu).
$$

The first pass selects $\nu$; CM then revises the weight of the correlated edge $\mu$ from its prior
$-\log q_\mu$ to a **posterior weight** $-\log P(\mu\mid \text{selection})$, and reruns MWPM.

---

## 2. The two rules

**Hard evidence — Bayes, $\alpha=1$ (ordinary CM).** Assume selection $\Rightarrow$ $\nu$ fired:

$$
P_{\text{Bayes}}(\mu) \;=\; P(\mu\mid\nu)\;=\;\frac{J}{q_\nu}.
$$

**Pearl's virtual evidence, $\alpha\in[0,1]$.** The selection updates the belief that $\nu$ fired to
$\alpha$. Propagating to $\mu$ (the evidence acts only through $\nu$, so $\mu\perp \text{evidence}\mid\nu$):

$$
\boxed{\,P_{\alpha}(\mu)\;=\;\alpha\,P(\mu\mid\nu)\;+\;(1-\alpha)\,P(\mu\mid\lnot\nu)\,}
$$

with

$$
P(\mu\mid\nu)=\frac{J}{q_\nu},\qquad
P(\mu\mid\lnot\nu)=\frac{q_\mu-J}{1-q_\nu}.
$$

Setting $\alpha=1$ recovers Bayes; $\alpha<1$ softens the update by retaining some belief that $\nu$ did
*not* fire.

---

## 3. The whole Pearl-vs-Bayes difference is one quantity

Subtract the two rules:

$$
P_{\alpha}(\mu)-P_{\text{Bayes}}(\mu)
=(1-\alpha)\big[P(\mu\mid\lnot\nu)-P(\mu\mid\nu)\big]
=-(1-\alpha)\,D,
$$

where we define the **discriminability (lever arm)**

$$
D \;\equiv\; P(\mu\mid\nu)-P(\mu\mid\lnot\nu).
$$

A short calculation collapses $D$ to a single, interpretable object:

$$
D=\frac{J}{q_\nu}-\frac{q_\mu-J}{1-q_\nu}
=\frac{J-q_\mu q_\nu}{q_\nu(1-q_\nu)}
=\frac{\operatorname{Cov}(\mu,\nu)}{\operatorname{Var}(\nu)} .
\tag{3.1}
$$

So $D$ is exactly the **regression coefficient of $\mathbf 1_\mu$ on $\mathbf 1_\nu$**. Equivalently
$D=\rho\,\sigma_\mu/\sigma_\nu$ with $\rho$ the Pearson correlation and $\sigma$ the standard deviations.
The sensitivity of the posterior to the trust parameter is

$$
\frac{\partial P_\alpha(\mu)}{\partial\alpha}=D .
$$

**Everything Pearl does differently from Bayes is proportional to $D$.** If $D\to0$, the two rules
coincide for *every* $\alpha$. The entire question — "why only at low $p$?" — therefore reduces to:
**how does $D$ scale with $p$?**

---

## 4. Asymptotics of $D$: a minimal shared‑mechanism model

Let $\mu$ and $\nu$ each be triggered by their own independent faults *and* by a single **shared
mechanism** $h$ (e.g. a $Y$ error or a hook fault that flips detectors making both edges plausible).
With rates linear in $p$,

$$
P(\text{own-}\mu)=a_\mu p,\quad P(\text{own-}\nu)=a_\nu p,\quad P(h)=s\,p ,
$$

and keeping leading order in $p$:

$$
q_\mu \approx (a_\mu+s)\,p,\qquad
q_\nu \approx (a_\nu+s)\,p,\qquad
J \approx s\,p\;+\;\underbrace{a_\mu a_\nu p^2}_{\text{coincidence}}\;\approx s\,p .
$$

The joint is dominated by the *shared* mechanism because two independent faults firing together is
$O(p^2)$. Hence

$$
P(\mu\mid\nu)=\frac{J}{q_\nu}\approx\frac{s}{a_\nu+s}=O(1),
\qquad
P(\mu\mid\lnot\nu)=\frac{q_\mu-J}{1-q_\nu}\approx a_\mu\,p=O(p).
\tag{4.1}
$$

The interpretation is sharp: **if $\nu$ did not fire, the shared mechanism did not fire, so $\mu$ can
only come from its own $O(p)$ fault**; whereas conditioning on $\nu$ makes $\mu$ an $O(1)$ event. The two
hypotheses about $\mu$ are separated by a factor $\sim 1/p$.

Therefore

$$
\boxed{\,D \approx \frac{s}{a_\nu+s} - a_\mu p \;\xrightarrow[p\to0]{}\; \frac{s}{a_\nu+s}=O(1)\,}
$$

$D$ saturates at an $O(1)$ value as $p\to0$ (maximal lever arm) and **shrinks toward $0$ as $p$ grows**,
because higher‑order ($\sim p^2$) coincidences and saturation pull $P(\mu\mid\lnot\nu)$ up toward
$P(\mu\mid\nu)$, diluting the correlation $\rho$ in (3.1).

It is convenient to package this in the **evidence contrast**

$$
r(p)\;\equiv\;\frac{P(\mu\mid\lnot\nu)}{P(\mu\mid\nu)}\;\approx\;\frac{a_\mu(a_\nu+s)}{s}\,p \;=\;O(p),
\qquad r\in[0,1],
$$

so $r\to 0$ at low $p$ and $r\to 1$ at high $p$. Note $D = P(\mu\mid\nu)\,(1-r)$.

---

## 5. Consequence in weight space (what MWPM actually sees)

MWPM is driven by weights, not probabilities. Compare the three weights of edge $\mu$:

| rule | weight of $\mu$ | low‑$p$ value |
|---|---|---|
| plain MWPM (prior) | $-\log q_\mu$ | $\log(1/p)+\text{const}$ — **large** |
| hard CM (Bayes, $\alpha=1$) | $-\log P(\mu\mid\nu)$ | $-\log\frac{s}{a_\nu+s}=O(1)$ — **small** |
| Pearl ($\alpha$) | $-\log P_\alpha(\mu)$ | $O(1)-\log\!\big[\alpha+(1-\alpha)r\big]$ |

Two gaps matter.

**(a) CM vs MWPM** (why correlations help at all):

$$
w_\mu^{\text{prior}}-w_\mu^{\text{Bayes}}
=\log\frac{P(\mu\mid\nu)}{q_\mu}
\approx \log\frac{1}{p}+\text{const}\;\xrightarrow[p\to0]{}\;\infty .
$$

Hard CM slashes $\mu$'s weight by $\sim\log(1/p)$: a *huge*, decode‑reordering move at low $p$. (This is
also why CM $\gg$ MWPM at low $p$.)

**(b) Pearl vs Bayes** (the object of this note):

$$
\boxed{\,G(p)\;\equiv\;w_\mu^{\text{Pearl}}-w_\mu^{\text{Bayes}}
=-\log\!\big[\alpha+(1-\alpha)\,r(p)\big]\,}
\tag{5.1}
$$

This is monotonically **decreasing in $r$**:

$$
r\to0\ (\text{low }p):\quad G\to -\log\alpha\;>0\quad(\text{maximal}),
\qquad
r\to1\ (\text{high }p):\quad G\to -\log 1 = 0 .
$$

So the **extra weight Pearl puts back on $\mu$ relative to hard CM is $-\log\alpha$ at low $p$ and
vanishes at high $p$**. At high $p$, $P(\mu\mid\nu)$ and $P(\mu\mid\lnot\nu)$ converge, the bracket in
(5.1) $\to 1$, and *Pearl becomes identical to Bayes for every $\alpha$*. There is simply nothing left to
soften.

Picture it on the measured weight histogram: the single‑fault families translate rigidly by
$-\log(p'/p)$, so the gaps between families are roughly $p$‑independent. But the **correlation contrast**
$r\propto p$ means the lever (b) that Pearl pulls shrinks from $-\log\alpha$ to $0$ as $p$ rises. The knob
controls a shift that only exists at low $p$.

---

## 6. From weights to LER: why a vanishing $G$ means a vanishing benefit

The reweight of $\mu$ can change the logical outcome only when it crosses the **weight margin** $m$
between the two best competing corrections in the second pass (these near‑ties are exactly the
syndromes that produce logical failures at low $p$). Decompose the expected effect by whether the
selected $\nu$ *truly* fired, with calibrated reliability $\pi=P(\nu\text{ fired}\mid\text{selection})$:

- **$\nu$ truly fired** (prob $\pi$): reweighting $\mu$ downward is *correct*. Bayes' larger move helps
  slightly more than Pearl's.
- **$\nu$ did not fire** (prob $1-\pi$): the downward reweight of $\mu$ is *wrong* and, when it exceeds
  $m$, flips the decode into a logical error. Bayes (full move) triggers these wrong flips more often
  than Pearl (damped move).

Writing the per‑edge expected logical‑error contribution $L(\alpha)$ and differentiating,

$$
\frac{\partial L}{\partial\alpha}\;\propto\;
\underbrace{D(p)}_{\text{size of the move}}\times
\underbrace{\big[(1-\pi)\,f_{\text{wrong}} - \pi\,f_{\text{right}}\big]}_{\text{net mis‑commit pressure}},
\tag{6.1}
$$

where $f_{\text{right}},f_{\text{wrong}}$ are the densities of near‑ties whose margin the move can cross.
The optimal $\alpha^\star$ solves $\partial L/\partial\alpha=0$, balancing the bracket. Two regimes:

- **Low $p$:** $D=O(1)$ and the wrong‑direction reweight is large, so the bracket is materially
  negative for $\alpha$ near $1$ — i.e. hard CM over‑commits. Reducing $\alpha$ from $1$ produces a
  *first‑order* LER reduction. The achievable gain is $O(1)\times D = O(1)$.
- **High $p$:** $D\to0$, so by (6.1) $\partial L/\partial\alpha\to 0$ **uniformly in $\alpha$**: the LER
  is flat in the trust parameter. Independently, near threshold the matching has no headroom — *no*
  reweighting scheme helps — so even the residual sensitivity cannot be cashed in.

Thus the gradient that Pearl exploits is gated by $D(p)$, which is the same $O(1)\!\to\!0$ quantity that
controlled (3.1) and (5.1). **Pearl can only improve on Bayes where $D$ is large, i.e. at low $p$.**

---

## 7. Corollary: why the optimal $\alpha$ is small (and not a literal probability)

At the optimum, $\alpha^\star$ is set by the bracket in (6.1), *not* by MWPM's marginal accuracy. Two
effects push it well below $1$:

1. **Conditioning on pivotal ties.** The bracket is evaluated only on near‑degenerate syndromes — the
   sole place the move changes the decode. On that subset MWPM's selection is barely better than a coin
   flip, so the effective trust is low.
2. **Shrinkage of an overconfident move.** Hard CM applies the *full* pairwise conditional under DEM
   independence assumptions; $\alpha^\star$ damps that overconfident, model‑mismatched move. It behaves
   like an optimal shrinkage coefficient, not a calibrated $P(\nu\text{ fired})$.

Consistency check: as $\alpha\to0$, $P_\alpha(\mu)\to P(\mu\mid\lnot\nu)\approx q_\mu$ — i.e. *no*
correlation correction (back to plain MWPM, which is worse than CM at low $p$). Since hard CM
($\alpha=1$) beats MWPM but Pearl beats hard CM, the LER‑vs‑$\alpha$ curve is **U‑shaped with an interior
minimum in $(0,1)$**, typically $\alpha^\star\sim 0.2$–$0.4$. This is "soften a lot, but do not discard
the correlation."

---

## 8. Empirical confirmation

Measured LER ratios (best‑$\alpha$ vs hard CM, $\alpha=1$), rotated surface code, no drift:

| $d$ | $p$ | regime | best $\alpha$ / CM | gain |
|---|---|---|---|---|
| 5 | $3\times10^{-4}$ | deep sub‑threshold | $0.773$ | $\approx 23\%$ |
| 5 | $5\times10^{-3}$ | near threshold | $\approx 0.96$ (flat) | $\lesssim$ noise |
| 7 | $5\times10^{-4}$ | deep sub‑threshold | $0.776$ | $\approx 22\%$ |
| 7 | $7\times10^{-4}$ | sub‑threshold | $0.927$ | $\approx 7\%$ |
| 7 | $\sim5\times10^{-3}$ | near threshold | $\approx 1$ (flat) | $\lesssim$ noise |

The gain tracks sub‑threshold depth $p/p_{\text{th}}$ and is essentially distance‑independent at fixed
depth — exactly as predicted by the $D(p)$ analysis (a local, bulk quantity). At $d=7$ the optimum sits
at the grid edge ($\alpha=0.4$, still dropping), consistent with the U‑shape of §7; the true
$\alpha^\star$ is somewhat lower.

---

## 9. Summary

- The entire difference between Pearl ($\alpha<1$) and hard Bayes ($\alpha=1$) is proportional to the
  **lever arm** $D=\operatorname{Cov}(\mu,\nu)/\operatorname{Var}(\nu)=P(\mu\mid\nu)\,(1-r)$.
- In a shared‑mechanism model, $P(\mu\mid\nu)=O(1)$ while $P(\mu\mid\lnot\nu)=O(p)$, so the **evidence
  contrast** $r\propto p$: the two hypotheses about $\mu$ are separated by a factor $\sim1/p$ at low $p$
  and merge ($r\to1$) at high $p$.
- Consequently the Pearl‑vs‑Bayes weight gap $G=-\log[\alpha+(1-\alpha)r]$ equals $-\log\alpha$ at low
  $p$ and **vanishes** at high $p$: near threshold Pearl $\equiv$ Bayes for any $\alpha$.
- The LER gradient $\partial L/\partial\alpha\propto D$ inherits this gating, so there is a real
  optimization to perform only at low $p$. Near threshold $D\to0$ *and* the code is out of headroom, so
  no trust setting matters.

**One line:** *Pearl's softening only bites where conditioning on the first‑pass selection is highly
informative — and that information ($D\sim1$, $r\to0$) exists only deep below threshold, vanishing
($D\to0$, $r\to1$) as $p$ approaches the threshold.*

---

### Assumptions and caveats

- Pairwise, single‑edge analysis with a global scalar $\alpha$; real CM aggregates many such updates and
  the DEM has higher‑order correlations not captured by the two‑variable model.
- The shared‑mechanism model captures the leading low‑$p$ scaling; exact constants depend on the circuit
  and the DEM decomposition.
- The LER argument (§6) is a decision‑theoretic sketch: it identifies the gating factor $D(p)$ rigorously
  but does not evaluate the matching combinatorics $f_{\text{right}},f_{\text{wrong}}$ explicitly.
