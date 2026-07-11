# Worked examples: where Pearl's CM ($\alpha<1$) decodes correctly but hard CM ($\alpha=1$) fails

Companion to `pearl_vs_bayes_low_p.md`. There we showed *that* the Pearl–Bayes gap is large only at
low $p$; here we exhibit **concrete error mechanisms and detector firings** in the low‑$p$ regime where
ordinary correlated matching (hard evidence, $\alpha=1$) produces a **logical failure** while Pearl's
soft evidence ($\alpha<1$) produces the **correct** output. Each example follows the same anatomy, which
we make precise first.

---

## 0. The mechanism in one inequality

Correlated matching is two‑pass. The **first pass** is plain MWPM; it returns a selected edge set $S$ and
(crucially) is *correct* on the syndromes considered here. For each selected $\nu\in S$ and each
correlated neighbour $\mu$, CM rewrites $\mu$'s weight to

$$
w_\mu(\alpha)=\log\frac{1-p_\mu^{\rm imp}(\alpha)}{p_\mu^{\rm imp}(\alpha)},\qquad
p_\mu^{\rm imp}(\alpha)=\alpha\,P(\mu\mid\nu)+(1-\alpha)\,P(\mu\mid\lnot\nu),
$$

and the **second pass** re‑runs MWPM with the rewritten weights.

Suppose the syndrome admits two competing matchings of **different logical class**:

$$
M_1\ (\text{correct, class }0),\qquad M_2\ (\text{wrong, class }1),
$$

with the correlated edge $\mu$ lying on $M_2$. Let the first pass pick $M_1$ (weight $w_1<w_2$). After
the discount, $w(M_2)=w_2-\big[W-w_\mu(\alpha)\big]$ where $W$ is $\mu$'s un‑discounted weight. The second
pass **flips to the wrong class** iff

$$
\boxed{\,w_2-\underbrace{\big[W-w_\mu(\alpha)\big]}_{\text{discount }\Delta(\alpha)}\;<\;w_1\,}
\qquad\Longleftrightarrow\qquad
\Delta(\alpha)\;>\;\underbrace{w_2-w_1}_{\text{margin }m}.
$$

The discount grows as $\alpha\to1$ (more trust → bigger reweight). So there is a window

$$
\Delta(\alpha_{\rm Pearl})\;<\;m\;<\;\Delta(1)
$$

in which **hard CM over‑discounts and flips ($\to$ failure), but Pearl's damped discount stays below the
margin ($\to$ success).** Every example below lands a realistic configuration in that window.

A clean special case: if $M_2$ differs from $M_1$ by exactly the single edge $\mu$ (e.g. the "long way"
around a logical loop), then $m=w_2-w_1=w_\mu^{\rm prior}-\,$(the edge $\mu$ replaces) and the flip
condition reduces to **$w_\mu(\alpha)<0$, i.e. $p_\mu^{\rm imp}(\alpha)>\tfrac12$**: hard CM turns $\mu$
into a *negative‑weight attractor* (the decoder treats it as "more likely fired than not"), while Pearl
keeps $p_\mu^{\rm imp}<\tfrac12$ so $\mu$ stays a cost.

---

## Example 1 — Spurious shared‑mechanism discount flips a boundary‑degenerate decode (fully worked)

**Geometry / detectors.** Four detectors $D_1,D_2,D_3,D_4$ fire. Two boundary‑respecting matchings
explain them:

| matching | edges | weight | logical class |
|---|---|---|---|
| $M_1$ (correct) | $(D_1\!-\!D_2),\ (D_3\!-\!D_4)$ | $2W$ | 0 |
| $M_2$ (wrong) | $(D_1\!-\!\partial),\ (\mu\!:\!D_2\!-\!D_3),\ (D_4\!-\!\partial)$ | $3W$ | 1 |

$M_1\oplus M_2$ is a chain from boundary to boundary — a **logical operator** — so the two matchings are
different classes. The decisive correlated edge is $\mu=(D_2\!-\!D_3)$.

**True error (low $p$).** A short chain of independent $X$ errors realises $M_1$ exactly: detectors
$D_1,D_2,D_3,D_4$ are lit by the *two‑edge* pattern. The minimum‑weight, most‑probable explanation is
$M_1$, and **$M_1$ is the correct correction.**

**Shared mechanism / why $\mu$ is "correlated".** A single hook / two‑qubit‑gate fault $H$ in the
extraction circuit lights $\{D_1,D_3\}$ and is decomposed by the DEM as $\nu\oplus\mu$ with
$\nu=(D_1\!-\!D_2)\in M_1$. Hence $H$ raises the **joint** $P(\mu,\nu)$, giving (take $p=10^{-3}$)

$$
P(\mu\mid\nu)=0.7\ \ (\text{O(1): dominated by }H),\qquad
P(\mu\mid\lnot\nu)\approx p=10^{-3}\ \ (\text{only }\mu\text{'s own fault}).
$$

**First pass.** $w_1=2W=13.82<w_2=3W=20.72$ (with $W=-\ln p=6.908$). MWPM picks $M_1$ — **correct** —
and selects $\nu=(D_1\!-\!D_2)$.

**Hard CM ($\alpha=1$).** $p_\mu^{\rm imp}=0.7\Rightarrow w_\mu=\ln\frac{0.3}{0.7}=-0.847<0$. The discount
is $\Delta(1)=W-w_\mu=7.76$, while the margin is $m=w_2-w_1=W=6.91$. Since $\Delta(1)=7.76>m=6.91$:

$$
w(M_2)=2W+w_\mu = 13.82-0.847 = 12.97\;<\;13.82=w(M_1)\ \Rightarrow\ \textbf{second pass picks }M_2\ \Rightarrow\ \textbf{LOGICAL FAILURE.}
$$

$\mu$ has become a negative‑weight attractor: hard CM "believes" the hook $H$ fired (because $\nu$ was
selected) and routes the correction the long way around the logical loop.

**Pearl ($\alpha=0.4$).** $p_\mu^{\rm imp}=0.4(0.7)+0.6(10^{-3})=0.281\Rightarrow
w_\mu=\ln\frac{0.719}{0.281}=+0.942>0$. Now $\Delta(0.4)=W-0.942=5.97<m=6.91$:

$$
w(M_2)=2W+0.942 = 14.76\;>\;13.82=w(M_1)\ \Rightarrow\ \textbf{second pass keeps }M_1\ \Rightarrow\ \textbf{SUCCESS.}
$$

**Critical trust.** The flip threshold is $p_\mu^{\rm imp}=\tfrac12$, i.e.
$\alpha^\star c+(1-\alpha^\star)p=\tfrac12\Rightarrow\alpha^\star\approx 0.5/0.7=0.71$. Any
$\alpha<0.71$ decodes correctly; $\alpha=1$ fails. This single configuration already shows why the
LER‑optimal $\alpha$ sits well below 1.

---

## Example 2 — A $Y$ / two‑qubit‑gate correlated pair over‑trusted

**Mechanism.** A depolarising fault on a CNOT during syndrome extraction flips **both** the data qubit
and the ancilla, producing two detection events that the matching graph carries as a *diagonal* space
edge $\mu$ and an adjacent edge $\nu$. Because they come from **one** physical fault (prob $\sim p$),

$$
P(\mu,\nu)\sim p,\quad P(\nu)\sim p \;\Rightarrow\; P(\mu\mid\nu)=O(1),
\qquad P(\mu\mid\lnot\nu)\sim p .
$$

**Detectors firing.** Suppose the *actual* error is a plain data‑qubit $X$ error that lights only $\nu$'s
endpoints $\{D_a,D_b\}$ — the diagonal partner $\mu$ (endpoints $\{D_b,D_c\}$) did **not** fire.

**First pass (correct).** MWPM matches $\{D_a,D_b\}$ with $\nu$ — the right, lowest‑weight explanation —
and selects $\nu$.

**Hard CM over‑commits.** Seeing $\nu$ selected, hard CM asserts the CNOT fault fired and discounts the
diagonal $\mu$ toward $w_\mu\le 0$. If a *third* detector $D_c$ is also lit (by an unrelated error), the
now‑cheap $\mu=(D_b\!-\!D_c)$ lets MWPM re‑route $\{D_a,D_b,D_c,\dots\}$ through the diagonal, crossing a
logical — a flip, exactly as in §0 with $m$ small because the alternative routing is only one diagonal
away. **Pearl** keeps $w_\mu>0$, the diagonal stays a genuine cost, and the original local matching
survives.

**Key point.** The selection of $\nu$ is *evidence* of the CNOT fault but not *proof*: $\nu$ fires just as
readily from an ordinary data $X$ error (probability $P(\mu\mid\lnot\nu)\sim p$ that $\mu$ accompanies
it). Hard CM ignores this; at low $p$ it is wrong with probability $\approx 1-P(\mu\mid\nu)=O(1)$ of the
time *given the geometry*, and when it is wrong the over‑discount is what creates the logical error.

---

## Example 3 — Space–time (measurement / hook) correlation across rounds

**Mechanism.** In the circuit‑level graph each detector lives at a $(\text{stabiliser},\text{round})$
node. A measurement fault produces a **vertical (time‑like) edge** $\nu$ between consecutive rounds; a
hook fault from the CX ordering produces a **diagonal space‑time edge** $\mu$ that shares the shared
mechanism’s decomposition, so $P(\mu\mid\nu)=O(1)$.

**Detectors firing.** A measurement flip in round $t$ lights the pair $\{(s,t),(s,t{+}1)\}$ — explained
by the vertical $\nu$. The true cause *is* the measurement error; no hook occurred, so $\mu$ did not fire.

**First pass (correct).** MWPM matches the vertical pair with $\nu$ and selects it.

**Hard CM.** Discounts the space‑time diagonal $\mu$ to a near‑free edge. When neighbouring rounds carry
an additional detection event, the cheap diagonal lets the matching "tunnel" diagonally across rounds and
space, producing a space‑time logical chain — a **time‑boundary logical failure**. **Pearl**'s damped
diagonal keeps the vertical (measurement‑error) explanation, which is correct.

This example matters because vertical/measurement edges are *frequently* selected (measurement noise is
ubiquitous), so a decoder that over‑trusts every vertical selection injects many spurious diagonal
discounts — most harmless, but a small, **decisive** fraction sit next to a competing logical chain.

---

## Why all three bite only at low $p$

In every example the damage is the **over‑discount** $\Delta(1)=W-w_\mu(1)$ relative to the margin $m$.
Two low‑$p$ facts make hard CM dangerous and Pearl helpful:

1. **The discount is large.** Because a shared mechanism gives $P(\mu\mid\nu)=O(1)$ while
   $P(\mu\mid\lnot\nu)\sim p$, the hard‑evidence implied probability is $O(1)$ and $w_\mu(1)$ is small or
   **negative** — a discount of order $W=-\log p$. The Pearl–hard gap per edge is exactly
   $-\log\!\big[\alpha+(1-\alpha)r\big]$ with $r=P(\mu\mid\lnot\nu)/P(\mu\mid\nu)=O(p)\to0$, i.e. it is at
   its maximum $-\log\alpha$. (See `pearl_vs_bayes_low_p.md`, §5.)
2. **Failures are decided by single near‑degenerate ties.** At low $p$ a logical error requires a rare
   coincidence in which a competing wrong‑class matching is within one discount of the correct one — a
   small margin $m$. One spurious $O(W)$ discount easily clears such a margin. These pivotal ties are the
   *dominant* failure channel at low $p$, so curing them (Pearl) moves the LER materially.

At high $p$, $P(\mu\mid\nu)\to P(\mu\mid\lnot\nu)$ ($r\to1$): $w_\mu(1)\to w_\mu^{\rm prior}$, the discount
$\Delta\to0$, hard CM and Pearl coincide, and in any case the dense‑error regime is not decided by single
edges — so neither the failure nor its cure exists. The examples simply cannot be constructed.

---

## Summary

| | true cause | first pass | hard CM ($\alpha=1$) | Pearl ($\alpha<1$) |
|---|---|---|---|---|
| **Ex.1** boundary loop | short $X$ chain ($M_1$) | $M_1$ ✓ | discounts $\mu$ to $w_\mu=-0.85$, picks long loop $M_2$ → **fail** | $w_\mu=+0.94$, keeps $M_1$ → **ok** |
| **Ex.2** CNOT/$Y$ pair | plain data $X$ (only $\nu$) | $\nu$ ✓ | trusts CNOT fault, over‑cheap diagonal reroutes → **fail** | diagonal stays a cost → **ok** |
| **Ex.3** measurement/hook | measurement flip (vertical $\nu$) | $\nu$ ✓ | over‑cheap space‑time diagonal tunnels → **fail** | vertical kept → **ok** |

**One sentence.** In each case the first pass is *right*, but selecting $\nu$ is only **evidence** — not
proof — that the shared mechanism fired; hard CM ($\alpha=1$) treats it as proof, applies an $O(-\log p)$
discount to a correlated edge $\mu$, and when that edge lies on a competing logical chain the over‑discount
crosses the (small, low‑$p$) margin and flips the decode; Pearl ($\alpha<1$) applies a damped discount
that stays below the margin and preserves the correct correction.

---

### Caveats

- The numbers in Example 1 are an explicit, self‑consistent gadget chosen to sit inside the success
  window $\Delta(\alpha_{\rm Pearl})<m<\Delta(1)$; real DEM probabilities/margins are circuit‑specific,
  but the *structure* (correct first pass + spurious $O(-\log p)$ discount + small competing‑class
  margin) is generic.
- Examples 2–3 are described at the mechanism level (which fault, which detectors, which competing
  routing); exact edge probabilities depend on the noise model and the hyperedge decomposition.
- The argument is per‑decision; the measured ~20% low‑$p$ LER reduction is the aggregate over many such
  pivotal syndromes.
