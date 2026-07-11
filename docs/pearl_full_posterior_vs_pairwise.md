# Why the pairwise reweight overstates $\mu$: full posterior vs. $P(e_\mu\mid e_\nu)$

Companion to `pearl_optimality_and_damping_mechanism.md` (this is the rigorous backing for mechanism
**M1** there) and `pearl_alpha_true_vs_optimal.md`. It explains *why shrinkage ($\alpha<1$) helps even
though MWPM's first‑pass selection is reliable* — the issue is the **inference from $\nu$ to $\mu$**, not
the **selection of $\nu$**.

Notation: $e_\mu,e_\nu$ are the indicator events "edge $\mu$ (resp. $\nu$) fired"; **extra** is the rest
of the syndrome beyond the selected neighbour $\nu$.

---

## 1. The two quantities

Correlated matching reweights $\mu$ with the **pairwise conditional**

$$
P(e_\mu\mid e_\nu)=\frac{P(e_\mu,e_\nu)}{P(e_\nu)} .
$$

The decode‑relevant quantity is the **full posterior given the whole syndrome**,
$P(e_\mu\mid e_\nu,\text{extra})$. CM uses the former as a proxy for the latter.

---

## 2. Factorisation: full posterior = pairwise $\times$ likelihood ratio

$$
P(e_\mu\mid e_\nu,\text{extra})
=\frac{P(e_\mu,e_\nu,\text{extra})}{P(e_\nu,\text{extra})}
=\frac{P(\text{extra}\mid e_\mu,e_\nu)\,P(e_\mu,e_\nu)}{P(\text{extra}\mid e_\nu)\,P(e_\nu)} .
$$

The $P(e_\nu)$ cancels and $P(e_\mu,e_\nu)/P(e_\nu)=P(e_\mu\mid e_\nu)$, leaving

$$
\boxed{\,P(e_\mu\mid e_\nu,\text{extra})=P(e_\mu\mid e_\nu)\,\cdot\, R,\qquad
R=\frac{P(\text{extra}\mid e_\mu,e_\nu)}{P(\text{extra}\mid e_\nu)}\,.}
$$

The reweight uses only $P(e_\mu\mid e_\nu)$; the correct value multiplies it by the **likelihood ratio
$R$** that the rest of the syndrome supplies.

---

## 3. Larger or smaller? — the sign of $R$

Because $P(\text{extra}\mid e_\nu)$ is a convex combination,

$$
P(\text{extra}\mid e_\nu)=P(\text{extra}\mid e_\mu,e_\nu)\,P(e_\mu\mid e_\nu)
+P(\text{extra}\mid \lnot e_\mu,e_\nu)\,P(\lnot e_\mu\mid e_\nu),
$$

the ratio satisfies

$$
R\gtrless 1\iff P(\text{extra}\mid e_\mu,e_\nu)\gtrless P(\text{extra}\mid \lnot e_\mu,e_\nu).
$$

- **$R>1$** ⇒ posterior **larger** than pairwise: the extra syndrome is *more* likely if $\mu$ also
  fired — $\mu$'s own detectors **corroborate** $\mu$.
- **$R<1$** ⇒ posterior **smaller**: the extra syndrome is *more* likely if $\mu$ did **not** fire — the
  rest of the syndrome is **counter‑evidence** to $\mu$.
- **$R=1$** ⇒ extra is independent of $\mu$ given $\nu$ — no change.

So the full posterior **sharpens** $P(e_\mu\mid e_\nu)$ toward $0$ or $1$ depending on whether $\mu$'s own
evidence agrees. The pairwise conditional is the value *before* looking at the rest of the syndrome.

---

## 4. In the regime where the reweight acts, $R<1$ (posterior is smaller)

The discount on $\mu$ can change the decode **only** when $\mu$'s detectors are lit and could be matched,
and the first‑pass MWPM has already found a minimum‑weight explanation of the *whole* syndrome —
typically **without** $\mu$. In that regime, **extra** already contains a competing, $\mu$‑free
explanation of $\mu$'s detectors. If $\mu$ *also* fired, those detectors would be flipped twice (by $\mu$
and by the alternative routing), i.e. you would need **additional errors** — a higher‑order, lower‑
probability configuration. Hence

$$
P(\text{extra}\mid e_\mu,e_\nu)<P(\text{extra}\mid \lnot e_\mu,e_\nu)
\;\Rightarrow\; R<1
\;\Rightarrow\; P(e_\mu\mid e_\nu,\text{extra})<P(e_\mu\mid e_\nu).
$$

**So precisely in the cases that drive over‑correction, the full posterior is smaller than the pairwise
conditional, and the reweight overstates $\mu$.** That is the overconfidence, made exact: $P(e_\mu\mid
e_\nu)$ omits the factor $R<1$ that the rest of the syndrome supplies.

The symmetric case is real too: when $\mu$'s detectors are lit in a way that is *not* well explained
without $\mu$ (so $\mu$ is the natural explanation), $R>1$ and the full discount is *justified* — this is
when correlated matching legitimately helps (Fowler's degeneracy‑breaking).

---

## 5. Why this makes shrinkage ($\alpha<1$) beneficial — and heaviest at low $p$

The pairwise rule applies the **same** discount to both cases:

- $R>1$ (corroboration): the full discount is right — $\mu$ should be matched.
- $R<1$ (counter‑evidence): the full discount **overshoots** — $\mu$ is redundant.

A scalar $\alpha<1$ is a blunt, global stand‑in for "multiply by a typical $R<1$": it damps the discount
to hedge the over‑correction ($R<1$) cases, at the cost of slightly under‑applying it in the
corroboration ($R>1$) cases. With an **asymmetric loss** (over‑discount $\to$ a near‑free attractor that
flips the decode across a logical; under‑discount $\to$ a correct edge left slightly expensive), this
trade favours shrinkage.

Why heaviest at low $p$: the discount magnitude is $\sim\log(1/p)$ (it collapses $\mu$ from weight
$\sim\log(1/p)$ toward $0$). So in the $R<1$ cases the **absolute weight overshoot grows as $p\to0$**, and
the asymmetric downside (a logical flip) grows with it. The optimal shrinkage therefore deepens as
$p\to0$ — **with no reference to how reliable $\nu$'s selection was.** This is exactly why the
LER‑optimal $\alpha^\star$ (shrink hard at low $p$) and the calibrated trust
$\alpha^{\rm cal}=P(e_\nu\text{ fired}\mid e_\nu\text{ selected})$ (≈1 at low $p$) trend in **opposite**
directions in $p$ (`pearl_alpha_true_vs_optimal.md`).

---

## 6. The honest ceiling: a decoder that computes $R$

A decoder that actually evaluates $R$ — i.e. uses the **whole** syndrome instead of a single pairwise
conditional — is **belief propagation / belief matching**: BP propagates the messages that constitute the
$R$ factor. Pearl‑CM with a tuned $\alpha$ is therefore the **cheap scalar approximation** to that $R$:
one global number replacing a per‑edge, syndrome‑dependent likelihood ratio. This is why:

- belief matching can beat CM where the correlation structure is rich/dense (near threshold, $R$ varies a
  lot per shot and the scalar $\alpha$ is too coarse), and
- CM/Pearl can beat belief matching deep sub‑threshold (simple degenerate $Y$‑chains, where the targeted
  pairwise discount + a single shrinkage scalar already captures $R$, and BP's loopy marginals hedge the
  degeneracy) — see the crossover in `ler_vs_p_beliefmatching` data.

---

## Summary

- $P(e_\mu\mid e_\nu,\text{extra})=P(e_\mu\mid e_\nu)\cdot R$ with $R=P(\text{extra}\mid e_\mu,e_\nu)/
  P(\text{extra}\mid e_\nu)$.
- $R\gtrless1$ iff the extra syndrome favours $e_\mu$ / $\lnot e_\mu$; the full posterior is a
  *sharpened* version of the pairwise conditional.
- In the reweight‑relevant regime (first pass explains $\mu$'s detectors without $\mu$), $R<1$, so the
  pairwise reweight **overstates** $\mu$.
- $\alpha<1$ is a scalar proxy for the missing $R<1$ factor; the overshoot — and hence the optimal
  shrinkage — grows as $p\to0$ because the discount magnitude is $\sim\log(1/p)$. None of this depends on
  the reliability of $\nu$'s selection, which is why optimal $\alpha$ and calibrated trust trend
  oppositely in $p$.
- Computing $R$ exactly (not via a scalar) is belief matching; Pearl‑CM is its cheap scalar approximation.
