# Computing $\alpha$: the calibrated ("true") value vs. the LER-optimal effective value

Companion to `pearl_vs_bayes_low_p.md` and `pearl_cm_success_examples.md`. This note defines the two
distinct quantities both called "$\alpha$", gives the recipe to compute each, reports whether they
match (they do **not**), and explains the mismatch.

---

## 0. Where $\alpha$ enters

In Pearl soft-evidence correlated matching, when the first pass selects a neighbour edge $\nu$, a
correlated edge $\mu$ is reweighted using

$$
p^{\rm imp}(\mu)=\alpha\,P(\mu\mid\nu)+(1-\alpha)\,P(\mu\mid\lnot\nu),
\qquad w^{\rm imp}(\mu)=\log\frac{1-p^{\rm imp}}{p^{\rm imp}},
$$

with $P(\mu\mid\nu)=P(\mu,\nu)/P(\nu)$ and $P(\mu\mid\lnot\nu)=(P(\mu)-P(\mu,\nu))/(1-P(\nu))$.
$\alpha=1$ is ordinary hard-evidence CM; $\alpha<1$ softens the trust placed in the selection of $\nu$.

There are **two different things** people mean by "$\alpha$":

| | symbol | definition | nature |
|---|---|---|---|
| **True / calibrated** | $\alpha^{\rm cal}$ | $P(\nu\text{ actually fired}\mid\nu\text{ selected by first-pass MWPM})$ | a *probability*, measurable from noise + decoder |
| **Optimal / effective** | $\alpha^\star$ | $\arg\min_\alpha \mathrm{LER}(\alpha)$ | a *tuning parameter*, defined by a downstream objective |

Pearl's derivation *suggests* you should plug in $\alpha=\alpha^{\rm cal}$. The empirical question is
whether $\alpha^{\rm cal}=\alpha^\star$. It does not.

---

## 1. How to compute the **true** (calibrated) $\alpha^{\rm cal}$

This is a Monte-Carlo conditional probability using DEM ground truth. (Implemented in
`notebooks/pearl_vs_cm_failure_analysis.ipynb`, Part B.)

1. **Ground truth — which mechanisms fired.** Sample the DEM with errors returned:
   ```python
   det, obs, errs = dem.compile_sampler().sample(shots=N, return_errors=True)   # errs: (N, n_errors)
   ```
2. **Map each DEM error to decoding-graph edge(s).** Split each error's targets into `^`-separated
   components; in a Tesseract-decomposed DEM each component flips $\le2$ detectors $\to$ one edge via
   `g.pair_to_idx_matrix`. Build a parity matrix $M$ ($n_{\rm errors}\times n_{\rm edges}$).
   ```python
   true_parity = (errs.astype(int) @ M) % 2      # (N, n_edges): edge i fired iff ==1
   ```
3. **First-pass selection.**
   ```python
   sel = matching.decode_batch(det, enable_correlations=False).astype(bool)   # (N, n_edges)
   ```
4. **Conditional probability**, per edge and pooled:
   $$
   \alpha^{\rm cal}_i=\frac{\#\{\text{shots}:\text{edge }i\text{ selected and fired}\}}{\#\{\text{shots}:\text{edge }i\text{ selected}\}},
   \qquad
   \alpha^{\rm cal}_{\rm pool}=\frac{\sum_i\#\{\text{sel}\wedge\text{fired}\}_i}{\sum_i\#\{\text{sel}\}_i}.
   $$

**Definitional choices that change the number** — state which you use:

- **"Fired" is parity** (mod-2 edge support of the true fault), the matching-relevant notion. A
  selection that is homologically equivalent but on a *different* edge counts as a miss.
- **Conditioning regime** (this is the crucial one):
  - *All shots* — pools in the trivial low-detection shots (most reliable), inflating the estimate.
  - *Non-bypassed shots* (detection count $>$ `bypass_threshold`) — the selections that actually feed
    a reweight. Filter `keep = det.sum(1) > bypass` before counting.
  - *Pivotal shots* (the reweight changes the logical decode, i.e. a near-degenerate tie) — the regime
    $\alpha$ truly governs.
- **Error bars:** $(\text{shot},\text{edge})$ events are correlated within a shot; a naive binomial CI
  is too tight — **bootstrap over shots**.

---

## 2. How to compute the **optimal effective** $\alpha^\star$

A 1-D minimisation of the logical error rate. (Implemented in `scripts/test_alpha_decode.py` /
`scripts/pearl_alpha_sweep.py`.)

1. Decode the **same** syndromes (common random numbers) with a grid of $\alpha$.
2. Accumulate logical errors until the reference (CM, $\alpha=1$) hits a target error count.
3. $\alpha^\star=\arg\min_\alpha \mathrm{LER}(\alpha)$. The curve is **U-shaped** in $\alpha$
   ($\alpha\to0$ reverts to no-correction $\approx$ MWPM; $\alpha=1$ is hard CM; the minimum is
   interior). Use enough errors ($\gtrsim$ a few hundred) and report the binomial std.

---

## 3. Do they match? **No.**

Measured for the rotated surface code, circuit-level depolarizing noise, $d=5$, $p=3\times10^{-4}$:

| quantity | value |
|---|---|
| $\alpha^{\rm cal}$, pooled, **all shots** | $\approx 0.993$ |
| $\alpha^{\rm cal}$, **non-bypassed** shots | somewhat lower, still high (~0.95–0.99) |
| selection reliability on **pivotal** shots | $\approx 0.17$ (measured: hard CM correct on $38/225$ of the shots where $\alpha{=}1$ and $\alpha{=}0.3$ disagree) |
| **$\alpha^\star$** (LER-optimal) | $\approx 0.2\text{–}0.4$ |

The calibrated value ($\approx0.99$) and the optimal value ($\approx0.3$) are nowhere near each other.
Plugging the *literal* Pearl probability ($0.99$) into the decoder is close to hard CM and is **worse**
than the optimal $0.3$.

---

## 4. Why the mismatch

Four compounding effects, in order of importance.

**(A) $\alpha$ only acts on pivotal cases, where the selection is far less reliable than its marginal.**
The reweight of $\mu$ changes the final decode **only** on rare near-degenerate syndromes (tiny
second-pass margin). Conditioned on "this reweight is pivotal," the first-pass selection is close to a
coin flip — measured $\approx0.17$ here, not $0.99$. $\alpha^\star$ is calibrated to *that* regime.
Formally, $\alpha^{\rm cal}$ is the **marginal** $P(\text{fired}\mid\text{selected})$; what matters is
$P(\text{fired}\mid\text{selected},\text{pivotal})$, and the two differ by orders of magnitude at low $p$.

**(B) The reliable 99% are $\alpha$-insensitive, so they cast no vote for high $\alpha$.** On the
non-pivotal shots (the vast majority) the correct matching wins regardless of $\alpha$ — the LER there
does not depend on $\alpha$. Only pivotal shots move the LER, and they pull $\alpha$ **down**. So
$\arg\min_\alpha\mathrm{LER}$ tracks the pivotal reliability, not the marginal:
$$
\frac{\partial\,\mathrm{LER}}{\partial\alpha}\ \text{is supported only on pivotal syndromes.}
$$

**(C) Asymmetric loss $\Rightarrow$ shrinkage below even the pivotal probability.** Hard CM applies the
*full* discount: at low $p$ it drops $w_\mu$ by $\sim\log(1/p)$, turning $\mu$ into a near-free /
negative-weight attractor. When the selection is wrong this large discount **reliably flips the decode
into a logical error** (big, irreversible downside); when it is right the discount is usually redundant
(small upside). Under a convex loss with an overconfident estimate, the optimum **shrinks** the
correction — exactly like ridge/regularisation pushing a coefficient below its unbiased value. This
pushes $\alpha^\star$ even lower than the pivotal $P(\text{fired})$.

**(D) The pairwise rule is itself overconfident.** $P(\mu\mid\nu)$ comes from a pairwise/independence
approximation of the DEM and is accumulated across multiple selected neighbours (the min-discount step).
It over-discounts relative to the true posterior, and $\alpha<1$ absorbs that model mismatch.

(A)+(B) explain why $\alpha^\star\ll\alpha^{\rm cal}$; (C)+(D) explain why $\alpha^\star$ lands *below*
even the pivotal reliability.

---

## 5. Practical guidance

- **Do not set $\alpha$ from the marginal calibration** $\alpha^{\rm cal}_{\rm pool}$ — it is the wrong
  number despite being the "literal" Pearl probability.
- **Either** tune $\alpha^\star$ by a cheap LER sweep (§2; the native decoder makes this minutes), **or**
  estimate a *pivotal-conditioned* calibration — restrict §1 to shots with a small second-pass margin —
  which lands much closer to $\alpha^\star$ and is the principled bridge between the two.
- The cheap sweep is the safe default; the pivotal calibration is the more *explanatory* route and a
  good thesis result in its own right.

---

## 6. Why the mismatch is the point, not a bug

The gap $\alpha^{\rm cal}\approx0.99$ vs. $\alpha^\star\approx0.3$ is the cleanest evidence that
ordinary correlated matching's **hard-evidence assumption is suboptimal precisely where it matters**.
If the assumption were right, $\alpha^\star$ would equal the calibrated $\alpha^{\rm cal}\approx1$.
Instead the optimum sits deep below it, showing that the correction must be **softened/shrunk** because
(i) it only acts in the regime where the selection is unreliable and (ii) over-committing there is
expensive. Reporting all three numbers — marginal ($\approx0.99$), pivotal ($\approx0.17$), optimal
($\approx0.3$) — and the mechanism above is a complete, defensible story.

### Caveats / scope
- Numbers are for one code (rotated surface), noise model (circuit-level depolarizing), basis (Z memory),
  $d=5$, $p=3\times10^{-4}$. The qualitative ordering $\alpha^\star\ll\alpha^{\rm cal}$ is expected to
  hold across low-$p$ settings but should be re-measured per regime.
- "Pivotal" is operationalised here as "the reweight changes the prediction" (e.g. the $\alpha{=}1$ vs
  $\alpha{<}1$ disagreement set); a margin-based definition (small second-pass weight gap) is a cleaner,
  decoder-internal alternative.
- $\alpha^{\rm cal}$ is a *marginal* per-edge/pooled probability; the per-edge distribution is broad
  (boundary/degenerate edges are less reliable), so quote the histogram, not just the pooled scalar.
