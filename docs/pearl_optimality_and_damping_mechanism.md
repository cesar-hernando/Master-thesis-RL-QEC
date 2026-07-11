# Is Pearl's rule optimal, or just a damped CM? And if so, why is damping needed?

Companion to `pearl_vs_bayes_low_p.md`, `pearl_cm_success_examples.md`, and
`pearl_alpha_true_vs_optimal.md`. This note argues that, in the regime where correlated matching (CM)
actually helps, **Pearl's rule is not doing anything its damped-Bayes shadow doesn't already do** — so
the soft-evidence story is not the operative mechanism. It then asks what the operative mechanism
*really* is, and lists the candidates.

---

## 1. What the experiments actually show

Two empirical facts (from `notebooks/pearl_vs_damped_bayes.ipynb` and
`notebooks/pearl_vs_cm_failure_analysis.ipynb`):

1. **At low $p$, Pearl is decode-identical to "damped Bayes."** Pearl uses
   $p^{\rm imp}=\alpha P(\mu\mid\nu)+(1-\alpha)P(\mu\mid\lnot\nu)$; dropping the second term gives
   $p^{\rm imp}=\alpha P(\mu\mid\nu)$. The two differ by $(1-\alpha)P(\mu\mid\lnot\nu)=O((1-\alpha)p)$,
   and they produce **the same logical predictions** at $p=3\times10^{-4}$ (0 disagreements in
   $1.5\times10^4$ shots). The Pearl-specific anti-evidence term is **decorative** where CM helps; it
   only bites near threshold, where nothing helps much.
2. **The optimal $\alpha$ is not a probability.** The calibrated trust
   $P(\nu\text{ fired}\mid\nu\text{ selected})\approx0.99$, but the LER-optimal $\alpha\approx0.3$.

Together: the part of Pearl's rule that distinguishes it from "scale the hard conditional by $\alpha$"
does no work, and the parameter that is supposed to be a soft-evidence probability isn't one. The
**parsimonious description** of what's running is therefore:

$$p^{\rm imp}=\alpha\,P(\mu\mid\nu),\qquad\text{equivalently}\qquad w_\mu \approx w_\mu^{\rm CM}-\log\alpha,$$

i.e. **apply a fraction of CM's discount** (a uniform reduction of the discount by $|\log\alpha|$ on the
edges that drive the decode). That is just **damped / regularized correlated matching**.

---

## 2. So is Pearl genuinely the optimal reweight? No.

- Pearl's derivation fixes a *specific functional form* (the mixture) and a *specific interpretation*
  ($\alpha=$ trust). At low $p$ the form collapses to damped CM (fact 1) and the interpretation is
  false (fact 2). Neither the form nor the principle is what produces the gain — so calling it
  "optimal" is unsupported; it is one (slightly baroque) **parameterization of damping**.
- The genuinely optimal reweight is the **true per-fault posterior given the whole syndrome**,
  $P(\mu\mid \mathbf{s})$ — i.e. maximum-likelihood decoding, which belief propagation / belief matching
  approximate by message passing. Hard CM and Pearl are both **rank-1 heuristics**: they replace
  $P(\mu\mid\mathbf s)$ with a single pairwise conditional $P(\mu\mid\nu)$ (or a min over neighbours).
  "Optimal among $\alpha$" is not "optimal" — the relevant ceiling is BP/belief matching and the ML
  bound, not $\alpha=1$.

**Bottom line:** Pearl $\equiv$ damped CM where it matters, and damped CM is a cheap scalar
regularization of an overconfident rank-1 correction — not the Bayes-optimal reweight.

---

## 3. Then why is damping necessary? Candidate mechanisms

If "soft evidence" is not the reason, what is? Four non-exclusive mechanisms, all of which push the
**optimal discount below the nominal CM discount**.

### M1 — The pairwise conditional is overconfident (it ignores counter-evidence)
CM reweights $\mu$ with $P(\mu\mid\nu)$, but the decode-relevant quantity is $P(\mu\mid\mathbf s)$, the
posterior given the *whole* syndrome. The single-pair conditional uses only the *supporting*
correlation and ignores the rest of the syndrome, which often provides *counter*-evidence against
$\mu$. So $P(\mu\mid\nu)\ge P(\mu\mid\mathbf s)$ systematically, and the nominal discount is too large.
$\alpha<1$ is a crude scalar proxy for "shrink because we omitted the counter-evidence that full
message passing would include." **Prediction:** belief matching (which does include it) should close
most of the CM$\to$best-$\alpha$ gap — i.e. $\mathrm{LER}_{\rm BM}\approx\mathrm{LER}_{\alpha^\star}$.

### M2 — Double use of the syndrome (the reweight is circular)
The selection of $\nu$ is itself the output of MWPM on syndrome $\mathbf s$, and the second pass
re-decodes the **same** $\mathbf s$ with $\mu$ reweighted toward the first pass's conclusion. The
"evidence" and the "data" are the same object, so hard CM **reinforces the first pass** — helpful when
it was right, self-confirming when it was wrong (degenerate ties). This is the matching analogue of
using the training set to set the prior: it produces optimism that must be regularized away.
**Prediction:** a sample-split / cross-fit scheme (form the reweight from one part of the information,
decode the other) would reduce the *need* for damping ($\alpha^\star\to1$).

### M3 — Asymmetric decision risk in the matching
The reweight only changes the decode at margins, and the loss is **asymmetric**: over-discounting turns
$\mu$ into a negative-weight *attractor* that flips the decode across a logical (catastrophic), whereas
under-discounting merely leaves a correct edge slightly expensive (usually still matched, mild cost).
Under such asymmetric/convex loss the expected-loss-minimising discount is **smaller** than the
unbiased estimate — classic shrinkage. M3 sets *how far* below 1 the optimum sits (it can push
$\alpha^\star$ even below the pivotal reliability). **Prediction:** an asymmetric penalty (only damp in
the direction that creates attractors) would beat a symmetric $\alpha$.

### M4 — Winner-take-all aggregation across neighbours
The analytic rule takes the **largest** discount (min weight) over all selected neighbours of $\mu$.
Picking the most favourable single piece of evidence is aggressive and compounds M1. $\alpha<1$ tempers
it. **Prediction:** replacing min-discount with an averaged/Bayesian combination changes $\alpha^\star$.

---

## 4. Which mechanism is it, most likely?

My assessment, in order of contribution:

- **M1 (overconfident partial conditional) and M2 (circular double-use) are the primary reasons damping
  is *needed*** — both make the nominal CM discount systematically too large relative to the true
  posterior on the same syndrome.
- **M3 (asymmetric loss) sets *how much* to damp** — it explains why $\alpha^\star$ lands well below
  even the calibrated/pivotal reliability, i.e. why you over-shrink.
- **M4 is a secondary amplifier.**

These are facets of one statement: *the hard conditional is not the posterior, and committing to it on
the very syndrome that produced it, under an asymmetric loss, over-corrects.* Damping is the one-scalar
fix. Pearl's mixture is just one way to write that scalar; the anti-evidence term is a low-$p$ no-op.

---

## 5. Consequences for the thesis / a paper

- **State it honestly.** "Soft evidence" is a useful *lens* and a clean way to motivate $\alpha$, but
  the operative mechanism in the helpful regime is **regularization of an overconfident, circularly-fed
  rank-1 correlation correction**, not Bayesian belief updating. Claiming Pearl's form is responsible is
  unfalsifiable at low $p$ (it is indistinguishable from damped CM there).
- **The decisive experiments** are the M1–M3 predictions above:
  1. **Belief matching comparison** (tests M1): does proper message passing reach best-$\alpha$ CM?
  2. **Sample-splitting / cross-fit reweight** (tests M2): does decoupling selection from re-decode
     push $\alpha^\star\to1$?
  3. **Asymmetric damping** (tests M3): does penalising only attractor-direction reweights beat scalar
     $\alpha$?
- **A more principled $\alpha$** would be *derived* from the overconfidence/asymmetry (e.g. calibrated
  on pivotal/near-tie shots, or fit as a shrinkage estimator), not tuned — see
  `pearl_alpha_true_vs_optimal.md`.

This is a stronger and more defensible story than "Pearl's rule is the optimal correlated reweight": you
show that a one-line damping captures the entire low-$p$ gain, you explain *why* damping is needed
(M1–M3), and you point at belief matching as the principled ceiling.

---

### Caveats / scope
- "Pearl $\equiv$ damped CM" is established **at the decode level for low $p$** (rotated surface code,
  circuit-level depolarizing, $d=5$). Near threshold the anti-evidence term is non-negligible and the
  two genuinely differ — but neither helps much there.
- M1–M4 are hypotheses with stated tests, not yet separately confirmed; the BM comparison and the
  sample-split experiment are the cheapest ways to attribute the effect.
- None of this diminishes the empirical result (a free ~30% LER reduction over hard CM at low $p$); it
  *reframes the explanation* from "soft evidence" to "regularization," which is both more accurate and
  more useful for choosing/deriving $\alpha$.
