# Fowler correlated matching vs. the conditional (PyMatching) reweight

**Question.** Fowler (2013, *Optimal complexity correction of correlated errors in the surface
code*, [arXiv:1310.0863](https://arxiv.org/abs/1310.0863)) describes the correlated-matching
second-pass reweight as: *believe the matched cylinder with certainty and uniformly scale the
probabilities of the mechanisms that produce it so they sum to 1.* The PyMatching fork (and this
repo's `analytical_cm_second_pass_weights`) instead use the pairwise conditional
`P(µ|ν) = corr(µ,ν) / occ(ν)`. **Are these the same rule?** If not, the difference could explain
performance gaps between PyMatching's correlated matching and the paper.

**Short answer.** They are the *same rule to leading order*. The joint (numerator) is identical; the
two differ only in the **normaliser**, an O(p) relative effect (Fowler uses a plain sum, PyMatching
uses the exact odd-parity marginal). At the decode level they agree except on rare near-degenerate
matchings, and PyMatching's built-in `enable_correlations` — which *is* Fowler's two-pass algorithm —
matches the analytical conditional decoder. The reweight formula is therefore **not** a source of any
large performance difference; look instead to the error *decomposition*, the noise model, or the
soft-evidence `alpha`.

---

## 1. Notation

A fault **mechanism** `m` is one physical fault (a gate, measurement, or reset error) with independent
probability `p_m`. After decomposition it flips a set of decoding-graph edges (its *components*). For a
first-pass-matched edge (cylinder) **ν** and a partner edge **µ** that shares a mechanism with ν:

- `M(ν)  = { m : ν ∈ components(m) }` — every mechanism that flips ν (ν's "causes").
- `M(µ,ν) = { m : {µ,ν} ⊆ components(m) }` — the **shared/correlated** mechanisms (e.g. a `Y` error
  whose `Z`-component is ν and `X`-component is µ, or a mixed two-qubit CNOT error).

Both rules produce an *implied probability* `p_imp(µ|ν)`, then map it to a weight
`w = log((1 − p_imp) / p_imp)` and take the **minimum weight over matched neighbours**. That weight
map and the min-over-neighbours step are **identical in both**; the only thing that differs is
`p_imp`.

---

## 2. The two rules

### Fowler — "scale the cylinder's causes to sum to 1"

Believing ν with certainty treats the mechanisms in `M(ν)` as **mutually exclusive and exhaustive**,
renormalised to total 1:

```
Z_ν = Σ_{m ∈ M(ν)} p_m              (plain SUM — the normaliser)
q_m = p_m / Z_ν                     (posterior of cause m)
```

The partner µ inherits the posterior mass of the shared causes:

```
                Σ_{m ∈ M(µ,ν)} p_m        S_both(µ,ν)
P_F(µ|ν) = ────────────────────────  =  ─────────────
                Σ_{m ∈ M(ν)}   p_m            Z_ν
                    (SUM)                     (SUM)
```

### PyMatching / analytical — the pairwise conditional

```
P_C(µ|ν) = corr(µ,ν) / occ(ν)
```

where both ingredients are **XOR-combined** — an edge fires iff an *odd* number of its mechanisms
fire (two faults on the same edge cancel). Writing `a ⊕ b = a(1−b) + b(1−a)`:

```
corr(µ,ν) = ⊕_{m ∈ M(µ,ν)} p_m  =  Σ p_m − 2 Σ_{i<j} p_i p_j + …     (odd-parity joint)
occ(ν)    = ⊕_{m ∈ M(ν)}   p_m  =  Σ p_m − 2 Σ_{i<j} p_i p_j + …     (odd-parity marginal)
```

`corr` and `occ` are exactly the `initial_corr_tracer` and `base_p` of `DecodingGraph`.

---

## 3. Where they differ — exactly

**Numerator (joint): identical in practice.** `S_both` (sum) and `corr` (odd-parity) differ only at
O(p²), and only if a pair `(µ,ν)` shares **≥2** mechanisms. At these code distances each correlated
pair has essentially one shared mechanism, so `⊕` of a single term equals that term equals the sum.
Measured directly: **`|corr − S_both| = 0` exactly.**

**Denominator (normaliser): this is the entire difference.** ν has *many* causes (several gate faults
+ measurement + boundary + the correlating error), so `N = |M(ν)|` is large and the parity correction
is non-trivial:

```
Z_ν − occ(ν) ≈ 2·e₂ = 2 Σ_{i<j} p_i p_j ≈ (N p)²        (e_k = k-th elementary symmetric poly)
```

Because `occ(ν) < Z_ν` (parity cancellations shrink the marginal below the raw sum) while the
numerators are equal:

```
P_C(µ|ν)      Z_ν
─────────  =  ──────  =  1 + O(N·p)   ≥ 1
P_F(µ|ν)     occ(ν)
```

So **the conditional rule always boosts µ slightly *more* than Fowler**, by a relative factor
`Z_ν / occ(ν) ≈ 1 + N·p`. This scales like `p`: larger near threshold, vanishing as `p → 0`.

---

## 4. The conceptual difference, in one line

- **Fowler** = the conditional under the approximation *"exactly one of ν's causes fired"*
  (mutually-exclusive causes ⇒ **sum** normaliser).
- **PyMatching** = the *exact* Bayesian conditional under *"independent mechanisms XOR-ing onto
  edges"* (true **odd-parity marginal** `P(ν)` in the denominator).

They coincide in the single-fault regime and part company at O(p²) precisely when ν has **multiple
simultaneously-plausible causes** — which is what more shots / higher `p` exposes.

---

## 5. What is *not* a Fowler-vs-conditional difference (shared approximations)

Two things could make "correlated matching" differ more, but they are choices *both* implementations
share here, so they do **not** distinguish the two formulas:

- **Multiple matched neighbours of µ:** taking the single strongest boost (`min` weight) rather than
  compounding beliefs from several matched cylinders.
- **Multi-component hyperedges** (one mechanism flipping ν and *two* partners µ₁, µ₂): the pairwise
  rule boosts µ₁, µ₂ independently and does not encode that they must fire *together*, whereas
  Fowler's "believe the whole cylinder" ties them. At surface-code level ≥3-component mechanisms are
  rare, but this is the one place a genuinely *structural* (not just O(p²)) difference could hide.

---

## 6. Empirical verification

Reproduce with
[`scripts/test_fowler_vs_conditional_reweighting.py`](../scripts/test_fowler_vs_conditional_reweighting.py)
(`--distance --p --shots --bypass`). It (Part 1) compares the two formulas on every correlated pair,
and (Part 2) decodes real shots with MWPM, PyMatching's built-in `enable_correlations`, the analytical
conditional two-pass, and a Fowler-normalised two-pass.

**Formula level** (d=5, p=2×10⁻³):

| quantity | value |
|---|---|
| `|corr − S_both|` (joint) | `0.0` (identical) |
| `|occ − Z_sum|` (normaliser) | `≈ 3×10⁻⁴` |
| `|P_fowler − P_cond|` | max `1.4×10⁻³`, mean `2.3×10⁻⁴` |
| relative `|P_fowler − P_cond| / P_cond` | max `1.4 %`, mean `0.4 %` |

**Decode level** (logical errors on the same shots):

| decoder | p=2×10⁻³ (10⁵ shots) | p=4×10⁻⁴ (10⁶ shots) |
|---|---|---|
| MWPM | 104 | 7 |
| CM built-in (α=1) | 66 | 5 |
| CM conditional 2-pass | 66 | 5 |
| CM Fowler 2-pass | 66 | 5 |

**0 disagreements** among conditional / Fowler / built-in at these shot counts — but that means
"below the ~10⁻⁶ resolution of those runs," **not exact equality**. The normaliser genuinely differs
at O(p), so conditional-vs-Fowler decode flips *do* exist on rare near-degenerate matchings; catching
them needs many shots, or a higher `p` where the gap is larger and near-ties are more common.

**High-`p` confirmation** (d=5, p=10⁻², 10⁶ shots, `bypass=0`) — the difference is real but performance-negligible:

| decoder | logical errors | LER |
|---|---|---|
| MWPM | 81 917 | 8.19×10⁻² |
| CM built-in (α=1) | 69 903 | 6.990×10⁻² |
| CM conditional 2-pass | 69 901 | 6.990×10⁻² |
| CM Fowler 2-pass | 69 921 | 6.992×10⁻² |

| disagreements / 10⁶ | count |
|---|---|
| conditional vs Fowler | **662** |
| Fowler vs built-in | 670 |
| built-in vs MWPM | 45 860 |

Three facts fall out:

1. **The disagreement is real** — 662/10⁶ conditional-vs-Fowler flips at p=10⁻², versus 0 detected at
   p≤2×10⁻³. So "equivalent" only ever meant "below resolution"; the normaliser difference does change
   the decode on rare near-ties.
2. **It is performance-negligible** — the flips go both ways and cancel: conditional LER 6.990×10⁻²
   vs Fowler 6.992×10⁻² is a 20-error difference on ~70 000, well inside the ±265 statistical noise.
   The normaliser does **not** meaningfully change decoding accuracy.
3. **The built-in sides with the conditional, not Fowler** — `Fowler-vs-built-in (670) ≈
   conditional-vs-Fowler (662)`, so PyMatching's `enable_correlations` uses the odd-parity marginal
   `occ` in the denominator (the exact conditional). Fowler's literal "sum to 1" (`Z_ν`) is the
   outlier — by a negligible margin.

The normaliser gap scales as predicted: `|occ − Z_sum|` grew `2.9×10⁻⁴ → 6.9×10⁻³` from p=2×10⁻³ to
p=10⁻² (≈24×, i.e. `(Np)²` for a 5× increase in `p`), and the decode-disagreement rate follows it down
toward 0 as `p → 0`. So at the low `p` of the α-analysis the two rules are indistinguishable in
practice.

> **Runtime note.** The p=10⁻², 10⁶-shot run above took ~13 min (the two-pass second stage is a
> per-shot Python loop, and `bypass=0` sends every non-trivial shot through it). A 10⁸-shot run at
> that `p` would be ~21 h; at low `p` it would find essentially nothing (rate → 0). Confirm the effect
> at high `p`, not with a huge low-`p` run.

---

## 7. Conclusions

1. **The reweight *formula* is not a confound.** Fowler's uniform-scaling and the pairwise
   `corr/occ` conditional share an identical joint and differ only by the normaliser (sum vs
   odd-parity marginal), an O(p) relative effect. The decode difference is real but tiny — measured at
   662/10⁶ near-tie flips at p=10⁻² (and undetectable at p≤2×10⁻³), with **no** significant LER change
   (6.990×10⁻² vs 6.992×10⁻²) — and it vanishes as `p → 0`.
2. **PyMatching's correlated matching *is* Fowler's.** The built-in `enable_correlations`, the repo's
   analytical two-pass, and a Fowler-normalised two-pass all decode identically at the shot counts
   tested. So the α-sweep results and the overcorrection / `alpha_best(p)` analysis are on **genuine
   Fowler correlated matching**, not an implementation artefact.
3. **If PyMatching CM differs from the paper's numbers, look elsewhere:**
   - the **error decomposition** (this repo uses stim's coordinate-based Tesseract decomposition to
     decide which errors share a mechanism; Fowler enumerates per-gate cylinders — different
     decompositions give a different `corr` structure with the *same* formula);
   - the **noise model** (circuit-level depolarizing here vs. the paper's model);
   - the **soft-evidence `alpha`** (the paper is hard evidence, `alpha = 1`; the `alpha`-sweep is an
     extension of this work, not part of Fowler).
