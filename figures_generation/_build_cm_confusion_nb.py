#!/usr/bin/env python
"""Builds figures_generation/cm_vs_mwpm_confusion.ipynb."""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# Where correlated matching goes wrong: weight-2 (d=5) and weight-3 (d=7) failures

Hard correlated matching (CM, $\alpha=1$) **loses a unit of effective distance** at low $p$: there
are low-weight error configurations that plain MWPM corrects but CM mis-decodes. This notebook

1. finds a **weight-2** configuration (two faults) at $d=5$ that MWPM corrects but hard CM fails,
   and draws **MWPM's matching (left) vs CM's matching (right)** on the 3-D decoding graph;
2. counts, for several $p$, how many weight-2 configurations confuse **CM** vs **MWPM** at $d=5$;
3. repeats for $d=7$ with **weight-3** configurations;
4. shows the mechanism: with CM($\alpha=1$) the **boosted (posterior) edge weight stays $O(1)$**
   while the **prior bulk edge weight grows as $\log(1/p)$** &mdash; broken scale invariance.
""")

code(r"""
%matplotlib inline
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "figures_generation" else os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "src"))
import stim, pymatching
from NeuralCM.decompose_errors import decompose_errors_for_stim_surface_code_coords
from NeuralCM.decoding_graph import DecodingGraph

FIG = os.path.join(ROOT, "plots", "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 110, "mathtext.fontset": "cm"})


def savepair(fig, name, dpi=200):
    "Save the figure as both PNG (raster) and PDF (vector) into plots/figures/."
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=dpi, bbox_inches="tight")


def build(D, P):
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=D, rounds=D,
        after_clifford_depolarization=P, before_measure_flip_probability=P,
        after_reset_flip_probability=P, before_round_data_depolarization=P)
    dem = decompose_errors_for_stim_surface_code_coords(circ.detector_error_model(decompose_errors=False))
    mwpm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=False)
    cm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    return circ, dem, mwpm, cm


def mechanisms(dem):
    "Return (mechs, det_masks, obs_arr, is_hyperedge); mechs[i] = (frozenset detectors, obs)."
    ND = dem.num_detectors; mechs, hyper = [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets, obs, sep = set(), 0, False
        for t in inst.targets_copy():
            if t.is_relative_detector_id(): dets ^= {t.val}
            elif t.is_logical_observable_id(): obs ^= 1
            elif t.is_separator(): sep = True
        mechs.append((frozenset(dets), obs)); hyper.append(sep)
    M = len(mechs)
    dm = np.zeros((M, ND), np.uint8); ob = np.zeros(M, np.uint8)
    for i, (d, o) in enumerate(mechs):
        dm[i, list(d)] = 1; ob[i] = o
    return mechs, dm, ob, np.array(hyper)


def gate_map(circ):
    "net-syndrome frozenset -> (gate, Pauli), to label the physical origin (CNOT = DEPOLARIZE2)."
    m = {}
    for e in circ.explain_detector_error_model_errors(reduce_to_one_representative_error=True):
        ds = frozenset(t.dem_target.val for t in e.dem_error_terms if t.dem_target.is_relative_detector_id())
        if e.circuit_error_locations:
            loc = e.circuit_error_locations[0]
            pl = "".join(sorted(g.gate_target.pauli_type for g in loc.flipped_pauli_product))
            m.setdefault(ds, (loc.instruction_targets.gate, pl))
    return m


def graph_edges(dem):
    "Graphlike structure for drawing: set of 2-detector edges and set of boundary detectors."
    edges, boundary = set(), set()
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        comp = [[]]
        for t in inst.targets_copy():
            if t.is_separator(): comp.append([])
            elif t.is_relative_detector_id(): comp[-1].append(t.val)
        for grp in comp:
            if len(grp) == 2: edges.add(tuple(sorted(grp)))
            elif len(grp) == 1: boundary.add(grp[0])
    return edges, boundary


# Every graphlike edge connects two detectors of the SAME stabilizer basis; a detector's basis is
# fixed by its (x, y) coordinate via ((x+y)//2) % 2.  type 0 = X-type (Z-stabilizer, detects X /
# bit-flip errors), type 1 = Z-type (X-stabilizer, detects Z / phase errors).  We colour fired
# detectors AND matched edges by this basis, so the X-graph and Z-graph corrections are separable.
TCOL = {0: "#1f77b4", 1: "#d62728"}
TNAME = {0: "X-type detector / edge  (bit-flip)", 1: "Z-type detector / edge  (phase)"}


def det_type(coords):
    return {k: int((c[0] + c[1]) // 2) % 2 for k, c in coords.items()}


def zoom_box(coords, fired, mg=1.6, tm=1.0):
    "Bounding box (xl, yl, tl) around the fired detectors, for a zoomed local view."
    fx = [coords[f][0] for f in fired]; fy = [coords[f][1] for f in fired]; ft = [coords[f][2] for f in fired]
    return (min(fx) - mg, max(fx) + mg), (min(fy) - mg, max(fy) + mg), (min(ft) - tm, max(ft) + tm)


def type_legend(fig, y=-0.02):
    h = [Line2D([0], [0], marker="o", ls="none", color=TCOL[t], ms=9, label=TNAME[t]) for t in (0, 1)]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, y))


def draw_matching(ax, coords, edges, typ, fired, matched, ok, title, view=(15, -72), zoom=None):
    "Circuit-level decoding graph in (x, y, t).  Faint full/local graph in grey; fired detectors and "
    "the decoder's matched edges coloured by X/Z basis.  `zoom` = (xl, yl, tl) crops to a local region."
    if zoom is None:
        keep = lambda k: True
    else:
        (xl, yl, tl) = zoom
        keep = lambda k: (xl[0] <= coords[k][0] <= xl[1] and yl[0] <= coords[k][1] <= yl[1]
                          and tl[0] <= coords[k][2] <= tl[1])
    for a, b in edges:                                       # faint graph
        if keep(a) and keep(b):
            xa, ya, ta = coords[a]; xb, yb, tb = coords[b]
            ax.plot([xa, xb], [ya, yb], [ta, tb], color="0.78", lw=0.6, alpha=0.5, zorder=1)
    for k, (x, y, t) in coords.items():                      # detector nodes
        if keep(k):
            ax.scatter([x], [y], [t], s=9, color="0.6", depthshade=False, zorder=2)
    fcx = np.mean([coords[f][0] for f in fired]); fcy = np.mean([coords[f][1] for f in fired])
    for f in fired:                                          # fired detectors: glow + solid, by type
        x, y, t = coords[f]; c = TCOL[typ[f]]
        ax.scatter([x], [y], [t], s=320, color=c, alpha=0.20, depthshade=False, zorder=3)
        ax.scatter([x], [y], [t], s=80, color=c, edgecolor="white", lw=0.8, depthshade=False, zorder=6)
    for a, b in matched:                                     # matched edges, coloured by type
        c = TCOL[typ[a if a != -1 else b]]
        if -1 in (a, b):
            n = a if b == -1 else b; x, y, t = coords[n]
            v = np.array([x - fcx, y - fcy]); v = v / (np.linalg.norm(v) + 1e-9) * 0.9
            ax.plot([x, x + v[0]], [y, y + v[1]], [t, t], color=c, lw=3.3, zorder=5, solid_capstyle="round")
        else:
            xa, ya, ta = coords[a]; xb, yb, tb = coords[b]
            ax.plot([xa, xb], [ya, yb], [ta, tb], color=c, lw=3.5, zorder=5, solid_capstyle="round")
    if zoom is not None:
        (xl, yl, tl) = zoom; ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*tl)
    ax.set_axis_off(); ax.view_init(elev=view[0], azim=view[1])
    ax.set_title(title, color=("#1e8449" if ok else "#c0392b"), fontsize=11)

print("helpers ready;  ROOT =", ROOT)
""")

md(r"""
## 1. A weight-2 configuration at $d=5$ that confuses hard CM

We build the $d=5$ circuit-level DEM at low $p$, enumerate its error **mechanisms**, and search
mechanism **pairs** (two faults) for one where CM($\alpha=1$) predicts the wrong logical outcome
while MWPM predicts the right one.
""")

code(r"""
D, P = 5, 2e-4
circ, dem, mwpm, cm = build(D, P)
mechs, dm, ob, hyper = mechanisms(dem)
M = len(mechs)

pair = None
for i in range(M):                                    # first pair where CM fails but MWPM succeeds
    s = dm[i] ^ dm[i + 1:]; obo = ob[i] ^ ob[i + 1:]
    pm = mwpm.decode_batch(s)[:, 0]
    c = cm.decode_batch(s, enable_correlations=True, alpha=1.0)[:, 0]
    sel = np.flatnonzero((c != obo) & (pm == obo))
    if len(sel):
        pair = (i, i + 1 + int(sel[0])); break
i, j = pair
syn = (dm[i] ^ dm[j]).astype(np.uint8); true_obs = int(ob[i] ^ ob[j])
gm = gate_map(circ)

print(f"d={D}, p={P:g}:  confusing weight-2 configuration = mechanisms {i} + {j}\n")
for x in (i, j):
    g, pl = gm.get(mechs[x][0], ("?", "?"))
    print(f"  mech {x}: detectors {sorted(mechs[x][0])}   obs={mechs[x][1]}   source: {g} ({pl})")
mp, cp = int(mwpm.decode(syn)[0]), int(cm.decode(syn, enable_correlations=True, alpha=1.0)[0])
print(f"\n  fired detectors : {np.flatnonzero(syn).tolist()}")
print(f"  TRUE obs flip   : {true_obs}")
print(f"  MWPM       -> {mp}   ({'CORRECT' if mp == true_obs else 'WRONG'})")
print(f"  CM(alpha=1)-> {cp}   ({'CORRECT' if cp == true_obs else 'WRONG'})")
print(f"\n  Both faults are CNOT (DEPOLARIZE2) cross-basis errors, i.e. hyperedges.")
""")

md(r"""
## 2. The two matchings side by side (zoomed, several view angles)

The **circuit-level decoding graph** in $(x,y,t)$, **zoomed to the fired region** and shown from
three viewpoints (columns) so the 3-D structure is unambiguous. Fired detectors are the **glowing**
nodes and each decoder's chosen correction is drawn in bold (short stubs = matches to the boundary).
Everything is **coloured by stabilizer basis**: <span style="color:#1f77b4">**X-type**</span>
(bit-flip / Z-stabilizer detectors and edges) vs <span style="color:#d62728">**Z-type**</span>
(phase / X-stabilizer). **Top row: MWPM** takes the minimum-weight correction (the true faults)
&rarr; correct. **Bottom row: CM($\alpha=1$)** re-weights the correlated edges after its first pass
and is pulled onto a different, distance-reducing correction &rarr; a logical error. Comparing the
two rows at the same viewpoint shows exactly which edge CM swaps.
""")

code(r"""
coords = {k: (c[0], c[1], c[2] if len(c) > 2 else 0.0) for k, c in circ.get_detector_coordinates().items()}
gedges, gbound = graph_edges(dem)
typ = det_type(coords)
fired = np.flatnonzero(syn).tolist()
edges_mwpm = mwpm.decode_to_edges_array(syn)
edges_cm = cm.decode_to_edges_array(syn, enable_correlations=True, alpha=1.0)
zb = zoom_box(coords, fired)
okm, okc = (mp == true_obs), (cp == true_obs)

VIEWS = [(18, -72), (8, 22), (34, -118)]
fig = plt.figure(figsize=(13.5, 8.6))
for cc, vw in enumerate(VIEWS):
    draw_matching(fig.add_subplot(2, 3, cc + 1, projection="3d"), coords, gedges, typ, fired,
                  edges_mwpm, okm, f"MWPM  -  view {cc + 1}", vw, zb)
    draw_matching(fig.add_subplot(2, 3, cc + 4, projection="3d"), coords, gedges, typ, fired,
                  edges_cm, okc, rf"CM ($\alpha=1$)  -  view {cc + 1}", vw, zb)
type_legend(fig)
fig.suptitle(f"Weight-2 configuration MWPM corrects (top) but CM $\\alpha{{=}}1$ fails (bottom)  "
             f"($d={D}$, $p={P:g}$)  —  zoomed, 3 view angles, coloured by X/Z basis", fontsize=12.5, y=0.99)
fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.07, wspace=0.02, hspace=0.10)
savepair(fig, "cm_vs_mwpm_weight2_matching", dpi=190)

# graphlike edges of the two actual fault mechanisms -> the "true underlying error"
def mech_component_edges(dem):
    per = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        comp = [[]]
        for t in inst.targets_copy():
            if t.is_separator(): comp.append([])
            elif t.is_relative_detector_id(): comp[-1].append(t.val)
        ed = []
        for grp in comp:
            if len(grp) == 2: ed.append([grp[0], grp[1]])
            elif len(grp) == 1: ed.append([grp[0], -1])
        per.append(ed)
    return per
me = mech_component_edges(dem)
true_edges = np.array(me[i] + me[j], dtype=int)

# standalone single-panel SVGs (view 1): the true error, MWPM's matching, CM's matching
for tag, matched, ok, ttl in [("true", true_edges, True, "true underlying error  -  view 1"),
                              ("mwpm", edges_mwpm, okm, "MWPM  -  view 1"),
                              ("cm", edges_cm, okc, r"CM ($\alpha=1$)  -  view 1")]:
    fs = plt.figure(figsize=(5.4, 5.6))
    draw_matching(fs.add_subplot(111, projection="3d"), coords, gedges, typ, fired,
                  matched, ok, ttl, VIEWS[0], zb)
    type_legend(fs)
    fs.savefig(os.path.join(FIG, f"cm_vs_mwpm_weight2_{tag}_view1.svg"), bbox_inches="tight")
    plt.close(fs)
plt.show()
""")

md(r"""
## 3. How many low-weight configurations confuse each decoder (vs $p$)

**Left ($d=5$, weight-2):** exhaustively decode **every** pair of mechanisms. Because $d=5$ has fault
distance 5, an optimal decoder corrects *all* 2-fault configurations &mdash; so **MWPM stays at 0**
(sub-threshold) while **CM($\alpha=1$) mis-decodes a few hundred**, all built from CNOT errors, with
the count *growing as $p\to0$*.

**Right ($d=7$, weight-3):** the fault distance is 7, so an optimal decoder corrects all 3-fault
configurations. Exhaustive triples are $\sim10^{10}$, so we **importance-sample** triples of
*hyperedge* mechanisms (the CNOT cross-basis errors responsible for the confusion) and scale the
measured failure **rate** by the number of such triples $C(H,3)\sim10^{10}$ to estimate the
**absolute number of confusing weight-3 configurations** &mdash; on the *same footing* as the $d=5$
exhaustive count (error bars are Poisson on the sampled hits). This matters: reported *per sampled
triple* the $d=7$ effect looks tiny, but that is only because the weight-3 space is astronomically
larger. In **absolute** terms the number of weight-3 configs that confuse CM is actually
$\sim\!10^{4}$&ndash;$10^{5}$, **~100&ndash;300$\times$ more than $d=5$'s few hundred** &mdash; the
effective-distance loss is *not* weaker at $d=7$. Again MWPM never fails, and CM's count rises as
$p\to0$.
""")

code(r"""
def count_2fault(D, P):
    "Exhaustive weight-2: (#CM failures, #CM failures with both faults CNOT, #MWPM failures)."
    circ, dem, mwpm, cm = build(D, P)
    mechs, dm, ob, hyper = mechanisms(dem)
    gm = gate_map(circ); M = len(mechs)
    cn = np.array([gm.get(mechs[x][0], ("?",))[0] == "DEPOLARIZE2" for x in range(M)])
    cmf = cmf_cnot = mwf = 0
    for i0 in range(0, M, 400):
        blocks, ii, jj = [], [], []
        for a in range(i0, min(i0 + 400, M)):
            b = np.arange(a + 1, M)
            if b.size:
                blocks.append(dm[a] ^ dm[a + 1:]); ii.append(np.full(b.size, a)); jj.append(b)
        if not blocks:
            continue
        s = np.vstack(blocks); ii = np.concatenate(ii); jj = np.concatenate(jj)
        obo = ob[ii] ^ ob[jj]
        pm = mwpm.decode_batch(s)[:, 0]; c = cm.decode_batch(s, enable_correlations=True, alpha=1.0)[:, 0]
        bc, bm = (c != obo), (pm != obo)
        cmf += int(bc.sum()); mwf += int(bm.sum())
        cmf_cnot += int((bc & cn[ii] & cn[jj]).sum())
    return cmf, cmf_cnot, mwf


def count_3fault_sampled(D, P, budget=90.0, seed=0):
    "Sampled weight-3 over ALL-hyperedge triples. The failure RATE among sampled triples is scaled by "
    "the number of such triples C(H,3) to estimate the absolute number of confusing weight-3 configs "
    "(directly comparable to the d=5 exhaustive count). Returns (cm_est, cm_err, mwpm_est, tot, cmf)."
    circ, dem, mwpm, cm = build(D, P)
    mechs, dm, ob, hyper = mechanisms(dem)
    H = np.flatnonzero(hyper); nH = len(H); nTri = nH * (nH - 1) * (nH - 2) / 6.0
    rng = np.random.default_rng(seed)
    tot = cmf = mwf = 0; t0 = time.time()
    while time.time() - t0 < budget:
        B = 400000
        ii = rng.choice(H, B); jj = rng.choice(H, B); kk = rng.choice(H, B)
        ok = (ii != jj) & (jj != kk) & (ii != kk); ii, jj, kk = ii[ok], jj[ok], kk[ok]
        s = dm[ii] ^ dm[jj] ^ dm[kk]; obo = ob[ii] ^ ob[jj] ^ ob[kk]
        pm = mwpm.decode_batch(s)[:, 0]; c = cm.decode_batch(s, enable_correlations=True, alpha=1.0)[:, 0]
        tot += len(ii); cmf += int((c != obo).sum()); mwf += int((pm != obo).sum())
    cm_est = cmf / tot * nTri; cm_err = cm_est / np.sqrt(max(cmf, 1))   # Poisson error on the estimate
    return cm_est, cm_err, mwf / tot * nTri, tot, cmf


# The d=7 sampling is expensive (~45 s/point), so the counts are computed once and cached.
# Re-executing the notebook to restyle the figure reloads the cache instantly; delete the
# .npz (or set RECOMPUTE = True) to regenerate the data.
CACHE = os.path.join(ROOT, "data", "cm_lowweight_counts_d5_d7.npz")
RECOMPUTE = False

if os.path.exists(CACHE) and not RECOMPUTE:
    z = np.load(CACHE)
    P5, cm5, mw5 = z["P5"], z["cm5"], z["mw5"]
    P7, cm7, cm7e, mw7 = z["P7"], z["cm7"], z["cm7e"], z["mw7"]
    print("loaded cached counts from", os.path.relpath(CACHE, ROOT),
          "(delete it or set RECOMPUTE=True to regenerate)")
else:
    PS5 = [2e-3, 1e-3, 4e-4, 2e-4, 1e-4]
    P5, cm5, mw5 = [], [], []
    print("d=5 (weight-2, exhaustive)")
    print(f"{'p':>8} {'CM fails':>9} {'(both CNOT)':>11} {'MWPM fails':>10}")
    for P in PS5:
        cf, cfc, mf = count_2fault(5, P)
        P5.append(P); cm5.append(cf); mw5.append(mf)
        print(f"{P:>8g} {cf:>9} {cfc:>11} {mf:>10}")

    PS7 = [2e-3, 1e-3, 4e-4, 2e-4, 1e-4]
    P7, cm7, cm7e, mw7 = [], [], [], []
    print("\nd=7 (weight-3, sampled all-hyperedge triples, scaled to absolute count)")
    print(f"{'p':>8} {'CM est':>12} {'+/-':>10} {'MWPM est':>10} {'sampled':>12} {'hits':>6}")
    for P in PS7:
        ce, cerr, me, tot, cmf = count_3fault_sampled(7, P)
        P7.append(P); cm7.append(ce); cm7e.append(cerr); mw7.append(me)
        print(f"{P:>8g} {ce:>12,.0f} {cerr:>10,.0f} {me:>10.0f} {tot:>12,} {cmf:>6}")

    P5, cm5, mw5 = map(np.array, (P5, cm5, mw5))
    P7, cm7, cm7e, mw7 = map(np.array, (P7, cm7, cm7e, mw7))
    np.savez(CACHE, P5=P5, cm5=cm5, mw5=mw5, P7=P7, cm7=cm7, cm7e=cm7e, mw7=mw7)
    print("saved counts to", os.path.relpath(CACHE, ROOT))

fig, (a5, a7) = plt.subplots(1, 2, figsize=(11.2, 4.4))
a5.plot(P5, cm5, "o-", color="#e67e22", label="CM")
a5.plot(P5, mw5, "s--", color="#2e86de", label="MWPM")
a5.set_title(r"$d=5$: weight-2 configurations (exhaustive)")
a5.set_ylabel("confusing configurations")

h_cm7 = a7.errorbar(P7, cm7 / 1e3, yerr=cm7e / 1e3, marker="o", ls="-", color="#e67e22",
                    capsize=3, elinewidth=1.0, label="CM")
h_mw7, = a7.plot(P7, mw7 / 1e3, "s--", color="#2e86de", label="MWPM")
a7.set_title(r"$d=7$: weight-3 configurations (sampled est.)")
a7.set_ylabel(r"estimated confusing configurations ($\times 10^{3}$)")
a5.legend()                                       # CM then MWPM (plot order)
a7.legend([h_cm7, h_mw7], ["CM", "MWPM"])         # force CM on top to match panel (a)
for ax, lab in zip((a5, a7), "ab"):
    ax.set_xscale("log"); ax.set_xlabel("Physical error rate $p$")
    ax.margins(y=0.12)
    hi = ax.get_ylim()[1]
    ax.set_ylim(-0.035 * hi, hi)          # small negative floor so the MWPM=0 markers stay visible
    ax.text(-0.14, 1.04, f"({lab})", transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.tight_layout()
savepair(fig, "cm_vs_mwpm_lowweight_counts_d5_d7")
plt.show()
print("\nNote: d=7's absolute count is ~100-300x larger than d=5's -- the weight-3 all-hyperedge")
print("space has C(H,3) ~ 1e10 triples, so the per-sample rate is tiny while the ABSOLUTE number")
print("of confusing configs is far larger. The 'less prominent' impression was a normalisation artifact.")
""")

md(r"""
## 3b. Above threshold: weight-$\frac{d+1}{2}$ configurations — does correlation help?

The counts above are **sub-threshold** (weight $\frac{d-1}{2}$: MWPM corrects all, hard CM breaks
some). At the **threshold** weight $\frac{d+1}{2}$ ($d=5\!\to\!3$, $d=7\!\to\!4$) MWPM itself starts
to fail. Here we sample uniformly over weight-$\frac{d+1}{2}$ fault combinations and compare the
failure rate of **MWPM**, **CM($\alpha=1$)** and **RCM($\alpha^{*}$)** on *identical* configurations
($\alpha^{*}$ from `reg_cm_alpha_scan_d3d5d7.csv`). The picture **flips**: the correlation information
now *helps* — CM corrects many configurations MWPM gets wrong, and RCM($\alpha^{*}$) helps most.
""")

code(r"""
import pandas as pd
scan = pd.read_csv(os.path.join(ROOT, "data", "reg_cm_alpha_scan_d3d5d7.csv"))


def best_alpha(d, p):
    s = scan[(scan.distance == d) & (np.abs(scan.p - p) < 1e-12) & (scan.decoder == "cm") & (scan.alpha < 1.0)]
    return float(s.loc[s.ler.idxmin()].alpha) if len(s) else 1.0


def count_threshold_sampled(D, P, budget, seed=0):
    "Uniformly sample weight-(D+1)/2 fault combos; failures per 1e6 for MWPM, CM(a=1), RCM(a*) on the same shots."
    w = (D + 1) // 2
    circ, dem, mwpm, cm = build(D, P)
    mechs, dm, ob, hyper = mechanisms(dem); M = len(mechs); a = best_alpha(D, P)
    rng = np.random.default_rng(seed); tot = nmw = ncm = nrcm = 0; t0 = time.time()
    while time.time() - t0 < budget:
        B = 100000
        idx = rng.integers(0, M, size=(w, B)); ok = np.ones(B, bool)
        for i in range(w):
            for j in range(i + 1, w):
                ok &= idx[i] != idx[j]
        idx = idx[:, ok]
        s = dm[idx[0]].copy(); obo = ob[idx[0]].copy()
        for r in range(1, w):
            s ^= dm[idx[r]]; obo ^= ob[idx[r]]
        pm = mwpm.decode_batch(s)[:, 0]
        c1 = cm.decode_batch(s, enable_correlations=True, alpha=1.0)[:, 0]
        ca = cm.decode_batch(s, enable_correlations=True, alpha=a)[:, 0]
        tot += idx.shape[1]
        nmw += int((pm != obo).sum()); ncm += int((c1 != obo).sum()); nrcm += int((ca != obo).sum())
    f = 1e6 / tot; err = lambda n: (n * f) / np.sqrt(max(n, 1))
    return dict(mw=nmw*f, cm=ncm*f, rcm=nrcm*f, mw_e=err(nmw), cm_e=err(ncm), rcm_e=err(nrcm), a=a)


PST = [2e-3, 1e-3, 4e-4, 2e-4, 1e-4]
CACHE_T = os.path.join(ROOT, "data", "cm_threshold_counts_d5_d7.npz")
if os.path.exists(CACHE_T):
    z = np.load(CACHE_T); T = {k: z[k] for k in z.files}
    print("loaded cached threshold counts from", CACHE_T)
else:
    def collect(D, budget):
        rows = [count_threshold_sampled(D, P, budget) for P in PST]
        return {k: np.array([r[k] for r in rows]) for k in ("mw", "cm", "rcm", "mw_e", "cm_e", "rcm_e", "a")}
    print("d=5 weight-3 (threshold) ..."); r5 = collect(5, 20)
    print("d=7 weight-4 (threshold) ..."); r7 = collect(7, 75)
    T = {"P": np.array(PST)}
    T.update({f"{k}5": v for k, v in r5.items()}); T.update({f"{k}7": v for k, v in r7.items()})
    np.savez(CACHE_T, **T)
    print("saved threshold counts to", CACHE_T)
for d in (5, 7):
    print(f"d={d}, weight-{(d+1)//2}:  MWPM {np.round(T['mw'+str(d)],0)}  CM {np.round(T['cm'+str(d)],0)}  RCM {np.round(T['rcm'+str(d)],0)}  (per 1e6)")
""")

code(r"""
C_CM, C_RCM, C_MW = "#e67e22", "#009E73", "#2e86de"
P = T["P"]
fig, (a5, a7) = plt.subplots(1, 2, figsize=(11.2, 4.4))
for ax, d, w in [(a5, 5, 3), (a7, 7, 4)]:
    s = str(d)
    ax.errorbar(P, T["mw"+s], yerr=T["mw_e"+s], marker="o", ls="--", color=C_MW,
                capsize=3, elinewidth=1.0, label="MWPM")
    ax.errorbar(P, T["cm"+s], yerr=T["cm_e"+s], marker="s", ls="-", color=C_CM,
                capsize=3, elinewidth=1.0, label=r"CM ($\alpha=1$)")
    ax.errorbar(P, T["rcm"+s], yerr=T["rcm_e"+s], marker="D", ls="-", color=C_RCM,
                capsize=3, elinewidth=1.0, label=r"RCM ($\alpha^{*}$)")
    ax.set_xscale("log"); ax.set_xlabel("Physical error rate $p$")
    ax.set_ylabel(r"failures per $10^{6}$ sampled configs")
    ax.set_title(rf"$d={d}$: weight-{w} configurations")
    ax.margins(y=0.12); ax.set_ylim(bottom=0)
a5.legend()                                   # legend only on the left panel
for ax, lab in zip((a5, a7), "ab"):
    ax.text(-0.14, 1.04, f"({lab})", transform=ax.transAxes, fontsize=13, fontweight="bold")
fig.tight_layout()
savepair(fig, "cm_vs_mwpm_rcm_threshold_counts_d5_d7")
plt.show()
""")

md(r"""
## 4. Distance 7, weight-3 configurations

At $d=7$ the fault distance is 7, so an optimal decoder corrects all **3-fault** configurations
(MWPM fails only at 4). Hard CM again loses a unit of distance and mis-decodes some **weight-3**
configurations. Exhaustive triples are $\sim10^{10}$, so we **importance-sample** triples of
*hyperedge* mechanisms (the CNOT cross-basis errors that cause the confusion), which finds
examples quickly. MWPM never fails on the sampled triples.
""")

code(r"""
D7, P7 = 7, 1e-3
circ7, dem7, mwpm7, cm7 = build(D7, P7)
mechs7, dm7, ob7, hyper7 = mechanisms(dem7)
gm7 = gate_map(circ7)
H = np.flatnonzero(hyper7)
rng = np.random.default_rng(0)
found, tot, mwpm_fail, t0 = [], 0, 0, time.time()
while time.time() - t0 < 60 and len(found) < 8:
    B = 200000
    ii = rng.choice(H, B); jj = rng.choice(H, B); kk = rng.choice(H, B)
    ok = (ii != jj) & (jj != kk) & (ii != kk); ii, jj, kk = ii[ok], jj[ok], kk[ok]
    s = dm7[ii] ^ dm7[jj] ^ dm7[kk]; obo = ob7[ii] ^ ob7[jj] ^ ob7[kk]
    pm = mwpm7.decode_batch(s)[:, 0]; c = cm7.decode_batch(s, enable_correlations=True, alpha=1.0)[:, 0]
    tot += len(ii); mwpm_fail += int((pm != obo).sum())
    for x in np.flatnonzero((c != obo) & (pm == obo)):
        if len(found) < 8:
            found.append((int(ii[x]), int(jj[x]), int(kk[x])))
print(f"d=7, p={P7:g}: sampled {tot:,} hyperedge-triples in {time.time()-t0:.0f}s")
print(f"  CM(alpha=1)-only weight-3 failures found: {len(found)}")
print(f"  MWPM failures among the same sampled triples: {mwpm_fail}")
tri = found[0]
print(f"\n  example confusing triple {tri}:")
for x in tri:
    g, pl = gm7.get(mechs7[x][0], ("?", "?"))
    print(f"    mech {x}: detectors {sorted(mechs7[x][0])}   source {g} ({pl})")
""")

code(r"""
syn7 = (dm7[tri[0]] ^ dm7[tri[1]] ^ dm7[tri[2]]).astype(np.uint8)
tobs7 = int(ob7[tri[0]] ^ ob7[tri[1]] ^ ob7[tri[2]])
coords7 = {k: (c[0], c[1], c[2] if len(c) > 2 else 0.0) for k, c in circ7.get_detector_coordinates().items()}
gedges7, gbound7 = graph_edges(dem7)
fired7 = np.flatnonzero(syn7).tolist()
em7 = mwpm7.decode_to_edges_array(syn7)
ec7 = cm7.decode_to_edges_array(syn7, enable_correlations=True, alpha=1.0)
mp7 = int(mwpm7.decode(syn7)[0]); cp7 = int(cm7.decode(syn7, enable_correlations=True, alpha=1.0)[0])
typ7 = det_type(coords7); zb7 = zoom_box(coords7, fired7)

fig = plt.figure(figsize=(13, 6.2))
draw_matching(fig.add_subplot(1, 2, 1, projection="3d"), coords7, gedges7, typ7, fired7, em7,
              mp7 == tobs7, f"MWPM\npredicts obs = {mp7}  (correct)", (18, -72), zb7)
draw_matching(fig.add_subplot(1, 2, 2, projection="3d"), coords7, gedges7, typ7, fired7, ec7,
              cp7 == tobs7, f"CM ($\\alpha=1$)\npredicts obs = {cp7}  (logical error)", (18, -72), zb7)
type_legend(fig, y=0.0)
fig.suptitle(f"Weight-3 configuration MWPM corrects but CM $\\alpha{{=}}1$ does not  "
             f"($d=7$, $p={P7:g}$)  —  zoomed, coloured by X/Z basis", fontsize=12.5)
fig.tight_layout(rect=[0, 0.04, 1, 1])
savepair(fig, "cm_vs_mwpm_weight3_matching_d7")
plt.show()
""")

md(r"""
## 5. Distance 3, weight-1: the confusion needs *two* faults (control)

Following the $d=5\!\to\!2$, $d=7\!\to\!3$ pattern one might expect $d=3$ to fail at **weight-1**.
It does **not**: at $d=3$ there are **zero** weight-1 configurations where CM fails but MWPM
succeeds (verified at every $p$). A single fault &mdash; even a $Y$/CNOT **hyperedge** &mdash; is
decoded *identically and correctly* by both. The CM over-correction needs a hyperedge to
over-boost **plus a second fault** to make the wrong path win, so the minimum confusing
configuration is **two faults**, independent of distance. (At $d=3$ the CM-only failures appear
only at weight-2, which is already MWPM's own failure threshold &mdash; which is why $d=3$ shows no
clean effective-distance loss.)
""")

code(r"""
D3, P3 = 3, 2e-4
circ3, dem3, mwpm3, cm3 = build(D3, P3)
mechs3, dm3, ob3, hyper3 = mechanisms(dem3); M3 = len(mechs3)
gm3 = gate_map(circ3)

# weight-1: does any single fault fool CM but not MWPM?
pm = mwpm3.decode_batch(dm3)[:, 0]; c = cm3.decode_batch(dm3, enable_correlations=True, alpha=1.0)[:, 0]
w1 = int(((c != ob3) & (pm == ob3)).sum())
cmonly2 = 0                                            # weight-2, for comparison
for a in range(M3):
    s = dm3[a] ^ dm3[a + 1:]; obo = ob3[a] ^ ob3[a + 1:]
    p2 = mwpm3.decode_batch(s)[:, 0]; c2 = cm3.decode_batch(s, enable_correlations=True, alpha=1.0)[:, 0]
    cmonly2 += int(((c2 != obo) & (p2 == obo)).sum())
print(f"d=3, p={P3:g}:  weight-1 configs where CM fails but MWPM succeeds: {w1}")
print(f"                weight-2 configs where CM fails but MWPM succeeds: {cmonly2}")
print("  -> the CM over-correction never appears at a single fault; it needs two.")

# a representative single (weight-1) fault -- a Y hyperedge -- decoded correctly by BOTH decoders
pick = next(i for i in range(M3) if hyper3[i] and len(mechs3[i][0]) >= 3
            and int(mwpm3.decode(dm3[i])[0]) == mechs3[i][1]
            and int(cm3.decode(dm3[i], enable_correlations=True, alpha=1.0)[0]) == mechs3[i][1])
s3 = dm3[pick].astype(np.uint8); tobs3 = int(ob3[pick])
g, pl = gm3.get(mechs3[pick][0], ("?", "?"))
print(f"\n  example weight-1 fault: mech {pick}  detectors {sorted(mechs3[pick][0])}  source {g} ({pl})")

coords3 = {k: (c[0], c[1], c[2] if len(c) > 2 else 0.0) for k, c in circ3.get_detector_coordinates().items()}
ge3, gb3 = graph_edges(dem3); fired3 = np.flatnonzero(s3).tolist()
em3 = mwpm3.decode_to_edges_array(s3)
ec3 = cm3.decode_to_edges_array(s3, enable_correlations=True, alpha=1.0)
mp3 = int(mwpm3.decode(s3)[0]); cp3 = int(cm3.decode(s3, enable_correlations=True, alpha=1.0)[0])
typ3 = det_type(coords3); zb3 = zoom_box(coords3, fired3, mg=2.0)

fig = plt.figure(figsize=(13, 6.2))
draw_matching(fig.add_subplot(1, 2, 1, projection="3d"), coords3, ge3, typ3, fired3, em3,
              mp3 == tobs3, f"MWPM\npredicts obs = {mp3}  (correct)", (18, -72), zb3)
draw_matching(fig.add_subplot(1, 2, 2, projection="3d"), coords3, ge3, typ3, fired3, ec3,
              cp3 == tobs3, f"CM ($\\alpha=1$)\npredicts obs = {cp3}  (correct)", (18, -72), zb3)
type_legend(fig, y=0.0)
fig.suptitle(f"Weight-1 fault: BOTH decoders correct it  ($d=3$, $p={P3:g}$)  —  the confusion needs two faults",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0.04, 1, 1])
savepair(fig, "cm_vs_mwpm_weight1_matching_d3")
plt.show()
""")

md(r"""
## 6. The mechanism: posterior $O(1)$ vs prior $\log\frac{1-p}{p}$

Why does CM over-correct at low $p$? An edge with error probability $p_e$ carries the matching weight
$w=\log\frac{1-p_e}{p_e}$. A **bulk** (or **boundary**) decoding edge has $p_e\sim p$, so
$w\sim\log(1/p)$ &mdash; it **grows** as $p\to0$. But CM($\alpha=1$) re-weights a correlated edge to
the **conditional** $P(e_\mu\!\mid\!e_\nu)=\Pr(e_\mu,e_\nu)/\Pr(e_\nu)$, a ratio of two $O(p)$
quantities, so it is $O(1)$ and its weight $\log\frac{1-P(e_\mu|e_\nu)}{P(e_\mu|e_\nu)}$ **stays
constant**. Below some $p$ the posterior edge is therefore far *cheaper* than the bulk/boundary
edges and the matcher routes through it wrongly. The curves below are **medians over the
decoding-graph edges**, taken straight from the DEM correlation statistics.
""")

code(r"""
def wlog(p):                                    # matching weight w = log((1-p)/p), median over edges
    p = np.asarray(p); p = p[(p > 0) & (p < 1)]
    return np.median(np.log((1.0 - p) / p)) if p.size else np.nan

PS = np.geomspace(1e-4, 1e-2, 9)
prior, post = [], []
for P in PS:
    _, dem, _, _ = build(5, P)
    g = DecodingGraph.from_dem(dem)
    occ = np.asarray(g.base_p)
    corr = np.asarray(g.initial_corr_tracer); src, dst = np.asarray(g.line_edge_index)
    prior.append(wlog(occ))
    pc = np.array([corr[k] / occ[dst[k]] for k in range(g.n_line_edges) if occ[dst[k]] > 0 and corr[k] > 0])
    post.append(wlog(pc))
prior, post = np.array(prior), np.array(post)

fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.plot(PS, prior, "o-", color="#2e86de", lw=2.0, ms=7,
        label=r"prior edge  $\log\dfrac{1-p_\mu}{p_\mu}$")
ax.plot(PS, post, "s-", color="#e67e22", lw=2.0, ms=7,
        label=r"CM posterior edge  $\log\dfrac{1-P(e_\mu|e_\nu)}{P(e_\mu|e_\nu)}$")
ax.set_xscale("log")
ax.set_xlabel("Physical error rate $p$", fontsize=15)
ax.set_ylabel(r"median edge weight  $w$", fontsize=15)
ax.tick_params(labelsize=13)
ax.set_ylim(top=10.6)                        # headroom so the bigger legend clears the data
ax.legend(fontsize=12, loc="upper right", framealpha=1.0, edgecolor="0.8",
          borderpad=0.5, labelspacing=0.35, handlelength=1.6, handletextpad=0.6)
fig.tight_layout()
savepair(fig, "cm_broken_scale_invariance")
plt.show()
print("prior     :", np.round(prior, 2))
print("posterior :", np.round(post, 2))
""")

md(r"""
## Takeaways

* At $d=5$, MWPM corrects **every** weight-2 configuration (sub-threshold), while hard
  CM($\alpha=1$) mis-decodes a few hundred of them &mdash; **all built from CNOT cross-basis
  (hyperedge) errors** &mdash; and the count grows as $p\to0$. The same happens at $d=7$ with
  weight-3 configurations.
* This is a genuine **loss of one unit of effective distance**, caused by CM re-weighting a
  correlated edge to an $O(1)$ posterior while the competing bulk edges carry $\log(1/p)$ weight.
* Damping the re-weight ($\alpha<1$) raises the boosted weight back toward the bulk scale and
  removes the failures &mdash; the subject of the regularized-CM analysis.
* At the **threshold** weight $\frac{d+1}{2}$ (where MWPM itself fails) the picture **flips**:
  correlation now *helps* &mdash; CM($\alpha=1$) corrects $\sim\!45\%$ of MWPM's failures and
  RCM($\alpha^{*}$) more still ($\S$3b). So regularized CM is favourable on **both** sides of
  threshold: it avoids CM's sub-threshold over-correction while keeping the above-threshold benefit.
""")

nb["cells"] = C
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cm_vs_mwpm_confusion.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(C), "cells")
