"""
Plot utilities for experimental decoder results (Google Willow d=3/5/7).

Consumes the CSV files written by evaluate_on_willow.py:
  - results CSV: one row per (distance, rounds, basis, patch, ...) with LER +
                 binomial SE for each of the three decoders.
  - calibration-trace CSV: per-iteration DGR mean/max/active-edges (Spitz
                 contributes a single summary row).

Three plot kinds are supported:

  python scripts/plot_experimental_results.py calibration \
      --trace-csv data/experimental/willow_calibration_trace.csv \
      --out plots/experimental/calibration_convergence.png

  python scripts/plot_experimental_results.py ler-bars \
      --results-csv data/experimental/willow_results.csv \
      --filter distance=5,basis=Z,rounds=10 \
      --out plots/experimental/ler_bars_d5_r10.png

  python scripts/plot_experimental_results.py vs \
      --results-csv data/experimental/willow_results.csv \
      --x rounds --group-by distance \
      --filter basis=Z \
      --out plots/experimental/ler_vs_rounds.png

Filters are key=value pairs, comma-separated. Values may be lists with '|', e.g.
"distance=3|5,basis=Z". Filtering is done as string comparison on the CSV cells, so
quote your values carefully.
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


DECODER_COLUMNS = {
    "Standard MWPM": ("ler_mwpm", "se_mwpm"),
    "Correlated Matching": ("ler_corr", "se_corr"),
    "Neural Correlated Matching": ("ler_neural", "se_neural"),
}
DECODER_COLORS = {
    "Standard MWPM": "tab:blue",
    "Correlated Matching": "tab:orange",
    "Neural Correlated Matching": "tab:green",
}


def read_csv(path: str) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_filter(filter_str: Optional[str]) -> Dict[str, List[str]]:
    if not filter_str:
        return {}
    out = {}
    for token in filter_str.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Bad filter token (expected key=value): {token!r}")
        k, v = token.split("=", 1)
        out[k.strip()] = [x.strip() for x in v.split("|")]
    return out


def filter_rows(rows: List[dict], filt: Dict[str, List[str]]) -> List[dict]:
    if not filt:
        return rows
    out = []
    for row in rows:
        ok = True
        for k, allowed in filt.items():
            if k not in row or str(row[k]) not in allowed:
                ok = False
                break
        if ok:
            out.append(row)
    return out


def safe_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


# =============================================================================
# Plot: calibration convergence (DGR iterations)
# =============================================================================

def plot_calibration_convergence(trace_csv_paths: List[str], out_path: str,
                                 filter_str: Optional[str], title: Optional[str]):
    filt = parse_filter(filter_str)
    all_rows = []
    for p in trace_csv_paths:
        all_rows.extend(read_csv(p))
    rows = filter_rows(all_rows, filt)
    if not rows:
        raise SystemExit("No rows matched the filter.")

    # Group by (dataset, distance, rounds, basis, patch, calibration)
    groups = defaultdict(list)
    for r in rows:
        key = (
            r.get("dataset", ""), r.get("distance", ""), r.get("rounds", ""),
            r.get("basis", ""), r.get("patch", ""), r.get("calibration", ""),
        )
        groups[key].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for key, grp in groups.items():
        grp = sorted(grp, key=lambda r: int(r["iteration"]))
        iters = [int(r["iteration"]) for r in grp]
        mean_p = [safe_float(r["mean_p"]) for r in grp]
        max_p = [safe_float(r["max_p"]) for r in grp]
        n_active = [int(r["n_active_edges"]) for r in grp]
        n_edges = [int(r["n_edges"]) for r in grp]

        label = "  ".join([
            key[0] or "?",
            f"d={key[1]}", f"r={key[2]}", f"b={key[3]}",
            f"p={key[4]}", f"cal={key[5]}",
        ])
        axes[0].plot(iters, mean_p, marker="o", label=label, linewidth=1.5)
        axes[1].plot(iters, max_p, marker="s", linewidth=1.5)
        axes[2].plot(iters, np.array(n_active) / np.array(n_edges),
                      marker="^", linewidth=1.5)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("mean p_e (log)")
    axes[0].set_xlabel("Calibration iteration")
    axes[0].set_title("Mean edge probability")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)

    axes[1].set_yscale("log")
    axes[1].set_ylabel("max p_e (log)")
    axes[1].set_xlabel("Calibration iteration")
    axes[1].set_title("Max edge probability")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)

    axes[2].set_ylabel("fraction of edges active")
    axes[2].set_xlabel("Calibration iteration")
    axes[2].set_title("Active edges / total edges")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].set_ylim(0, 1.05)

    # Compact legend on the leftmost axis
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle(title or "Calibration convergence", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out_path}")


# =============================================================================
# Plot: LER bar chart (compare three decoders at fixed config)
# =============================================================================

def plot_ler_bars(results_csv_paths: List[str], out_path: str,
                  filter_str: Optional[str], title: Optional[str]):
    filt = parse_filter(filter_str)
    rows = []
    for p in results_csv_paths:
        rows.extend(read_csv(p))
    rows = filter_rows(rows, filt)
    if not rows:
        raise SystemExit("No rows matched the filter.")

    n_rows = len(rows)
    decoders = list(DECODER_COLUMNS.keys())
    bar_w = 0.25
    x = np.arange(n_rows)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * n_rows), 5))

    for i, dec in enumerate(decoders):
        ler_col, se_col = DECODER_COLUMNS[dec]
        lers = np.array([safe_float(r[ler_col]) for r in rows])
        ses = np.array([safe_float(r[se_col]) for r in rows])
        ax.bar(x + i * bar_w, lers, bar_w, yerr=ses, capsize=3,
               label=dec, color=DECODER_COLORS[dec])

    # x-axis labels: key params per row
    labels = []
    for r in rows:
        parts = [
            f"d{r.get('distance','')}",
            f"r{r.get('rounds','')}",
            f"{r.get('basis','')}",
            r.get("patch", ""),
        ]
        labels.append("\n".join(p for p in parts if p))

    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("Logical Error Rate (LER)")
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title(title or "LER per decoder")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out_path}")


# =============================================================================
# Plot: LER vs distance OR rounds (multi-decoder)
# =============================================================================

def plot_ler_vs_x(results_csv_paths: List[str], out_path: str,
                  x_field: str, group_by: List[str],
                  filter_str: Optional[str], title: Optional[str],
                  pool_replicates: bool = True):
    """Plot LER vs `x_field` (e.g. 'rounds' or 'distance'), one line per decoder
    per group_by combination.

    If pool_replicates is True, rows with the same (group_by..., x_field) are pooled
    by summing errors and shots, giving correct binomial CIs across whatever the
    pooled dimension is (typically different patches).
    """
    filt = parse_filter(filter_str)
    rows = []
    for p in results_csv_paths:
        rows.extend(read_csv(p))
    rows = filter_rows(rows, filt)
    if not rows:
        raise SystemExit("No rows matched the filter.")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Group key = (group_by_values...) excluding x_field
    grouped = defaultdict(list)
    for r in rows:
        key = tuple(r.get(g, "") for g in group_by)
        grouped[key].append(r)

    for key, grp_rows in grouped.items():
        # Bucket by x value, optionally pooling across patches.
        per_x = defaultdict(list)
        for r in grp_rows:
            per_x[r.get(x_field, "")].append(r)

        xs_sorted = sorted(per_x.keys(), key=lambda v: (safe_float(v), v))
        x_vals = []
        for dec in DECODER_COLUMNS:
            x_vals_dec, lers, ses = [], [], []
            for x_str in xs_sorted:
                bucket = per_x[x_str]
                if pool_replicates and len(bucket) > 1:
                    err_col = "err_" + ("mwpm" if dec == "Standard MWPM"
                                        else "corr" if dec == "Correlated Matching"
                                        else "neural")
                    n_total = sum(int(safe_float(r["n_test"])) for r in bucket)
                    err_total = sum(int(safe_float(r[err_col])) for r in bucket)
                    if n_total == 0:
                        continue
                    p = err_total / n_total
                    se = float(np.sqrt(p * (1 - p) / n_total))
                else:
                    r = bucket[0]
                    ler_col, se_col = DECODER_COLUMNS[dec]
                    p = safe_float(r[ler_col])
                    se = safe_float(r[se_col])
                x_vals_dec.append(safe_float(x_str))
                lers.append(p)
                ses.append(se)

            label_suffix = "  ".join(f"{g}={v}" for g, v in zip(group_by, key) if v)
            label = f"{dec}" + (f"  ({label_suffix})" if label_suffix else "")
            ax.errorbar(x_vals_dec, lers, yerr=ses, marker="o", capsize=3,
                         linewidth=1.5, label=label,
                         color=DECODER_COLORS[dec],
                         linestyle={"Standard MWPM": "-",
                                    "Correlated Matching": "--",
                                    "Neural Correlated Matching": ":"}[dec])

    ax.set_xlabel(x_field)
    ax.set_ylabel("Logical Error Rate (LER)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(title or f"LER vs {x_field}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibration", help="Plot calibration convergence over iterations.")
    p_cal.add_argument("--trace-csv", nargs="+", required=True,
                       help="One or more calibration-trace CSV files.")
    p_cal.add_argument("--filter", default=None)
    p_cal.add_argument("--out", default="plots/experimental/calibration_convergence.png")
    p_cal.add_argument("--title", default=None)

    p_bar = sub.add_parser("ler-bars",
                            help="Grouped bar chart of the three decoders for matching rows.")
    p_bar.add_argument("--results-csv", nargs="+", required=True)
    p_bar.add_argument("--filter", default=None)
    p_bar.add_argument("--out", default="plots/experimental/ler_bars.png")
    p_bar.add_argument("--title", default=None)

    p_x = sub.add_parser("vs",
                          help="LER vs rounds or distance (one curve per decoder per group).")
    p_x.add_argument("--results-csv", nargs="+", required=True)
    p_x.add_argument("--x", choices=["rounds", "distance", "n_test"], default="rounds",
                     help="Field on x-axis.")
    p_x.add_argument("--group-by", nargs="*", default=["dataset", "distance"],
                     help="Fields that distinguish curves. Default: dataset, distance.")
    p_x.add_argument("--filter", default=None)
    p_x.add_argument("--no-pool", action="store_true",
                     help="Do NOT pool across patches; plot raw per-row points.")
    p_x.add_argument("--out", default="plots/experimental/ler_vs_x.png")
    p_x.add_argument("--title", default=None)

    args = parser.parse_args()

    if args.cmd == "calibration":
        plot_calibration_convergence(args.trace_csv, args.out, args.filter, args.title)
    elif args.cmd == "ler-bars":
        plot_ler_bars(args.results_csv, args.out, args.filter, args.title)
    elif args.cmd == "vs":
        plot_ler_vs_x(args.results_csv, args.out, args.x, args.group_by,
                      args.filter, args.title,
                      pool_replicates=not args.no_pool)


if __name__ == "__main__":
    main()
