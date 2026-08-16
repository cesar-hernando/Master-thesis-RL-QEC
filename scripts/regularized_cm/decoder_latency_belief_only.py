#!/usr/bin/env python
"""Run ONLY Belief Matching's latency benchmark, at the fixed shot count already locked in
by an existing decoder_latency CSV's other rows (MWPM/CM/RCM/Tesseract), and merge the
result back into that same CSV.

Exists because Belief Matching needs the `beliefmatching` package, which wasn't installed
when data/latency/decoder_latency_d{3,5,7}_cluster.csv were generated on the cluster (see
job_scripts/decoder_latency_cluster.job) -- decoder_latency_benchmark.py silently skips a
decoder whose package isn't importable rather than failing the whole run, so MWPM/CM/RCM/
Tesseract came back fine and Belief Matching just never got a row. Rather than rerunning
everything, this reads N from the CSV's existing MWPM row (same N the other decoders already
used) and fills in just the missing decoder -- decoder_latency_benchmark.py's own
merge-before-write logic (see its main()) only overwrites the row(s) for the --decoders you
pass, so this cannot clobber the four rows already there.

Thin wrapper, not a reimplementation: shells out to decoder_latency_benchmark.py itself with
--decoders belief --fixed-shots N, so the actual timing/statistics code lives in exactly one
place.

Usage:
    python scripts/regularized_cm/decoder_latency_belief_only.py --distance 7 \
        --csv data/latency/decoder_latency_d7_cluster.csv
"""
import argparse
import csv
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN_SCRIPT = os.path.join(_HERE, "decoder_latency_benchmark.py")


def read_n_shots(csv_path, decoder):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["decoder"] == decoder:
                return int(row["n_shots"])
    sys.exit(f"no '{decoder}' row found in {csv_path} to read the fixed shot count from")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", type=int, required=True)
    ap.add_argument("--csv", required=True,
                    help="existing decoder_latency CSV to read N from and write the belief "
                         "row into (its other rows are left untouched)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--p", type=float, default=1e-3)
    ap.add_argument("--bp-iters", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=999999.0,
                    help="wall-clock safety cap (default: effectively uncapped -- Belief "
                         "Matching is the slowest, highest-variance decoder here, and this "
                         "repo's convention has been to prefer a full, uncapped sample over "
                         "a fast-but-truncated one; pass a smaller value to bound it instead)")
    ap.add_argument("--n-shots-from", default="mwpm",
                    help="which decoder's row in --csv to read the fixed shot count from "
                         "(default: mwpm -- present in every one of these CSVs and always "
                         "the number the others were locked to)")
    ap.add_argument("--out-plot", default=None,
                    help="default: plots/<csv stem>.png (mirrors decoder_latency_benchmark.py)")
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        sys.exit(f"{a.csv} doesn't exist -- run the other four decoders first "
                 f"(job_scripts/decoder_latency_cluster.job) so there's an N to read.")
    n_shots = read_n_shots(a.csv, a.n_shots_from)

    out_plot = a.out_plot or f"plots/{os.path.splitext(os.path.basename(a.csv))[0]}.png"

    cmd = [sys.executable, _MAIN_SCRIPT,
          "--distance", str(a.distance), "--rounds", str(a.rounds), "--p", str(a.p),
          "--bp-iters", str(a.bp_iters), "--fixed-shots", str(n_shots),
          "--decoders", "belief", "--max-seconds", str(a.max_seconds),
          "--out-csv", a.csv, "--out-plot", out_plot]
    print(f"N = {n_shots:,} (from '{a.n_shots_from}' row in {a.csv})")
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
