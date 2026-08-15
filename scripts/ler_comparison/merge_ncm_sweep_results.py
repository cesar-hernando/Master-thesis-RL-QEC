"""Merge the per-task CSVs produced by job_scripts/ncm_ler_sweep_parallel.job (one row each,
written by ncm_ler_sweep_task.py) into the master data/ncm_ler_sweep_d3d5d7.csv, then refresh
the per-distance legacy CSVs the thesis figure scripts read.

Safe to run more than once / while the array job is still finishing other tasks: points
already present in the master CSV are skipped rather than duplicated or overwritten, and each
append goes through append_row() (same fsync'd, crash-safe write the serial sweep uses).

Usage:
    python scripts/ler_comparison/merge_ncm_sweep_results.py
    python scripts/ler_comparison/merge_ncm_sweep_results.py \
        --task-csv-glob "data/ncm_ler_sweep_parallel/*.csv" \
        --master-csv data/ncm_ler_sweep_d3d5d7.csv
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncm_ler_sweep_d3d5d7 import COLUMNS, key, load_done, append_row, write_legacy_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-csv-glob", type=str, default="data/ncm_ler_sweep_parallel/*.csv")
    ap.add_argument("--master-csv", type=str, default="data/ncm_ler_sweep_d3d5d7.csv")
    ap.add_argument("--legacy-split-dir", type=str, default="data")
    ap.add_argument("--legacy-prefix", type=str, default="ler_vs_p_ncm_sweep")
    args = ap.parse_args()

    task_files = sorted(glob.glob(args.task_csv_glob))
    if not task_files:
        sys.exit(f"no task csvs matched {args.task_csv_glob}")

    done = load_done(args.master_csv)
    n_written = n_skipped = n_bad = 0
    for path in task_files:
        with open(path, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split(",")
            if header != COLUMNS:
                print(f"  [!] {path}: unexpected header, skipping")
                n_bad += 1
                continue
            for line in f:
                if not line.strip():
                    continue
                vals = line.rstrip("\n").split(",")
                row = dict(zip(COLUMNS, vals))
                k = key(int(row["distance"]), float(row["p"]))
                if k in done:
                    print(f"  [-] d={row['distance']} p={float(row['p']):.4e}: already in "
                          f"{args.master_csv}, skipping ({path})")
                    n_skipped += 1
                    continue
                append_row(args.master_csv, row)   # values are already the right strings
                done.add(k)
                n_written += 1
                print(f"  [+] d={row['distance']} p={float(row['p']):.4e}: merged from {path}")

    print(f"\nmerged {n_written} point(s), skipped {n_skipped} already-present, "
         f"{n_bad} bad file(s) -> {args.master_csv}")
    if args.legacy_split_dir:
        write_legacy_split(args.master_csv, args.legacy_split_dir, args.legacy_prefix)
        print(f"refreshed legacy per-distance CSVs in {args.legacy_split_dir}/")


if __name__ == "__main__":
    main()
