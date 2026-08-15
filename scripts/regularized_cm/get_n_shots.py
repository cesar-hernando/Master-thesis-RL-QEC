"""Print the n_shots value from one decoder's row in a decoder_latency_benchmark.py CSV.

Used by job_scripts/decoder_latency_cluster.job to read back the shot count Tesseract's
adaptive calibration settled on, so it can be passed as --fixed-shots to the other four
decoders -- mirrors the local Windows+WSL workflow (Tesseract calibrates N, everyone else
runs at that same fixed N for a like-for-like, same-sample-size comparison).

Usage:
    python scripts/regularized_cm/get_n_shots.py --csv data/decoder_latency_d7.csv --decoder tesseract
"""
import argparse
import csv
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--decoder", required=True)
    a = ap.parse_args()

    with open(a.csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["decoder"] == a.decoder:
                print(row["n_shots"])
                return

    sys.exit(f"no '{a.decoder}' row found in {a.csv}")


if __name__ == "__main__":
    main()
