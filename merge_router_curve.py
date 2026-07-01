#!/usr/bin/env python3
import argparse
import csv


def read_by_rate(path):
    with open(path, newline="") as f:
        return {int(row["rate_id"]): row for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description="Merge adaptive router AP and G-PCC metrics into curve point pairs.")
    parser.add_argument("--ap_csv", required=True)
    parser.add_argument("--gpcc_csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ap_rows = read_by_rate(args.ap_csv)
    gpcc_rows = read_by_rate(args.gpcc_csv)
    rows = []
    for rate_id in sorted(set(ap_rows) & set(gpcc_rows)):
        ap = ap_rows[rate_id]
        gpcc = gpcc_rows[rate_id]
        rows.append({
            "rate_id": rate_id,
            "threshold": gpcc.get("threshold") or ap.get("threshold"),
            "bpp": gpcc.get("bpp"),
            "seg_time": gpcc.get("seg_time"),
            "fg_enc_time": gpcc.get("fg_enc_time"),
            "bg_enc_time": gpcc.get("bg_enc_time"),
            "gpcc_enc_time": gpcc.get("gpcc_enc_time"),
            "enc_time": gpcc.get("enc_time"),
            "dec_time": gpcc.get("dec_time"),
            "Car_3d_AP_R40_moderate": ap.get("Car_3d_AP_R40_moderate"),
            "Pedestrian_3d_AP_R40_moderate": ap.get("Pedestrian_3d_AP_R40_moderate"),
            "Cyclist_3d_AP_R40_moderate": ap.get("Cyclist_3d_AP_R40_moderate"),
        })
    if not rows:
        raise RuntimeError("No matching rate_id between AP and router G-PCC CSVs")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(rows)} router curve rows -> {args.out}")


if __name__ == "__main__":
    main()
