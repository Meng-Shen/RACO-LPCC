#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def norm_frame_id(x):
    return str(x).strip().zfill(6)


def read_split(path):
    with open(path) as f:
        return [norm_frame_id(line) for line in f if line.strip()]


def read_split_details(path):
    table = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frame_id = norm_frame_id(row.get("filename") or row.get("frame_id"))
            label = int(row.get("combo_id") or row["rate_id"])
            table[(frame_id, label)] = row
    return table


def read_labels(path):
    labels = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            labels[norm_frame_id(row["frame_id"])] = int(row["jucp_label"])
    return labels


def main():
    parser = argparse.ArgumentParser(description="Aggregate router bpp/time from existing Split-GPCC per-frame details.")
    parser.add_argument("--split_details_csv", required=True, help="split_all_details.csv from Split-GPCC")
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    frame_ids = read_split(args.split_file)
    details = read_split_details(args.split_details_csv)
    manifest = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    avg_rows = []
    for item in sorted(manifest["label_csvs"], key=lambda x: int(x["rate_id"])):
        rate_id = int(item["rate_id"])
        threshold = item.get("threshold", "")
        labels = read_labels(item["path"])
        rows = []
        for frame_id in frame_ids:
            if frame_id not in labels:
                raise KeyError(
                    f"Missing router label for frame_id={frame_id} in {item['path']}. "
                    "Generate router labels with the same split used by split_details_csv."
                )
            label = int(labels[frame_id])
            key = (frame_id, label)
            if key not in details:
                raise KeyError(f"Missing Split-GPCC detail for frame={frame_id}, combo/label={label}")
            src = details[key]
            out = dict(src)
            out["rate_id"] = rate_id
            out["threshold"] = threshold
            out["jucp_label"] = label
            rows.append(out)
            detail_rows.append(out)

        total_bits = sum(int(float(r["bits"])) for r in rows)
        total_points = sum(int(float(r["num_points"])) for r in rows)
        avg_rows.append({
            "rate_id": rate_id,
            "threshold": threshold,
            "num_frames": len(rows),
            "total_points": total_points,
            "total_bits": total_bits,
            "bpp": round(total_bits / total_points, 6) if total_points else 0.0,
            "seg_time": round(sum(float(r.get("seg_time", 0.0)) for r in rows) / len(rows), 6),
            "fg_enc_time": round(sum(float(r.get("fg_enc_time", 0.0)) for r in rows) / len(rows), 6),
            "bg_enc_time": round(sum(float(r.get("bg_enc_time", 0.0)) for r in rows) / len(rows), 6),
            "gpcc_enc_time": round(sum(float(r.get("gpcc_enc_time", 0.0)) for r in rows) / len(rows), 6),
            "enc_time": round(sum(float(r.get("enc_time", 0.0)) for r in rows) / len(rows), 6),
            "dec_time": round(sum(float(r.get("dec_time", 0.0)) for r in rows) / len(rows), 6),
        })

    detail_csv = out_dir / "router_all_details.csv"
    with open(detail_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    avg_csv = out_dir / "router_average_results.csv"
    with open(avg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(avg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(avg_rows)

    print(f"Detail CSV: {detail_csv}")
    print(f"Average CSV: {avg_csv}")


if __name__ == "__main__":
    main()
