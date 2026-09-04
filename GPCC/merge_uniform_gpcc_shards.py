#!/usr/bin/env python3
"""Merge six independently encoded whole-frame GPCC scale shards."""

import argparse
import csv
from pathlib import Path


def norm_frame_id(value):
    return str(value).strip().zfill(6)


def parse_scale(value):
    value = str(value).strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_scales(text):
    scales = [parse_scale(item) for item in str(text).split(",") if item.strip()]
    if not scales:
        raise ValueError("--scales must contain at least one value")
    return scales


def read_split(path):
    with open(path) as handle:
        return [norm_frame_id(line) for line in handle if line.strip()]


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_root", required=True)
    parser.add_argument("--scales", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--details_out", required=True)
    parser.add_argument("--average_out", required=True)
    args = parser.parse_args()

    expected_frame_ids = read_split(args.split_file)
    expected_set = set(expected_frame_ids)
    order = {frame_id: idx for idx, frame_id in enumerate(expected_frame_ids)}
    detail_rows = []
    average_rows = []

    for rate_id, scale in enumerate(parse_scales(args.scales)):
        source = Path(args.shard_root) / f"scale_{rate_id}" / "gpcc_baseline_details.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Missing GPCC scale shard: {source}")
        rows = read_csv(source)
        actual_ids = [norm_frame_id(row["filename"]) for row in rows]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_set:
            raise ValueError(
                f"Scale {rate_id} frame mismatch: actual={len(actual_ids)} "
                f"unique={len(set(actual_ids))} expected={len(expected_frame_ids)}"
            )
        rows.sort(key=lambda row: order[norm_frame_id(row["filename"])])
        for row in rows:
            row["rate_id"] = rate_id
            row["posQuantscale"] = scale
            row["scale"] = scale
            detail_rows.append(row)

        total_bits = sum(int(float(row["bits"])) for row in rows)
        total_points = sum(int(float(row["num_points"])) for row in rows)
        average_rows.append(
            {
                "rate_id": rate_id,
                "posQuantscale": scale,
                "scale": scale,
                "num_frames": len(rows),
                "total_points": total_points,
                "total_bits": total_bits,
                "bpp": round(total_bits / total_points, 6) if total_points else 0.0,
                "enc_time": round(sum(float(row["enc_time"]) for row in rows) / len(rows), 6),
                "dec_time": round(sum(float(row["dec_time"]) for row in rows) / len(rows), 6),
            }
        )

    details_out = Path(args.details_out).resolve()
    details_out.parent.mkdir(parents=True, exist_ok=True)
    detail_fields = [
        "filename", "rate_id", "posQuantscale", "scale", "num_points",
        "bits", "bpp", "enc_time", "dec_time",
    ]
    with open(details_out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(detail_rows)

    average_out = Path(args.average_out).resolve()
    average_out.parent.mkdir(parents=True, exist_ok=True)
    with open(average_out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(average_rows[0].keys()))
        writer.writeheader()
        writer.writerows(average_rows)

    print(f"Merged {len(average_rows)} GPCC scales and {len(detail_rows)} detail rows")
    print(f"Details: {details_out}")
    print(f"Average: {average_out}")


if __name__ == "__main__":
    main()
