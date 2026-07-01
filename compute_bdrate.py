import argparse
import csv
import math
from pathlib import Path

from BDrate import robust_bdrate


CLASSES = {
    "Car": "Car_3d_AP_R40_moderate",
    "Pedestrian": "Pedestrian_3d_AP_R40_moderate",
    "Cyclist": "Cyclist_3d_AP_R40_moderate",
}


DEFAULT_COMPARISONS = (
    ("JUQP Router", "Baseline G-PCC", "juqp", "baseline"),
    ("JUQP Router", "Split-GPCC", "juqp", "split"),
    ("Split-GPCC", "Baseline G-PCC", "split", "baseline"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-class BD-rate from AP-bpp curve CSVs. "
            "Negative values mean the compared method uses fewer bits than the reference at the same AP."
        )
    )
    parser.add_argument(
        "--baseline_csv",
        default="point_pairs/baseline_fov/baseline_gpcc_curve.csv",
        help="Baseline G-PCC AP-bpp curve CSV.",
    )
    parser.add_argument(
        "--split_csv",
        default="point_pairs/split_gpcc_fov/split_gpcc_curve.csv",
        help="Split-GPCC AP-bpp curve CSV.",
    )
    parser.add_argument(
        "--juqp_csv",
        default="point_pairs/router_gpcc_fov/router_gpcc_curve.csv",
        help="JUQP Router AP-bpp curve CSV.",
    )
    parser.add_argument(
        "--out_csv",
        default="",
        help="Optional output CSV path for BD-rate results.",
    )
    return parser.parse_args()


def to_float(row, column, path):
    value = row.get(column)
    if value in (None, ""):
        raise ValueError(f"Missing column value '{column}' in {path}: {row}")
    return float(value)


def read_curve_points(path):
    path = Path(path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    curves = {cls: [] for cls in CLASSES}
    for row in rows:
        bpp = to_float(row, "bpp", path)
        if bpp <= 0:
            continue
        for cls, ap_column in CLASSES.items():
            ap = to_float(row, ap_column, path)
            curves[cls].append((bpp, ap))

    for cls, points in curves.items():
        if len(points) < 2:
            raise ValueError(f"Need at least two AP-bpp points for {cls} in {path}")
        points.sort(key=lambda item: item[0])
    return curves


def format_value(value):
    if math.isnan(value):
        return "nan"
    return f"{value:.2f}"


def write_results(path, results):
    fieldnames = ["comparison", "class", "bd_rate_percent"]
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main():
    args = parse_args()
    curves = {
        "baseline": read_curve_points(args.baseline_csv),
        "split": read_curve_points(args.split_csv),
        "juqp": read_curve_points(args.juqp_csv),
    }

    results = []
    for compared_name, reference_name, compared_key, reference_key in DEFAULT_COMPARISONS:
        comparison = f"{compared_name} vs {reference_name}"
        for cls in CLASSES:
            bd_rate = robust_bdrate(curves[reference_key][cls], curves[compared_key][cls])
            results.append({
                "comparison": comparison,
                "class": cls,
                "bd_rate_percent": bd_rate,
            })

    print("=== BD-rate from AP-bpp curve points ===")
    print("Negative values mean bitrate saving of the compared method relative to the reference.\n")
    print(f"{'Comparison':<36} {'Class':<12} {'BD-rate (%)':>12}")
    print("-" * 62)
    for row in results:
        print(
            f"{row['comparison']:<36} "
            f"{row['class']:<12} "
            f"{format_value(row['bd_rate_percent']):>12}"
        )

    if args.out_csv:
        serializable = [
            {
                "comparison": row["comparison"],
                "class": row["class"],
                "bd_rate_percent": format_value(row["bd_rate_percent"]),
            }
            for row in results
        ]
        write_results(args.out_csv, serializable)
        print(f"\nWrote: {args.out_csv}")


if __name__ == "__main__":
    main()
