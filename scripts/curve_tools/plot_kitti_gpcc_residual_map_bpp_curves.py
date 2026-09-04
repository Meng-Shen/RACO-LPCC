#!/usr/bin/env python3
"""Plot fixed/routed G-PCC with two residual-routing variants."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AP_COLUMNS = (
    "Car_3d_AP_R40_moderate",
    "Pedestrian_3d_AP_R40_moderate",
    "Cyclist_3d_AP_R40_moderate",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def route_curve(path: Path) -> tuple[list[float], list[float]]:
    rows = read_rows(path)
    return (
        [float(row["bpp"]) for row in rows],
        [sum(float(row[key]) for key in AP_COLUMNS) / 3.0 for row in rows],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-csv", required=True, type=Path)
    parser.add_argument("--plain-router-csv", required=True, type=Path)
    parser.add_argument("--residual-router-csv", required=True, type=Path)
    parser.add_argument("--plain-route-then-residual-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    fixed = read_rows(args.fixed_csv)
    curves = {
        "Fixed G-PCC": (
            [float(row["bpp"]) for row in fixed],
            [float(row["gpcc_baseline"]) for row in fixed],
        ),
        "LRproxy routed G-PCC": route_curve(args.plain_router_csv),
        "Fixed G-PCC + residual": (
            [float(row["bpp"]) for row in fixed],
            [float(row["scratch_x100"]) for row in fixed],
        ),
        "Residual-aware LRproxy route": route_curve(args.residual_router_csv),
        "Original G-PCC route then residual": route_curve(
            args.plain_route_then_residual_csv
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = args.output_dir / "five_curve_map_bpp.csv"
    with combined_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "point_id", "bpp", "mAP_R40_moderate"))
        writer.writeheader()
        for method, (rates, maps) in curves.items():
            for point_id, (rate, score) in enumerate(zip(rates, maps)):
                writer.writerow({
                    "method": method,
                    "point_id": point_id,
                    "bpp": f"{rate:.6f}",
                    "mAP_R40_moderate": f"{score:.6f}",
                })

    fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=180)
    styles = {
        "Fixed G-PCC": ("#1f77b4", "--", "o"),
        "LRproxy routed G-PCC": ("#1f77b4", "-", "s"),
        "Fixed G-PCC + residual": ("#d62728", "--", "o"),
        "Residual-aware LRproxy route": ("#d62728", "-", "s"),
        "Original G-PCC route then residual": ("#9467bd", "-.", "D"),
    }
    for method, (rates, maps) in curves.items():
        color, linestyle, marker = styles[method]
        ax.plot(rates, maps, label=method, color=color, linestyle=linestyle,
                marker=marker, linewidth=2.2, markersize=6)
    ax.set_xlabel("BPP (total G-PCC bits / total original points)")
    ax.set_ylabel("Moderate 3D mAP_R40 (%)")
    ax.set_title("KITTI FOV / PV-RCNN")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    output_png = args.output_dir / "five_curve_map_bpp.png"
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(output_png)
    print(combined_csv)


if __name__ == "__main__":
    main()
