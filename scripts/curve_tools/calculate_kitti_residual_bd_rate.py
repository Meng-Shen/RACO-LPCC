#!/usr/bin/env python3
"""Calculate pairwise BD-rate against fixed G-PCC from the five-curve CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


DEFAULT_ANCHOR = "Fixed G-PCC"


def read_curves(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["method"], []).append(
                (float(row["mAP_R40_moderate"]), float(row["bpp"]))
            )
    curves = {}
    for method, values in grouped.items():
        # For repeated quality values retain the lower rate (the RD envelope).
        best_by_quality: dict[float, float] = {}
        for quality, rate in values:
            best_by_quality[quality] = min(rate, best_by_quality.get(quality, math.inf))
        ordered = sorted(best_by_quality.items())
        curves[method] = (
            np.asarray([item[0] for item in ordered], dtype=np.float64),
            np.log(np.asarray([item[1] for item in ordered], dtype=np.float64)),
        )
    return curves


def pchip_derivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = np.diff(x)
    delta = np.diff(y) / h
    n = len(x)
    d = np.zeros(n, dtype=np.float64)
    if n == 2:
        d[:] = delta[0]
        return d
    for k in range(1, n - 1):
        if delta[k - 1] == 0 or delta[k] == 0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
            d[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    def endpoint(h0: float, h1: float, m0: float, m1: float) -> float:
        value = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        if np.sign(value) != np.sign(m0):
            return 0.0
        if np.sign(m0) != np.sign(m1) and abs(value) > abs(3.0 * m0):
            return 3.0 * m0
        return value

    d[0] = endpoint(h[0], h[1], delta[0], delta[1])
    d[-1] = endpoint(h[-1], h[-2], delta[-1], delta[-2])
    return d


def integrate_pchip(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    d = pchip_derivatives(x, y)
    total = 0.0
    for i, h in enumerate(np.diff(x)):
        left = max(lo, x[i])
        right = min(hi, x[i + 1])
        if right <= left:
            continue
        slope = (y[i + 1] - y[i]) / h
        c2 = (3.0 * slope - 2.0 * d[i] - d[i + 1]) / h
        c3 = (d[i] + d[i + 1] - 2.0 * slope) / (h * h)
        a, b = left - x[i], right - x[i]
        primitive = lambda t: y[i] * t + d[i] * t**2 / 2.0 + c2 * t**3 / 3.0 + c3 * t**4 / 4.0
        total += primitive(b) - primitive(a)
    return total


def bd_rate_pchip(anchor, test) -> tuple[float, float, float]:
    xa, ya = anchor
    xt, yt = test
    lo, hi = max(xa[0], xt[0]), min(xa[-1], xt[-1])
    if hi <= lo:
        raise ValueError("curves have no overlapping quality range")
    mean_delta = (
        integrate_pchip(xt, yt, lo, hi) - integrate_pchip(xa, ya, lo, hi)
    ) / (hi - lo)
    return (math.exp(mean_delta) - 1.0) * 100.0, lo, hi


def bd_rate_cubic(anchor, test) -> float:
    xa, ya = anchor
    xt, yt = test
    lo, hi = max(xa[0], xt[0]), min(xa[-1], xt[-1])
    pa = np.polyfit(xa, ya, 3)
    pt = np.polyfit(xt, yt, 3)
    ia, it = np.polyint(pa), np.polyint(pt)
    mean_delta = ((np.polyval(it, hi) - np.polyval(it, lo))
                  - (np.polyval(ia, hi) - np.polyval(ia, lo))) / (hi - lo)
    return (math.exp(float(mean_delta)) - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate PCHIP and cubic BD-rate values from a multi-method "
            "mAP-BPP CSV."
        )
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR)
    args = parser.parse_args()

    input_csv = args.input_csv.resolve()
    output = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else input_csv.with_name("bd_rate_vs_fixed_gpcc.csv")
    )
    curves = read_curves(input_csv)
    if args.anchor not in curves:
        raise KeyError(f"Anchor curve not found: {args.anchor}")
    anchor = curves[args.anchor]
    rows = []
    for method, curve in curves.items():
        if method == args.anchor:
            continue
        pchip, lo, hi = bd_rate_pchip(anchor, curve)
        cubic = bd_rate_cubic(anchor, curve)
        rows.append((pchip, method, cubic, lo, hi))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "rank_by_rate_saving", "anchor", "method", "bd_rate_pchip_percent",
            "bd_rate_cubic_percent", "overlap_map_min", "overlap_map_max",
        ))
        writer.writeheader()
        for rank, (pchip, method, cubic, lo, hi) in enumerate(sorted(rows), 1):
            writer.writerow({
                "rank_by_rate_saving": rank,
                "anchor": args.anchor,
                "method": method,
                "bd_rate_pchip_percent": f"{pchip:.6f}",
                "bd_rate_cubic_percent": f"{cubic:.6f}",
                "overlap_map_min": f"{lo:.6f}",
                "overlap_map_max": f"{hi:.6f}",
            })
            print(f"{rank}\t{method}\tPCHIP={pchip:.6f}%\tCubic={cubic:.6f}%\toverlap=[{lo:.6f},{hi:.6f}]")
    print(output)


if __name__ == "__main__":
    main()
