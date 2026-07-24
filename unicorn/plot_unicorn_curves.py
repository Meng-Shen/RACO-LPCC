#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


CLASSES = {
    'Car': 'Car_3d_AP_R40_moderate',
    'Pedestrian': 'Pedestrian_3d_AP_R40_moderate',
    'Cyclist': 'Cyclist_3d_AP_R40_moderate',
}

METRICS = {
    'ap_bpp': ('bpp', 'Bits Per Point (bpp)', '{cls} 3D AP R40 Moderate (%)'),
    'ap_enctime': ('enc_time', 'Encoding Time (s/frame)', '{cls} 3D AP R40 Moderate (%)'),
    'ap_dectime': ('dec_time', 'Decoding Time (s/frame)', '{cls} 3D AP R40 Moderate (%)'),
}

PSNR_METRICS = {
    'd1_psnr_bpp': ('d1_psnr', 'psnr_p2point', 'D1 PSNR (dB)', 'D1 PSNR-bpp Curve'),
    'd2_psnr_bpp': ('d2_psnr', 'psnr_p2plane', 'D2 PSNR (dB)', 'D2 PSNR-bpp Curve'),
}

METHODS = {
    'Baseline G-PCC': {'color': '#ff7f0e', 'marker': 'X', 'linestyle': '--'},
    'Split-GPCC': {'color': '#1f77b4', 'marker': 's', 'linestyle': '-.'},
    'JUQP Router': {'color': '#2ca02c', 'marker': 'o', 'linestyle': '-'},
    'RENO': {'color': '#d62728', 'marker': '^', 'linestyle': ':'},
    'Unicorn': {'color': '#9467bd', 'marker': 'D', 'linestyle': '-'},
}


def read_curve_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'No rows found in {path}')
    return rows


def maybe_read(path):
    if path is None or str(path).strip() == '':
        return None
    path = Path(path)
    if not path.exists():
        print(f'[!] Optional curve CSV not found, skipping: {path}')
        return None
    return read_curve_csv(path)


def to_float(row, key):
    value = row.get(key, '')
    if value in ('', None):
        return None
    return float(value)


def to_float_any(row, keys):
    for key in keys:
        value = row.get(key, '')
        if value not in ('', None):
            return float(value)
    return None


def collect_points(rows, x_key, y_key):
    points = []
    for row in rows:
        x = to_float(row, x_key)
        y = to_float(row, y_key)
        if x is not None and y is not None:
            points.append((x, y))
    points.sort(key=lambda p: p[0])
    return points


def collect_points_any(rows, x_key, y_keys):
    points = []
    for row in rows:
        x = to_float(row, x_key)
        y = to_float_any(row, y_keys)
        if x is not None and y is not None:
            points.append((x, y))
    points.sort(key=lambda p: p[0])
    return points


def plot_one(out_dir, formats, dpi, metric_key, datasets, cls):
    x_key, xlabel, ylabel = METRICS[metric_key]
    y_key = CLASSES[cls]
    plt.figure(figsize=(8.5, 6))
    plotted = []
    for name, rows in datasets.items():
        points = collect_points(rows, x_key, y_key)
        if not points:
            continue
        style = METHODS[name]
        plt.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            markersize=6,
            linewidth=2.2,
            label=name,
        )
        plotted.extend(points)
    if not plotted:
        raise ValueError(f'No plottable points for {cls} {metric_key}')
    y_values = [p[1] for p in plotted]
    y_min, y_max = min(y_values), max(y_values)
    pad = max((y_max - y_min) * 0.08, 1.0)
    plt.ylim(max(0, y_min - pad), y_max + pad)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel.format(cls=cls), fontsize=13)
    plt.title(f'{cls} {metric_key.replace("_", "-")} Curve', fontsize=15, pad=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    saved = []
    for fmt in formats:
        out_path = out_dir / f'{metric_key}_{cls.lower()}.{fmt}'
        plt.savefig(out_path, dpi=dpi, facecolor='white')
        saved.append(out_path)
    plt.close()
    return saved


def split_psnr_rows(rows):
    datasets = {}
    method_aliases = {
        'Baseline G-PCC': 'Baseline G-PCC',
        'Split-GPCC': 'Split-GPCC',
        'Split GPCC': 'Split-GPCC',
        'JUQP Router': 'JUQP Router',
        'Router-GPCC': 'JUQP Router',
        'Router GPCC': 'JUQP Router',
    }
    for row in rows:
        method = method_aliases.get(row.get('method', ''), row.get('method', ''))
        if method in METHODS:
            datasets.setdefault(method, []).append(row)
    return datasets


def plot_psnr_one(out_dir, formats, dpi, metric_key, datasets):
    primary_y, fallback_y, ylabel, title = PSNR_METRICS[metric_key]
    plt.figure(figsize=(8.5, 6))
    plotted = []
    for name, rows in datasets.items():
        points = collect_points_any(rows, 'bpp', (primary_y, fallback_y))
        if not points:
            continue
        style = METHODS[name]
        plt.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            markersize=6,
            linewidth=2.2,
            label=name,
        )
        plotted.extend(points)
    if not plotted:
        print(f'[!] No plottable points for {metric_key}, skipping.')
        plt.close()
        return []
    y_values = [p[1] for p in plotted]
    y_min, y_max = min(y_values), max(y_values)
    pad = max((y_max - y_min) * 0.08, 1.0)
    plt.ylim(max(0, y_min - pad), y_max + pad)
    plt.xlabel('Bits Per Point (bpp)', fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=15, pad=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    saved = []
    for fmt in formats:
        out_path = out_dir / f'{metric_key}.{fmt}'
        plt.savefig(out_path, dpi=dpi, facecolor='white')
        saved.append(out_path)
    plt.close()
    return saved


def main():
    parser = argparse.ArgumentParser(description='Plot AP curves with Baseline, Split, JUQP, RENO, and Unicorn.')
    parser.add_argument('--baseline_csv', default='point_pairs/baseline_fov/baseline_gpcc_curve.csv')
    parser.add_argument('--split_csv', default='point_pairs/split_gpcc_fov/split_gpcc_curve.csv')
    parser.add_argument('--juqp_csv', default='point_pairs/router_gpcc_fov/router_gpcc_curve.csv')
    parser.add_argument('--reno_csv', default='point_pairs/reno_fov/reno_full_curve.csv')
    parser.add_argument('--unicorn_csv', default='point_pairs/unicorn_fov/unicorn_full_curve.csv')
    parser.add_argument('--psnr_csv', default='point_pairs/psnr_bpp/all_methods_psnr_bpp_curve.csv')
    parser.add_argument('--out_dir', default='plots_unicorn')
    parser.add_argument('--formats', default='png')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    datasets = {
        'Baseline G-PCC': read_curve_csv(args.baseline_csv),
        'Split-GPCC': read_curve_csv(args.split_csv),
    }
    optional = [
        ('JUQP Router', args.juqp_csv),
        ('RENO', args.reno_csv),
        ('Unicorn', args.unicorn_csv),
    ]
    for name, path in optional:
        rows = maybe_read(path)
        if rows is not None:
            datasets[name] = rows

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip().lstrip('.') for fmt in args.formats.split(',') if fmt.strip()]
    saved = []
    for cls in CLASSES:
        for metric_key in METRICS:
            saved.extend(plot_one(out_dir, formats, args.dpi, metric_key, datasets, cls))

    psnr_datasets = {}
    psnr_rows = maybe_read(args.psnr_csv)
    if psnr_rows is not None:
        psnr_datasets.update(split_psnr_rows(psnr_rows))
    if 'RENO' in datasets:
        psnr_datasets['RENO'] = datasets['RENO']
    if 'Unicorn' in datasets:
        psnr_datasets['Unicorn'] = datasets['Unicorn']
    for metric_key in PSNR_METRICS:
        saved.extend(plot_psnr_one(out_dir, formats, args.dpi, metric_key, psnr_datasets))

    print(f'[+] Saved {len(saved)} plots to {out_dir}')
    for path in saved:
        print(path)


if __name__ == '__main__':
    main()
