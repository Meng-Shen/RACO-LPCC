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
    'ap_bpp': {
        'x_column': 'bpp',
        'filename': 'ap_bpp',
        'xlabel': 'Bits Per Point (bpp)',
        'ylabel': '{cls} 3D AP R40 Moderate (%)',
        'title': '{cls} AP-bpp Curve',
    },
    'ap_enctime': {
        'x_column': 'enc_time',
        'filename': 'ap_enctime',
        'xlabel': 'Encoding Time (s/frame)',
        'ylabel': '{cls} 3D AP R40 Moderate (%)',
        'title': '{cls} AP-Encoding Time Curve',
    },
    'ap_dectime': {
        'x_column': 'dec_time',
        'filename': 'ap_dectime',
        'xlabel': 'Decoding Time (s/frame)',
        'ylabel': '{cls} 3D AP R40 Moderate (%)',
        'title': '{cls} AP-Decoding Time Curve',
    },
}

METHODS = {
    'Baseline G-PCC': {
        'color': '#ff7f0e',
        'marker': 'X',
        'linestyle': '--',
    },
    'Split-GPCC': {
        'color': '#1f77b4',
        'marker': 's',
        'linestyle': '-.',
    },
    'JUQP Router': {
        'color': '#2ca02c',
        'marker': 'o',
        'linestyle': '-',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot AP-bpp, AP-enctime, and AP-dectime curves for baseline G-PCC, Split-GPCC, and JUQP Router.'
    )
    parser.add_argument(
        '--baseline_csv',
        default='point_pairs/baseline_fov/baseline_gpcc_curve.csv',
        help='Path to baseline curve CSV.',
    )
    parser.add_argument(
        '--split_csv',
        default='point_pairs/split_gpcc_fov/split_gpcc_curve.csv',
        help='Path to Split-GPCC curve CSV.',
    )
    parser.add_argument(
        '--juqp_csv',
        default='point_pairs/router_gpcc_fov/router_gpcc_curve.csv',
        help='Path to JUQP Router curve CSV. Set to an empty string to disable.',
    )
    parser.add_argument(
        '--out_dir',
        default='plots',
        help='Directory where all curve images will be saved.',
    )
    parser.add_argument(
        '--formats',
        default='png',
        help='Comma-separated output image formats, e.g. png,pdf.',
    )
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI for raster formats.')
    return parser.parse_args()


def read_curve_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f'No rows found in {path}')
    return rows


def maybe_read_curve_csv(path):
    if path is None or str(path).strip() == '':
        return None
    path = Path(path)
    if not path.exists():
        print(f'[!] Optional curve CSV not found, skipping: {path}')
        return None
    return read_curve_csv(path)


def to_float(row, column):
    value = row.get(column, '')
    if value is None or value == '':
        return None
    return float(value)


def collect_points(rows, x_column, y_column):
    points = []
    for row in rows:
        x = to_float(row, x_column)
        y = to_float(row, y_column)
        if x is None or y is None:
            continue
        points.append((x, y))
    points.sort(key=lambda item: item[0])
    return points


def adaptive_ylim(curves, bottom_zero=True):
    values = [y for points in curves for _, y in points]
    if not values:
        return None

    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        pad = max(abs(y_max) * 0.05, 1.0)
    else:
        pad = (y_max - y_min) * 0.08

    bottom = 0 if bottom_zero else y_min - pad
    if not bottom_zero:
        bottom = max(0, bottom)
    top = y_max + pad
    return bottom, top


def plot_one(out_dir, formats, dpi, metric_key, datasets, cls=None):
    metric = METRICS[metric_key]
    x_column = metric['x_column']
    y_column = CLASSES[cls]

    plt.figure(figsize=(8.5, 6))
    plotted = []

    for method_name, rows in datasets.items():
        points = collect_points(rows, x_column, y_column)
        if not points:
            continue
        x_vals = [x for x, _ in points]
        y_vals = [y for _, y in points]
        style = METHODS[method_name]
        plt.plot(
            x_vals,
            y_vals,
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            markersize=6,
            linewidth=2.2,
            label=method_name,
        )
        plotted.append(points)

    if not plotted:
        raise ValueError(f'No plottable points for {cls} {metric_key}')

    plt.xlabel(metric['xlabel'], fontsize=13)
    plt.ylabel(metric['ylabel'].format(cls=cls), fontsize=13)
    plt.title(metric['title'].format(cls=cls), fontsize=15, pad=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11)

    ylim = adaptive_ylim(plotted, bottom_zero=True)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()

    saved = []
    stem = f'{metric["filename"]}_{cls.lower()}'
    for fmt in formats:
        out_path = out_dir / f'{stem}.{fmt}'
        plt.savefig(out_path, dpi=dpi, facecolor='white')
        saved.append(out_path)
    plt.close()
    return saved


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip().lstrip('.') for fmt in args.formats.split(',') if fmt.strip()]

    datasets = {
        'Baseline G-PCC': read_curve_csv(args.baseline_csv),
        'Split-GPCC': read_curve_csv(args.split_csv),
    }
    juqp_rows = maybe_read_curve_csv(args.juqp_csv)
    if juqp_rows is not None:
        datasets['JUQP Router'] = juqp_rows

    saved_paths = []
    for cls in CLASSES:
        for metric_key in ('ap_bpp', 'ap_enctime', 'ap_dectime'):
            saved_paths.extend(plot_one(out_dir, formats, args.dpi, metric_key, datasets, cls=cls))

    print(f'[+] Saved {len(saved_paths)} plots to {out_dir}')
    for path in saved_paths:
        print(path)


if __name__ == '__main__':
    main()
