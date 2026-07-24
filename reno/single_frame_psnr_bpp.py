#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHOD_STYLE = {
    'RENO': {'color': '#d62728', 'marker': '^', 'linestyle': '-'},
    'Baseline G-PCC': {'color': '#ff7f0e', 'marker': 'X', 'linestyle': '--'},
    'Split-GPCC': {'color': '#1f77b4', 'marker': 's', 'linestyle': '-.'},
}


def read_csv(path, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def frame_id_from_value(value):
    stem = Path(str(value)).stem
    return stem.zfill(6) if stem.isdigit() else stem


def to_float(value, default=None):
    if value is None or value == '':
        return default
    return float(value)


def to_int(value, default=None):
    if value is None or value == '':
        return default
    return int(float(value))


def bpp_by_rate(rows, frame_id):
    out = {}
    for row in rows:
        if frame_id_from_value(row.get('filename', '')) != frame_id:
            continue
        rate_id = str(to_int(row.get('rate_id')))
        out[rate_id] = row
    return out


def psnr_by_rate(rows, frame_id):
    out = {}
    for row in rows:
        if frame_id_from_value(row.get('filename', '')) != frame_id:
            continue
        rate_id = str(to_int(row.get('rate_id')))
        out[rate_id] = row
    return out


def collect_gpcc(method, frame_id, bpp_detail_csv, psnr_detail_csv):
    bpp_rows = bpp_by_rate(read_csv(bpp_detail_csv), frame_id)
    psnr_rows = psnr_by_rate(read_csv(psnr_detail_csv), frame_id)
    rows = []
    for rate_id in sorted(set(bpp_rows) & set(psnr_rows), key=lambda x: int(x)):
        bpp_row = bpp_rows[rate_id]
        psnr_row = psnr_rows[rate_id]
        scale = bpp_row.get('scale') or psnr_row.get('scale') or bpp_row.get('posQuantscale') or ''
        rows.append({
            'method': method,
            'filename': frame_id,
            'rate_id': int(rate_id),
            'scale': scale,
            'bpp': to_float(bpp_row.get('bpp')),
            'psnr_d1': to_float(psnr_row.get('psnr_p2point')),
            'psnr_d2': '',
            'enc_time': to_float(bpp_row.get('enc_time'), ''),
            'dec_time': to_float(bpp_row.get('dec_time'), ''),
        })
    return rows


def collect_reno(frame_id, reno_detail_csv):
    rows = []
    for row in read_csv(reno_detail_csv):
        if frame_id_from_value(row.get('filename', '')) != frame_id:
            continue
        rows.append({
            'method': 'RENO',
            'filename': frame_id,
            'rate_id': to_int(row.get('rate_id')),
            'scale': row.get('scale_label') or row.get('scale') or row.get('posQ') or '',
            'bpp': to_float(row.get('bpp')),
            'psnr_d1': to_float(row.get('d1_psnr')),
            'psnr_d2': to_float(row.get('d2_psnr'), ''),
            'enc_time': to_float(row.get('enc_time'), ''),
            'dec_time': to_float(row.get('dec_time'), ''),
        })
    rows.sort(key=lambda r: r['rate_id'])
    return rows


def plot_metric(rows, out_dir, metric_key, ylabel, filename):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 6))
    plotted = False
    for method in METHOD_STYLE:
        points = [r for r in rows if r['method'] == method and r.get('bpp') not in ('', None) and r.get(metric_key) not in ('', None)]
        points.sort(key=lambda r: float(r['bpp']))
        if not points:
            continue
        style = METHOD_STYLE[method]
        x_vals = [float(r['bpp']) for r in points]
        y_vals = [float(r[metric_key]) for r in points]
        plt.plot(
            x_vals,
            y_vals,
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            linewidth=2.2,
            markersize=6,
            label=method,
        )
        for r, x, y in zip(points, x_vals, y_vals):
            plt.annotate(str(r['rate_id']), (x, y), fontsize=7, xytext=(3, 3), textcoords='offset points')
        plotted = True
    if not plotted:
        raise ValueError(f'No plottable rows for {metric_key}')
    plt.xlabel('Bits Per Point (bpp)', fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(ylabel + '-bpp', fontsize=15, pad=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    path = out_dir / filename
    plt.savefig(path, dpi=300, facecolor='white')
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description='Merge and plot one-frame RENO/G-PCC/Split-GPCC PSNR-bpp rows.')
    parser.add_argument('--frame', required=True)
    parser.add_argument('--reno_detail_csv', required=True)
    parser.add_argument('--baseline_detail_csv', default='point_pairs/baseline_fov/gpcc/gpcc_baseline_details.csv')
    parser.add_argument('--split_detail_csv', default='point_pairs/split_gpcc_fov/gpcc/split_all_details.csv')
    parser.add_argument('--baseline_psnr_csv', default='point_pairs/psnr_bpp/baseline_psnr_details.csv')
    parser.add_argument('--split_psnr_csv', default='point_pairs/psnr_bpp/split_psnr_details.csv')
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    frame_id = frame_id_from_value(args.frame)
    rows = []
    rows.extend(collect_reno(frame_id, args.reno_detail_csv))
    rows.extend(collect_gpcc('Baseline G-PCC', frame_id, args.baseline_detail_csv, args.baseline_psnr_csv))
    rows.extend(collect_gpcc('Split-GPCC', frame_id, args.split_detail_csv, args.split_psnr_csv))

    if not rows:
        raise ValueError(f'No rows found for frame {frame_id}')

    rows.sort(key=lambda r: (r['method'], int(r['rate_id'])))
    fields = ['method', 'filename', 'rate_id', 'scale', 'bpp', 'psnr_d1', 'psnr_d2', 'enc_time', 'dec_time']
    out_dir = Path(args.out_dir)
    merged_csv = out_dir / f'{frame_id}_single_frame_psnr_bpp.csv'
    write_csv(merged_csv, rows, fields)

    saved = [plot_metric(rows, out_dir, 'psnr_d1', 'D1 PSNR (dB)', f'{frame_id}_d1_psnr_bpp.png')]
    d2_path = out_dir / f'{frame_id}_d2_psnr_bpp.png'
    if any(r.get('psnr_d2') not in ('', None) for r in rows):
        saved.append(plot_metric(rows, out_dir, 'psnr_d2', 'D2 PSNR (dB)', d2_path.name))
    elif d2_path.exists():
        d2_path.unlink()

    print(f'Merged CSV: {merged_csv}')
    for path in saved:
        print(f'Plot: {path}')


if __name__ == '__main__':
    main()
