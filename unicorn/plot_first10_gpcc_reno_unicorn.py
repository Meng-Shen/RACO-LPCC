#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


STYLES = {
    'G-PCC': {'color': '#ff7f0e', 'marker': 'X', 'linestyle': '--'},
    'Split-GPCC': {'color': '#1f77b4', 'marker': 's', 'linestyle': '-.'},
    'RENO': {'color': '#d62728', 'marker': '^', 'linestyle': '-'},
    'Unicorn': {'color': '#9467bd', 'marker': 'D', 'linestyle': '-'},
    'Unicorn no SR + offset': {'color': '#17becf', 'marker': 'P', 'linestyle': '--'},
    'Unicorn SR + no offset': {'color': '#2ca02c', 'marker': 'o', 'linestyle': '-.'},
    'Unicorn no SR + no offset': {'color': '#7f7f7f', 'marker': 's', 'linestyle': ':'},
}


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def read_frames(path):
    with open(path) as f:
        return {line.strip().zfill(6) for line in f if line.strip()}


def frame_id(value):
    return Path(str(value)).stem.zfill(6)


def to_float(value):
    return float(value)


def to_int(value):
    return int(float(value))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'method', 'rate_id', 'rate_label', 'num_frames', 'total_points',
        'total_bits', 'bpp', 'enc_time', 'dec_time', 'd1_psnr', 'd2_psnr',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_detail(method, detail_csv, frames, psnr_csv=None):
    detail_rows = [
        row for row in read_csv(detail_csv)
        if frame_id(row.get('filename', '')) in frames
    ]

    psnr_by_case = {}
    if psnr_csv:
        for row in read_csv(psnr_csv):
            fid = frame_id(row.get('filename', ''))
            if fid in frames:
                psnr_by_case[(fid, to_int(row['rate_id']))] = to_float(row['psnr_p2point'])

    grouped = {}
    for row in detail_rows:
        fid = frame_id(row['filename'])
        rid = to_int(row['rate_id'])
        psnr = row.get('d1_psnr', '')
        if psnr == '' and psnr_by_case:
            psnr = psnr_by_case.get((fid, rid), '')
        if psnr == '':
            continue
        grouped.setdefault(rid, []).append((row, to_float(psnr)))

    out = []
    for rid in sorted(grouped):
        rows = grouped[rid]
        frame_count = len({frame_id(row['filename']) for row, _ in rows})
        if frame_count != len(frames):
            print(
                f'[!] Skipping incomplete {method} rate {rid}: '
                f'{frame_count}/{len(frames)} frames',
            )
            continue
        total_points = sum(to_int(row['num_points']) for row, _ in rows)
        total_bits = sum(to_int(row['bits']) for row, _ in rows)
        enc_times = [to_float(row['enc_time']) for row, _ in rows if row.get('enc_time', '') != '']
        dec_times = [to_float(row['dec_time']) for row, _ in rows if row.get('dec_time', '') != '']
        d2_values = [to_float(row['d2_psnr']) for row, _ in rows if row.get('d2_psnr', '') != '']
        first = rows[0][0]
        label = first.get('scale_label') or first.get('rate_label') or first.get('scale') or first.get('posQuantscale') or ''
        out.append({
            'method': method,
            'rate_id': rid,
            'rate_label': label,
            'num_frames': frame_count,
            'total_points': total_points,
            'total_bits': total_bits,
            'bpp': round(total_bits / total_points, 6) if total_points else 0.0,
            'enc_time': round(sum(enc_times) / len(enc_times), 6) if enc_times else '',
            'dec_time': round(sum(dec_times) / len(dec_times), 6) if dec_times else '',
            'd1_psnr': round(sum(psnr for _, psnr in rows) / len(rows), 6),
            'd2_psnr': round(sum(d2_values) / len(d2_values), 6) if len(d2_values) == len(rows) else '',
        })
    return out


def plot(rows, out_path, metric, ylabel):
    plt.figure(figsize=(8.5, 6))
    plotted = False
    for method, style in STYLES.items():
        points = [
            row for row in rows
            if row['method'] == method and row.get(metric, '') not in ('', None)
        ]
        points.sort(key=lambda row: float(row['bpp']))
        if not points:
            continue
        x_vals = [float(row['bpp']) for row in points]
        y_vals = [float(row[metric]) for row in points]
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
        plotted = True
        for row, x, y in zip(points, x_vals, y_vals):
            plt.annotate(str(row['rate_id']), (x, y), fontsize=7, xytext=(3, 3), textcoords='offset points')

    if not plotted:
        plt.close()
        return None
    plt.xlabel('Bits Per Point (bpp)', fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(f'First 10 KITTI Frames {ylabel}-bpp', fontsize=15, pad=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close()
    return out_path


def save_image_variants(png_path):
    try:
        from PIL import Image
    except ImportError:
        return []
    im = Image.open(png_path).convert('RGB')
    saved = []
    for suffix, kwargs in (('.jpg', {'quality': 95}), ('.pdf', {})):
        out = png_path.with_suffix(suffix)
        im.save(out, **kwargs)
        saved.append(out)
    return saved


def main():
    parser = argparse.ArgumentParser(description='Aggregate and plot first-10-frame PSNR-bpp curves.')
    parser.add_argument('--split_file', default='point_pairs/unicorn_first10/first10_split.txt')
    parser.add_argument('--gpcc_detail_csv', default='point_pairs/baseline_fov/gpcc/gpcc_baseline_details.csv')
    parser.add_argument('--gpcc_psnr_csv', default='point_pairs/psnr_bpp/baseline_psnr_details.csv')
    parser.add_argument('--split_detail_csv', default='point_pairs/split_gpcc_fov/gpcc/split_all_details.csv')
    parser.add_argument('--split_psnr_csv', default='point_pairs/psnr_bpp/split_psnr_details.csv')
    parser.add_argument(
        '--skip_split_gpcc',
        action='store_true',
        help='Plot only G-PCC, RENO, and Unicorn.',
    )
    parser.add_argument('--reno_detail_csv', default='point_pairs/reno_fov/reno/reno_details.csv')
    parser.add_argument('--unicorn_detail_csv', default='point_pairs/unicorn_first10/unicorn/unicorn_details.csv')
    parser.add_argument('--unicorn_no_sr_offset_detail_csv', default='')
    parser.add_argument('--unicorn_sr_no_offset_detail_csv', default='')
    parser.add_argument('--unicorn_no_sr_no_offset_detail_csv', default='')
    parser.add_argument('--out_dir', default='point_pairs/unicorn_first10')
    args = parser.parse_args()

    frames = read_frames(args.split_file)
    if len(frames) != 10:
        raise ValueError(f'Expected exactly 10 unique frames in {args.split_file}, found {len(frames)}')
    rows = []
    rows.extend(aggregate_detail('G-PCC', args.gpcc_detail_csv, frames, psnr_csv=args.gpcc_psnr_csv))
    if not args.skip_split_gpcc:
        rows.extend(aggregate_detail('Split-GPCC', args.split_detail_csv, frames, psnr_csv=args.split_psnr_csv))
    rows.extend(aggregate_detail('RENO', args.reno_detail_csv, frames))
    rows.extend(aggregate_detail('Unicorn', args.unicorn_detail_csv, frames))
    optional_unicorn = [
        ('Unicorn no SR + offset', args.unicorn_no_sr_offset_detail_csv),
        ('Unicorn SR + no offset', args.unicorn_sr_no_offset_detail_csv),
        ('Unicorn no SR + no offset', args.unicorn_no_sr_no_offset_detail_csv),
    ]
    for method, detail_csv in optional_unicorn:
        if not detail_csv:
            continue
        detail_path = Path(detail_csv)
        if detail_path.exists():
            rows.extend(aggregate_detail(method, detail_path, frames))
        else:
            print(f'[!] Optional Unicorn ablation CSV not found, skipping: {detail_path}')
    rows.sort(key=lambda row: (row['method'], int(row['rate_id'])))
    if not rows:
        raise ValueError('No complete first-10-frame rows were found to aggregate.')

    out_dir = Path(args.out_dir)
    csv_path = out_dir / 'first10_gpcc_reno_unicorn_psnr_bpp.csv'
    png_path = out_dir / 'first10_gpcc_reno_unicorn_d1_psnr_bpp.png'
    write_csv(csv_path, rows)
    saved = []
    d1_path = plot(rows, png_path, 'd1_psnr', 'D1 PSNR (dB)')
    if d1_path is not None:
        saved.append(d1_path)
        saved.extend(save_image_variants(d1_path))
    d2_png_path = out_dir / 'first10_gpcc_reno_unicorn_d2_psnr_bpp.png'
    d2_path = plot(rows, d2_png_path, 'd2_psnr', 'D2 PSNR (dB)')
    if d2_path is not None:
        saved.append(d2_path)
        saved.extend(save_image_variants(d2_path))

    print(f'CSV: {csv_path}')
    for path in saved:
        print(f'Plot: {path}')


if __name__ == '__main__':
    main()
