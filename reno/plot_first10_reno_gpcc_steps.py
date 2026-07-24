#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


FIELDS = [
    'method',
    'rate_id',
    'quant_step_mm',
    'num_frames',
    'total_points',
    'total_bits',
    'bpp',
    'enc_time',
    'dec_time',
    'd1_psnr',
]


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def read_frames(path):
    with open(path) as f:
        return {line.strip().zfill(6) for line in f if line.strip()}


def frame_id(value):
    return Path(str(value)).stem.zfill(6)


def format_number(value):
    value = float(value)
    if math.isinf(value):
        return 'inf'
    return f'{value:.6f}'.rstrip('0').rstrip('.')


def aggregate(method, detail_csv, frames, psnr_csv=None):
    rows = [
        row for row in read_csv(detail_csv)
        if frame_id(row['filename']) in frames
    ]
    psnr_by_case = {}
    if psnr_csv:
        for row in read_csv(psnr_csv):
            fid = frame_id(row['filename'])
            if fid in frames:
                psnr_by_case[(fid, int(row['rate_id']))] = float(row['psnr_p2point'])

    grouped = {}
    for row in rows:
        grouped.setdefault(int(row['rate_id']), []).append(row)

    out = []
    for rate_id in sorted(grouped):
        rate_rows = grouped[rate_id]
        if len({frame_id(row['filename']) for row in rate_rows}) != len(frames):
            raise ValueError(
                f'{method} rate {rate_id} does not cover all {len(frames)} frames'
            )
        total_points = sum(int(row['num_points']) for row in rate_rows)
        total_bits = sum(int(row['bits']) for row in rate_rows)
        if method == 'G-PCC':
            scale = float(rate_rows[0]['posQuantscale'])
            quant_step_mm = 1.0 / scale
            d1_values = [
                psnr_by_case[(frame_id(row['filename']), rate_id)]
                for row in rate_rows
            ]
        else:
            quant_step_mm = float(rate_rows[0]['posQ'])
            d1_values = [float(row['d1_psnr']) for row in rate_rows]
        enc_times = [float(row['enc_time']) for row in rate_rows]
        dec_times = [float(row['dec_time']) for row in rate_rows]
        out.append({
            'method': method,
            'rate_id': rate_id,
            'quant_step_mm': format_number(quant_step_mm),
            'num_frames': len(rate_rows),
            'total_points': total_points,
            'total_bits': total_bits,
            'bpp': format_number(total_bits / total_points),
            'enc_time': format_number(sum(enc_times) / len(enc_times)),
            'dec_time': format_number(sum(dec_times) / len(dec_times)),
            'd1_psnr': format_number(sum(d1_values) / len(d1_values)),
        })
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, out_path, annotate_steps=False):
    styles = {
        'G-PCC': {'color': '#ff7f0e', 'marker': 'X', 'linestyle': '--'},
        'RENO': {'color': '#d62728', 'marker': '^', 'linestyle': '-'},
    }
    finite_psnr = [
        float(row['d1_psnr'])
        for row in rows
        if math.isfinite(float(row['d1_psnr']))
    ]
    finite_min = min(finite_psnr)
    finite_max = max(finite_psnr)
    inf_display = finite_max + max(4.0, 0.08 * (finite_max - finite_min))

    plt.figure(figsize=(9.5, 6.5))
    for method, style in styles.items():
        points = sorted(
            (row for row in rows if row['method'] == method),
            key=lambda row: float(row['bpp']),
        )
        x_values = [float(row['bpp']) for row in points]
        raw_y = [float(row['d1_psnr']) for row in points]
        y_values = [value if math.isfinite(value) else inf_display for value in raw_y]
        plt.plot(
            x_values,
            y_values,
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            linewidth=2.2,
            markersize=7,
            label=method,
        )

    if annotate_steps:
        by_step = {}
        for row in rows:
            raw_y = float(row['d1_psnr'])
            by_step.setdefault(float(row['quant_step_mm']), []).append((
                float(row['bpp']),
                raw_y if math.isfinite(raw_y) else inf_display,
                raw_y,
            ))
        for step, values in sorted(by_step.items()):
            x = sum(value[0] for value in values) / len(values)
            y = sum(value[1] for value in values) / len(values)
            is_infinite = any(not math.isfinite(value[2]) for value in values)
            label = f'Δ={format_number(step)} mm'
            if is_infinite:
                label += ' (∞)'
            offset = (-85, 7) if is_infinite else (4, 6)
            plt.annotate(
                label,
                (x, y),
                fontsize=7,
                xytext=offset,
                textcoords='offset points',
            )

    plt.xlabel('Bits Per Point (bpp)', fontsize=13)
    plt.ylabel('D1 PSNR (dB)', fontsize=13)
    plt.title('First 10 KITTI Frames: RENO vs G-PCC', fontsize=15, pad=12)
    plt.ylim(finite_min - 3.0, inf_display + 3.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf', '.jpg'):
        path = out_path.with_suffix(suffix)
        kwargs = {'dpi': 300, 'facecolor': 'white'}
        if suffix == '.jpg':
            kwargs['pil_kwargs'] = {'quality': 95}
        plt.savefig(path, **kwargs)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot first-10-frame RENO and G-PCC D1 PSNR-bpp curves.'
    )
    parser.add_argument('--split_file', required=True)
    parser.add_argument('--gpcc_detail_csv', required=True)
    parser.add_argument('--gpcc_psnr_csv', required=True)
    parser.add_argument('--reno_detail_csv', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument(
        '--annotate_steps',
        action='store_true',
        help='Annotate each curve point with its quantization step.',
    )
    args = parser.parse_args()

    frames = read_frames(args.split_file)
    if len(frames) != 10:
        raise ValueError(f'Expected 10 frames, found {len(frames)}')

    rows = []
    rows.extend(aggregate(
        'G-PCC', args.gpcc_detail_csv, frames, psnr_csv=args.gpcc_psnr_csv
    ))
    rows.extend(aggregate('RENO', args.reno_detail_csv, frames))
    rows.sort(key=lambda row: (row['method'], int(row['rate_id'])))

    out_dir = Path(args.out_dir)
    csv_path = out_dir / 'first10_reno_gpcc_steps_1_2048_psnr_bpp.csv'
    plot_path = out_dir / 'first10_reno_gpcc_steps_1_2048_d1_psnr_bpp.png'
    write_csv(csv_path, rows)
    plot(rows, plot_path, annotate_steps=args.annotate_steps)
    print(f'CSV: {csv_path}')
    print(f'Plot: {plot_path}')


if __name__ == '__main__':
    main()
