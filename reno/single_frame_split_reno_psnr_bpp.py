#!/usr/bin/env python3
import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(Path(__file__).resolve().parent))

from reno_rates import (  # noqa: E402
    compute_psnr,
    configure_torchsparse,
    decode_bitstream,
    encode_tensor,
    load_model,
    parse_rates,
    points_to_sparse,
    read_kitti_bin,
    write_bitstream,
)


FG_CLASSES = [1]


METHOD_STYLE = {
    'RENO': {'color': '#d62728', 'marker': '^', 'linestyle': '-'},
    'Split-RENO': {'color': '#1f77b4', 'marker': 's', 'linestyle': '-.'},
}


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_reno_one(net, ref_xyz, posq, bitstream_path, tmp_dir, frame_id, rate_id, resolution, device):
    x, offset = points_to_sparse(ref_xyz, posq, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    base_coords, base_feats, byte_stream = encode_tensor(net, x)
    write_bitstream(bitstream_path, posq, offset, base_coords, base_feats, byte_stream)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    enc_time = time.perf_counter() - t0

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    dec_xyz = decode_bitstream(net, bitstream_path, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dec_time = time.perf_counter() - t0

    d1_psnr, _ = compute_psnr(ref_xyz, dec_xyz, frame_id, rate_id, tmp_dir, resolution)
    return {
        'bits': bitstream_path.stat().st_size * 8,
        'dec_xyz': dec_xyz,
        'enc_time': enc_time,
        'dec_time': dec_time,
        'd1_psnr': d1_psnr,
    }


def run_split_reno_one(net, ref_xyz, labels, posq, bitstream_dir, tmp_dir, frame_id, rate_id, resolution, device):
    labels = labels[:len(ref_xyz)]
    fg_mask = np.isin(labels, FG_CLASSES)
    bg_mask = ~fg_mask
    subsets = [('fg', fg_mask), ('bg', bg_mask)]

    total_bits = 0
    total_enc_time = 0.0
    total_dec_time = 0.0
    decoded = []
    subset_counts = {}

    for name, mask in subsets:
        subset_xyz = ref_xyz[mask]
        subset_counts[f'{name}_points'] = int(len(subset_xyz))
        if len(subset_xyz) == 0:
            continue
        bitstream_path = bitstream_dir / f'rate_{rate_id}' / f'{frame_id}_{name}.bin'
        result = run_reno_one(
            net, subset_xyz, posq, bitstream_path, tmp_dir / name, frame_id,
            f'{rate_id}_{name}', resolution, device,
        )
        total_bits += int(result['bits'])
        total_enc_time += float(result['enc_time'])
        total_dec_time += float(result['dec_time'])
        decoded.append(result['dec_xyz'])

    if not decoded:
        raise ValueError(f'No foreground/background points found for {frame_id}')

    dec_xyz = np.concatenate(decoded, axis=0)
    d1_psnr, _ = compute_psnr(ref_xyz, dec_xyz, frame_id, f'{rate_id}_split_reno', tmp_dir / 'merged', resolution)
    return {
        'bits': total_bits,
        'dec_xyz': dec_xyz,
        'enc_time': total_enc_time,
        'dec_time': total_dec_time,
        'd1_psnr': d1_psnr,
        **subset_counts,
    }


def plot_curve(rows, out_path):
    plt.figure(figsize=(8.5, 6))
    for method, style in METHOD_STYLE.items():
        points = [r for r in rows if r['method'] == method]
        points.sort(key=lambda row: float(row['bpp']))
        if not points:
            continue
        plt.plot(
            [float(r['bpp']) for r in points],
            [float(r['d1_psnr']) for r in points],
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            linewidth=2.2,
            markersize=6,
            label=method,
        )
        for r in points:
            plt.annotate(str(r['rate_id']), (float(r['bpp']), float(r['d1_psnr'])),
                         fontsize=7, xytext=(3, 3), textcoords='offset points')

    plt.xlabel('Bits Per Point (bpp)', fontsize=13)
    plt.ylabel('D1 PSNR (dB)', fontsize=13)
    plt.title('000001 D1 PSNR-bpp', fontsize=15, pad=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run one-frame RENO and foreground/background Split-RENO PSNR-bpp curves.')
    parser.add_argument('--reno_root', default='/public/DATA/sm/RENO')
    parser.add_argument('--frame_bin', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--scales', default='1/64,1/128,1/256,1/512,1/1024,1/2048')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--resolution', type=int, default=80000)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    configure_torchsparse()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    net = load_model(args.reno_root, args.ckpt, args.channels, args.kernel_size, device)

    frame_bin = Path(args.frame_bin)
    frame_id = frame_bin.stem
    points = read_kitti_bin(frame_bin)
    ref_xyz = points[:, :3]
    labels = np.load(args.mask)
    if len(labels) < len(ref_xyz):
        raise ValueError(f'Mask has {len(labels)} labels, but point cloud has {len(ref_xyz)} points')

    out_dir = Path(args.out_dir)
    rows = []

    with torch.no_grad():
        for rate in parse_rates(args.scales):
            rate_id = int(rate['rate_id'])
            posq = float(rate['posQ'])
            reno_result = run_reno_one(
                net, ref_xyz, posq,
                out_dir / 'bitstreams_reno' / f'rate_{rate_id}' / f'{frame_id}.bin',
                out_dir / 'tmp' / 'reno',
                frame_id, rate_id, args.resolution, device,
            )
            split_result = run_split_reno_one(
                net, ref_xyz, labels, posq,
                out_dir / 'bitstreams_split_reno',
                out_dir / 'tmp' / 'split_reno',
                frame_id, rate_id, args.resolution, device,
            )

            for method, result in [('RENO', reno_result), ('Split-RENO', split_result)]:
                rows.append({
                    'method': method,
                    'filename': frame_id,
                    'rate_id': rate_id,
                    'scale': rate['label'],
                    'posQ': posq,
                    'num_points': len(ref_xyz),
                    'fg_points': int(np.isin(labels[:len(ref_xyz)], FG_CLASSES).sum()),
                    'bg_points': int((~np.isin(labels[:len(ref_xyz)], FG_CLASSES)).sum()),
                    'bits': int(result['bits']),
                    'bpp': round(float(result['bits']) / len(ref_xyz), 6),
                    'decoded_points': len(result['dec_xyz']),
                    'd1_psnr': round(float(result['d1_psnr']), 6),
                    'enc_time': round(float(result['enc_time']), 6),
                    'dec_time': round(float(result['dec_time']), 6),
                })
            print(f"[{frame_id}] rate {rate_id}: RENO bpp={rows[-2]['bpp']} psnr={rows[-2]['d1_psnr']} | "
                  f"Split-RENO bpp={rows[-1]['bpp']} psnr={rows[-1]['d1_psnr']}", flush=True)

    fields = [
        'method', 'filename', 'rate_id', 'scale', 'posQ', 'num_points', 'fg_points', 'bg_points',
        'bits', 'bpp', 'decoded_points', 'd1_psnr', 'enc_time', 'dec_time',
    ]
    csv_path = out_dir / f'{frame_id}_reno_vs_split_reno_psnr_bpp.csv'
    plot_path = out_dir / f'{frame_id}_reno_vs_split_reno_d1_psnr_bpp.png'
    write_csv(csv_path, rows, fields)
    plot_curve(rows, plot_path)
    print(f'CSV: {csv_path}')
    print(f'Plot: {plot_path}')


if __name__ == '__main__':
    main()
