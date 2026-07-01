#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from data_utils.geometry.inout import write_ply_o3d
from extention.pc_error_geo import pc_error


FG_CLASSES = [1]
PSNR_KEY = 'mseF,PSNR (p2point)'
MSE_KEY = 'mseF      (p2point)'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute PSNR-bpp curve point pairs for Baseline G-PCC, Split-GPCC, and JUQP Router.'
    )
    parser.add_argument('--testdata', default='OpenPCDet/data/kitti_fov/training/velodyne')
    parser.add_argument('--split_file', default='OpenPCDet/data/kitti_fov/ImageSets/val.txt')
    parser.add_argument('--mask_dir', default='point_pairs/split_gpcc_fov/seg_masks')
    parser.add_argument('--baseline_curve_csv', default='point_pairs/baseline_fov/baseline_gpcc_curve.csv')
    parser.add_argument('--split_curve_csv', default='point_pairs/split_gpcc_fov/split_gpcc_curve.csv')
    parser.add_argument('--router_curve_csv', default='point_pairs/router_gpcc_fov/router_gpcc_curve.csv')
    parser.add_argument('--router_details_csv', default='point_pairs/router_gpcc_fov/gpcc/router_all_details.csv')
    parser.add_argument('--out_dir', default='point_pairs/psnr_bpp')
    parser.add_argument('--tmp_dir', default='point_pairs/psnr_tmp')
    parser.add_argument('--resolution', type=int, default=80000)
    parser.add_argument('--methods', default='baseline,split,router',
                        help='Comma-separated subset: baseline,split,router')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Only use the first N frames from split_file. Useful for smoke tests.')
    parser.add_argument('--keep_details', action='store_true',
                        help='Also write per-frame PSNR details for each method.')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_split_file(path):
    with open(path) as f:
        return [line.strip().zfill(6) for line in f if line.strip()]


def read_kitti_bin(path):
    points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def to_float(value):
    return float(str(value).strip())


def parse_scale_pair(value):
    parts = [part.strip() for part in str(value).split(',')]
    if len(parts) != 2:
        raise ValueError(f'Expected fg,bg scale pair, got: {value}')
    return to_float(parts[0]), to_float(parts[1])


def quantize_subset(coords_scaled, scale):
    if len(coords_scaled) == 0:
        return np.empty((0, 3), dtype=np.int32)
    if scale >= 1.0:
        return coords_scaled.astype(np.int32)
    quantized = np.round(coords_scaled.astype(np.float64) * scale).astype(np.int32)
    unique_quantized = np.unique(quantized, axis=0)
    reconstructed = unique_quantized.astype(np.float64) / scale
    return np.round(reconstructed).astype(np.int32)


def quantize_global(coords_scaled, scale):
    return quantize_subset(coords_scaled, scale)


def quantize_split(coords_scaled, labels, scale_fg, scale_bg):
    labels = labels[:len(coords_scaled)]
    fg_mask = np.isin(labels, FG_CLASSES)
    dec_fg = quantize_subset(coords_scaled[fg_mask], scale_fg)
    dec_bg = quantize_subset(coords_scaled[~fg_mask], scale_bg)
    if len(dec_fg) and len(dec_bg):
        return np.concatenate([dec_fg, dec_bg], axis=0)
    return dec_fg if len(dec_fg) else dec_bg


def psnr_for_coords(ref_coords, dec_coords, frame_id, tag, tmp_dir, resolution):
    ref_ply = tmp_dir / f'{frame_id}_{tag}_ref.ply'
    dec_ply = tmp_dir / f'{frame_id}_{tag}_dec.ply'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        write_ply_o3d(str(ref_ply), ref_coords + 1, dtype='int32')
        write_ply_o3d(str(dec_ply), dec_coords + 1, dtype='int32')
        results = pc_error(str(ref_ply), str(dec_ply), resolution=resolution, normal=False, show=False)
        if PSNR_KEY not in results:
            raise RuntimeError(f'pc_error did not return {PSNR_KEY} for {frame_id} {tag}: {results}')
        return float(results[PSNR_KEY]), float(results.get(MSE_KEY, 0.0))
    finally:
        for path in (ref_ply, dec_ply):
            if path.exists():
                path.unlink()


def iter_progress(items, **kwargs):
    if tqdm is None:
        return items
    return tqdm(items, **kwargs)


def load_frames(testdata, split_file, max_frames=None):
    testdata = Path(testdata)
    frame_ids = read_split_file(split_file)
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    frames = []
    for frame_id in iter_progress(frame_ids, desc='Loading frames', unit='frame'):
        bin_path = testdata / f'{frame_id}.bin'
        if not bin_path.exists():
            raise FileNotFoundError(bin_path)
        coords_raw = read_kitti_bin(bin_path)
        coords_mm = np.round(coords_raw.astype(np.float64) * 1000).astype(np.int32)
        coords_scaled = coords_mm - coords_mm.min(axis=0)
        frames.append((frame_id, coords_scaled))
    return frames


def aggregate_rate(rows):
    if not rows:
        return None
    psnr_values = [row['psnr_p2point'] for row in rows]
    mse_values = [row['mse_p2point'] for row in rows]
    return round(float(np.mean(psnr_values)), 6), round(float(np.mean(mse_values)), 12)


def compute_baseline(curve_rows, frames, tmp_dir, resolution):
    detail_rows = []
    curve_out = []
    for row in iter_progress(curve_rows, desc='Baseline PSNR', unit='rate'):
        rate_id = int(row['rate_id'])
        scale = to_float(row.get('scale') or row['posQuantscale'])
        per_frame = []
        frame_iter = iter_progress(frames, desc=f'Baseline r{rate_id}', unit='frame', leave=False)
        for frame_id, coords_scaled in frame_iter:
            dec_coords = quantize_global(coords_scaled, scale)
            psnr, mse = psnr_for_coords(coords_scaled, dec_coords, frame_id, f'baseline_r{rate_id}', tmp_dir, resolution)
            detail = {
                'method': 'Baseline G-PCC',
                'filename': frame_id,
                'rate_id': rate_id,
                'scale': scale,
                'bpp': to_float(row['bpp']),
                'psnr_p2point': psnr,
                'mse_p2point': mse,
            }
            detail_rows.append(detail)
            per_frame.append(detail)
        avg_psnr, avg_mse = aggregate_rate(per_frame)
        curve_out.append({
            'method': 'Baseline G-PCC',
            'rate_id': rate_id,
            'scale': scale,
            'posQuantscale': row.get('posQuantscale') or scale,
            'bpp': row['bpp'],
            'psnr_p2point': avg_psnr,
            'mse_p2point': avg_mse,
        })
    return curve_out, detail_rows


def compute_split(curve_rows, frames, mask_dir, tmp_dir, resolution):
    detail_rows = []
    curve_out = []
    mask_dir = Path(mask_dir)
    for row in iter_progress(curve_rows, desc='Split-GPCC PSNR', unit='rate'):
        rate_id = int(row['rate_id'])
        scale_fg = to_float(row['posQ_fg'])
        scale_bg = to_float(row['posQ_bg'])
        per_frame = []
        frame_iter = iter_progress(frames, desc=f'Split r{rate_id}', unit='frame', leave=False)
        for frame_id, coords_scaled in frame_iter:
            mask_path = mask_dir / f'{frame_id}.npy'
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            labels = np.load(mask_path)
            dec_coords = quantize_split(coords_scaled, labels, scale_fg, scale_bg)
            psnr, mse = psnr_for_coords(coords_scaled, dec_coords, frame_id, f'split_r{rate_id}', tmp_dir, resolution)
            detail = {
                'method': 'Split-GPCC',
                'filename': frame_id,
                'rate_id': rate_id,
                'posQ_fg': scale_fg,
                'posQ_bg': scale_bg,
                'bpp': to_float(row['bpp']),
                'psnr_p2point': psnr,
                'mse_p2point': mse,
            }
            detail_rows.append(detail)
            per_frame.append(detail)
        avg_psnr, avg_mse = aggregate_rate(per_frame)
        curve_out.append({
            'method': 'Split-GPCC',
            'rate_id': rate_id,
            'posQ_fg': scale_fg,
            'posQ_bg': scale_bg,
            'posQuantscale': row.get('posQuantscale') or f'{scale_fg},{scale_bg}',
            'bpp': row['bpp'],
            'psnr_p2point': avg_psnr,
            'mse_p2point': avg_mse,
        })
    return curve_out, detail_rows


def group_router_details(path):
    grouped = {}
    for row in read_csv(path):
        grouped.setdefault(int(row['rate_id']), {})[str(row['filename']).zfill(6)] = row
    return grouped


def compute_router(curve_rows, frames, mask_dir, router_details_csv, tmp_dir, resolution):
    detail_rows = []
    curve_out = []
    mask_dir = Path(mask_dir)
    router_by_rate = group_router_details(router_details_csv)
    for row in iter_progress(curve_rows, desc='JUQP Router PSNR', unit='rate'):
        rate_id = int(row['rate_id'])
        frame_details = router_by_rate.get(rate_id)
        if frame_details is None:
            raise KeyError(f'Missing router detail rows for rate_id={rate_id}')
        per_frame = []
        frame_iter = iter_progress(frames, desc=f'Router r{rate_id}', unit='frame', leave=False)
        for frame_id, coords_scaled in frame_iter:
            if frame_id not in frame_details:
                raise KeyError(f'Missing router detail row for frame={frame_id}, rate_id={rate_id}')
            detail_row = frame_details[frame_id]
            scale_fg = to_float(detail_row['posQ_fg'])
            scale_bg = to_float(detail_row['posQ_bg'])
            mask_path = mask_dir / f'{frame_id}.npy'
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            labels = np.load(mask_path)
            dec_coords = quantize_split(coords_scaled, labels, scale_fg, scale_bg)
            psnr, mse = psnr_for_coords(coords_scaled, dec_coords, frame_id, f'router_r{rate_id}', tmp_dir, resolution)
            detail = {
                'method': 'JUQP Router',
                'filename': frame_id,
                'rate_id': rate_id,
                'jucp_label': detail_row.get('jucp_label', ''),
                'posQ_fg': scale_fg,
                'posQ_bg': scale_bg,
                'bpp': to_float(row['bpp']),
                'psnr_p2point': psnr,
                'mse_p2point': mse,
            }
            detail_rows.append(detail)
            per_frame.append(detail)
        avg_psnr, avg_mse = aggregate_rate(per_frame)
        curve_out.append({
            'method': 'JUQP Router',
            'rate_id': rate_id,
            'threshold': row.get('threshold', ''),
            'bpp': row['bpp'],
            'psnr_p2point': avg_psnr,
            'mse_p2point': avg_mse,
        })
    return curve_out, detail_rows


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    tmp_dir = Path(args.tmp_dir)
    methods = {item.strip().lower() for item in args.methods.split(',') if item.strip()}
    frames = load_frames(args.testdata, args.split_file, max_frames=args.max_frames)
    print(f'[i] Loaded {len(frames)} frames from {args.split_file}', flush=True)

    all_curve_rows = []
    jobs = []
    if 'baseline' in methods:
        curve_rows = read_csv(args.baseline_curve_csv)
        print(f'[i] Baseline jobs: {len(curve_rows)} rates x {len(frames)} frames', flush=True)
        jobs.append(('baseline', compute_baseline(curve_rows, frames, tmp_dir / 'baseline', args.resolution)))
    if 'split' in methods:
        curve_rows = read_csv(args.split_curve_csv)
        print(f'[i] Split-GPCC jobs: {len(curve_rows)} rates x {len(frames)} frames', flush=True)
        jobs.append(('split', compute_split(curve_rows, frames, args.mask_dir, tmp_dir / 'split', args.resolution)))
    if 'router' in methods:
        curve_rows = read_csv(args.router_curve_csv)
        print(f'[i] JUQP Router jobs: {len(curve_rows)} rates x {len(frames)} frames', flush=True)
        jobs.append(('router', compute_router(
            curve_rows, frames, args.mask_dir, args.router_details_csv, tmp_dir / 'router', args.resolution)))

    for name, (curve_rows, detail_rows) in jobs:
        curve_path = out_dir / f'{name}_psnr_bpp_curve.csv'
        write_csv(curve_path, curve_rows, list(curve_rows[0].keys()) if curve_rows else [])
        print(f'[+] Wrote {curve_path}')
        all_curve_rows.extend(curve_rows)
        if args.keep_details:
            detail_path = out_dir / f'{name}_psnr_details.csv'
            write_csv(detail_path, detail_rows, list(detail_rows[0].keys()) if detail_rows else [])
            print(f'[+] Wrote {detail_path}')

    all_path = out_dir / 'all_methods_psnr_bpp_curve.csv'
    all_fields = ['method', 'rate_id', 'bpp', 'psnr_p2point', 'mse_p2point']
    for optional in ('scale', 'posQuantscale', 'posQ_fg', 'posQ_bg', 'threshold'):
        if any(optional in row for row in all_curve_rows):
            all_fields.append(optional)
    write_csv(all_path, all_curve_rows, all_fields)
    print(f'[+] Wrote {all_path}')


if __name__ == '__main__':
    main()
