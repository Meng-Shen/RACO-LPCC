import argparse
import csv
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.append(str(ROOT_DIR))

from data_utils.geometry.inout import write_ply_o3d
from extension.gpcc_geo import gpcc_decode, gpcc_encode


DEFAULT_SCALES_STR = "1/64,1.5/128,1/128,1.5/256,1/256,1.5/512,1/512"


@contextmanager
def suppress_stderr():
    fd = sys.stderr.fileno()
    saved_fd = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, fd)
        os.close(devnull)
        os.close(saved_fd)


def parse_scale_value(value):
    value = str(value).strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_scales(scales_str):
    scales = []
    for item in str(scales_str).split(","):
        item = item.strip()
        if item:
            scales.append(parse_scale_value(item))
    if not scales:
        raise ValueError("--scales must contain at least one quantization scale")
    return scales


def read_kitti_bin(bin_path):
    points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def progress_bar(total, **kwargs):
    if tqdm is None:
        return None
    return tqdm(total=total, **kwargs)


def collect_files(testdata, split_file):
    testdata = Path(testdata)
    if testdata.is_file():
        return [testdata]

    if split_file:
        with open(split_file, 'r') as f:
            frame_ids = [line.strip() for line in f if line.strip()]
        files = [testdata / f'{frame_id}.bin' for frame_id in frame_ids]
        return [path for path in files if path.exists()]

    return sorted(testdata.rglob('*.bin'))


def encode_decode_one_scale(
        frame_id, coords_scaled, num_points, scale, tmp_dir, cfg_path,
        encode_only=False):
    ref_ply = tmp_dir / f'{frame_id}_ref.ply'
    bitstream = tmp_dir / f'{frame_id}_scale_{scale:.12g}.bin'
    dec_ply = tmp_dir / f'{frame_id}_scale_{scale:.12g}_dec.ply'

    write_ply_o3d(str(ref_ply), coords_scaled, normal=True, knn=16)
    try:
        with suppress_stderr():
            enc_log = gpcc_encode(
                str(ref_ply), str(bitstream), posQuantscale=scale, cfgdir=str(cfg_path))
            dec_log = {} if encode_only else gpcc_decode(str(bitstream), str(dec_ply))

        bits = bitstream.stat().st_size * 8 if bitstream.exists() else 0
        return {
            'bits': bits,
            'bpp': bits / num_points if num_points else 0.0,
            'enc_time': float(enc_log.get('Processing time (wall)', 0.0)) if isinstance(enc_log, dict) else 0.0,
            'dec_time': float(dec_log.get('Processing time (wall)', 0.0)) if isinstance(dec_log, dict) else 0.0,
        }
    finally:
        for path in (ref_ply, bitstream, dec_ply):
            if path.exists():
                path.unlink()


def main():
    parser = argparse.ArgumentParser(description='Measure baseline whole-frame G-PCC bpp/enc_time/dec_time.')
    parser.add_argument('--testdata', required=True, help='KITTI velodyne directory or one .bin file')
    parser.add_argument('--split_file', default=None, help='Optional ImageSets split file, e.g. val.txt')
    parser.add_argument('--scales', default=DEFAULT_SCALES_STR, help='Comma-separated posQuantscale list')
    parser.add_argument('--results', default='GPCC/results_gpcc_baseline', help='Output result directory')
    parser.add_argument('--tmp_dir', default='GPCC/tmp_gpcc_baseline', help='Temporary file directory')
    parser.add_argument('--cfg', default='extension/kitti.cfg', help='G-PCC cfg file')
    parser.add_argument(
        '--encode_only', action='store_true',
        help='Skip G-PCC decoding when only bitstream size/bpp is required.')
    args = parser.parse_args()

    scales = parse_scales(args.scales)
    result_dir = Path(args.results)
    tmp_dir = Path(args.tmp_dir)
    cfg_path = Path(args.cfg).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(args.testdata, args.split_file)
    if not files:
        raise FileNotFoundError('No input .bin files found for baseline G-PCC measurement.')

    detail_rows = []
    progress = progress_bar(total=len(files) * len(scales), desc='Baseline G-PCC', unit='job')
    try:
        for bin_path in files:
            frame_id = bin_path.stem
            coords_raw = read_kitti_bin(bin_path)
            num_points = len(coords_raw)
            if num_points == 0:
                if progress is not None:
                    progress.update(len(scales))
                continue

            coords_mm = np.round(coords_raw.astype(np.float64) * 1000).astype(np.int32)
            coords_scaled = coords_mm - coords_mm.min(axis=0)

            for rate_id, scale in enumerate(scales):
                if progress is not None:
                    progress.set_postfix(frame=frame_id, scale=f'{scale:.6g}')
                stats = encode_decode_one_scale(
                    frame_id, coords_scaled, num_points, scale, tmp_dir, cfg_path,
                    encode_only=args.encode_only)
                detail_rows.append({
                    'filename': frame_id,
                    'rate_id': rate_id,
                    'posQuantscale': scale,
                    'scale': scale,
                    'num_points': num_points,
                    'bits': stats['bits'],
                    'bpp': round(stats['bpp'], 6),
                    'enc_time': round(stats['enc_time'], 6),
                    'dec_time': round(stats['dec_time'], 6),
                })
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    detail_csv = result_dir / 'gpcc_baseline_details.csv'
    with open(detail_csv, 'w', newline='') as f:
        fieldnames = ['filename', 'rate_id', 'posQuantscale', 'scale', 'num_points', 'bits', 'bpp', 'enc_time', 'dec_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    grouped = {}
    for row in detail_rows:
        grouped.setdefault(row['rate_id'], []).append(row)

    avg_rows = []
    for rate_id in sorted(grouped):
        rows = grouped[rate_id]
        total_bits = sum(int(r['bits']) for r in rows)
        total_points = sum(int(r['num_points']) for r in rows)
        avg_rows.append({
            'rate_id': rate_id,
            'posQuantscale': rows[0]['posQuantscale'],
            'scale': rows[0]['scale'],
            'num_frames': len(rows),
            'total_points': total_points,
            'total_bits': total_bits,
            'bpp': round(total_bits / total_points, 6) if total_points else 0.0,
            'enc_time': round(sum(float(r['enc_time']) for r in rows) / len(rows), 6),
            'dec_time': round(sum(float(r['dec_time']) for r in rows) / len(rows), 6),
        })

    avg_csv = result_dir / 'gpcc_baseline_average.csv'
    with open(avg_csv, 'w', newline='') as f:
        fieldnames = [
            'rate_id', 'posQuantscale', 'scale', 'num_frames',
            'total_points', 'total_bits', 'bpp', 'enc_time', 'dec_time'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(avg_rows)

    print(f'Detail CSV: {detail_csv}')
    print(f'Average CSV: {avg_csv}')


if __name__ == '__main__':
    main()
