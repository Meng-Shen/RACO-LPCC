#!/usr/bin/env python3
import argparse
import csv
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import MinkowskiEngine as ME
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils.geometry.inout import write_ply_o3d  # noqa: E402
from extention.pc_error_geo import pc_error  # noqa: E402


def configure_unicorn_args():
    sys.argv = ['unicorn_rates_direct.py', '--only_global_topk', '0']


def add_unicorn_to_path(unicorn_root):
    unicorn_root = Path(unicorn_root).resolve()
    sys.path.insert(0, str(unicorn_root))
    sys.path.insert(0, str(unicorn_root / 'lossy_geometry'))


def parse_rate_list(value):
    rates = []
    for rate_id, item in enumerate(str(value).split(',')):
        item = item.strip()
        if not item:
            continue
        scale_ae, scale_sr, posqscale = item.split(':')
        scale_ae = int(scale_ae)
        if scale_ae != 0:
            raise ValueError('Direct KITTI flow trains/evaluates lossless+SR+offset; use scale_AE=0.')
        rates.append({
            'rate_id': rate_id,
            'scale_AE': scale_ae,
            'scale_SR': int(scale_sr),
            'posQuantscale': float(posqscale),
            'rate_label': item,
        })
    return rates


def read_rate_config(path):
    rates = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            scale_ae = int(row['scale_AE'])
            if scale_ae != 0:
                raise ValueError('Direct KITTI flow trains/evaluates lossless+SR+offset; use scale_AE=0.')
            rates.append({
                'rate_id': int(row['rate_id']),
                'scale_AE': scale_ae,
                'scale_SR': int(row['scale_SR']),
                'posQuantscale': float(row['posQuantscale']),
                'rate_label': row.get('rate_label') or f"{scale_ae}:{row['scale_SR']}:{row['posQuantscale']}",
            })
    if not rates:
        raise ValueError(f'No rate points found in {path}')
    return rates


def read_split(path):
    with open(path) as f:
        return [line.strip().zfill(6) for line in f if line.strip()]


def collect_files(velodyne, split_file):
    velodyne = Path(velodyne)
    files = [velodyne / f'{frame_id}.bin' for frame_id in read_split(split_file)]
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
    return files


def load_sparse_from_bin(path, posq, device):
    points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)[:, :3]
    coords_mm = np.round(points.astype(np.float64) * 1000.0).astype(np.int64)
    offset = coords_mm.min(axis=0)
    coords = np.round((coords_mm - offset).astype(np.float64) / float(posq)).astype(np.int32)
    coords = np.unique(coords, axis=0).astype(np.int32)
    coords_t = torch.from_numpy(coords).int()
    feats_t = torch.ones((coords_t.shape[0], 1), dtype=torch.float32)
    coords_b, feats_b = ME.utils.sparse_collate([coords_t], [feats_t])
    x = ME.SparseTensor(features=feats_b, coordinates=coords_b, device=device)
    return x, points, offset.astype(np.float64)


def sparse_from_coords(coords, device):
    coords_t = torch.from_numpy(coords.astype(np.int32)).int()
    feats_t = torch.ones((coords_t.shape[0], 1), dtype=torch.float32)
    coords_b, feats_b = ME.utils.sparse_collate([coords_t], [feats_t])
    return ME.SparseTensor(features=feats_b, coordinates=coords_b, device=device)


def load_model(ctor, ckpt_path, device, **kwargs):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    model = ctor(**kwargs).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def pack_coords(coords):
    coords = coords.astype(np.int32)
    min_v = coords.min(axis=0).astype(np.int32)
    rel = (coords - min_v).astype(np.int16)
    return min_v.tobytes() + rel.tobytes()


def unpack_coords(bitstream):
    min_v = np.frombuffer(bitstream[:12], dtype=np.int32).reshape(1, 3)
    rel = np.frombuffer(bitstream[12:], dtype=np.int16).reshape(-1, 3).astype(np.int32)
    return rel + min_v


@torch.no_grad()
def lossless_encode(model, x):
    bitstreams = []
    while len(x) > 32:
        x_low = model.downsampler(x)
        bitstream = model.upsampler.encode(x_low, x)
        bitstreams.append(bitstream)
        coords = torch.div(x_low.C, 2, rounding_mode='floor')
        feats = torch.ones((len(coords), 1), dtype=torch.float32, device=x.device)
        x = ME.SparseTensor(features=feats, coordinates=coords, device=x.device)
    bitstreams.append(pack_coords(x.C[:, 1:].detach().cpu().numpy()))
    return bitstreams, (4 + 4 * len(bitstreams)) * 8 + sum(len(b) * 8 for b in bitstreams)


@torch.no_grad()
def lossless_decode(model, bitstreams, device):
    streams = list(bitstreams)[::-1]
    x = sparse_from_coords(unpack_coords(streams[0]), device)
    for bitstream in streams[1:]:
        x = ME.SparseTensor(features=x.F, coordinates=x.C * 2, tensor_stride=2, device=x.device)
        x = model.upsampler.decode(x, bitstream)
    return x


@torch.no_grad()
def run_one_rate(x_raw, model_lossless, model_sr, model_offset, scale_sr, posqscale, use_offset=True):
    from data_utils.geometry.quantize import quantize_sparse_tensor

    start = time.time()
    phase_start = start
    x = quantize_sparse_tensor(x_raw, factor=1 / posqscale, quant_mode='round')
    quant_time = time.time() - phase_start

    phase_start = time.time()
    pooling = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
    x_tp = x
    num_points_list = [len(x_tp)]
    for _ in range(scale_sr):
        x_tp = pooling(x_tp)
        num_points_list.append(len(x_tp))
    pooling_time = time.time() - phase_start

    phase_start = time.time()
    for _ in range(scale_sr):
        x_low = model_sr.downsampler(x)
        coords = torch.div(x_low.C, 2, rounding_mode='floor')
        feats = torch.ones((len(coords), 1), dtype=torch.float32, device=x.device)
        x = ME.SparseTensor(features=feats, coordinates=coords, device=x.device)
    sr_down_time = time.time() - phase_start

    phase_start = time.time()
    bitstreams, lossless_bits = lossless_encode(model_lossless, x)
    lossless_enc_time = time.time() - phase_start
    enc_time = time.time() - start

    start_dec = time.time()
    phase_start = start_dec
    x = lossless_decode(model_lossless, bitstreams, x_raw.device)
    lossless_dec_time = time.time() - phase_start

    phase_start = time.time()
    for num_points in num_points_list[::-1][1:]:
        x = ME.SparseTensor(features=x.F, coordinates=x.C * 2, tensor_stride=2, device=x.device)
        x = model_sr.upsampler.upsample(x, num_points)
    sr_up_time = time.time() - phase_start

    phase_start = time.time()
    if use_offset and model_offset is not None:
        dec_coords = model_offset.upscale(x, posQuantscale=posqscale)
    else:
        x_dec = quantize_sparse_tensor(x, factor=posqscale)
        dec_coords = x_dec.C[:, 1:].detach().cpu().numpy()
    offset_time = time.time() - phase_start
    dec_time = time.time() - start_dec
    side_bits = 32 * (1 + len(num_points_list))
    timings = {
        'quant_time': quant_time,
        'pooling_time': pooling_time,
        'sr_down_time': sr_down_time,
        'lossless_enc_time': lossless_enc_time,
        'lossless_dec_time': lossless_dec_time,
        'sr_up_time': sr_up_time,
        'offset_time': offset_time,
    }
    return dec_coords.astype(np.float64), int(lossless_bits + side_bits), enc_time, dec_time, timings


def write_bitstream_stub(path, bits):
    path.parent.mkdir(parents=True, exist_ok=True)
    nbytes = int((bits + 7) // 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', int(bits)))
        if nbytes > 8:
            f.write(bytes(nbytes - 8))


def points_to_bin(points_xyz, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    intensity = np.zeros((points_xyz.shape[0], 1), dtype=np.float32)
    points = np.concatenate([points_xyz.astype(np.float32), intensity], axis=1)
    points.astype(np.float32).tofile(out_path)


def compute_psnr(ref_xyz_m, dec_xyz_m, frame_id, rate_id, tmp_dir, resolution):
    ref_coords = np.round(ref_xyz_m.astype(np.float64) * 1000.0).astype(np.int32)
    dec_coords = np.round(dec_xyz_m.astype(np.float64) * 1000.0).astype(np.int32)
    ref_ply = tmp_dir / f'{frame_id}_r{rate_id}_ref.ply'
    dec_ply = tmp_dir / f'{frame_id}_r{rate_id}_dec.ply'
    write_ply_o3d(str(ref_ply), ref_coords, dtype='int32', normal=True, knn=16)
    write_ply_o3d(str(dec_ply), dec_coords, dtype='int32')
    try:
        return pc_error(str(ref_ply), str(dec_ply), resolution=resolution, normal=True, show=False)
    finally:
        for path in (ref_ply, dec_ply):
            if path.exists():
                path.unlink()


def progress_bar(*args, **kwargs):
    if tqdm is None:
        return None
    return tqdm(*args, **kwargs)


class GpuContendedError(RuntimeError):
    pass


class GpuGuard:
    def __init__(self, gpu_id, enabled=True, interval=1.0):
        self.gpu_id = gpu_id
        self.enabled = enabled and bool(gpu_id)
        self.interval = max(float(interval), 0.2)
        self.event = threading.Event()
        self.message = ''
        self.thread = None

    def start(self):
        if not self.enabled:
            return
        self.check()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.event.is_set():
            time.sleep(self.interval)
            try:
                self.check()
            except GpuContendedError as exc:
                self.message = str(exc)
                self.event.set()

    def check(self):
        if not self.enabled:
            return
        own_pids = own_process_ids()
        others = [(pid, mem) for pid, mem in query_gpu_compute_pids(self.gpu_id) if pid not in own_pids]
        if others:
            detail = ', '.join(f'pid={pid} mem={mem}MiB' if mem else f'pid={pid}' for pid, mem in others)
            raise GpuContendedError(
                f'GPU {self.gpu_id} has other compute process(es): {detail}. '
                'Stopping before recording the current case; completed rows remain resumable.'
            )

    def raise_if_triggered(self):
        if self.event.is_set():
            raise GpuContendedError(self.message or f'GPU {self.gpu_id} contention detected.')


def read_completed_cases(detail_csv):
    if not detail_csv.exists():
        return set(), []
    completed = set()
    rows = []
    with open(detail_csv, newline='') as f:
        for row in csv.DictReader(f):
            try:
                key = (row['filename'], int(row['rate_id']))
            except (KeyError, TypeError, ValueError):
                continue
            completed.add(key)
            rows.append(row)
    return completed, rows


def ensure_detail_schema(detail_csv, fieldnames):
    if not detail_csv.exists():
        return
    with open(detail_csv, newline='') as f:
        reader = csv.DictReader(f)
        old_fieldnames = reader.fieldnames or []
        rows = list(reader)
    if old_fieldnames == fieldnames:
        return
    detail_csv.replace(detail_csv.with_suffix(detail_csv.suffix + '.bak'))
    with open(detail_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, '') for name in fieldnames})


def own_process_ids():
    ids = {os.getpid(), os.getppid()}
    for child_file in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = child_file.read_text(errors='replace')
            rparen = text.rfind(')')
            if rparen < 0:
                continue
            fields = text[rparen + 2:].split()
            ppid = int(fields[1])
            if ppid in ids:
                ids.add(int(child_file.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return ids


def query_gpu_compute_pids(gpu_id):
    try:
        output = subprocess.check_output(
            [
                'nvidia-smi',
                f'--id={gpu_id}',
                '--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits',
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    pids = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if not parts or not parts[0]:
            continue
        try:
            pids.append((int(parts[0]), parts[1] if len(parts) > 1 else ''))
        except ValueError:
            continue
    return pids


def write_average_csv(results_dir, detail_rows):
    grouped = {}
    for row in detail_rows:
        grouped.setdefault(int(row['rate_id']), []).append(row)
    avg_rows = []
    for rate_id in sorted(grouped):
        rows = grouped[rate_id]
        total_bits = sum(int(r['bits']) for r in rows)
        total_points = sum(int(r['num_points']) for r in rows)
        d1 = [float(r['d1_psnr']) for r in rows if r['d1_psnr'] != '']
        d2 = [float(r['d2_psnr']) for r in rows if r['d2_psnr'] != '']
        avg_rows.append({
            'rate_id': rate_id,
            'rate_label': rows[0]['rate_label'],
            'scale_AE': rows[0]['scale_AE'],
            'scale_SR': rows[0]['scale_SR'],
            'posQuantscale': rows[0]['posQuantscale'],
            'posQ': rows[0]['posQ'],
            'num_frames': len(rows),
            'total_points': total_points,
            'total_bits': total_bits,
            'bpp': round(total_bits / total_points, 6) if total_points else 0.0,
            'enc_time': round(sum(float(r['enc_time']) for r in rows) / len(rows), 6),
            'dec_time': round(sum(float(r['dec_time']) for r in rows) / len(rows), 6),
            'd1_psnr': round(sum(d1) / len(d1), 6) if d1 else '',
            'd2_psnr': round(sum(d2) / len(d2), 6) if d2 else '',
        })

    avg_csv = results_dir / 'unicorn_average.csv'
    if not avg_rows:
        return avg_csv
    with open(avg_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(avg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(avg_rows)
    return avg_csv


def main():
    parser = argparse.ArgumentParser(description='Run direct Unicorn lossless+SR evaluation on KITTI .bin files, optionally with offset refinement.')
    parser.add_argument('--unicorn_root', default='/public/DATA/sm/Unicorn')
    parser.add_argument('--testdata', required=True)
    parser.add_argument('--split_file', required=True)
    parser.add_argument('--train_posq', type=float, default=64.0)
    parser.add_argument('--results', required=True)
    parser.add_argument('--tmp_dir', required=True)
    parser.add_argument('--bitstream_dir', required=True)
    parser.add_argument('--decoded_dir', required=True)
    parser.add_argument('--rates', default='0:0:1,0:1:1,0:2:1,0:3:1,0:1:2,0:2:2,0:3:2,0:2:4')
    parser.add_argument('--rate_config_csv', default='')
    parser.add_argument('--ckptdir_low', required=True)
    parser.add_argument('--ckptdir_sr_low', required=True)
    parser.add_argument('--ckptdir_offset', default='', help='Offset checkpoint (required unless --disable_offset is set).')
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--offset_channels', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--block_layers', type=int, default=3)
    parser.add_argument('--block_type', default='conv')
    parser.add_argument('--resolution', type=int, default=80000)
    parser.add_argument('--no_psnr', action='store_true')
    parser.add_argument('--disable_offset', action='store_true', help='Skip the Unicorn offset module and only rescale decoded lattice coordinates.')
    parser.add_argument('--resume', action='store_true', help='Append to existing detail CSV and skip completed frame/rate rows.')
    parser.add_argument('--gpu_guard_id', default='', help='Physical nvidia-smi GPU id to monitor for other compute processes.')
    parser.add_argument('--gpu_guard_interval', type=float, default=1.0)
    parser.add_argument('--disable_gpu_guard', action='store_true')
    args = parser.parse_args()

    configure_unicorn_args()
    add_unicorn_to_path(args.unicorn_root)
    from lossless_geometry.model import PCCModel as PCCModelLossless
    from lossy_geometry.model import PCCModel as PCCModelLossy

    if not args.disable_offset and not args.ckptdir_offset:
        parser.error('--ckptdir_offset is required unless --disable_offset is set')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_lossless = load_model(
        PCCModelLossless, args.ckptdir_low, device,
        channels=args.channels, kernel_size=args.kernel_size,
        block_layers=args.block_layers, stage=8, scale=1, block_type=args.block_type,
    )
    model_sr = load_model(
        PCCModelLossy, args.ckptdir_sr_low, device,
        channels=args.channels, kernel_size=args.kernel_size,
        block_layers=args.block_layers, stage=1, scale=1,
        enc_type='pooling', block_type=args.block_type,
    )
    model_offset = None
    if not args.disable_offset:
        from lossy_geometry.model_offset import OffsetModel

        model_offset = load_model(
            OffsetModel, args.ckptdir_offset, device,
            channels=args.offset_channels, kernel_size=args.kernel_size,
            block_layers=args.block_layers, posQuantscaleList=[1, 2, 4, 8, 16, 32, 64],
        )
    gpu_guard = GpuGuard(
        args.gpu_guard_id,
        enabled=not args.disable_gpu_guard,
        interval=args.gpu_guard_interval,
    )
    gpu_guard.start()

    files = collect_files(args.testdata, args.split_file)
    rates = read_rate_config(args.rate_config_csv) if args.rate_config_csv else parse_rate_list(args.rates)
    results_dir = Path(args.results)
    tmp_dir = Path(args.tmp_dir)
    bitstream_dir = Path(args.bitstream_dir)
    decoded_dir = Path(args.decoded_dir)
    for path in (results_dir, tmp_dir, bitstream_dir, decoded_dir):
        path.mkdir(parents=True, exist_ok=True)

    detail_csv = results_dir / 'unicorn_details.csv'
    detail_fieldnames = [
        'filename',
        'rate_id',
        'rate_label',
        'scale_AE',
        'scale_SR',
        'posQuantscale',
        'posQ',
        'use_sr',
        'use_offset',
        'num_points',
        'decoded_points',
        'bits',
        'bpp',
        'enc_time',
        'dec_time',
        'quant_time',
        'pooling_time',
        'sr_down_time',
        'lossless_enc_time',
        'lossless_dec_time',
        'sr_up_time',
        'offset_time',
        'd1_psnr',
        'd2_psnr',
    ]
    if args.resume:
        ensure_detail_schema(detail_csv, detail_fieldnames)
    completed_cases, detail_rows = read_completed_cases(detail_csv) if args.resume else (set(), [])
    total_cases = len(files) * len(rates)
    pending_cases = total_cases - len(completed_cases)
    progress = progress_bar(total=pending_cases, desc='Unicorn rates', unit='case')
    mode = 'a' if args.resume and detail_csv.exists() else 'w'
    try:
        with open(detail_csv, mode, newline='') as detail_file:
            detail_writer = csv.DictWriter(detail_file, fieldnames=detail_fieldnames)
            if mode == 'w':
                detail_writer.writeheader()
            detail_file.flush()
            for bin_path in files:
                frame_id = bin_path.stem
                x_raw, ref_xyz, offset = load_sparse_from_bin(bin_path, args.train_posq, device)
                input_num_points = int(ref_xyz.shape[0])
                for rate in rates:
                    if (frame_id, int(rate['rate_id'])) in completed_cases:
                        continue
                    gpu_guard.check()
                    gpu_guard.raise_if_triggered()
                    if progress is not None:
                        progress.set_postfix(frame=frame_id, rate=rate['rate_label'])
                    dec_coords, bits, enc_time, dec_time, timings = run_one_rate(
                        x_raw, model_lossless, model_sr, model_offset,
                        rate['scale_SR'], rate['posQuantscale'],
                        use_offset=not args.disable_offset)
                    gpu_guard.raise_if_triggered()
                    dec_xyz = (dec_coords * args.train_posq + offset) / 1000.0
                    points_to_bin(dec_xyz, decoded_dir / f"rate_{rate['rate_id']}" / f'{frame_id}.bin')
                    write_bitstream_stub(bitstream_dir / f"rate_{rate['rate_id']}" / f'{frame_id}.bin', bits)
                    psnr = {} if args.no_psnr else compute_psnr(ref_xyz, dec_xyz, frame_id, rate['rate_id'], tmp_dir, args.resolution)
                    detail_row = {
                        'filename': frame_id,
                        'rate_id': rate['rate_id'],
                        'rate_label': rate['rate_label'],
                        'scale_AE': rate['scale_AE'],
                        'scale_SR': rate['scale_SR'],
                        'posQuantscale': rate['posQuantscale'],
                        'posQ': args.train_posq,
                        'use_sr': int(rate['scale_SR'] > 0),
                        'use_offset': int(not args.disable_offset),
                        # Match G-PCC's bpp denominator: count the original
                        # input points, not the deduplicated sparse voxels.
                        'num_points': input_num_points,
                        'decoded_points': int(dec_xyz.shape[0]),
                        'bits': bits,
                        'bpp': round(bits / float(input_num_points), 6),
                        'enc_time': round(enc_time, 6),
                        'dec_time': round(dec_time, 6),
                        'quant_time': round(timings['quant_time'], 6),
                        'pooling_time': round(timings['pooling_time'], 6),
                        'sr_down_time': round(timings['sr_down_time'], 6),
                        'lossless_enc_time': round(timings['lossless_enc_time'], 6),
                        'lossless_dec_time': round(timings['lossless_dec_time'], 6),
                        'sr_up_time': round(timings['sr_up_time'], 6),
                        'offset_time': round(timings['offset_time'], 6),
                        'd1_psnr': psnr.get('mseF,PSNR (p2point)', ''),
                        'd2_psnr': psnr.get('mseF,PSNR (p2plane)', ''),
                    }
                    print(
                        '[TIME] '
                        f"frame={frame_id} rate={rate['rate_label']} "
                        f"enc={enc_time:.6f}s dec={dec_time:.6f}s "
                        f"quant={timings['quant_time']:.6f}s "
                        f"pool={timings['pooling_time']:.6f}s "
                        f"sr_down={timings['sr_down_time']:.6f}s "
                        f"lossless_enc={timings['lossless_enc_time']:.6f}s "
                        f"lossless_dec={timings['lossless_dec_time']:.6f}s "
                        f"sr_up={timings['sr_up_time']:.6f}s "
                        f"offset={timings['offset_time']:.6f}s",
                        flush=True,
                    )
                    detail_rows.append(detail_row)
                    detail_writer.writerow(detail_row)
                    detail_file.flush()
                    torch.cuda.empty_cache()
                    if progress is not None:
                        progress.update(1)
    except GpuContendedError as exc:
        print(f'[GPU-GUARD] {exc}', file=sys.stderr)
        avg_csv = write_average_csv(results_dir, detail_rows)
        print(f'Unicorn detail CSV: {detail_csv}')
        print(f'Partial Unicorn average CSV: {avg_csv}')
        raise SystemExit(75)
    finally:
        if progress is not None:
            progress.close()

    avg_csv = write_average_csv(results_dir, detail_rows)
    print(f'Unicorn detail CSV: {detail_csv}')
    print(f'Unicorn average CSV: {avg_csv}')
    print(f'Decoded bins: {decoded_dir}')


if __name__ == '__main__':
    main()
