#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torchac
from torchsparse import SparseTensor
from torchsparse.nn import functional as F

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RENO_ROOT = Path('/public/DATA/sm/RENO')
sys.path.append(str(ROOT_DIR))

from data_utils.geometry.inout import write_ply_o3d  # noqa: E402
from extension.pc_error_geo import pc_error  # noqa: E402


def add_reno_to_path(reno_root):
    reno_root = Path(reno_root).resolve()
    if str(reno_root) not in sys.path:
        sys.path.insert(0, str(reno_root))
    return reno_root


def parse_number(value):
    value = str(value).strip()
    if '/' in value:
        num, den = value.split('/', 1)
        return float(num) / float(den)
    return float(value)


def parse_rates(value):
    rates = []
    for rate_id, item in enumerate(str(value).replace(';', ',').split(',')):
        label = item.strip()
        if not label:
            continue
        scale = parse_number(label)
        if scale <= 0:
            raise ValueError(f'Invalid scale: {label}')
        rates.append({
            'rate_id': rate_id,
            'scale': scale,
            'posQ': 1.0 / scale,
            'label': label,
        })
    if not rates:
        raise ValueError('--scales must contain at least one value')
    return rates


def read_split(path):
    with open(path) as f:
        return [line.strip().zfill(6) for line in f if line.strip()]


def collect_files(testdata, split_file):
    testdata = Path(testdata)
    if testdata.is_file():
        return [testdata]
    if split_file:
        return [testdata / f'{frame_id}.bin' for frame_id in read_split(split_file)
                if (testdata / f'{frame_id}.bin').exists()]
    return sorted(testdata.rglob('*.bin'))


def read_kitti_bin(path):
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)


def iter_progress(items, **kwargs):
    if tqdm is None:
        return items
    return tqdm(items, **kwargs)


def configure_torchsparse():
    conv_config = F.conv_config.get_default_conv_config()
    conv_config.kmap_mode = 'hashmap'
    F.conv_config.set_global_conv_config(conv_config)


def sparse_to_device(x, device):
    if torch.device(device).type == 'cuda':
        return x.cuda()
    return x


def load_model(reno_root, ckpt, channels, kernel_size, device):
    add_reno_to_path(reno_root)
    from network import Network

    net = Network(channels=channels, kernel_size=kernel_size)
    state = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(state)
    net.to(device).eval()

    with torch.no_grad():
        random_coords = torch.randint(low=0, high=2048, size=(2048, 3), dtype=torch.int32, device=device)
        warm = SparseTensor(
            coords=torch.cat((random_coords[:, 0:1] * 0, random_coords), dim=-1),
            feats=torch.ones((2048, 1), device=device),
        )
        warm = sparse_to_device(warm, device)
        net(warm)
    return net


def points_to_sparse(points_xyz, posq, device):
    coords_mm = np.round(points_xyz.astype(np.float64) * 1000.0).astype(np.int64)
    offset = coords_mm.min(axis=0).astype(np.int64)
    coords_scaled = coords_mm - offset
    xyz = torch.from_numpy(np.round(coords_scaled.astype(np.float64) / float(posq)).astype(np.int32))
    coords = torch.cat((xyz[:, 0:1] * 0, xyz), dim=-1).int()
    feats = torch.ones((coords.shape[0], 1), dtype=torch.float)
    return sparse_to_device(SparseTensor(coords=coords, feats=feats), device), offset


def encode_tensor(net, x):
    import kit.op as op

    data_ls = []
    while True:
        x = net.fog(x)
        data_ls.append((x.coords.clone(), x.feats.clone()))
        if x.coords.shape[0] < 64:
            break
    data_ls = data_ls[::-1]

    byte_stream_ls = []
    for depth in range(len(data_ls) - 1):
        x_c, x_o = data_ls[depth]
        gt_x_up_c, gt_x_up_o = data_ls[depth + 1]
        gt_x_up_c, gt_x_up_o = op.sort_CF(gt_x_up_c, gt_x_up_o)

        x_f = net.prior_embedding(x_o.int()).view(-1, net.channels)
        x_prior = SparseTensor(coords=x_c, feats=x_f)
        x_prior = net.prior_resnet(x_prior)

        x_up_c, x_up_f = net.fcg(x_c, x_o, x_prior.feats)
        x_up_c, x_up_f = op.sort_CF(x_up_c, x_up_f)

        x_up_f = net.target_embedding(x_up_f, x_up_c)
        x_up = SparseTensor(coords=x_up_c, feats=x_up_f)
        x_up = net.target_resnet(x_up)

        gt_s0 = torch.remainder(gt_x_up_o, 16)
        gt_s1 = torch.div(gt_x_up_o, 16, rounding_mode='floor')

        prob_s0 = net.pred_head_s0(x_up.feats)
        prob_s1 = net.pred_head_s1(x_up.feats + net.pred_head_s1_emb(gt_s0[:, 0].long()))

        prob = torch.cat((prob_s0, prob_s1), dim=0)
        symbols = torch.cat((gt_s0, gt_s1), dim=0)
        cdf = torch.cat((prob[:, 0:1] * 0, prob.cumsum(dim=-1)), dim=-1)
        cdf = torch.clamp(cdf, min=0, max=1)
        cdf_norm = op._convert_to_int_and_normalize(cdf, True).cpu()
        symbols = symbols[:, 0].to(torch.int16).cpu()

        half = symbols.shape[0] // 2
        byte_stream_ls.append(torchac.encode_int16_normalized_cdf(cdf_norm[:half], symbols[:half]))
        byte_stream_ls.append(torchac.encode_int16_normalized_cdf(cdf_norm[half:], symbols[half:]))

    base_coords, base_feats = data_ls[0]
    byte_stream = op.pack_byte_stream_ls(byte_stream_ls)
    return base_coords[:, 1:].cpu().numpy(), base_feats.cpu().numpy(), byte_stream


def write_bitstream(path, posq, offset, base_coords, base_feats, byte_stream):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(np.array(posq, dtype=np.float32).tobytes())
        f.write(np.asarray(offset, dtype=np.int32).tobytes())
        f.write(np.array(base_coords.shape[0], dtype=np.int32).tobytes())
        f.write(np.asarray(base_coords, dtype=np.int32).tobytes())
        f.write(np.asarray(base_feats, dtype=np.uint8).tobytes())
        f.write(byte_stream)


def decode_bitstream(net, path, device):
    import kit.op as op

    with open(path, 'rb') as f:
        posq = np.frombuffer(f.read(4), dtype=np.float32)[0]
        offset = np.frombuffer(f.read(12), dtype=np.int32).astype(np.float64)
        base_len = np.frombuffer(f.read(4), dtype=np.int32)[0]
        base_coords = np.frombuffer(f.read(base_len * 4 * 3), dtype=np.int32)
        base_feats = np.frombuffer(f.read(base_len), dtype=np.uint8)
        byte_stream = f.read()

    base_coords = torch.tensor(base_coords.reshape(-1, 3), dtype=torch.int32, device=device)
    base_feats = torch.tensor(base_feats.reshape(-1, 1), dtype=torch.int32, device=device)
    batch_col = torch.zeros((base_coords.shape[0], 1), dtype=torch.int32, device=device)
    x = sparse_to_device(SparseTensor(coords=torch.cat((batch_col, base_coords), dim=-1), feats=base_feats), device)

    byte_stream_ls = op.unpack_byte_stream(byte_stream)
    for stream_idx in range(0, len(byte_stream_ls), 2):
        stream_s0 = byte_stream_ls[stream_idx]
        stream_s1 = byte_stream_ls[stream_idx + 1]

        x_o = x.feats.int()
        x.feats = net.prior_embedding(x_o).view(-1, net.channels)
        x = net.prior_resnet(x)

        x_up_c, x_up_f = net.fcg(x.coords, x_o, x_F=x.feats)
        x_up_c, x_up_f = op.sort_CF(x_up_c, x_up_f)

        x_up_f = net.target_embedding(x_up_f, x_up_c)
        x_up = SparseTensor(coords=x_up_c, feats=x_up_f)
        x_up = net.target_resnet(x_up)

        prob_s0 = net.pred_head_s0(x_up.feats)
        cdf_s0 = torch.cat((prob_s0[:, 0:1] * 0, prob_s0.cumsum(dim=-1)), dim=-1)
        cdf_s0 = torch.clamp(cdf_s0, min=0, max=1)
        cdf_s0 = op._convert_to_int_and_normalize(cdf_s0, True).cpu()
        occ_s0 = torchac.decode_int16_normalized_cdf(cdf_s0, stream_s0).to(device)

        prob_s1 = net.pred_head_s1(x_up.feats + net.pred_head_s1_emb(occ_s0.long()))
        cdf_s1 = torch.cat((prob_s1[:, 0:1] * 0, prob_s1.cumsum(dim=-1)), dim=-1)
        cdf_s1 = torch.clamp(cdf_s1, min=0, max=1)
        cdf_s1 = op._convert_to_int_and_normalize(cdf_s1, True).cpu()
        occ_s1 = torchac.decode_int16_normalized_cdf(cdf_s1, stream_s1).to(device)

        x = sparse_to_device(SparseTensor(coords=x_up_c, feats=(occ_s1 * 16 + occ_s0).unsqueeze(-1)), device)

    scan = net.fcg(x.coords, x.feats)
    offset_t = torch.tensor(offset, dtype=torch.float32, device=scan.device)
    scan = (scan[:, 1:] * float(posq) + offset_t) * 0.001
    return scan.float().cpu().numpy()


@contextmanager
def temp_ply_pair(tmp_dir, frame_id, tag, ref_coords, dec_coords):
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ref_ply = tmp_dir / f'{frame_id}_{tag}_ref.ply'
    dec_ply = tmp_dir / f'{frame_id}_{tag}_dec.ply'
    write_ply_o3d(str(ref_ply), ref_coords + 1, dtype='int32')
    write_ply_o3d(str(dec_ply), dec_coords + 1, dtype='int32')
    try:
        yield ref_ply, dec_ply
    finally:
        for path in (ref_ply, dec_ply):
            if path.exists():
                path.unlink()


def compute_psnr(ref_xyz, dec_xyz, frame_id, posq, tmp_dir, resolution):
    coords_mm = np.round(ref_xyz.astype(np.float64) * 1000.0).astype(np.int32)
    offset = coords_mm.min(axis=0)
    ref_coords = coords_mm - offset
    dec_coords = np.round(dec_xyz.astype(np.float64) * 1000.0).astype(np.int32) - offset
    with temp_ply_pair(tmp_dir, frame_id, f'posQ_{posq}', ref_coords, dec_coords) as (ref_ply, dec_ply):
        results = pc_error(str(ref_ply), str(dec_ply), resolution=resolution, normal=False, show=False)
    return float(results.get('mseF,PSNR (p2point)', -1.0)), ''


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Run RENO KITTI bpp/time/PSNR measurement like GPCC/baseline_rates.py.')
    parser.add_argument('--reno_root', default=str(DEFAULT_RENO_ROOT))
    parser.add_argument('--testdata', required=True, help='KITTI velodyne directory or one .bin file')
    parser.add_argument('--split_file', default=None)
    parser.add_argument(
        '--scales',
        default='1/64,1.5/128,1/128,1.5/256,1/256,1.5/512,1/512,1/2048',
        help='Comma-separated quantization scales. RENO uses posQ=1/scale internally.'
    )
    parser.add_argument('--posqs', default=None, help='Deprecated alias: comma-separated RENO posQ values.')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--results', default='experiment_results/reno_fov/reno')
    parser.add_argument('--tmp_dir', default='experiment_results/reno_fov/tmp')
    parser.add_argument('--bitstream_dir', default='experiment_results/reno_fov/bitstreams')
    parser.add_argument('--kitti_root', default='OpenPCDet/data/kitti_fov')
    parser.add_argument('--resolution', type=int, default=80000)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no_psnr', action='store_true')
    args = parser.parse_args()

    configure_torchsparse()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    net = load_model(args.reno_root, args.ckpt, args.channels, args.kernel_size, device)
    if args.posqs:
        rates = []
        for rate_id, item in enumerate(str(args.posqs).replace(';', ',').split(',')):
            label = item.strip()
            if label:
                posq = parse_number(label)
                rates.append({'rate_id': rate_id, 'scale': 1.0 / posq, 'posQ': posq, 'label': label})
    else:
        rates = parse_rates(args.scales)
    files = collect_files(args.testdata, args.split_file)
    if not files:
        raise FileNotFoundError('No KITTI .bin files found.')

    result_dir = Path(args.results)
    bitstream_dir = Path(args.bitstream_dir)
    tmp_dir = Path(args.tmp_dir)
    detail_rows = []

    jobs = [(path, rate) for path in files for rate in rates]
    progress = iter_progress(jobs, desc='RENO KITTI', unit='job')
    with torch.no_grad():
        for bin_path, rate in progress:
            posq = rate['posQ']
            rate_id = rate['rate_id']
            frame_id = Path(bin_path).stem
            points = read_kitti_bin(bin_path)
            ref_xyz = points[:, :3]
            num_points = ref_xyz.shape[0]
            if num_points == 0:
                continue

            if tqdm is not None:
                progress.set_postfix(frame=frame_id, rate=rate_id, scale=rate['label'])

            x, offset = points_to_sparse(ref_xyz, posq, device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            base_coords, base_feats, byte_stream = encode_tensor(net, x)
            bitstream = bitstream_dir / f'rate_{rate_id}' / f'{frame_id}.bin'
            write_bitstream(bitstream, posq, offset, base_coords, base_feats, byte_stream)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            enc_time = time.perf_counter() - t0
            enc_peak_memory_mib = (
                torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                if device.type == 'cuda' else 0.0)

            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            dec_xyz = decode_bitstream(net, bitstream, device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dec_time = time.perf_counter() - t0
            dec_peak_memory_mib = (
                torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                if device.type == 'cuda' else 0.0)

            d1_psnr, d2_psnr = '', ''
            if not args.no_psnr:
                d1_psnr, d2_psnr = compute_psnr(ref_xyz, dec_xyz, frame_id, rate_id, tmp_dir, args.resolution)

            bits = bitstream.stat().st_size * 8
            detail_rows.append({
                'filename': frame_id,
                'rate_id': rate_id,
                'scale': rate['scale'],
                'scale_label': rate['label'],
                'posQ': posq,
                'num_points': num_points,
                'decoded_points': dec_xyz.shape[0],
                'bits': bits,
                'bpp': round(bits / num_points, 6),
                'enc_time': round(enc_time, 6),
                'dec_time': round(dec_time, 6),
                'enc_peak_memory_mib': round(enc_peak_memory_mib, 6),
                'dec_peak_memory_mib': round(dec_peak_memory_mib, 6),
                'd1_psnr': d1_psnr if d1_psnr == '' else round(d1_psnr, 6),
                'd2_psnr': d2_psnr if d2_psnr == '' else round(d2_psnr, 6),
            })

    detail_csv = result_dir / 'reno_details.csv'
    write_csv(detail_csv, detail_rows, [
        'filename', 'rate_id', 'scale', 'scale_label', 'posQ', 'num_points', 'decoded_points',
        'bits', 'bpp', 'enc_time', 'dec_time', 'enc_peak_memory_mib',
        'dec_peak_memory_mib', 'd1_psnr', 'd2_psnr'
    ])

    grouped = {}
    for row in detail_rows:
        grouped.setdefault(int(row['rate_id']), []).append(row)

    avg_rows = []
    for rate_id in sorted(grouped):
        rows = grouped[rate_id]
        total_bits = sum(int(r['bits']) for r in rows)
        total_points = sum(int(r['num_points']) for r in rows)
        avg_rows.append({
            'rate_id': rate_id,
            'scale': rows[0]['scale'],
            'scale_label': rows[0]['scale_label'],
            'posQ': rows[0]['posQ'],
            'num_frames': len(rows),
            'total_points': total_points,
            'total_bits': total_bits,
            'bpp': round(total_bits / total_points, 6) if total_points else 0.0,
            'enc_time': round(sum(float(r['enc_time']) for r in rows) / len(rows), 6),
            'dec_time': round(sum(float(r['dec_time']) for r in rows) / len(rows), 6),
            'enc_peak_memory_mib': round(sum(
                float(r['enc_peak_memory_mib']) for r in rows) / len(rows), 6),
            'dec_peak_memory_mib': round(sum(
                float(r['dec_peak_memory_mib']) for r in rows) / len(rows), 6),
            'd1_psnr': '' if args.no_psnr else round(np.mean([float(r['d1_psnr']) for r in rows if r['d1_psnr'] != '']), 6),
            'd2_psnr': '' if args.no_psnr or any(r['d2_psnr'] == '' for r in rows) else round(np.mean([float(r['d2_psnr']) for r in rows]), 6),
        })

    avg_csv = result_dir / 'reno_average.csv'
    write_csv(avg_csv, avg_rows, [
        'rate_id', 'posQ', 'num_frames', 'total_points', 'total_bits',
        'scale', 'scale_label', 'bpp', 'enc_time', 'dec_time',
        'enc_peak_memory_mib', 'dec_peak_memory_mib', 'd1_psnr', 'd2_psnr'
    ])

    print(f'Detail CSV: {detail_csv}')
    print(f'Average CSV: {avg_csv}')


if __name__ == '__main__':
    main()
