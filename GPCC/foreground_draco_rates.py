#!/usr/bin/env python3
import argparse
import csv
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    from fg_box_occupancy_codec import build_boxes
except ImportError:
    build_boxes = None


FG_CLASSES = (1,)
DEFAULT_SCALES = "1/64"


def parse_scale(value):
    value = str(value).strip()
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den)
    return float(value)


def parse_scales(text):
    scales = [parse_scale(item) for item in str(text).split(",") if item.strip()]
    if not scales:
        raise ValueError("--scales must contain at least one value")
    return scales


def read_kitti_bin(path):
    points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def collect_files(testdata, split_file=None):
    testdata = Path(testdata)
    if testdata.is_file():
        return [testdata]
    if split_file:
        with open(split_file) as f:
            frame_ids = [line.strip().zfill(6) for line in f if line.strip()]
        return [testdata / f"{frame_id}.bin" for frame_id in frame_ids if (testdata / f"{frame_id}.bin").exists()]
    return sorted(testdata.rglob("*.bin"))


def quantize_foreground(coords_raw, labels, scale, local_origin=True):
    fg_mask = np.isin(labels[:len(coords_raw)], FG_CLASSES)
    if not fg_mask.any():
        return np.empty((0, 3), dtype=np.int32)

    coords_mm = np.round(coords_raw.astype(np.float64) * 1000).astype(np.int32)
    coords_scaled = coords_mm - coords_mm.min(axis=0)
    qcoords = np.round(coords_scaled[fg_mask].astype(np.float64) * float(scale)).astype(np.int32)
    qcoords = np.unique(qcoords, axis=0)
    if local_origin and len(qcoords):
        qcoords = qcoords - qcoords.min(axis=0)
    return qcoords.astype(np.int32, copy=False)


def write_ascii_ply(path, coords):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(coords)}\n")
        f.write("property int x\n")
        f.write("property int y\n")
        f.write("property int z\n")
        f.write("end_header\n")
        for x, y, z in coords.astype(np.int64, copy=False):
            f.write(f"{int(x)} {int(y)} {int(z)}\n")


def read_ascii_ply_xyz(path):
    with open(path, "rb") as f:
        line = f.readline().decode("ascii", errors="replace").strip()
        if line != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        num_vertices = None
        fmt = None
        properties = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Truncated PLY header: {path}")
            line = line.decode("ascii", errors="replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("property") and num_vertices is not None:
                parts = line.split()
                if len(parts) >= 3:
                    properties.append((parts[1], parts[2]))
            if line == "end_header":
                break
        if num_vertices is None:
            raise ValueError(f"Missing vertex count in PLY: {path}")
        if fmt == "ascii":
            coords = []
            for _ in range(num_vertices):
                parts = f.readline().decode("ascii", errors="replace").split()
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            return np.round(np.asarray(coords, dtype=np.float64)).astype(np.int32)
        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format from Draco decoder: {fmt}")

        dtype_map = {
            "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
            "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
            "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
            "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        }
        dtype = []
        for prop_type, prop_name in properties:
            if prop_type not in dtype_map:
                raise ValueError(f"Unsupported PLY property type: {prop_type}")
            dtype.append((prop_name, dtype_map[prop_type]))
        data = np.frombuffer(f.read(), dtype=np.dtype(dtype), count=num_vertices)
        coords = np.stack([data["x"], data["y"], data["z"]], axis=1)
    return np.round(np.asarray(coords, dtype=np.float64)).astype(np.int32)


def sort_rows(coords):
    if len(coords) == 0:
        return coords.reshape(0, 3).astype(np.int32)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    return coords[order]


def draco_encode(encoder, in_ply, out_drc, compression_level=10, position_quantization_bits=0):
    cmd = [
        encoder,
        "-point_cloud",
        "-i", str(in_ply),
        "-o", str(out_drc),
        "-cl", str(compression_level),
    ]
    if position_quantization_bits is not None:
        cmd.extend(["-qp", str(position_quantization_bits)])
    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    enc_time = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"Draco encoder failed for {in_ply}\ncommand: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return enc_time


def draco_decode(decoder, in_drc, out_ply):
    cmd = [decoder, "-i", str(in_drc), "-o", str(out_ply)]
    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dec_time = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"Draco decoder failed for {in_drc}\ncommand: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return dec_time


def main():
    parser = argparse.ArgumentParser(description="Measure Draco foreground point-cloud bits.")
    parser.add_argument("--testdata", required=True, help="KITTI velodyne directory or one .bin file")
    parser.add_argument("--mask_dir", required=True, help="Directory containing <frame_id>.npy masks")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--results", default="point_pairs/foreground_draco")
    parser.add_argument("--tmp_dir", default="point_pairs/foreground_draco/tmp")
    parser.add_argument("--draco_encoder", default="draco_encoder")
    parser.add_argument("--draco_decoder", default="draco_decoder")
    parser.add_argument("--compression_level", type=int, default=10)
    parser.add_argument("--position_quantization_bits", type=int, default=0,
                        help="Draco -qp value. 0 is intended to avoid additional Draco quantization.")
    parser.add_argument("--no_local_origin", action="store_true",
                        help="Keep full-frame quantized coordinates instead of translating foreground to local origin.")
    parser.add_argument("--verify", action="store_true", help="Decode and compare quantized coordinates.")
    parser.add_argument("--box_mode", action="store_true",
                        help="Build foreground boxes and compress each box-local point set with Draco.")
    parser.add_argument("--micro_radius", type=int, default=8)
    parser.add_argument("--merge_gap", type=int, default=16)
    parser.add_argument("--attach_gap", type=int, default=16)
    parser.add_argument("--min_box_points", type=int, default=32)
    parser.add_argument("--small_cluster_threshold", type=int, default=4)
    parser.add_argument("--max_side", type=int, default=80)
    args = parser.parse_args()

    encoder = shutil.which(args.draco_encoder) or (args.draco_encoder if Path(args.draco_encoder).exists() else None)
    if encoder is None:
        raise FileNotFoundError(
            f"Missing Draco encoder: {args.draco_encoder}. Install Draco and pass --draco_encoder /path/to/draco_encoder."
        )
    decoder = None
    if args.verify:
        decoder = shutil.which(args.draco_decoder) or (args.draco_decoder if Path(args.draco_decoder).exists() else None)
        if decoder is None:
            raise FileNotFoundError(
                f"Missing Draco decoder: {args.draco_decoder}. Install Draco and pass --draco_decoder /path/to/draco_decoder."
            )

    scales = parse_scales(args.scales)
    files = collect_files(args.testdata, args.split_file)
    mask_dir = Path(args.mask_dir)
    results_dir = Path(args.results)
    tmp_dir = Path(args.tmp_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for bin_path in files:
        frame_id = bin_path.stem
        mask_path = mask_dir / f"{frame_id}.npy"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        coords_raw = read_kitti_bin(bin_path)
        labels = np.load(mask_path)[:len(coords_raw)]
        for rate_id, scale in enumerate(scales):
            qcoords = quantize_foreground(
                coords_raw, labels, scale, local_origin=not args.no_local_origin)
            enc_time = 0.0
            dec_time = ""
            verify = ""
            boxes = []
            residual_points = 0
            if args.box_mode:
                if build_boxes is None:
                    raise ImportError("box_mode requires fg_box_occupancy_codec.py on PYTHONPATH")
                boxes, residual = build_boxes(
                    qcoords,
                    micro_radius=args.micro_radius,
                    merge_gap=args.merge_gap,
                    attach_gap=args.attach_gap,
                    min_box_points=args.min_box_points,
                    small_cluster_threshold=args.small_cluster_threshold,
                    max_side=args.max_side,
                    max_boxes=0,
                )
                residual_points = len(residual)
                bits = 0
                dec_time_sum = 0.0
                for box_id, box in enumerate(boxes):
                    local = box.points - box.lo
                    ply_path = tmp_dir / f"{frame_id}_box{box_id}_s{scale:.12g}.ply"
                    drc_path = tmp_dir / f"{frame_id}_box{box_id}_s{scale:.12g}.drc"
                    dec_path = tmp_dir / f"{frame_id}_box{box_id}_s{scale:.12g}_dec.ply"
                    write_ascii_ply(ply_path, local)
                    enc_time += draco_encode(
                        encoder, ply_path, drc_path,
                        compression_level=args.compression_level,
                        position_quantization_bits=args.position_quantization_bits,
                    )
                    bits += drc_path.stat().st_size * 8 if drc_path.exists() else 0
                    if args.verify:
                        dec_time_sum += draco_decode(decoder, drc_path, dec_path)
                        dec = read_ascii_ply_xyz(dec_path)
                        if not np.array_equal(sort_rows(dec), sort_rows(local)):
                            raise RuntimeError(
                                f"Draco box verification failed for frame={frame_id}, scale={scale}, box={box_id}")
                    for path in (ply_path, drc_path, dec_path):
                        if path.exists():
                            path.unlink()
                if args.verify:
                    dec_time = dec_time_sum
                    verify = True
            else:
                ply_path = tmp_dir / f"{frame_id}_fg_s{scale:.12g}.ply"
                drc_path = tmp_dir / f"{frame_id}_fg_s{scale:.12g}.drc"
                dec_path = tmp_dir / f"{frame_id}_fg_s{scale:.12g}_dec.ply"

                write_ascii_ply(ply_path, qcoords)
                enc_time = draco_encode(
                    encoder, ply_path, drc_path,
                    compression_level=args.compression_level,
                    position_quantization_bits=args.position_quantization_bits,
                )
                if args.verify:
                    dec_time = draco_decode(decoder, drc_path, dec_path)
                    dec = read_ascii_ply_xyz(dec_path)
                    verify = bool(np.array_equal(sort_rows(dec), sort_rows(qcoords)))
                    if not verify:
                        raise RuntimeError(f"Draco verification failed for frame={frame_id}, scale={scale}")

                bits = drc_path.stat().st_size * 8 if drc_path.exists() else 0
                for path in (ply_path, drc_path, dec_path):
                    if path.exists():
                        path.unlink()
            rows.append({
                "filename": frame_id,
                "rate_id": rate_id,
                "scale": scale,
                "fg_qpoints": len(qcoords),
                "mode": "box" if args.box_mode else "whole_fg",
                "boxes": len(boxes),
                "box_points": int(sum(len(box.points) for box in boxes)),
                "residual_points": int(residual_points),
                "bits": bits,
                "bpp_fg_qpoint": round(bits / len(qcoords), 6) if len(qcoords) else 0.0,
                "bpp_original_point": round(bits / len(coords_raw), 6) if len(coords_raw) else 0.0,
                "enc_time": round(enc_time, 6),
                "dec_time": round(dec_time, 6) if dec_time != "" else "",
                "verify": verify,
            })

    detail_csv = results_dir / "foreground_draco_details.csv"
    with open(detail_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["rate_id"], []).append(row)
    avg_rows = []
    for rate_id in sorted(grouped):
        group = grouped[rate_id]
        total_bits = sum(int(row["bits"]) for row in group)
        total_fg = sum(int(row["fg_qpoints"]) for row in group)
        avg_rows.append({
            "rate_id": rate_id,
            "scale": group[0]["scale"],
            "num_frames": len(group),
            "total_fg_qpoints": total_fg,
            "total_bits": total_bits,
            "bpp_fg_qpoint": round(total_bits / total_fg, 6) if total_fg else 0.0,
            "enc_time": round(sum(float(row["enc_time"]) for row in group) / len(group), 6),
        })
    avg_csv = results_dir / "foreground_draco_average.csv"
    with open(avg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(avg_rows[0].keys()) if avg_rows else [])
        writer.writeheader()
        writer.writerows(avg_rows)

    print(f"Detail CSV: {detail_csv}")
    print(f"Average CSV: {avg_csv}")


if __name__ == "__main__":
    main()
