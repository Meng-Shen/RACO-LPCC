#!/usr/bin/env python3
"""Lossless-after-quantization foreground cluster geometry codec.

The codec stores foreground geometry as quantized integer coordinates. Points are
clustered in quantized space, each cluster stores one absolute origin and local
coordinates are delta-coded in Morton order, then compressed with zlib.
"""

import argparse
import csv
import struct
import time
import zlib
from collections import deque
from pathlib import Path

import numpy as np


MAGIC = b"FGC1"
VERSION = 1
FG_CLASSES = (1,)


def parse_scale(value):
    value = str(value).strip()
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den)
    return float(value)


def read_kitti_bin(path):
    points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def quantize_coords(coords_raw, scale):
    coords_mm = np.round(coords_raw.astype(np.float64) * 1000).astype(np.int32)
    coords_scaled = coords_mm - coords_mm.min(axis=0)
    qcoords = np.round(coords_scaled.astype(np.float64) * float(scale)).astype(np.int32)
    return np.unique(qcoords, axis=0)


def zigzag_encode(value):
    value = int(value)
    return (value << 1) ^ (value >> 63)


def zigzag_decode(value):
    value = int(value)
    return (value >> 1) ^ -(value & 1)


def put_uvarint(out, value):
    value = int(value)
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def get_uvarint(data, offset):
    shift = 0
    value = 0
    while True:
        if offset >= len(data):
            raise ValueError("Truncated varint payload")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift > 70:
            raise ValueError("Invalid varint payload")


def morton_key(coords):
    coords = coords.astype(np.uint64, copy=False)
    max_bits = int(coords.max()).bit_length() if len(coords) else 0
    keys = np.zeros(len(coords), dtype=object)
    for bit in range(max_bits):
        keys += ((coords[:, 0] >> bit) & 1).astype(object) << (3 * bit)
        keys += ((coords[:, 1] >> bit) & 1).astype(object) << (3 * bit + 1)
        keys += ((coords[:, 2] >> bit) & 1).astype(object) << (3 * bit + 2)
    return keys


def sort_morton(coords):
    if len(coords) <= 1:
        return coords.astype(np.int32, copy=True)
    keys = morton_key(coords)
    order = sorted(range(len(coords)), key=lambda i: (keys[i], int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])))
    return coords[np.asarray(order, dtype=np.int64)].astype(np.int32, copy=False)


def cluster_quantized(qcoords, connect_radius=2, min_cluster_points=1):
    """Return connected components in quantized space.

    Two points are connected when their Chebyshev distance is <= connect_radius.
    With connect_radius=1 this is standard 26-neighbor voxel connectivity. KITTI
    foreground is often sparse after quantization, so 2-4 is usually more useful.
    """
    if len(qcoords) == 0:
        return []
    if connect_radius < 1:
        raise ValueError("connect_radius must be >= 1")

    qcoords = np.unique(qcoords.astype(np.int32, copy=False), axis=0)
    if len(qcoords) <= 2048:
        diff = qcoords[:, None, :].astype(np.int32) - qcoords[None, :, :].astype(np.int32)
        adjacency = np.max(np.abs(diff), axis=2) <= int(connect_radius)
        visited = np.zeros(len(qcoords), dtype=bool)
        clusters = []
        for start in range(len(qcoords)):
            if visited[start]:
                continue
            visited[start] = True
            queue = deque([start])
            members = []
            while queue:
                idx = queue.popleft()
                members.append(idx)
                neighbors = np.flatnonzero(adjacency[idx] & ~visited)
                if len(neighbors):
                    visited[neighbors] = True
                    queue.extend(int(n) for n in neighbors)
            clusters.append(qcoords[np.asarray(members, dtype=np.int64)])

        large = [c for c in clusters if len(c) >= min_cluster_points]
        small = [c for c in clusters if len(c) < min_cluster_points]
        if small:
            large.extend(small)
        large.sort(key=lambda c: (-len(c), int(c[:, 0].min()), int(c[:, 1].min()), int(c[:, 2].min())))
        return large

    cell_size = int(connect_radius) + 1
    buckets = {}
    cell_coords = np.floor_divide(qcoords, cell_size).astype(np.int32)
    for i, cell in enumerate(cell_coords):
        buckets.setdefault(tuple(map(int, cell)), []).append(i)

    visited = np.zeros(len(qcoords), dtype=bool)
    neighbor_cells = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]

    clusters = []
    for start in range(len(qcoords)):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        members = []
        while queue:
            idx = queue.popleft()
            members.append(idx)
            base_cell = cell_coords[idx]
            point = qcoords[idx]
            for dx, dy, dz in neighbor_cells:
                cell_key = (
                    int(base_cell[0] + dx),
                    int(base_cell[1] + dy),
                    int(base_cell[2] + dz),
                )
                for nxt in buckets.get(cell_key, ()):
                    if visited[nxt]:
                        continue
                    if int(np.max(np.abs(qcoords[nxt] - point))) <= connect_radius:
                        visited[nxt] = True
                        queue.append(nxt)
        clusters.append(qcoords[np.asarray(members, dtype=np.int64)])

    large = [c for c in clusters if len(c) >= min_cluster_points]
    small = [c for c in clusters if len(c) < min_cluster_points]
    if small:
        large.extend(small)
    large.sort(key=lambda c: (-len(c), int(c[:, 0].min()), int(c[:, 1].min()), int(c[:, 2].min())))
    return large


def encode_local_coords(local_coords, compression_level=9):
    local_coords = sort_morton(local_coords)
    raw = bytearray()
    prev = np.zeros(3, dtype=np.int64)
    for point in local_coords.astype(np.int64, copy=False):
        delta = point - prev
        for value in delta:
            put_uvarint(raw, zigzag_encode(value))
        prev = point
    return zlib.compress(bytes(raw), level=compression_level)


def decode_local_coords(payload, num_points):
    raw = zlib.decompress(payload)
    coords = np.empty((num_points, 3), dtype=np.int32)
    offset = 0
    prev = np.zeros(3, dtype=np.int64)
    for i in range(num_points):
        delta = np.empty(3, dtype=np.int64)
        for axis in range(3):
            value, offset = get_uvarint(raw, offset)
            delta[axis] = zigzag_decode(value)
        point = prev + delta
        coords[i] = point.astype(np.int32)
        prev = point
    if offset != len(raw):
        raise ValueError("Trailing bytes in local coordinate payload")
    return coords


def encode_qcoords(qcoords, out_path, connect_radius=2, min_cluster_points=1, compression_level=9):
    qcoords = np.unique(qcoords.astype(np.int32, copy=False), axis=0)
    clusters = cluster_quantized(qcoords, connect_radius=connect_radius, min_cluster_points=min_cluster_points)

    cluster_records = []
    for cluster in clusters:
        origin = cluster.min(axis=0).astype(np.int32)
        local = cluster - origin
        payload = encode_local_coords(local, compression_level=compression_level)
        cluster_records.append((origin, len(cluster), payload))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<BIII", VERSION, int(len(qcoords)), int(len(cluster_records)), int(connect_radius)))
        for origin, num_points, payload in cluster_records:
            f.write(struct.pack("<3iII", int(origin[0]), int(origin[1]), int(origin[2]), int(num_points), int(len(payload))))
            f.write(payload)
    return {
        "points": int(len(qcoords)),
        "clusters": int(len(cluster_records)),
        "bits": int(out_path.stat().st_size * 8),
    }


def decode_qcoords(path):
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError("Not an FGC1 bitstream")
        version, total_points, num_clusters, _connect_radius = struct.unpack("<BIII", f.read(13))
        if version != VERSION:
            raise ValueError(f"Unsupported FGC version: {version}")
        decoded = []
        for _ in range(num_clusters):
            origin_x, origin_y, origin_z, num_points, payload_size = struct.unpack("<3iII", f.read(20))
            payload = f.read(payload_size)
            if len(payload) != payload_size:
                raise ValueError("Truncated cluster payload")
            local = decode_local_coords(payload, num_points)
            origin = np.asarray([origin_x, origin_y, origin_z], dtype=np.int32)
            decoded.append(local + origin)
    if not decoded:
        return np.empty((0, 3), dtype=np.int32)
    qcoords = np.concatenate(decoded, axis=0)
    qcoords = np.unique(qcoords.astype(np.int32, copy=False), axis=0)
    if len(qcoords) != total_points:
        raise ValueError(f"Decoded {len(qcoords)} unique points, expected {total_points}")
    return qcoords


def encode_frame(bin_path, mask_path, out_path, scale, connect_radius=2, min_cluster_points=1):
    coords_raw = read_kitti_bin(bin_path)
    labels = np.load(mask_path)[: len(coords_raw)]
    fg_mask = np.isin(labels, FG_CLASSES)
    qcoords = quantize_coords(coords_raw[fg_mask], scale)
    return encode_qcoords(qcoords, out_path, connect_radius=connect_radius, min_cluster_points=min_cluster_points)


def collect_files(testdata, split_file):
    testdata = Path(testdata)
    if testdata.is_file():
        return [testdata]
    if split_file:
        with open(split_file) as f:
            frame_ids = [line.strip().zfill(6) for line in f if line.strip()]
        return [testdata / f"{frame_id}.bin" for frame_id in frame_ids if (testdata / f"{frame_id}.bin").exists()]
    return sorted(testdata.rglob("*.bin"))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "filename", "scale", "fg_qpoints", "clusters", "bits", "bpp_original_points", "enc_time", "verify"
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Encode foreground masks with a cluster local-coordinate codec.")
    parser.add_argument("--testdata", required=True, help="KITTI velodyne directory or one .bin file")
    parser.add_argument("--mask_dir", required=True, help="Directory containing <frame_id>.npy foreground masks")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--scale", default="1/64", help="Quantization scale used before lossless coding")
    parser.add_argument("--connect_radius", type=int, default=2, help="Chebyshev connectivity radius in quantized units")
    parser.add_argument("--min_cluster_points", type=int, default=1)
    parser.add_argument("--out_dir", default="point_pairs/fg_cluster_codec")
    parser.add_argument("--verify", action="store_true", help="Decode each bitstream and check exact quantized-coordinate recovery")
    args = parser.parse_args()

    scale = parse_scale(args.scale)
    files = collect_files(args.testdata, args.split_file)
    out_dir = Path(args.out_dir)
    bitstream_dir = out_dir / "bitstreams"
    rows = []

    for bin_path in files:
        frame_id = bin_path.stem
        mask_path = Path(args.mask_dir) / f"{frame_id}.npy"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        coords_raw = read_kitti_bin(bin_path)
        num_original_points = len(coords_raw)
        labels = np.load(mask_path)[:num_original_points]
        fg_mask = np.isin(labels, FG_CLASSES)
        qcoords = quantize_coords(coords_raw[fg_mask], scale)
        out_path = bitstream_dir / f"{frame_id}.fgc"

        start = time.time()
        stats = encode_qcoords(
            qcoords,
            out_path,
            connect_radius=args.connect_radius,
            min_cluster_points=args.min_cluster_points,
        )
        enc_time = time.time() - start

        ok = ""
        if args.verify:
            dec = decode_qcoords(out_path)
            ref = np.unique(qcoords, axis=0)
            ok = bool(np.array_equal(sort_morton(dec), sort_morton(ref)))
            if not ok:
                raise RuntimeError(f"Verify failed for {frame_id}")

        rows.append({
            "filename": frame_id,
            "scale": scale,
            "fg_qpoints": stats["points"],
            "clusters": stats["clusters"],
            "bits": stats["bits"],
            "bpp_original_points": round(stats["bits"] / num_original_points, 6) if num_original_points else 0.0,
            "bpp_fg_qpoint": round(stats["bits"] / stats["points"], 6) if stats["points"] else 0.0,
            "connect_radius": args.connect_radius,
            "enc_time": round(enc_time, 6),
            "verify": ok,
        })

    write_csv(out_dir / "fg_cluster_details.csv", rows)
    total_bits = sum(int(row["bits"]) for row in rows)
    total_points = 0
    for bin_path in files:
        total_points += len(read_kitti_bin(bin_path))
    avg_rows = [{
        "scale": scale,
        "num_frames": len(rows),
        "total_bits": total_bits,
        "total_points": total_points,
        "bpp": round(total_bits / total_points, 6) if total_points else 0.0,
        "total_fg_qpoints": sum(int(row["fg_qpoints"]) for row in rows),
        "total_clusters": sum(int(row["clusters"]) for row in rows),
    }]
    write_csv(out_dir / "fg_cluster_average.csv", avg_rows)
    print(f"Detail CSV: {out_dir / 'fg_cluster_details.csv'}")
    print(f"Average CSV: {out_dir / 'fg_cluster_average.csv'}")


if __name__ == "__main__":
    main()
