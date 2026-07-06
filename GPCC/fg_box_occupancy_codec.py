#!/usr/bin/env python3
"""Foreground box occupancy codec, lossless after quantization.

This is an experimental codec for foreground points only:
1. quantize foreground points to integer coordinates;
2. build object-like boxes from foreground points;
3. encode occupied voxels inside each generated box with combinatorial coding;
4. encode leftover residual points with global Morton-delta coding.
"""

import argparse
import csv
import math
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fg_cluster_codec import (
    FG_CLASSES,
    MAGIC as _OLD_MAGIC,
    cluster_quantized,
    get_uvarint,
    morton_key,
    parse_scale,
    put_uvarint,
    quantize_coords,
    read_kitti_bin,
    sort_morton,
    zigzag_decode,
    zigzag_encode,
)


MAGIC = b"FBO1"
VERSION = 1


@dataclass
class Component:
    points: np.ndarray

    @property
    def count(self):
        return int(len(self.points))

    @property
    def lo(self):
        return self.points.min(axis=0).astype(np.int32)

    @property
    def hi(self):
        return self.points.max(axis=0).astype(np.int32)

    @property
    def span(self):
        return (self.hi - self.lo + 1).astype(np.int32)


def unique_points(points):
    if len(points) == 0:
        return points.reshape(0, 3).astype(np.int32)
    return np.unique(points.astype(np.int32, copy=False), axis=0)


def make_component(points):
    return Component(unique_points(points))


def bbox_gap(a, b):
    alo, ahi = a.lo, a.hi
    blo, bhi = b.lo, b.hi
    gap = np.maximum(0, np.maximum(alo - bhi, blo - ahi))
    return int(gap.max()), gap.astype(np.int32)


def merged_component(a, b):
    return make_component(np.concatenate([a.points, b.points], axis=0))


def can_merge(a, b, max_side, max_volume=None):
    merged = merged_component(a, b)
    span = merged.span
    if int(span.max()) > max_side:
        return False, merged
    if max_volume is not None and int(np.prod(span.astype(np.int64))) > max_volume:
        return False, merged
    return True, merged


def build_boxes(
    qcoords,
    micro_radius=8,
    merge_gap=16,
    attach_gap=16,
    min_box_points=8,
    small_cluster_threshold=4,
    max_side=80,
    max_boxes=0,
):
    """Build a small set of object-like boxes plus residual points.

    The first pass intentionally over-segments foreground points into micro
    components. The second pass merges nearby component bounding boxes under a
    car-like maximum side constraint. Tiny components are attached to the nearest
    accepted box only when this does not enlarge the box too much; otherwise they
    are encoded as residual points. By default the number of boxes is not fixed,
    because the number of foreground objects varies by frame.
    """
    qcoords = unique_points(qcoords)
    if len(qcoords) == 0:
        return [], qcoords

    micro = [make_component(c) for c in cluster_quantized(qcoords, connect_radius=micro_radius)]
    large = [c for c in micro if c.count >= small_cluster_threshold]
    small = [c for c in micro if c.count < small_cluster_threshold]

    changed = True
    while changed and len(large) > 1:
        changed = False
        best = None
        for i in range(len(large)):
            for j in range(i + 1, len(large)):
                gap, gap_vec = bbox_gap(large[i], large[j])
                if gap > merge_gap:
                    continue
                ok, merged = can_merge(large[i], large[j], max_side=max_side)
                if not ok:
                    continue
                score = (gap, int(np.prod(merged.span.astype(np.int64))), -merged.count, i, j)
                if best is None or score < best[0]:
                    best = (score, i, j, merged)
        if best is not None:
            _, i, j, merged = best
            large = [c for k, c in enumerate(large) if k not in (i, j)]
            large.append(merged)
            changed = True

    boxes = [c for c in large if c.count >= min_box_points]
    residual = [c.points for c in large if c.count < min_box_points]

    for comp in small:
        best = None
        for i, box in enumerate(boxes):
            gap, _ = bbox_gap(comp, box)
            if gap > attach_gap:
                continue
            ok, merged = can_merge(comp, box, max_side=max_side)
            if not ok:
                continue
            score = (gap, int(np.prod(merged.span.astype(np.int64))), i)
            if best is None or score < best[0]:
                best = (score, i, merged)
        if best is None:
            residual.append(comp.points)
        else:
            _, i, merged = best
            boxes[i] = merged

    boxes.sort(key=lambda c: (-c.count, int(c.lo[0]), int(c.lo[1]), int(c.lo[2])))

    if max_boxes and len(boxes) > max_boxes:
        keep = boxes[:max_boxes]
        for comp in boxes[max_boxes:]:
            residual.append(comp.points)
        boxes = keep

    residual_points = unique_points(np.concatenate(residual, axis=0)) if residual else np.empty((0, 3), dtype=np.int32)
    return boxes, residual_points


def comb_payload_size(num_voxels, num_points):
    if num_points == 0:
        return 0
    states = math.comb(int(num_voxels), int(num_points))
    return max(1, (states.bit_length() + 7) // 8)


def rank_combination(indices):
    """Combinadic rank for sorted indices."""
    rank = 0
    for i, idx in enumerate(indices, start=1):
        rank += math.comb(int(idx), i)
    return rank


def unrank_combination(rank, num_voxels, num_points):
    combo = [0] * num_points
    upper = int(num_voxels) - 1
    rank = int(rank)
    for i in range(num_points, 0, -1):
        lo, hi = i - 1, upper
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            value = math.comb(mid, i)
            if value <= rank:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        combo[i - 1] = best
        rank -= math.comb(best, i)
        upper = best - 1
    return np.asarray(combo, dtype=np.int64)


def encode_box_payload(local_points, size):
    size = np.asarray(size, dtype=np.int64)
    local_points = unique_points(local_points)
    idx = (
        local_points[:, 0].astype(np.int64)
        + local_points[:, 1].astype(np.int64) * size[0]
        + local_points[:, 2].astype(np.int64) * size[0] * size[1]
    )
    idx = np.sort(idx)
    num_voxels = int(size[0] * size[1] * size[2])
    payload_size = comb_payload_size(num_voxels, len(idx))
    rank = rank_combination(idx)
    return rank.to_bytes(payload_size, "little"), num_voxels


def decode_box_payload(payload, origin, size, num_points):
    size = np.asarray(size, dtype=np.int64)
    num_voxels = int(size[0] * size[1] * size[2])
    rank = int.from_bytes(payload, "little")
    idx = unrank_combination(rank, num_voxels, num_points)
    z = idx // (size[0] * size[1])
    rem = idx - z * size[0] * size[1]
    y = rem // size[0]
    x = rem - y * size[0]
    local = np.stack([x, y, z], axis=1).astype(np.int32)
    return local + np.asarray(origin, dtype=np.int32)


def encode_residual(points):
    points = sort_morton(unique_points(points))
    raw = bytearray()
    prev = np.zeros(3, dtype=np.int64)
    for point in points.astype(np.int64, copy=False):
        delta = point - prev
        for value in delta:
            put_uvarint(raw, zigzag_encode(value))
        prev = point
    return zlib.compress(bytes(raw), level=9)


def decode_residual(payload, num_points):
    if num_points == 0:
        return np.empty((0, 3), dtype=np.int32)
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
        raise ValueError("Trailing bytes in residual payload")
    return coords


def encode_qcoords(qcoords, out_path, **box_kwargs):
    qcoords = unique_points(qcoords)
    boxes, residual = build_boxes(qcoords, **box_kwargs)

    box_records = []
    theoretical_bits = 0.0
    for box in boxes:
        origin = box.lo
        size = box.span
        local = box.points - origin
        payload, num_voxels = encode_box_payload(local, size)
        theoretical_bits += math.log2(math.comb(num_voxels, box.count)) if box.count else 0.0
        box_records.append((origin, size, box.count, payload, num_voxels))

    residual_payload = encode_residual(residual)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<BIII", VERSION, int(len(qcoords)), int(len(box_records)), int(len(residual))))
        for origin, size, count, payload, _num_voxels in box_records:
            if int(size.max()) > 256:
                raise ValueError(
                    f"Box side exceeds uint8-coded limit 256: origin={origin.tolist()}, size={size.tolist()}"
                )
            if int(count) > 65535:
                raise ValueError(f"Box point count exceeds uint16 limit: {count}")
            f.write(struct.pack(
                "<3i3BHI",
                int(origin[0]), int(origin[1]), int(origin[2]),
                int(size[0] - 1), int(size[1] - 1), int(size[2] - 1),
                int(count), int(len(payload)),
            ))
            f.write(payload)
        f.write(struct.pack("<I", int(len(residual_payload))))
        f.write(residual_payload)

    return {
        "fg_qpoints": int(len(qcoords)),
        "boxes": int(len(box_records)),
        "box_points": int(sum(r[2] for r in box_records)),
        "residual_points": int(len(residual)),
        "bits": int(out_path.stat().st_size * 8),
        "box_payload_bits": int(sum(len(r[3]) * 8 for r in box_records)),
        "residual_payload_bits": int(len(residual_payload) * 8),
        "box_theoretical_bits": float(theoretical_bits),
        "box_spans": ";".join("x".join(str(int(v)) for v in r[1]) for r in box_records),
        "box_counts": ";".join(str(int(r[2])) for r in box_records),
    }


def decode_qcoords(path):
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError("Not an FBO1 bitstream")
        version, total_points, num_boxes, residual_points = struct.unpack("<BIII", f.read(13))
        if version != VERSION:
            raise ValueError(f"Unsupported FBO version: {version}")
        decoded = []
        for _ in range(num_boxes):
            data = f.read(21)
            ox, oy, oz, sx, sy, sz, count, payload_size = struct.unpack("<3i3BHI", data)
            sx, sy, sz = sx + 1, sy + 1, sz + 1
            payload = f.read(payload_size)
            decoded.append(decode_box_payload(payload, (ox, oy, oz), (sx, sy, sz), count))
        residual_payload_size = struct.unpack("<I", f.read(4))[0]
        residual_payload = f.read(residual_payload_size)
        decoded.append(decode_residual(residual_payload, residual_points))
    points = unique_points(np.concatenate(decoded, axis=0)) if decoded else np.empty((0, 3), dtype=np.int32)
    if len(points) != total_points:
        raise ValueError(f"Decoded {len(points)} unique points, expected {total_points}")
    return points


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Encode foreground with box occupancy combinatorial coding.")
    parser.add_argument("--testdata", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--scale", default="1/64")
    parser.add_argument("--out_dir", default="point_pairs/fg_box_occupancy")
    parser.add_argument("--micro_radius", type=int, default=8)
    parser.add_argument("--merge_gap", type=int, default=16)
    parser.add_argument("--attach_gap", type=int, default=16)
    parser.add_argument("--min_box_points", type=int, default=8)
    parser.add_argument("--small_cluster_threshold", type=int, default=4)
    parser.add_argument("--max_side", type=int, default=80)
    parser.add_argument("--max_boxes", type=int, default=0,
                        help="Maximum generated boxes per frame. 0 means no limit.")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    scale = parse_scale(args.scale)
    bin_path = Path(args.testdata)
    frame_id = bin_path.stem
    mask_path = Path(args.mask_dir) / f"{frame_id}.npy"
    coords_raw = read_kitti_bin(bin_path)
    labels = np.load(mask_path)[:len(coords_raw)]
    qcoords = quantize_coords(coords_raw[np.isin(labels, FG_CLASSES)], scale)

    out_dir = Path(args.out_dir)
    out_path = out_dir / "bitstreams" / f"{frame_id}.fbo"
    start = time.time()
    stats = encode_qcoords(
        qcoords,
        out_path,
        micro_radius=args.micro_radius,
        merge_gap=args.merge_gap,
        attach_gap=args.attach_gap,
        min_box_points=args.min_box_points,
        small_cluster_threshold=args.small_cluster_threshold,
        max_side=args.max_side,
        max_boxes=args.max_boxes,
    )
    enc_time = time.time() - start

    verify = ""
    if args.verify:
        dec = decode_qcoords(out_path)
        verify = bool(np.array_equal(sort_morton(dec), sort_morton(qcoords)))
        if not verify:
            raise RuntimeError(f"Verify failed for {frame_id}")

    row = {
        "filename": frame_id,
        "scale": scale,
        "fg_qpoints": stats["fg_qpoints"],
        "boxes": stats["boxes"],
        "box_points": stats["box_points"],
        "residual_points": stats["residual_points"],
        "bits": stats["bits"],
        "bpp_original_points": round(stats["bits"] / len(coords_raw), 6) if len(coords_raw) else 0.0,
        "box_payload_bits": stats["box_payload_bits"],
        "residual_payload_bits": stats["residual_payload_bits"],
        "box_theoretical_bits": round(stats["box_theoretical_bits"], 3),
        "box_counts": stats["box_counts"],
        "box_spans": stats["box_spans"],
        "micro_radius": args.micro_radius,
        "merge_gap": args.merge_gap,
        "max_side": args.max_side,
        "enc_time": round(enc_time, 6),
        "verify": verify,
    }
    write_csv(out_dir / "fg_box_occupancy_details.csv", [row])
    print(f"Detail CSV: {out_dir / 'fg_box_occupancy_details.csv'}")
    print(f"Bitstream : {out_path}")
    print(row)


if __name__ == "__main__":
    main()
