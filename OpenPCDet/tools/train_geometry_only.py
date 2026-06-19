import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune geometry-only PV-RCNN from an xyz+intensity checkpoint.')
    parser.add_argument(
        '--cfg_file',
        default='cfgs/kitti_models/pv_rcnn_geometry.yaml',
        help='Geometry-only OpenPCDet config.')
    parser.add_argument(
        '--pretrained_model',
        default='ckpt/latest_model.pth',
        help='Existing 4-channel xyz+intensity checkpoint used for fine-tuning.')
    parser.add_argument(
        '--extra_tag',
        default='geometry_only',
        help='Output tag for this fine-tuning run.')
    args, remaining = parser.parse_known_args()

    train_py = os.path.join(os.path.dirname(__file__), 'train.py')
    cmd = [
        sys.executable,
        train_py,
        '--cfg_file',
        args.cfg_file,
        '--pretrained_model',
        args.pretrained_model,
        '--extra_tag',
        args.extra_tag,
    ] + remaining
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
