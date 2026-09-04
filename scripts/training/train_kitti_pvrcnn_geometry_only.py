import argparse
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[2]
    tools_dir = project_root / 'OpenPCDet' / 'tools'
    launcher = project_root / 'integrations' / 'openpcdet' / 'run_tool.py'
    parser = argparse.ArgumentParser(
        description='Fine-tune the KITTI-FOV geometry-only PV-RCNN model.')
    parser.add_argument(
        '--cfg_file',
        default=str(project_root / 'integrations' / 'openpcdet' / 'configs' /
                    'kitti_models' / 'pv_rcnn_fov_geometry.yaml'),
        help='Geometry-only OpenPCDet config.')
    parser.add_argument(
        '--pretrained_model',
        default=str(tools_dir / 'ckpt' / 'model_non_reflectance.pth'),
        help='Checkpoint used to initialize or resume geometry-only fine-tuning.')
    parser.add_argument(
        '--extra_tag',
        default='geometry_only',
        help='Output tag for this fine-tuning run.')
    args, remaining = parser.parse_known_args()

    cmd = [
        sys.executable,
        str(launcher),
        'train.py',
        '--cfg_file',
        args.cfg_file,
        '--pretrained_model',
        args.pretrained_model,
        '--extra_tag',
        args.extra_tag,
    ] + remaining
    raise SystemExit(subprocess.call(cmd, cwd=str(tools_dir)))


if __name__ == '__main__':
    main()
