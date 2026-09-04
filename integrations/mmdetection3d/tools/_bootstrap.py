"""Path bootstrap shared by project-owned MMDetection3D entry points."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MMDET_ROOT = PROJECT_ROOT / "mmdetection3d"


def bootstrap_paths() -> None:
    for path in (PROJECT_ROOT, MMDET_ROOT):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
