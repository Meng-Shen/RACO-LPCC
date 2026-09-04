#!/usr/bin/env python3
"""Run an upstream OpenPCDet tool after installing project compatibility."""

from __future__ import annotations

import runpy
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.openpcdet import install_openpcdet_compat


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_tool.py TOOL.py [tool arguments ...]")
    tool = PROJECT_ROOT / "OpenPCDet" / "tools" / sys.argv[1]
    if not tool.is_file() or tool.parent != PROJECT_ROOT / "OpenPCDet" / "tools":
        raise SystemExit(f"invalid OpenPCDet tool: {sys.argv[1]}")
    install_openpcdet_compat()
    os.chdir(tool.parent)
    sys.argv = [str(tool), *sys.argv[2:]]
    runpy.run_path(str(tool), run_name="__main__")


if __name__ == "__main__":
    main()
