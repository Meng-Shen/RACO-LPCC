# OpenPCDet integration

OpenPCDet remains the source of detector, backbone, voxel encoder, and point
head implementations.  `bootstrap.py` only installs three thin compatibility
wrappers needed by the project's very coarse XYZ-only experiments: safe point
padding, empty raw-point features, and geometry-only checkpoint conversion.

Run an upstream entry point through `run_tool.py`, for example:

```bash
python integrations/openpcdet/run_tool.py train.py --cfg_file ...
```
