# Supplementary dataset experiments

This directory collects the source code and the minimum experiment configurations for the three supplementary datasets used by the project:

- `nuscenes/`: geometry-only CenterPoint, six-scale label export, route-proxy training, evaluation and plotting.
- `shapenet55/pointmae/`: the main Point-MAE classification pipeline.
- `shapenet55/dgcnn_legacy/`: the earlier DGCNN classification pipeline, retained only for reproducibility.
- `sunrgbd/`: SUN-RGBD VoteNet/TinyPoint pipeline, including the runnable shell entrypoints under `run/`.
- `shared/tiny_point_router/`: shared classification-router utilities.

Only source files and configuration files are stored in this repository. Datasets, pretrained detectors/classifiers, labels, checkpoints, caches, logs and figures stay in external experiment roots and are intentionally not copied here.

The original experiment roots on node-177 are:

```text
nuScenes:  /home/sm/raco_rate_aware_nuscenes_20260822
ShapeNet:  /home/sm/raco_rate_aware_shapenet55_pointmae_20260825
SUN-RGBD:  /home/sm/sunrgbd_lite_s3_20260828
```

The copied scripts retain their historical external-asset defaults so old checkpoints and cached labels remain usable. When running on another host, change the root variables in the selected entry script (or expose equivalent paths through its command-line arguments) rather than placing datasets inside this repository.

Basic validation from the repository root:

```bash
python -c "import ast,pathlib; ps=list(pathlib.Path('supplementary_experiments').rglob('*.py')); [ast.parse(p.read_text()) for p in ps]; print('python AST OK:', len(ps))"
find supplementary_experiments -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

The full pipelines are intentionally not started by the consolidation step. They require the external datasets, detector/classifier weights and previously generated labels described above.
