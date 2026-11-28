# Nine-run voxel study

These are the complete outputs of the nine-run study from trainer commit
`b595369ec80b00aeadd3e71261b0f4ff1d3b7dc7`, run on 28 November 2026.
The baseline, interleaved, and context variants each use seeds 20260920,
20260921, and 20260922. See the nine-run study section of `trainer/README.md`
for the design and interpretation of the stationary diagnostic.

`voxel-study-v1-setup/comparison.json` collects the car test and stationary
diagnostic reports for all runs. Each run directory also contains the model
bundle, best and last checkpoints, resolved config and split, normalizer,
training history, selection metrics, calibration plot, and both reports. The
setup directory preserves the generated configurations and job list.

The `dataset-metadata` directory preserves the manifests, schemas, and available
split and preparation metadata. The actual dataset chunks are not included;
transfer `data/training/car_training_options_v1/` and
`data/training/unified_v1/` separately to reproduce evaluations or training.
The manifests contain chunk checksums. Original configurations and `jobs.json`
contain absolute paths from the training computer and require adjustment on a
different machine. Model bundles can be loaded on CPU or CUDA independently of
those paths. Environment files record the Python packages and GPU environment.

Stationary recordings are development diagnostics, and repeated review makes
subsequent ARC-6 comparisons exploratory. `SHA256SUMS` covers the exported
files other than itself.
