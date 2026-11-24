# Published training artifacts

This directory contains curated, version-controlled exports from selected training
runs. It is deliberately separate from `runs/`, which remains ignored because it
can contain large or numerous local experiment artifacts.

Each published run should include the portable model bundle, its resolved
configuration and split, the metrics used for model selection, epoch history, and
any small diagnostic plots needed to interpret the result. Training data and local
environments must not be committed.

- [Car v4 to GrandTour comparison](car-v4-to-grandtour-v1/README.md): L1, G1, and GT checkpoints and cross-dataset results.
