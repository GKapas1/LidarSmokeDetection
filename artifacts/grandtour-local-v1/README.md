# GrandTour-only local MLP (v1)

This export contains the portable model bundle and reproducibility artifacts for
the `grandtour-local-v1` experiment. Training used only the `grandtour_ros1`
source domain and the saved chronological split. The interrupted GPU run was
resumed from epoch 6 on CPU; the selected checkpoint is epoch 8.

The held-out GrandTour test partition contains 522 frames, 6,727,294 valid
points, and 2,535 smoke-impacted points. Its results are in
`test_metrics.json`: average precision 0.29384, ROC-AUC 0.97692, precision
0.45786, recall 0.22288, F1 0.29981, and false-positive rate 0.0000995.

For a fair baseline comparison, `../local-default/grandtour_only_test_metrics.json`
evaluates the prior mixed-domain model on exactly the same held-out frames.
