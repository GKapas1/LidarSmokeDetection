# Neural network trainer implementation and experiment plan

Prepared 2026-09-17 for the current repository and `data/training/unified_v1`.

This is a build plan, not an implemented trainer or a report of neural-network results. Commands, modules, budgets, and acceptance criteria below are proposed interfaces. The proposal supplies the research requirements; implementation choices here are recommendations based on the actual data and the user's constraints.

## 1. Decisions and intended outcome

Build a separate, configuration-driven `trainer/` Python package that reads the canonical dataset manifest, creates reproducible partitions, trains four compact model families, compares them fairly, calibrates their probabilities, and exports a predictor that a ROS 2 node can use without importing the labeler.

The agreed priorities are:

- Use the existing data for development now; collect independent sessions for the final thesis evaluation.
- Keep the initial training campaign to hours on an older gaming laptop. Use a hard wall-clock budget, and measure throughput before scheduling experiments.
- Prioritize real-time prediction on very light hardware. A compact model with inexpensive CPU preprocessing is the deployment candidate.
- Document suitable HPC resources and use HPC only when the measured experiment cost or scientific benefit justifies it.
- Keep the pose requirement undecided: support a LiDAR-only path, and compare an optional pose-assisted temporal path before making it a deployment dependency.

The trainer predicts `p_smoke = P(label = 1 | available sensor observations)`. The published reliability is `1 - p_smoke`. With today's labels, this means agreement with the reviewed smoke-impact labeling policy on observed returns. It does not establish the probability that odometry will succeed, and it does not measure missing returns or reliability in empty space.

### Proposal coverage

| Proposal requirement | Planned implementation and evidence |
| --- | --- |
| §3.2.1: local per-voxel statistics and MLP/CNN | Compact voxel MLP, starting at 0.10 m voxels |
| §3.2.1: whole-scan 3D CNN or transformer | Small transformer with a bounded global attention bottleneck over occupied voxel tokens |
| §3.2.2: local temporal processing | Causal voxel feature sequences with a small temporal convolutional network (TCN); compare sensor-frame and pose-aligned sequences |
| §3.2.2: global temporal processing | Whole-scan spatial encoder plus causal temporal processing of scan representations and local histories |
| §4.1: train/validation/test; ROC, PR, AUC, F1 | Versioned split plans, common full-frame evaluation, curves and metrics by domain |
| §4.1: calibrated probabilities | Separate calibration data, reliability diagrams, Brier score and log loss |
| §4.2: qualitative maps and temporal behavior | Point-cloud overlays and fixed sequence visualizations |
| §6.1: feature, architecture, temporal and spatial ablations | Preregistered experiment matrix with matched seeds and budgets |
| §3.3/§6.1: reusable ROS 2 reliability node | Shared preprocessing/predictor API, export bundle, streaming replay and latency acceptance tests |

Source: [thesis proposal](../GKapas_ThesisProposal.pdf), especially pages 2–3 and 5. The proposal names architecture families, not a mandatory implementation or exhaustive model zoo. Its literature examples do not require reproducing every cited system.

## 2. What is actually available

The entry point is `data/training/unified_v1/dataset_manifest.json`. There is no neural trainer in the repository yet. The existing `labeler/src/smoke_labeler/unified_dataset.py` provides a useful reference loader and strict schema validator.

| Acquisition domain | Acquisition sessions | Storage chunks | Frames | All points | Smoke points | Smoke / nonignored | Ignored / all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stationary ROS 2 | 1 | 2 | 1,472 | 29,475,456 | 304,729 | 1.6153% | 35.9971% |
| GrandTour ARC-6 | 1 | 7 | 3,609 | 47,306,636 | 221,980 | 0.4695% | 0.0499% |
| Total | 2 | 9 | 5,081 | 76,782,092 | 526,709 | 0.7963% | 13.8495% |

There are 65,621,439 unimpacted points and 10,633,944 ignored points. Approximately 99.20% of nonignored points are negative, so accuracy alone would reward an almost useless all-negative classifier.

The nine NPZ files occupy approximately 778 MiB. Their point arrays alone expand to approximately 2.53 GB before caches, worker copies, features and training tensors. Do not load a complete copy into every data-loader worker.

### Observed details that affect the design

1. **Chunks are not independent recordings.** All seven GrandTour files belong to ARC-6 session `2024-11-18-17-13-09`. Their `recording_id` values are storage identifiers such as `chunk_000`. The stationary clean control and smoke recording share one scene/session and labeling reference.
2. **Each GrandTour chunk resets time to zero.** `source_time_selection_s.start` is a requested extraction boundary, not necessarily the timestamp of the first retained scan. Recover exact integer timestamps from the source export before joining chunks. Simply adding 60 seconds per chunk is unsafe.
3. **The streams run at approximately 10 Hz.** Stationary scans contain about 20,000 points; GrandTour scans usually contain about 13,000. The inspected chunks have no internal frame gap above 0.25 s. Reconstruct and validate continuity across chunks separately.
4. **Sensor metadata differ between sources.** All inspected GrandTour tags equal zero; stationary tags have multiple values. Both contain line values 0–3. Observed intensity ranges are 0–171 for stationary and up to 255 for GrandTour. Identical array names do not guarantee identical measurement characteristics.
5. **Point times have small numerical offsets.** GrandTour contains negative offsets as small as about -0.12 microseconds. Define a documented tolerance; do not reject these as whole-frame timing failures.
6. **Label coverage is very different.** Approximately 36% of stationary points are ignored, versus 0.05% in GrandTour. Evaluation must show coverage and per-domain results; a pooled number can hide these differences.
7. **Labels are partly automatic.** Stationary labeling compares against a clean angular reference; ARC-6 uses map alignment and reviewed regions. The ARC-6 person near the source is a reviewed hard negative. Geometry/location shortcuts and teacher-policy errors are plausible.
8. **Poses are not in the unified contract.** ARC-6 DLIO poses and deskewed clouds exist upstream, but only raw sensor-frame XYZ is exported as model input. A pose-assisted experiment requires an explicit sidecar and an equivalent runtime source.
9. **Existing clean-validation numbers concern the labeler.** They are not neural-network results. Half of the stationary clean control was used to calibrate labeling thresholds; the whole control is now exported with valid points marked clean. ARC-5 clean validation has already been inspected and is outside `data/training`; it is at most an additional diagnostic after an explicit export, not a fresh final test.

Evidence: canonical schema/manifest, array inspections of all nine chunks, source GrandTour timestamps, stationary `dataset_summary.json`, and `labeler/GRANDTOUR_IMPLEMENTATION.md`. No model performance was used to choose this plan.

## 3. Split strategy and data use

Model comparison, ablation and hyperparameter optimization should share one experimental protocol. They do not need independently randomized datasets for each purpose. Final testing must remain separate from all three.

### 3.1 Partition roles

| Role | Allowed uses | Prohibited uses |
| --- | --- | --- |
| Training | Gradients, feature normalization, class weights, sampling statistics, training-only augmentation and hard-negative mining | Treating its metrics as evidence of generalization |
| Selection validation | Checkpoint selection, early stopping, architecture selection, HPO, ablations, decision-threshold selection | Fitting model weights or preprocessing statistics |
| Calibration | Fit the fixed probability calibrator after model selection | Choosing architectures, HPO trials or training epochs |
| Locked test | Evaluate a frozen, declared set of models/ablations for final reporting | Retuning any part of the pipeline after looking at results |

After fitting the calibrator on calibration data, threshold selection may use the already designated selection-validation predictions transformed by that calibrator. It is acceptable for selection validation to be reused for tuning; report it as selection performance. Evaluate calibration quality on test, not on the data used to fit the calibrator. Never select the winning random seed using test results.

If a final model is refitted on training plus selection data, use a preselected epoch/update count and regenerate calibration predictions from that newly fitted checkpoint. Never reuse a calibrator fitted to a different checkpoint. The initial implementation should avoid this complication and retain the selected training-only checkpoint.

### 3.2 Current data: explicitly a development protocol

A conventional independent four-way session split is impossible with only two acquisition sessions. Implement two clearly named protocols:

**A. `pilot_temporal_v1`: mixed-domain, purged chronological partitions.** Use this for building the trainer, inexpensive HPO and the initial four-model comparison. It measures transfer to later portions of familiar recordings, not unseen rooms or independent sessions. This is an explicit pilot exception to the repository's preferred whole-recording/session split policy; the final protocol below must enforce that policy.

For each of the three actual streams (stationary clean, stationary smoke, complete ARC-6), start with chronological boundaries at 55%, 70% and 85% of retained duration. Roles are 55% training, 15% selection, 15% calibration and 15% test before purging. Remove two seconds on **each side** of every boundary. Build windows only after applying these assignments; every context frame and target must belong to the same partition. If the maximum supported history exceeds two seconds, increase the purge before generating any experiments. The purge removes direct temporal overlap; it does not remove shared-scene correlation.

The resulting planning audit is below. These counts precede temporal warm-up exclusion and deployment-validity filtering; the implementation must regenerate and freeze the exact frame IDs and counts.

| Stream | Training frames / smoke points | Selection frames / smoke points | Calibration frames / smoke points | Pilot test frames / smoke points |
| --- | ---: | ---: | ---: | ---: |
| Stationary clean | 291 / 0 | 45 / 0 | 45 / 0 | 64 / 0 |
| Stationary smoke | 479 / 170,049 | 96 / 30,131 | 96 / 31,300 | 116 / 36,175 |
| ARC-6 | 1,965 / 146,768 | 502 / 19,718 | 501 / 52,725 | 521 / 2,535 |
| Total | 2,735 / 316,817 | 643 / 49,849 | 642 / 84,025 | 701 / 38,710 |

The clean control is negative-only by design; do not demand both classes in every stream. Require both classes in each aggregate detection/calibration partition and publish class counts by domain. The ARC-6 pilot-test prevalence is particularly low. This is an honest feature of the chronological test, not a reason to move boundaries after observing model results.

For split generation, define duration as `last_retained_time - first_retained_time + median_frame_interval`. Use integer timestamps for joining GrandTour chunks, subtract the first timestamp before converting to seconds, and use exact half-open intervals. The retained ARC-6 duration under this definition is approximately 360.900 s; it differs from the raw bag duration and the nominal 0–362 s extraction range.

Use identical target-frame IDs for all four models, based on the longest predeclared temporal history. Separately report startup/all-frame behavior so temporal warm-up is not hidden. A single-scan model must not obtain extra test frames in the principal paired comparison.

**B. `domain_transfer_v1`: secondary stress tests.** Train/tune/calibrate only within the source domain and evaluate the other domain without adapting normalization, thresholds or calibration. Run stationary → GrandTour and GrandTour → stationary after fixing the recipes. Each direction is a separate experiment. These are domain-shift diagnostics on one session per domain, not two independent estimates of generalization. Because the mixed-domain pilot uses both domains for development, do not describe these as untouched external validation. Do not use their scores for further tuning while still calling them tests.

Do not run random point splits, random frame splits, or nine-fold cross-validation over the nine files. Do not treat different training seeds as independent data splits. No standard `GroupKFold` call alone can repair an incorrectly defined group.

### 3.3 Final data: independent sessions and a held-out environment

Plan approximately 8–12 independent acquisition episodes across at least three environments and multiple days, with stationary and moving acquisition represented where possible. This is a collection target, not a claim of statistical sufficiency. Capture clean controls, smoke onset/steady/clearing periods, different plume locations/densities and distances, and hard negatives such as people and low-reflectivity surfaces.

Record `environment_id`, `acquisition_session_id`, `parent_recording_id`, `smoke_episode_id`, `reference_family_id`, sensor configuration, and label-policy version. Decide the claim before the split:

- For **new-session performance in familiar environments**, group complete acquisition/reference families; scene overlap is allowed but explicitly reported.
- For **new-environment performance**, reserve an entire environment, including its related acquisitions and reference family, as final test. Reference maps may be used offline to create that environment's labels; they must not enter training or runtime inference. Freeze label-generation rules before inspecting model predictions there.

Aim roughly for 60% training, 15% selection, 10% calibration and 15% final test by acquisition groups, but group integrity and class support take precedence over percentages. The grouped split generator must fail with an explanation when the available groups cannot support the requested roles. Collect more groups or reduce the claim rather than quietly splitting a group.

With enough development groups, add three-fold grouped inner validation for HPO confirmation, leaving calibration and final test untouched. Fit preprocessing anew within every fold. If the available data cannot support these folds, keep a fixed grouped selection set and state the limitation; nested cross-validation is optional, not a reason to multiply laptop cost.

Freeze the final test before its predictions are inspected. A predeclared set of four core models and a small number of primary ablations can all be evaluated in one final campaign. Any subsequent test-driven redesign requires new test data or explicitly exploratory reporting.

### 3.4 Split artifacts and checks

Write a separate `split_plan.json`; leave the source manifest unchanged. It must contain dataset and chunk hashes, grouping definitions, protocol/version, seed, exact frame memberships, discarded boundary frames, purge/history settings, per-role counts, and a machine-readable reason for every exclusion.

Check that context and target frame sets are disjoint across roles; every grouped final split has disjoint group IDs; duplicate source frames do not occur in multiple roles; and every target has only permitted causal context. Split inspection may show counts and fixed label-distribution summaries before training, but must not repeatedly move partitions to improve model scores.

These separation rules follow the distinction between training, model selection and held-out evaluation in the [scikit-learn cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html). The specific split proportions and purges above are project design choices.

## 4. Data ingestion, caching and feature contract

### 4.1 Immutable ingestion

1. Resolve the manifest and relative paths, require `status=complete` and a supported schema, and verify each chunk checksum once during indexing.
2. Validate required arrays, dtypes, lengths, finite values, label vocabulary, frame-pointer endpoints, frame-index agreement, increasing timestamps and metadata agreement with the manifest.
3. Build a compact frame index with stable source identity, chunk/frame IDs, point slices, exact stream time, acquisition group and class counts.
4. Recover a GrandTour timestamp sidecar from `source_path` using the original `frame_time_ns`, checking source identity and correspondence. Do not import source labels, world coordinates or reference distances as features. If exact timing cannot be recovered, reset history at chunk boundaries instead of inventing continuity.
5. Create a read-only, uncompressed NPY/memory-mapped cache, sharded by source chunk. Compressed NPZ files remain the source of truth. Stream conversion one chunk at a time; avoid loading every chunk simultaneously.
6. Key caches by source hash, schema and deterministic feature version. Key fitted transformations separately by training split hash. Use atomic writes and reject stale/incomplete cache entries.

The index and cache are runtime data under `data/training_cache/`; they are not additions to the canonical dataset. Store small configuration and split recipes in Git, not the large cache.

### 4.2 No label information in inference features

Allowed inputs are `xyz`, `intensity`, optional `tag`/`line`, point offsets and causal history. Frame timestamps organize sequences and provide time deltas; absolute date, frame number, session, filename, domain and condition must not be learned inputs.

`label != 255` is a loss/evaluation mask only. Do not remove points from voxel features merely because their label is ignored: that would make inference depend on labeler information unavailable online. Instead, define runtime validity independently using finite values and the configured sensor/range contract. Compute occupancy and neighborhoods from all runtime-valid returns, then apply the separate supervision mask. Nonignored points excluded by runtime filtering count toward an explicit coverage/exclusion report.

Forbidden inputs also include reference maps, nearest-reference distances, automatic labels, confidence/review fields, clean-map identifiers and absolute world coordinates. Offline overlays may join this information for error analysis without feeding it into the model. Annotation-derived hard-negative categories may stratify evaluation but may not become predictor features.

### 4.3 Initial representations

Use occupied voxels only. Start with 0.10 m cubic voxels in sensor coordinates; never allocate a dense cube covering the entire 30 m radius.

The common feature encoder starts with:

- `log1p(point_count)` and occupancy density tied to the chosen voxel volume;
- intensity mean and standard deviation;
- range mean and standard deviation;
- voxel center/point centroid in sensor coordinates and within-voxel coordinate spread;
- a geometry-valid indicator for statistics that require multiple points.

Add covariance eigenvalue geometry only as an ablation after profiling its CPU cost. For singleton/degenerate voxels, use finite defaults plus the geometry-valid indicator. Exclude tags from the primary comparison because of the observed source mismatch. Add tag bit features, line features and point-offset summaries in controlled feature ablations; do not assume a categorical tag number has ordinal meaning.

Use fixed physical units for XYZ/range and train-fitted normalization for scalar statistics. Save the exact ordered feature schema and normalization constants with each checkpoint. Do not normalize every scan to a unit sphere, which would remove physical scale. Do not normalize each domain separately at evaluation time unless domain identity is an explicit deployment requirement in a separately named experiment.

Voxel sizes of 0.05, 0.10 and 0.20 m form the initial resolution study. Changing voxel size requires recomputing occupancy/statistics and the point-to-voxel inverse map. Cache all three only if disk space and preparation time justify it.

### 4.4 Targets and mixed voxels

Keep per-point labels as the authoritative target. For a voxel, let `n` be its number of supervised points and `k` the smoke-positive count. Its soft target is `q = k/n` for `n > 0`. Do not turn every voxel with one smoke point into an entirely positive target.

For the shared voxel-output models, optimize point-equivalent binary cross-entropy: weight each voxel's soft-target loss by `n`, with separate positive/negative terms when class weighting is enabled. A voxel with `n=0` contributes context but no loss. Inference broadcasts the voxel probability to each of its original points using the inverse map. This makes the main local/global comparison use the same output resolution.

This shared resolution cannot distinguish differently labeled points inside one voxel. Report the mixed-voxel rate and resolution effects. A point-refinement head using per-point features plus the voxel embedding is a later ablation, applied to both local and global encoders for a fair comparison. All variants still return one score per original point.

## 5. Models and temporal alignment

Implement a small model registry with the same batch and prediction interfaces. The main experiment is a 2 × 2 matrix, not an open-ended architecture search.

| ID | Spatial context | Time | Initial implementation | Purpose |
| --- | --- | --- | --- | --- |
| L1 | Individual occupied voxel | Current scan | Shared MLP, widths 64 → 64 → 32, normalization/activation, one logit | Cheapest required neural baseline and first deployment candidate |
| G1 | Complete current scan | Current scan | Shared voxel encoder, 16–32 global latent tokens, 1–2 small attention blocks, voxel-to-global decoding | Required whole-scan transformer comparison |
| LT | Voxel history | Past and current scans | Same local encoder plus a causal TCN over voxel feature histories, then current-voxel logit | Tests temporal benefit without global context |
| GT | Whole scans plus voxel histories | Past and current scans | G1 scan encoder, causal processing of per-scan global latents, current-voxel decoder with the LT history branch | Tests temporal benefit with global context |

For G1, every occupied input voxel contributes to global summaries and every current voxel is decoded. Cross-attention to a fixed number of latent tokens bounds attention storage approximately by occupied-voxel count × latent count rather than voxel count squared. Decode query voxels in chunks if necessary. GT processes the complete sequence through the same scan encoder and preserves region-specific outputs; a scan-level smoke classifier alone would not satisfy the proposal.

The exact width/token counts are initialization choices, not guaranteed performance or memory figures. Start with a target under roughly 250,000 parameters for L1/LT and under roughly one million for G1/GT, then profile. Preprocessing and memory access can dominate latency even for a small model. A larger sparse 3D CNN or point transformer is a secondary HPC experiment only if the compact global model leaves a clear research question unanswered.

Add inexpensive controls: all-negative and training-prior predictors, logistic regression on the same local features, and a small global-pooling MLP. The latter checks whether any global gain requires attention. A map-distance teacher is not a deployable LiDAR-only baseline because it uses privileged information. An unsupervised clean-trained autoencoder/VAE is optional; it is not required before the four supervised cells are complete.

### 5.1 Causal temporal contract

Use `T=5` initially: about 0.4 s of history at 10 Hz. Compare `T=1,3,5,10` only within predeclared history/purge limits. Predict the final/current scan; no future scans, centered smoothing or bidirectional recurrent layers.

Build windows on actual timestamps with stride one initially. Reject/reset on an acquisition boundary, split boundary, missing timestamp, non-increasing time or gap above 0.25 s. Adjacent storage chunks may share history only after exact continuity is verified. Cold starts use an explicit history-valid mask and a documented current-scan fallback; never copy history across sessions or partitions. Absent historical occupancy is missing context, not a clean observation.

For fair temporal ablations, score the same target frames and preserve the same spatial encoder, feature set and training exposure. Compare against a frozen single-scan predictor with simple causal probability smoothing to check whether temporal gains require a learned temporal model. Smoothing must follow the same alignment and reset rules.

### 5.2 Pose-free versus pose-assisted choices

| Choice | What it does | Benefit | Limitation / deployment cost |
| --- | --- | --- | --- |
| Single scan | Operates in the current sensor frame | No pose or history dependency; simplest ROS 2 node | Cannot use temporal persistence |
| LiDAR-only temporal | Uses voxel histories in the sensor coordinate grid plus causal global scan features | Works with existing unified inputs and avoids coupling to odometry | Under motion, a fixed voxel is not a fixed physical region; report as unregistered temporal context, not world-voxel tracking |
| Pose-assisted temporal | Transforms earlier observations into the current sensor frame before forming histories | Makes local histories correspond more closely to physical space | Requires time synchronization, extrinsics, pose quality checks, extra CPU work and a runtime pose source |

Implement the LiDAR-only four-model matrix first. Stationary recordings supply a useful identity-alignment case. For moving ARC-6, add pose-assisted LT/GT as a separate comparison once the base trainer works. This is the extension that directly tests the proposal's tracked-voxel interpretation on a moving sensor.

If `T_W_L(t)` maps LiDAR coordinates at time `t` into an odometry frame, align a past point with `p_current = inverse(T_W_L(current)) @ T_W_L(past) @ p_past`. Retain relative motion only; neither global pose nor map coordinates become learned features.

The pose sidecar must specify integer timestamps, frame names, extrinsics, interpolation validity and provenance. A LiDAR-derived pose source makes the system dependent on that odometry's failure modes; it is no longer a predictor with only a point-cloud subscription. Offline smoothed trajectories or interpolation using poses that would arrive later than the decision deadline must not silently stand in for online poses. Distinguish an offline alignment upper-bound experiment from a deployable pose experiment.

Raw points also contain within-scan motion. Frame-to-frame alignment does not itself deskew them. If deskewing is introduced, make it a separate preprocessing option and ablation with causal point times and an equivalent runtime implementation. Do not feed upstream deskewed clouds into training while deploying on raw clouds.

Default deployment decision: prefer L1 or the LiDAR-only temporal model unless pose-assisted inference produces a meaningful held-out improvement and the intended robot already provides sufficiently timely poses. On invalid/missing poses, use an explicitly evaluated fallback and publish the active mode; never treat failed alignment as zero motion.

## 6. Training engine and practical defaults

Use Python 3.11+, PyTorch, NumPy/SciPy for initial preprocessing, scikit-learn for metrics/calibration/baselines, and Optuna for budgeted HPO. TOML configs fit the existing repository. Keep tracking local with JSON/CSV logs, checkpoints and an SQLite study database; a remote experiment service is optional. Pin a tested environment after checking the actual laptop GPU/driver; do not prescribe an unverified CUDA build.

Start with a small explicit PyTorch training loop rather than adding distributed-training machinery. CPU operation must support data checks, baselines, replay and at least the smallest neural model. CUDA and mixed precision are optional acceleration paths, with numerical checks before enabling them.

Initial training recipe:

- AdamW, learning rate `1e-3`, weight decay `1e-4`, gradient clipping at norm 1, and a modest learning-rate schedule.
- Batch by a maximum number of occupied voxels/points, not only by a fixed count of scans. Begin with one scan/window per batch and gradient accumulation if needed. Do not silently crop global context to resolve an out-of-memory error.
- Start with natural frame prevalence, domain-balanced frame selection within the **training** partition, and moderate positive weighting. Define the training objective explicitly as an average over the selected domains/frames with point-equivalent voxel losses. Record per-domain exposure.
- Try `pos_weight` values 1, 4, 8 and 16; default 8 is provisional. Do not automatically apply the approximately 125:1 overall negative-to-positive ratio together with aggressive positive oversampling. Their combined effect would change the objective substantially.
- Compare focal loss only after weighted BCE works. Neither a weighted-loss sigmoid nor a resampled-model sigmoid is automatically a calibrated probability.
- Start with at most 30 epochs for the pilot, a minimum of 5 before early stopping, and patience 5 selection evaluations; wall-clock budgets override these caps. Define an epoch and examples/updates seen in the run record.
- Selection uses full, naturally distributed validation frames, with no positive oversampling, random point dropping or training augmentation.

Use only conservative, physically defensible augmentation initially: consistent yaw rotation of a whole scan/history when appropriate to the scene, small coordinate noise, and small intensity perturbations validated against sensor behavior. Keep the sensor origin and range semantics intact. The same transform must be applied to all temporal frames and poses consistently. Density-altering dropout changes the phenomenon being measured and belongs in a separate robustness experiment, not an unchecked default augmentation. Do not synthesize smoke without a validated labeling model.

Checkpoint `last` and `best_selection`, optimizer/scheduler/scaler states, epoch/update count, RNG/sampler state, and resolved config. Support resuming an interrupted run; save checkpoints atomically. Fail or mark a trial invalid on nonfinite loss/gradients, invalid output shape, zero supervised points or incompatible data hashes. Do not overwrite a previous experiment's outputs.

PyTorch's [reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html) supports recording and seeding Python, NumPy, PyTorch and loader workers, and using deterministic operations where available. Record hardware/library versions and nondeterministic exceptions; matching seeds do not guarantee bitwise equality across hardware.

## 7. HPO and model comparison under a laptop budget

### 7.1 First campaign: hard cap of four hours

Four hours is a proposed initial compute cap, not a throughput prediction. The laptop GPU model/VRAM and eventual deployment CPU are still unknown; the current environment's `nvidia-smi` could not communicate with a driver. Measure the actual training machine before choosing batch sizes.

| Stage | Maximum initial allocation | Work |
| --- | ---: | --- |
| Profile | 20 min | Cache/loader measurement, smallest model, peak RAM/VRAM, preprocessing and inference time |
| Baselines and sanity | 20 min | Constant/logistic baselines and tiny-subset fitting checks |
| Four core cells | 60 min | One capped initial run per family, about 15 min each |
| Focused tuning | 80 min | About 20 min per family; sequential trials, as many complete valid trials as measured throughput permits |
| Confirm and report | 40 min | Refit/confirm promising recipes, calibration and full selection evaluation |
| Reserve | 20 min | Checkpoint completion, failures and artifact generation |

Cache construction may require a separate one-time preparation budget if measured I/O makes it exceed the first allocation. The scheduler must expose that cost and reduce the campaign rather than silently exceed the requested cap. If a global model cannot complete a useful run, record it as incomplete; a severely undertrained run is not evidence that global processing is worse. Keep the pilot test closed until the candidate set and evaluation recipe are frozen.

Prioritize a valid L1 pipeline and an interpretable four-cell screening result over a large number of trials. The initial hours-scale campaign will generally be exploratory; repeated seeds and a thesis-quality HPO campaign can follow after the resource decision.

### 7.2 Search protocol

Use Optuna with a seeded TPE sampler, a persistent local study, and a simple median pruner after at least five completed warm-up trials. If the budget cannot support that many trials, use a small fixed/random search without pretending the pruner is informative. See the official [Optuna sampler and pruning guide](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html).

Search architecture families separately so cheap local models cannot consume the whole study. Use the same split, objective, preprocessing rules and documented per-family compute cap. Report both best-achieved performance under the cap and actual training exposure; an equal-time screen is not an equal-convergence architecture comparison.

The primary HPO objective is the mean of per-domain smoke average precision (AP) on selection data, using fixed weights of 0.5/0.5 for the current two domains. This avoids allowing GrandTour's larger point count to determine every choice. Also record per-domain AP and pooled AP. If a domain/fold has no positives, mark its AP undefined and reject that detection fold for HPO rather than silently inventing a score. Clean-only controls contribute false-positive metrics, not AP.

| Hyperparameter | Initial search values/range |
| --- | --- |
| Learning rate | Log scale `1e-4` to `3e-3` |
| Weight decay | `0`, `1e-5`, `1e-4`, `1e-3` |
| Hidden width | 32, 64, 96 |
| Dropout | 0, 0.1, 0.2 |
| Positive weight | 1, 4, 8, 16 |
| Spatial resolution | Fix 0.10 m for initial HPO; study 0.05/0.20 m separately |
| Temporal history | Fix T=5 for initial temporal HPO; study T=1/3/10 separately |
| Global tokens | 16 or 32 |
| Attention depth | 1 or 2 |

Do not search every feature subset, architecture, resolution and loss combination at once. Tune optimizer/width first; use explicit ablations for the research questions. Make runtime/memory feasibility constraints visible. Keep any slower research winner separate from the deployable winner.

After the pilot, propose a measured budget for 12–20 trials per family, then repeat the best configuration per family with three seeds, increasing to five only if uncertainty matters. Confirm promising trials at full training budget. A pruned trial must not enter the final comparison as if it were fully trained. HPO remains confined to training and selection data.

### 7.3 When HPC is worthwhile

An appropriate starting allocation is one CUDA-capable GPU with 16 GB VRAM minimum and preferably 24 GB, 8–16 CPU cores, 32–64 GB RAM, and fast local scratch storage. These are planning targets, not a compatibility guarantee. CPU inference constraints remain unchanged when training on HPC.

Use HPC if measured cost prevents finishing the four families with repeated seeds, if new sessions multiply the training set, or if longer history/global context is scientifically justified by the pilot. Prefer independent single-GPU jobs for trials and seeds; multi-GPU training is unnecessary initially. Use job arrays and a shared database only with an appropriate concurrent backend; do not have multiple nodes write a local SQLite file.

Estimate requested GPU hours from actual seconds per update/epoch and the surviving experiment matrix, with a stated margin for preprocessing and evaluation. Start with a small allocation and checkpoint on scheduler termination. Access to HPC is not authorization to submit jobs from this planning task.

## 8. Ablation plan

Run ablations against fixed selected recipes on the same split and seed set. Change one factor at a time unless a specific interaction is the question. Retrain when a feature or architectural component is removed; zeroing a feature only at test time measures distribution shift, not a training ablation.

| Priority | Comparison | Question / control |
| --- | --- | --- |
| Required | L1 vs G1; LT vs GT | Does full-scan context improve regional prediction? Include a wider local parameter-matched control when practical |
| Required | L1 vs LT; G1 vs GT | Does causal history help with the same target frames and preprocessing? |
| Required | Geometry/counts vs geometry + intensity | How much signal comes from reflectivity? |
| Required | With vs without absolute sensor-frame position; retain range | Is the stationary model memorizing smoke location? Validate on relocated/new-session smoke |
| Required | 0.05 / 0.10 / 0.20 m voxels | Resolution, mixed-label voxels, speed and accuracy tradeoff |
| Required | T=1 / 3 / 5 / 10 | Benefit, compute and behavior across temporal horizons |
| Required for moving temporal claim | Unregistered vs pose-aligned history | Does physical alignment justify its runtime dependency? |
| Secondary | Tags/line/point-time summaries added separately | Useful sensor information or dataset-specific shortcut? |
| Secondary | Weighted BCE vs unweighted BCE; focal if needed | How does imbalance handling change detection and probability quality? |
| Secondary | No calibration vs fixed post-hoc calibration | Does calibration improve held-out probability quality? |
| Secondary | Single-domain vs mixed-domain training | Transfer and possible negative transfer between acquisitions |
| Secondary | Neural temporal model vs causal smoothing | Learned temporal structure versus simple filtering |
| Optional | Point refinement, covariance features, global pooling vs attention | Which extra costs buy meaningful improvement? |

The laptop runs the four core cells and a few cheap feature controls first. The final campaign covers the required rows with at least three seeds where feasible, after estimating cost. Do not form the full Cartesian product. Select representative settings using development data and predeclare the final reporting subset.

Treat label-policy variants as a separate dataset-version sensitivity experiment, not a feature ablation. Any relabeling must be independent of held-out model predictions and regenerate affected hashes/splits.

## 9. Evaluation, calibration and uncertainty

### 9.1 Common evaluation

Evaluate complete target scans with natural class prevalence, original point order and no positive subsampling. Streaming accumulators can compute confusion/Brier metrics; save score/label arrays or sorted shards for exact AP/ROC computation. Approximate binned curves are allowed during HPO only if labeled approximate and checked against exact finalist metrics.

Report per domain, per recording/session, and pooled:

- Smoke AP as the primary ranking metric, plus PR curves; distinguish AP from trapezoidal PR-AUC.
- ROC-AUC/ROC curves as required by the proposal, with class prevalence beside them.
- Smoke precision, recall, F1, IoU and confusion counts at a fixed selection-derived threshold.
- Macro summaries across independent sessions when they become available, as well as point-weighted pooled metrics.
- Clean-control false-positive point rate, false positives per frame, and the distribution of falsely removed point fractions. If defining an alarm, freeze its minimum count/fraction and duration before testing.
- Results by fixed range bands, density/occupancy bands, motion condition, and reviewed hard-negative subsets where available.
- Raw and calibrated Brier score, unweighted binary log loss, reliability diagrams and a clearly specified binning scheme. Show class-conditional errors and prediction histograms because majority negatives dominate pooled Brier/ECE.
- Runtime validity coverage, annotation coverage, ignored counts, startup exclusions and any unevaluated points.

For voxel reporting, use a fixed evaluation grid (initially 0.10 m) independent of each model's internal resolution. Aggregate point predictions on that grid over the same nonignored, runtime-valid points used to calculate the target fraction. Report error against the smoke fraction `q`; for binary voxel F1/PR define positive as `q >= 0.5`, require at least one labeled point, and show mixed-voxel/support counts. This scoring mask is evaluation-only; published region summaries still use runtime-valid returns without labels. Boundary-sensitive alternative definitions may be supplementary but must not replace the common metric to favor a model.

For negative-only clean subsets, ROC-AUC and positive AP/recall are undefined: show false-positive metrics and explicit `N/A` values. For slices with zero predicted positives, define metric conventions and retain the raw counts.

### 9.2 Calibration and operating threshold

Fit a monotone sigmoid calibrator `p = sigmoid(a*z + b)`, with positive slope, on the independent calibration partition after the checkpoint is frozen. Use unweighted calibration loss at natural prevalence. A bias term is useful because class weighting/resampling can shift logits. Temperature-only scaling is a useful lower-complexity comparison; avoid isotonic calibration initially given the small number of independent episodes.

Use one pooled calibrator by default and report its performance per domain. A different calibrator per domain would require identifying that domain at runtime and is not the default deployment contract. Calibration under a changed smoke prevalence or new environment is an empirical question, not guaranteed by the fitting procedure. The [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html) describes independent calibration and warns that Brier/log loss also reflect discrimination; [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) provide the temperature-scaling reference.

Maintain two selection-derived operating points: maximum smoke F1 for comparability and maximum recall subject to a declared false-positive ceiling for deployment. The acceptable false-removal rate on clean structure remains to be set with the eventual application. Until then, show several fixed ceilings, for example 0.1%, 0.5% and 1%, as sensitivity results rather than claiming any is safe. Store thresholds in probability space with the exact calibrator version; never optimize them on test.

### 9.3 Statistical and label uncertainty

Three seeds measure training variation; they do not create three independent datasets. For final confidence intervals, use a paired bootstrap at the independent acquisition-group/session level and compare models on the same sampled groups. With current data, a paired block bootstrap over predeclared time blocks can describe within-recording variation, but must not be presented as uncertainty over new environments. Avoid point-wise bootstrapping millions of correlated returns.

For pseudo-label evaluation, call the outcome agreement with the reviewed labels. Before final model predictions are inspected, manually adjudicate a separate small stratified audit set containing clean structure, smoke, boundaries and hard negatives across the held-out groups. Record uncertainty and label corrections independently of model preference. A case-enriched audit set needs its own sampling description/weights; it cannot silently replace natural-prevalence test metrics.

All-zero/all-prior controls must appear beside probability metrics. A model that scores well only in stationary familiar geometry should not be described as generally smoke-aware.

### 9.4 Qualitative artifacts

Export PLY/NPZ predictions with XYZ, original point ID, smoke/reliability probability, predicted class and ground truth when available. Render fixed examples covering positive smoke, negative structure, the person near the source, voxel boundaries, sparse/far returns, and temporal startup/reset. Create short sequences showing plume entry and clearing, with identical color scales across models. Include representative failures rather than selecting only attractive maps.

## 10. Deployment contract and latency

The offline trainer should not depend on ROS. Export a bundle containing architecture/config, weights, feature order, normalization, runtime-validity rules, voxel size, history/pose policy, calibration, thresholds, expected fields/units, software versions and a compact model card with data/split hashes.

Expose a shared predictor API such as `predict_frame(frame, timestamp, optional_pose) -> probabilities, valid_mask, diagnostics`. Return one score per original point in original order. Unsupported/invalid outputs need an explicit validity field; never represent lack of evidence as reliability 1. Keep annotation ignore masks out of the live API.

The ROS 2 adapter subscribes to PointCloud2, converts available fields into this contract, and publishes reliability with matching timestamps/frame IDs. Stationary training originated from Livox CustomMsg while GrandTour used PointCloud2; test both conversion routes against the canonical arrays. If intensity/tag/line/time fields are missing, use a model declared compatible with those missing fields or a defined fallback. Do not silently fabricate metadata for a model that requires it.

At the observed 10 Hz rate, an initial acceptance target is end-to-end p95 latency below 100 ms, with a preferred engineering target below 50 ms for headroom. These are provisional until the deployment CPU and scan-rate requirement are known. Measure conversion, voxelization/features, alignment, model, inverse mapping, calibration and serialization separately and together; include queueing and p99/max latency. A fast neural forward pass alone is insufficient.

Benchmark batch size one on the actual low-power target, with a fixed CPU thread budget, representative full-size scans and sustained replay long enough to expose thermal throttling. Record RAM, peak working memory, startup latency and dropped frames. Use bounded queues/history and reset appropriately after drops. Past-only temporal history adds no future-frame wait; its warm-up and reaction/smoothing behavior still need measurement.

Start with CPU float32. Consider optimized export/ONNX, quantization or a compiled feature extractor only if profiling identifies a bottleneck. Recheck probabilities, calibration and task metrics after any numerical conversion; do not assume export support for dynamic sparse operations. Distillation from a stronger global teacher to a small local student is optional after a real accuracy/latency gap is demonstrated, using training data only.

A ROS 2 node is the proposal's integration deliverable, but full SLAM modification and navigation-performance claims are outside this trainer build. The export/replay interface should make node development straightforward later.

## 11. Proposed code and artifact layout

```text
trainer/
  pyproject.toml
  README.md
  configs/
    data.toml
    splits/pilot_temporal_v1.toml
    models/{local,global,local_temporal,global_temporal}.toml
    experiments/{pilot,comparison,ablations,hpo}.toml
  src/smoke_trainer/
    cli.py
    config.py
    data/{schema,index,cache,splits,windows,sampling}.py
    features/{voxel,statistics,normalization,alignment}.py
    models/{registry,local,global,temporal}.py
    training/{engine,losses,checkpoint,search}.py
    evaluation/{metrics,calibration,thresholds,bootstrap,report}.py
    inference/{predictor,bundle,replay}.py
  tests/
data/training_cache/             # ignored, derived and replaceable
experiments/splits/              # small frozen plans and reports
runs/<experiment_id>/            # ignored checkpoints/predictions/logs
reports/<experiment_id>/         # figures and concise experiment report
```

The filenames indicate responsibilities, not a requirement to create dozens of nearly empty files immediately. Keep the existing labeler independent. Reuse its validated schema behavior through a small shared contract or an explicitly tested adapter; avoid a second drifting interpretation of NPZ fields and units.

Every run should record a UUID, Git revision and dirty-tree patch/hash, full resolved config, dependency lock/environment, dataset/cache/split hashes, feature/model version, seeds, sampling/exposure counts, device, timing, checkpoint selection, calibration provenance and failure status. Use small local JSON/CSV artifacts and TensorBoard optionally. Git history alone is insufficient when the working tree already contains uncommitted changes.

### Proposed command flow

These commands describe the future interface; they do not exist yet.

```bash
smoke-train inspect --manifest data/training/unified_v1/dataset_manifest.json
smoke-train cache --config trainer/configs/data.toml
smoke-train split --config trainer/configs/splits/pilot_temporal_v1.toml
smoke-train verify-split --plan experiments/splits/pilot_temporal_v1/split_plan.json
smoke-train fit --config trainer/configs/models/local.toml --split-plan <plan>
smoke-train study --config trainer/configs/experiments/pilot.toml --budget-hours 4
smoke-train compare --experiment <id> --partition selection
smoke-train calibrate --run <run_id> --partition calibration
smoke-train freeze-evaluation --experiment <id>
smoke-train evaluate --frozen-plan <path> --partition test
smoke-train export --run <run_id> --output <bundle_path>
smoke-train replay --bundle <bundle_path> --manifest <manifest_path>
```

The frozen evaluation plan lists checkpoint/config hashes and operating points. Its purpose is reproducibility and preventing accidental tuning on test; it need not impose a repeated interactive approval prompt.

## 12. Implementation sequence and acceptance criteria

| Phase | Build | Done when |
| --- | --- | --- |
| 1. Contract/index | Ingest, validate, index, exact time sidecar and bounded cache | All nine chunks account for 5,081 frames/76,782,092 points; corruption/unsupported schema fail clearly; memory stays bounded |
| 2. Splits/windows | Pilot grouped-stream split, strict grouped final split, causal windows | Frozen counts are reproducible; no forbidden frame/group overlap; boundary/gap/reset tests pass |
| 3. Features/baselines | Voxel statistics, inverse mapping, normalization and baselines | Hand-constructed mixed/ignored/singleton cases behave correctly; features do not depend on labels; train-only normalization is proved |
| 4. First trainer | L1, masked point-equivalent loss, checkpoint/resume, metrics | Fits a tiny deliberately simple fixture; ignored-label changes do not affect gradients; run resumes correctly; full-frame predictions align |
| 5. Required model matrix | G1, LT, GT, common comparison | All four pass the same I/O tests and run on identical target IDs; causal-history behavior and global-context access are demonstrated |
| 6. Experiment tooling | Budget scheduler, HPO, ablations, calibration, reports | Four-hour cap is honored with resumable status; trial/fold preprocessing stays isolated; no test access during studies |
| 7. Pose study and deployment | Optional pose sidecar/alignment, bundle/replay, runtime profiling | Synthetic rigid-motion alignment is correct; missing/late-pose fallback works; offline and streaming outputs agree |
| 8. Thesis campaign | New grouped data, repeated seeds, frozen test, qualitative analysis | Required comparisons have data/compute provenance and uncertainty; claims match actual held-out groups and target latency |

Suggested engineering order is phases 1–4 first, followed by a laptop pilot to determine how aggressively to scope phases 5–7. Training compute can be hours; implementing and validating the full system is a separate multi-stage engineering effort. Do not imply that all eight phases will be implemented in four hours.

### Essential tests

- Schema/manifest mismatch, corrupt checksum, duplicate frame and invalid pointer/time detection.
- An ARC-6 chunk boundary with valid continuity and another with a synthetic gap; no crossing of clean/smoke recordings or split boundaries.
- A future-frame perturbation has no effect on a current prediction; online history reset matches offline window behavior.
- Ignored labels have zero loss/gradient contribution and never change inference features; all-ignore targets are skipped safely without division by zero.
- Mixed-voxel aggregated loss equals its expanded per-point calculation, including positive weighting.
- A held-out outlier does not change training normalization or sampling weights.
- Original point order/count survives voxelization, batched prediction and inverse mapping.
- Point/voxel metrics on tiny known examples, including negative-only data and no predicted positives.
- Calibration/threshold provenance rejects training/test data in the wrong roles; fitted calibration remains tied to its checkpoint.
- Streaming and exported predictor probabilities agree within stated numerical tolerances, including startup and missing-field cases.

Run synthetic/unit tests plus a small real-data integration test before expensive campaigns. Do not rerun the entire dataset for every low-impact documentation change.

## 13. Remaining decisions and recommended next milestone

The user has settled current-data development, future independent data collection, hours-scale laptop training and lightweight real-time deployment. These details remain open without blocking phases 1–4:

| Decision | Working assumption | Resolve by |
| --- | --- | --- |
| Laptop GPU/VRAM/driver and usable RAM | CPU-capable implementation; automatic profiling; sequential jobs | Inspect the actual training machine before the pilot |
| Final CPU/device and required scan rate | 10 Hz, p95 <100 ms, aim <50 ms; batch one | Obtain a representative target before choosing the deployed winner |
| Whether pose is a runtime dependency | LiDAR-only default; pose-assisted research comparison | Compare benefit and end-to-end cost, then choose with the user |
| Cost of falsely rejecting clean structure vs missing smoke | F1 operating point plus several clean-FP ceilings | Agree with the downstream use before final threshold selection |
| Which new sessions/environments can be collected | Approximately 8–12 episodes, ≥3 environments and multiple days | Make a collection/grouping plan before freezing final test data |

The first implementation milestone is a working manifest-to-L1 experiment with verified splits, full-frame metrics, calibration provenance and replay timing. It should already produce the same artifacts the later global and temporal models use. That provides an early deployable candidate and tests the experimental foundation before the more expensive architecture work.
