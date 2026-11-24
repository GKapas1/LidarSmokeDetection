# Preparing the November RC-car recordings

The new `car-prepare` workflow reads your ROS 2 MCAPs directly on the laptop.
It does not need ROS installed, a running car, or a repaired `metadata.yaml`.
It prepares reference alignment and motion review. Section 6 includes an
experimental automatic labeling and export workflow for the moving hall bags.

The stationary `dataset` command assumes an unchanged LiDAR pose and must not be
used on these drives. GrandTour's commands assume its paired raw/deskewed clouds
and DLIO poses; those are not present in the car bags. Your neural network can
still consume raw single scans or short windows. The accurate geometry is needed
offline to create trustworthy supervision.

## 1. Prepare the data once

From the repository root:

```bash
labeler/.venv/bin/smoke-label car-prepare \
  data/raw_car/smoke \
  --output data/raw_car/prepared/20261113_v1
```

This output was created on 13 November 2026. Open `REPORT.md` there first.
For a new preparation run, use a new output directory: the command refuses to
overwrite existing reviews. The 20 hall/corridor/tree recordings are included;
unrelated old test directories are listed as ignored in `manifest.json`.

No raw directories need renaming. Canonical IDs map the `staionary` typo and
the trailing `ű` to readable names while retaining original source paths and
MCAP SHA-256 checksums. In particular, renaming a ROS bag's MCAP without updating
its metadata would break ROS replay. The empty corridor-smoke-2 metadata does
not prevent this workflow: its messages are read from the MCAP itself. If ROS
replay is needed later, repair/reindex a copy with compatible ROS tooling.

The derived layout is:

```text
20261113_v1/
  REPORT.md
  manifest.json
  effective_config.json
  run_provenance.json
  recordings/<canonical_id>/
    audit.json
    motion.npz
    stationary.ply       # stationary recordings only
    stationary.npz
    frame_000000.ply      # moving recordings: three individual scans
    frame_XXXXXX.ply
    frame_YYYYYY.ply
  layouts/<hall|corridor|tree>/alignment.json
```

Stationary clouds use every fifth scan and every fourth range-valid point
(0.3–30 m), reduced to 4 cm voxels seen in at least three distinct sampled scans.
This is a candidate clean surface map, not proof that everything in it is static:
a person standing still or the car itself may survive the filter. Inspect/remove
those regions. Moving previews are separate single scans; do not merge them in
raw sensor coordinates and interpret the result as a room map. Reflectivity is a
PLY scalar field; use the same display scale when comparing recordings.

`motion.npz` preserves odometry translation/quaternions, raw IMU measurements,
integer nanosecond timestamps, frame point counts, per-scan maximum point offsets,
the measured mount transform, and full odometry coverage flags. It is an audit
and future pose-processing input, not a training dataset. Per-point XYZ, intensity,
tag, line and offsets remain intact in the original MCAPs.

## 2. Align the stationary references, one layout at a time

Start with **hall**. In CloudCompare open:

```text
recordings/hall_clean_stationary1/stationary.ply
recordings/hall_clean_stationary2/stationary.ply
recordings/hall_clean_stationary3/stationary.ply
```

1. Keep stationary1 fixed. Its LiDAR coordinate system defines the layout frame.
2. Align stationary2 to stationary1 using corresponding corners/structural
   features; refine if useful. Check walls and vertical features as well as the
   floor. A low residual on the floor alone does not constrain yaw/translation.
3. Align stationary3 into the same fixed frame. Prefer surfaces visible in both
   clouds, and exclude people, the car, reflective artifacts and changing objects.
4. Save the **total** 4×4 transform from each original LiDAR cloud to the fixed
   layout frame in `layouts/hall/alignment.json`, under `T_layout_lidar`.
   Each matrix is a JSON list of four rows. For stationary1 retain identity.
   Set each `alignment_reviewed` to `true` only after reviewing it.
5. If several transforms were applied successively, compose them in order:
   `T_total = T_last @ ... @ T_first`. A final ICP correction alone is not the
   complete original-to-layout transform. Keep global shift and scaling disabled
   for these small, metre-scale clouds.

The files also contain `unverified_odom_guess` for a possible initial placement.
It is **not** accepted as an alignment: wheel odometry alone previously produced
poor cross-view agreement. The exported stationary clouds are in original LiDAR
coordinates, so do not add the 12 cm mount offset again to a transform obtained
by aligning these clouds. The mount is needed when mapping through `base_link`.

Exclusions can be entered in each reference's `exclude_boxes_sensor_frame`:

```json
{"min_m": [-0.5, -0.5, -0.4], "max_m": [0.5, 0.5, -0.1]}
```

These numbers are an illustration, not a recommended car crop. Set bounds from
the actual cloud, in that original sensor frame. Excluding a region from the
reference does **not** authorize labeling unmatched smoke-run points there as
smoke; uncertain target regions must also be ignored later.

Repeat independently for corridor (four viewpoints) and tree (three). Do not
combine changed room layouts into one clean reference.

## 3. Record the rebuilt tree layout explicitly

The tree and smoke machine were rebuilt between clean and smoke runs. Similar
placement is useful but does not establish centimetre-level correspondence.
Use unchanged walls/floor/other fixed structures for alignment, and identify a
region covering both the old and new tree/machine positions. That region needs
target-side exclusion (`255`) or explicit review of the actual object returns
during labeling. Never classify the whole moved-object region as smoke merely
because it is distant from the old reference.

`layout_matches_smoke_confirmed` starts false. Set it true only after reviewing
unchanged geometry and documenting any differences and the required exclusions
in `notes`. For the tree layout, leave it false until that review is complete.
This flag is a human assessment; the preparation code cannot verify the layout
from bag names or reflectivity statistics.

## 4. Build each reviewed reference

After the hall review:

```bash
labeler/.venv/bin/smoke-label car-reference \
  data/raw_car/prepared/20261113_v1 \
  --layout hall \
  --output data/raw_car/prepared/references/hall_v1
```

Use `corridor` and `tree` with their own output directories later. This validates
the transforms, rejects unreviewed/missing alignments and modified source clouds,
applies exclusions, and writes `reference.ply`, `reference.npz`, a copy of the
alignment settings and reference hashes. Inspect the merged PLY for doubled
walls and misaligned box edges. The generated reference is still conditional on
the review and is not evidence that a moving trajectory is accurate.

## 5. Validate moving geometry before producing labels

Run a bounded clean-control pilot after the reference alignments are saved. The
initial pose file records one manually reviewed raw-scan-to-layout transform. In
the hall recording, frame 0 precedes wheel-odometry coverage, so the reviewed
pose is attached to frame 1 only because the audit confirms that the car remained
stationary across those first two frames.

```bash
labeler/.venv/bin/smoke-label car-register-clean \
  data/raw_car/prepared/20261113_v1 \
  --recording hall_clean_moving1 \
  --reference data/raw_car/prepared/references/hall_v1 \
  --initial-pose data/raw_car/prepared/20261113_v1/recordings/hall_clean_moving1/initial_pose_reviewed.json \
  --frames 150 \
  --output data/raw_car/prepared/registration/hall_clean_moving1_pilot_v1
```

This interpolates wheel odometry at each point timestamp for within-scan motion
compensation, propagates the reviewed starting pose, and applies bounded trimmed
scan-to-map ICP. It writes `trajectory.npz`, per-frame `metrics.csv`, and three
CloudCompare overlays. In an overlay, scalar field `source=0` is the reference
and `source=1` is the registered scan; `nearest_map_distance_m` is diagnostic.
Inspect coincident walls and corners in the start, middle and end overlays.
Unmapped but real surfaces may have a large distance and must not become smoke
labels merely because they are absent from the stationary reference.

- Estimate a pose for each moving scan against the appropriate clean map, using
  wheel odometry as an initial guess and observed fixed structure for correction.
  Use smoke-resistant registration; retain diagnostics and reject poor fits.
- Correct motion within each roughly 100 ms scan using per-point times and a
  validated trajectory. IMU data are measurements, not an already-corrected pose.
  Do not extrapolate beyond recorded trajectory coverage or bridge large gaps.
- Inspect beginning, middle and end overlays, especially during sharp turns and
  dense smoke. One rigid transform for an entire drifting drive is insufficient.
- Keep clean moving bags out of stationary-map construction. Use them to measure
  false smoke detections, uncovered regions and registration failures. Select
  thresholds from development clean controls; retain independent controls when
  making evaluation claims.
- Only create smoke candidates after those checks. Distance from a clean map
  alone also detects moved objects, occlusions, people and pose errors. Unsupported
  geometry and ambiguous cases should receive `255`, with reviewed smoke regions
  and hard negatives recorded separately.

The November IMU samples have acceleration magnitude around 1 at rest, consistent
with g units. The [Livox driver publishes the SDK acceleration values directly](https://github.com/Livox-SDK/livox_ros_driver2/blob/master/src/lddc.cpp).
Confirm units before a processor expecting m/s² (conversion is approximately
9.81 per g); do not feed these values unchanged into such a processor. Raw IMU
orientation can be constant and must not be assumed to be a fused attitude.
Preparation flags these facts and does not alter raw measurements or integrate
them into a trajectory.

Zero-range returns exist in these bags (about 25% of recorded points in hall
smoke 2). The audit counts them, and the preview range filter removes them.
They need the same runtime-validity filter and ignore handling during training;
their absence of a measured location does not make them positive smoke points.
All bags also have one or two boundary scans lacking full recorded odometry
coverage; `full_odom_coverage` identifies them for later rejection.

### 5.1 Directional review of the hall ghost returns

The reviewed overlays contain faint displaced wall copies in clean frames 1 and
150. Do not expand the reference with these returns based on recurrence alone.
The exact physical cause is unresolved. A second decoder (`rosbags`, using the
embedded MCAP schema) reproduced the XYZ, offsets and timebase of frames 1, 76
and 150 exactly, ruling out disagreement with the custom CDR reader on these
frames. Seeing the artifact in a raw frame does not by itself establish multipath
or exclude every acquisition/timing issue.

Run the bounded diagnostic using the existing registered poses:

```bash
labeler/.venv/bin/python -m smoke_labeler.car_ray_review \
  --prepared data/raw_car/prepared/20261113_v1 \
  --recording hall_clean_moving1 \
  --reference data/raw_car/prepared/references/hall_v1 \
  --registration data/raw_car/prepared/registration/hall_clean_moving1_pilot_v1 \
  --frames 1 76 150 \
  --output data/raw_car/prepared/registration/hall_clean_moving1_ray_review_v1
```

That output already exists. Choose a new directory to rerun. Open
`frame_000150_review.ply` in CloudCompare and enable RGB colors (disable scalar
field display if it overrides colors). The map is blue-gray; scan points are
green near mapped surfaces, magenta beyond them, orange before them, and gray
when uncertain. Categories are diagnostic and **not class labels**. The NPZs
retain original point indices and raw sensor XYZ along with review coordinates.

This prototype uses per-point motion poses, approximating the sensor origin in
5 ms groups (under 1.7 mm maximum error in these three frames). It compares up to
eight map neighbors within 0.75 degrees and requires four with at most 0.25 m
depth spread. Residuals within 0.20 m are near-reference; differences beyond
0.50 m are before/behind. These are exploratory settings, not calibrated label
thresholds. Sparse directions and mixed depths remain uncertain. Angular map
neighbors do not prove an opaque wall: openings, glass, edges and changed objects
still require review. This is not a mesh ray-tracing or certified occlusion test.

The pilot found 805/183/1540 before-reference candidates in clean frames
1/76/150 respectively, and 691/19/1366 behind-reference candidates. Most points
remain uncertain under the strict support rule. The colored frame-150 preview
places much of the external duplicate in the behind-reference category, but
also shows structured before-reference discrepancies elsewhere. Because these
are clean controls, the before-reference candidates demonstrate that the rule
is not yet suitable as an automatic smoke labeler. They may include map/pose
errors, uncovered objects and sensor artifacts; review must establish the cause.

Frame 150's before- and behind-reference points are not independent effects.
Their median acquisition offsets are 36.59 ms and 36.41 ms, and both have a
second concentration near the end of the roughly 100 ms frame. Contributions
are distributed across all four laser lines. Visually and temporally, the orange
interior band and magenta exterior band are opposite sides of the same displaced
scan portions. Treating only the magenta points as artifacts would leave false
orange smoke candidates.

Exploratory 10 ms slice registration did not provide a physically credible
repair. The worst 90--100 ms slice requested about 0.77 m vertical translation
and 5.6 degrees of pitch while still retaining a 0.33 m median nearest-map
distance. The RC car cannot undergo that motion within 10 ms. Do not write these
independent slice fits into the trajectory. For labeling, reject the affected
geometry at the sub-scan or frame level until a sensor/timestamp cause and a
validated correction are established.

A constant LiDAR-to-odometry clock offset was tested from -0.5 to +0.5 s in
0.1 s steps, with deskewing and registration repeated for frame 150. Median map
distance remained 0.133--0.134 m, the 90th percentile remained 2.237--2.242 m,
and the fraction within 0.20 m remained 0.589--0.590. A fixed clock shift does
not repair the displaced component. This does not rule out corrupted individual
point times or non-rigid sensor effects, but it rules out the simplest constant
clock-offset explanation for this frame.

Next validation sequence:

1. Review the orange and magenta bands as one displaced acquisition pattern.
   Preserve the original map and raw points. Correct only demonstrable global
   pose/map mistakes; mark failed sub-scans or frames unknown rather than fitting
   an unconstrained independent pose to each time slice.
2. Extend registration and QC over the clean development drive. Measure false
   candidates **and coverage**; ignoring every difficult point is not success.
   Reserve another clean drive for validation rather than tuning on both.
3. Review a short smoke segment at per-point level. Before-reference distance
   alone cannot distinguish smoke from people, moved objects or artifacts.
   Accept positives only with supported geometry and reviewed smoke evidence;
   preserve ambiguous points as `255`. Do not assume all behind-wall returns in
   smoke are non-smoke: smoke itself can perturb measured ranges.
4. Scale to all runs only after both checks pass. Export raw frame identity and
   coordinates; retain geometry/review evidence as teacher-only metadata.

No additional CloudCompare alignment of the existing clean references is called
for solely by this artifact. The current pilot is not a validated labeled dataset.

### 5.2 Full clean-drive frame audit

The complete `hall_clean_moving1` trajectory and conservative frame audit are in:

```text
data/raw_car/prepared/registration/hall_clean_moving1_full_v1/
data/raw_car/prepared/registration/hall_clean_moving1_audit_v3/
```

The audit processed frames 1--463; frame 0 lacks full odometry coverage. It
flagged 129 keep candidates, 217 review candidates and 117 reject candidates.
The safest contiguous ranges for initial temporal work are 23--101 and 281--329.
See `AUDIT_REPORT.md`, `audit_timeline.png`, and
`recommended_frame_policy.json` in the audit directory. The conservative policy
excludes frame 1, frames 106--214 and 240--268; all other non-keep ranges require
review. These are candidate geometry flags, not class labels, and raw data remain
unchanged. Validate the rule on `hall_clean_moving2` before using it automatically
on smoke recordings.

## 6. Review labels and export with original frame identity

### Automatic hall pseudo-labels

The automatic labeler combines directional clean-map comparison with raw
reflectivity and consecutive-scan support. A positive must be at least 60 cm
shorter than a supported clean-map range, at least 35 cm from a mapped point,
and have raw reflectivity at most 5. Both adjacent scans must have at least
three such returns within 30 cm. The frame must pass the map-match check.
For smoky recordings, only closely matched, brighter background returns get
class `0`. Uncertain points, including poorly registered frames, get `255`.
For known clean recordings, valid returns get `0` and any fires of the same
positive rule are counted in the audit. The optional clean-drive veto removes
positive candidates within 30 cm of false detections from the development
clean drive. Use a different clean drive to measure how the veto generalizes.

From the repository root, register each smoke drive. The 180-degree wheel
mount yaw is the existing hall alignment hypothesis. The solver estimates a
level starting pose and saves the alternative fits for review.

```bash
root=data/raw_car/prepared
for recording in hall_smoke1 hall_smoke2; do
  labeler/.venv/bin/python -m smoke_labeler.car_align "$root/20261113_v1" \
    --recording "$recording" --reference "$root/references/hall_v1" \
    --wheel-mount-yaw-deg 180 --output "$root/registration/${recording}_auto_v1"
done
```

Generate all four chunks with the same labeling rule. Use a fresh output
directory for each run.

```bash
labeler/.venv/bin/python -m smoke_labeler.car_label "$root/20261113_v1" \
  --reference "$root/references/hall_v1" --legacy-wheel-mount-yaw-deg 180 \
  --clean-veto-recording hall_clean_moving1 \
  --registration hall_clean_moving1 "$root/registration/hall_yaw_validation_20261121/hall_clean_moving1" \
  --registration hall_clean_moving2 "$root/registration/hall_yaw_validation_20261121/hall_clean_moving2" \
  --registration hall_smoke1 "$root/registration/hall_smoke1_auto_v1" \
  --registration hall_smoke2 "$root/registration/hall_smoke2_auto_v1" \
  --output "$root/labels/hall_pseudo_v3"
```

`dataset_manifest.json` uses the trainer's unified format. `label_audit.json`
and the per-recording CSVs show coverage and candidate counts after each
stage. The peak-frame PLYs show the mapped scene in blue and candidates in
orange for inspection, with original MCAP point indices on the candidates.
Treat the smoke positives as pseudo-labels for development. A clean-drive
false-positive count does not measure recall or establish precision in smoke.
The low-reflectivity rule favors dim smoke, so evaluation needs independently
labeled examples, including brighter smoke.

The current hall map has reviewed alignments and a confirmed layout match.
Corridor still needs reviewed reference alignment and a clean-to-smoke layout
check. Tree needs those checks plus exclusion or review of the moved objects.

The car labeling/export adapter uses the existing class contract:
`0` unimpacted, `1` smoke-impacted, `255` ignore. Keep original raw sensor-frame
XYZ, point order, frame boundaries, reflectivity and per-point times as model
inputs; keep map coordinates/distances, pose quality and review regions as
teacher-only metadata. Deskewed world coordinates support labeling, not an
unannounced change to the predictor inputs.

The smoke impacts returned points; attenuation or missing returns do not directly
provide a positive point label. Lower reflectivity is evidence to inspect, not a
label threshold by itself.

### Using raw reflectivity during a manual review

You can inspect individual frames before a moving trajectory is available. For
example, from the repository root, export three consecutive hall smoke frames:

```bash
labeler/.venv/bin/python labeler/scripts/review_intensity.py \
  data/raw_car/smoke/mid360_hall_smoke1 --frames 149 150 151 --cutoff 5 \
  --output data/raw_car/prepared/diagnostics/hall_smoke1_review
```

In CloudCompare, first inspect the full scan with uniform coloring, then enable
RGB. Amber highlights raw reflectivity at or below the chosen cutoff. Gray shows
the other returns. Inspect one frame at a time in its own sensor coordinates.
Review the same structure in the neighboring scans and compare against a clean
drive with the same display settings. Also inspect gray regions for plumes.
The cutoff of 5 reproduces the pilot visualization. Dark walls, edges and other
real surfaces can also appear amber.

For a selected region, retain the recording, `frame_index` and
`source_point_index` with the review decision. The PLY preserves these scalar
fields for every measured point from 0.3 to 25 m. The accompanying
`review.json` records MCAP hashes, timestamps and counts. Use supported clean-map
geometry and manual review when assigning `1` to smoke-impacted points or `0`
to unimpacted points. Unreviewed and ambiguous points stay `255`. Keep the
highlight flags and reference-map evidence in the offline review artifacts.

Keep whole recordings together when splitting, with temporal windows confined to
each recording and split. These layouts were collected in one hall/session:
holding out a layout is useful development evidence but not an independent-day
or new-building generalization test. Leave splits unassigned until the usable
labels, geometry failures and positive-class coverage are known.

## Joint hall surface registration

The offline batch experiment runs from the repository root.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 labeler/.venv/bin/python \
  -m smoke_labeler.car_joint_align \
  --output data/raw_car/prepared/registration/hall_joint_new
```

Use a new output directory for each run. The solver stratifies measured points
by laser line, azimuth and acquisition time, preserves timestamp identity, and
recomputes motion compensation between batch iterations. It uses finite local
surface neighborhoods from the existing reviewed hall map, a deterministic
spatial holdout, floor initialization, wheel displacement and gyro yaw factors.
The two clean drives and two smoky drives are fitted in their respective pairs,
with their matching parked endpoints connected by a soft relative-pose factor.
Only surface factors receive robust outlier weights. Motion factors retain
quadratic penalties. The batch solver stops on small pose updates after at least
four passes, with a default limit of sixty passes.
Automatic smoky motion calibration is estimated on drive 1 and applied to both
smoky drives. Drive 2 and alternating scan pairs provide calibration checks.
Relative gyro timing is estimated on clean drive 1 and frozen for all drives.
`--gyro-offset-s 0` runs the timing ablation. The recorded estimate compares
gyro(t + offset) with wheel(t), and is an effective relative timing parameter.

Outputs include source hashes, a saved solver snapshot, map surface normals and
station support, continuous pose corrections, per-frame metrics, geometry
conditioning diagnostics, and individual scan/map overlays. `trajectory.npz`
stores the actual mount transform and the time-varying correction used for
per-point acquisition poses. Use both when projecting raw returns.

The default starting poses come from the previous candidate trajectories. Supply
`--anchors PATH.json` with a dictionary mapping recording IDs to reviewed 4x4
first-covered-frame transforms to replace those anchors. All transforms use
column vectors and metres. The first covered frames are 1, 2, 1 and 1 for clean
1, clean 2, smoke 1 and smoke 2 respectively.

The spatial holdout excludes patches from the new fit. Its map still comes from
the same stationary recordings used by previous initial trajectories, so this is
an internal consistency check. Surface support and directional ray support are
reported with their errors. Sparse angular nearest-neighbor ray tests retain
visibility and sampling ambiguities. Gyro and wheel residuals have exploratory
weights. Eigenvalues measure geometry conditioning and do not establish calibrated
pose uncertainty. Automatic anchors and mount estimates require independent
landmark review before treating trajectories as verified labeling supervision.

The joint registration CSV includes `local_fit_unsettled`. A final supporting
node update above 1 cm or 0.002 rad flags its interpolation interval for review.
These numerical stability flags accompany the global automatic-anchor status.
