"""Offline planar multi-hypothesis registration experiment for car recordings."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from scipy.integrate import trapezoid
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .bag import _custom_msg_cdr_arrays
from .car import _fresh_directory, _sha256
from .car_registration import _OdomInterpolator, _deskew_to_start, _load_initial_pose, _write_overlay
from .geometry import transform_points
from .provenance import write_run_provenance


def _estimate_start_pose(scans, tree, map_xyz):
    """Fit a level starting pose from several wheel-aligned scans."""
    _, _, _, start_odom = scans[0]
    local = np.concatenate([
        transform_points(np.linalg.solve(start_odom, odom), points[::8])
        for _, _, points, odom in scans[:61:15]
    ])
    bounds = list(zip(*np.quantile(map_xyz[:, :2], [.01, .99], axis=0)))
    bounds.append((-np.pi, np.pi))

    def transform(parameters):
        x, y, yaw = parameters
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
        pose[:2, 3] = [x, y]
        return pose

    def score(parameters):
        d = tree.query(transform_points(transform(parameters), local))[0]
        return np.minimum(d, .45).mean()

    fits = [differential_evolution(score, bounds, seed=seed, popsize=12, maxiter=110,
                                   tol=.002, polish=False) for seed in (17, 31, 42)]
    best = min(fits, key=lambda fit: fit.fun)
    return transform(best.x), [
        {"pose_xy_yaw": fit.x.tolist(), "clipped_mean_distance_m": float(fit.fun)}
        for fit in fits
    ]


def align(prepared, recording_id, reference, initial_pose, output, keyframe_stride=10, wheel_mount_yaw_deg=0.0):
    prepared, reference, output = map(Path, (prepared, reference, output))
    initial_pose = None if initial_pose is None else Path(initial_pose)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    recording = next(r for r in manifest['recordings'] if r['id'] == recording_id)
    reference_info = json.loads((reference / 'reference_manifest.json').read_text())
    if reference_info['layout'] != recording['layout']:
        raise ValueError('Reference layout does not match recording')
    for source in recording['source_files']:
        if _sha256(Path(source['path'])) != source['sha256']:
            raise ValueError('Recording has changed since preparation')
    with np.load(reference / 'reference.npz') as data:
        map_xyz = data['xyz'].astype(float)
    tree = cKDTree(map_xyz)
    with np.load(prepared / 'recordings' / recording_id / 'motion.npz') as data:
        motion = {key: data[key] for key in data.files}
    motion["T_base_lidar"][:3, :3] = (
        motion["T_base_lidar"][:3, :3] @ Rotation.from_euler("z", wheel_mount_yaw_deg, degrees=True).as_matrix()
    )
    if initial_pose is None:
        covered = np.flatnonzero(motion["full_odom_coverage"])
        if not len(covered):
            raise ValueError("No LiDAR frames with full odometry coverage")
        start_index = int(covered[0])
    else:
        start_index, start_pose, _ = _load_initial_pose(initial_pose)
    odom = _OdomInterpolator(motion)
    start_time = int(motion['frame_time_ns'][start_index])
    scans = []
    index = -1
    for source in recording['source_files']:
        with Path(source['path']).open('rb') as stream:
            for _, _, message in make_reader(stream, validate_crcs=True).iter_messages(topics=['/livox/lidar']):
                index += 1
                if index < start_index or not motion['full_odom_coverage'][index]:
                    continue
                xyz, _, _, _, offsets, timestamp = _custom_msg_cdr_arrays(message.data, point_stride=12)
                if int(timestamp) != int(motion['frame_time_ns'][index]):
                    raise ValueError('LiDAR timestamps differ from prepared motion data')
                ranges = np.linalg.norm(xyz, axis=1)
                valid = np.isfinite(xyz).all(axis=1) & (ranges >= .3) & (ranges <= 25)
                xyz = _deskew_to_start(xyz[valid], offsets[valid], int(timestamp), odom)
                scans.append((index, int(timestamp), xyz, odom.pose(int(timestamp))))
    if not scans:
        raise ValueError('No covered scans')
    start_fits = None
    if initial_pose is None:
        start_pose, start_fits = _estimate_start_pose(scans, tree, map_xyz)
    layout_odom = start_pose @ np.linalg.inv(odom.pose(start_time))
    scans = [(index, timestamp, xyz, layout_odom @ pose) for index, timestamp, xyz, pose in scans]
    key_indices = list(range(0, len(scans), keyframe_stride))
    if key_indices[-1] != len(scans) - 1:
        key_indices.append(len(scans) - 1)
    # Follow wheel motion and accept only small, well-supported map corrections.
    # The limits are exploratory and the output remains subject to review.
    corrections = [np.eye(4)]
    key_reasons = ['automatic_start_review' if initial_pose is None else 'reviewed_start']
    gyro_t = (motion['imu_time_ns'].astype(np.int64) - start_time) * 1e-9
    gyro_z = motion['imu_angular_velocity_raw'][:, 2]
    for k in range(1, len(key_indices)):
        j = key_indices[k]
        previous_j = key_indices[k - 1]
        frame, timestamp, points, base = scans[j]
        previous_time = scans[previous_j][1]
        previous_base = scans[previous_j][3]
        predicted = corrections[-1] @ base
        sample = points[::max(1, len(points) // 400)]
        elapsed = (timestamp - previous_time) * 1e-9
        first = (previous_time - start_time) * 1e-9
        last = (timestamp - start_time) * 1e-9
        times = np.r_[first, gyro_t[(gyro_t > first) & (gyro_t < last)], last]
        gyro_turn = trapezoid(np.interp(times, gyro_t, gyro_z), times)
        wheel_turn = Rotation.from_matrix(previous_base[:3, :3].T @ base[:3, :3]).as_euler('xyz')[2]
        gyro_agrees = abs(np.arctan2(np.sin(gyro_turn - wheel_turn), np.cos(gyro_turn - wheel_turn))) < .15 + .1 * elapsed

        def pose(x):
            result = predicted.copy()
            result[:3, :3] = Rotation.from_euler('z', x[2]).as_matrix() @ predicted[:3, :3]
            result[:2, 3] += x[:2]
            return result

        def score(x):
            distance = tree.query(transform_points(pose(x), sample), workers=1)[0]
            return float(np.mean(np.minimum(distance, .75)))

        limit = min(.35, .08 + .18 * elapsed)
        angle_limit = min(.12, .03 + .06 * elapsed)
        seeds = [np.zeros(3), np.array([limit / 2, 0, 0]), np.array([-limit / 2, 0, 0])]
        fits = [minimize(score, seed, method='Powell',
                         bounds=[(-limit, limit), (-limit, limit), (-angle_limit, angle_limit)],
                         options={'maxiter': 18, 'xtol': .01, 'ftol': .002}) for seed in seeds]
        best = min(fits, key=lambda fit: fit.fun)
        distances = tree.query(transform_points(pose(best.x), sample), workers=1)[0]
        azimuth = np.arctan2(sample[:, 1], sample[:, 0])
        sectors = np.floor((azimuth + np.pi) * 8 / (2 * np.pi)).astype(int).clip(0, 7)
        supported = sum(np.count_nonzero((sectors == n) & (distances < .25)) >= 20 for n in range(8))
        if gyro_agrees and supported >= 3 and np.mean(distances < .25) >= .5:
            corrections.append(pose(best.x) @ np.linalg.inv(base))
            key_reasons.append('map_match_review')
        else:
            corrections.append(corrections[-1].copy())
            key_reasons.append('wheel_only_gyro_disagreement' if not gyro_agrees else 'wheel_only_weak_map')
        print(f'keyframe {frame}: {key_reasons[-1]}, supported sectors {supported}', flush=True)
    from scipy.spatial.transform import Slerp
    key_times = np.array([scans[j][1] for j in key_indices], dtype=np.int64)
    seconds = (key_times - key_times[0]) * 1e-9
    rotations = Slerp(seconds, Rotation.from_matrix(np.array(corrections)[:, :3, :3]))
    _fresh_directory(output)
    write_run_provenance(output, {'keyframe_stride': keyframe_stride, 'seed': 42,
                                  'wheel_mount_yaw_deg': wheel_mount_yaw_deg},
        {'command': 'car-align', 'recording': recording_id, 'reference': str(reference.resolve()),
         'initial_pose': str(initial_pose.resolve()) if initial_pose else None,
         'reference_sha256': _sha256(reference / 'reference.npz'),
         'motion_sha256': _sha256(prepared / 'recordings' / recording_id / 'motion.npz')})
    if start_fits is not None:
        (output / 'initial_pose_automatic.json').write_text(json.dumps({
            'recording': recording_id, 'frame_index': start_index, 'T_layout_lidar': start_pose.tolist(),
            'fits': start_fits, 'status': 'automatic_candidate',
        }, indent=2) + '\n')
    rows, poses = [], []
    for frame, timestamp, points, base in scans:
        t = (timestamp - key_times[0]) * 1e-9
        correction = np.eye(4)
        correction[:3, :3] = rotations([t]).as_matrix()[0]
        correction[:3, 3] = [np.interp(t, seconds, np.array(corrections)[:, axis, 3]) for axis in range(3)]
        current = correction @ base
        world = transform_points(current, points)
        d = tree.query(world, workers=1)[0]
        key_pos = min(np.searchsorted(key_times, timestamp, side='right') - 1, len(key_times) - 1)
        rows.append({'frame_index': frame, 'frame_time_ns': timestamp, 'median_m': np.median(d), 'p90_m': np.percentile(d, 90), 'within_0_20': np.mean(d < .2), 'status': 'review', 'pose_source': key_reasons[key_pos]})
        poses.append(current)
        if frame in (1, 76, 150, 264, 410):
            _write_overlay(output / f'overlay_frame_{frame:06d}.ply', map_xyz, world, d)
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output / 'trajectory.npz', frame_index=[r['frame_index'] for r in rows], frame_time_ns=[r['frame_time_ns'] for r in rows], T_layout_lidar=poses)
    steps = np.linalg.norm(np.diff(np.asarray(poses)[:, :3, 3], axis=0), axis=1)
    summary = {'recording': recording_id, 'frames': len(rows),
               'median_fraction_within_0_20': float(np.median([r['within_0_20'] for r in rows])),
               'maximum_adjacent_translation_m': float(steps.max(initial=0)),
               'status': 'experimental_review_required'}
    (output / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared'); parser.add_argument('--recording', required=True)
    parser.add_argument('--reference', required=True); parser.add_argument('--initial-pose')
    parser.add_argument('--output', required=True); parser.add_argument('--keyframe-stride', type=int, default=10)
    parser.add_argument('--wheel-mount-yaw-deg', type=float, default=0.0)
    args = parser.parse_args()
    if args.keyframe_stride < 1:
        parser.error('keyframe stride must be positive')
    print(json.dumps(align(args.prepared, args.recording, args.reference, args.initial_pose, args.output,
                           args.keyframe_stride, args.wheel_mount_yaw_deg), indent=2))

if __name__ == '__main__':
    main()
