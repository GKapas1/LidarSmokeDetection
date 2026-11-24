"""Offline joint surface registration for the prepared hall MCAP recordings.

Run with ``python -m smoke_labeler.car_joint_align --output DIRECTORY``.
Calibration and absolute anchors remain candidates until independently reviewed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from scipy.integrate import trapezoid
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from .bag import _custom_msg_cdr_arrays
from .car import _sha256, validate_transform
from .car_registration import _OdomInterpolator, _deskew_to_start, _write_overlay
from .car_ray_review import query_directional_ranges
from .geometry import transform_points


def balanced_indices(xyz, line, offsets, maximum, seed):
    """Stratify across laser, azimuth and time, preserving original point IDs."""
    valid = np.flatnonzero(np.isfinite(xyz).all(axis=1)
                           & (np.linalg.norm(xyz, axis=1) >= .3)
                           & (np.linalg.norm(xyz, axis=1) <= 25))
    az = np.floor((np.arctan2(xyz[valid, 1], xyz[valid, 0]) + np.pi) * 8 / (2*np.pi)).clip(0, 7).astype(int)
    time = np.floor(offsets[valid] * 50).clip(0, 4).astype(int)
    cells = line[valid].astype(int) * 40 + az * 5 + time
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(valid))
    ordered = permutation[np.argsort(cells[permutation], kind='stable')]
    groups = np.split(ordered, np.flatnonzero(np.diff(cells[ordered])) + 1)
    quota = max(1, maximum // len(groups))
    chosen = np.concatenate([group[:quota] for group in groups])
    remaining = np.setdiff1d(np.arange(len(valid)), chosen, assume_unique=False)
    if len(chosen) < maximum and len(remaining):
        chosen = np.r_[chosen, rng.choice(remaining, min(maximum-len(chosen), len(remaining)), replace=False)]
    return np.sort(valid[chosen[:maximum]])


def floor_plane(xyz, seed=42):
    sample = xyz[(xyz[:, 2] > -.8) & (xyz[:, 2] < -.09)
                 & (np.linalg.norm(xyz[:, :2], axis=1) > .7)
                 & (np.linalg.norm(xyz[:, :2], axis=1) < 7)]
    if len(sample) < 80:
        return None
    rng = np.random.default_rng(seed)
    best = np.zeros(len(sample), bool)
    for _ in range(400):
        p = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(p[1]-p[0], p[2]-p[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        if abs(normal[2]) < .94:
            continue
        inside = np.abs((sample-p[0]) @ normal) < .025
        if inside.sum() > best.sum():
            best = inside
    if best.sum() < 80:
        return None
    center = sample[best].mean(axis=0)
    _, _, vh = np.linalg.svd(sample[best]-center, full_matrices=False)
    normal = vh[-1] * np.sign(vh[-1, 2])
    return normal, float(-normal @ center)


def surface_map(reference):
    """Use measured local patches, with a fixed spatial holdout before fitting."""
    xyz = reference.astype(float)
    tree = cKDTree(xyz)
    distance, neighbors = tree.query(xyz, k=16)
    local = xyz[neighbors]
    centered = local-local.mean(axis=1, keepdims=True)
    eig, vec = np.linalg.eigh(centered.transpose(0, 2, 1) @ centered / 16)
    keep = (distance[:, -1] < .45) & (eig[:, 0] < .0016) & (eig[:, 0] < .08*eig[:, 1]) & (eig[:, 1] > .0001)
    center = local.mean(axis=1)[keep]
    normals = vec[:, :, 0][keep]
    tiles = np.floor(center / .8).astype(int)
    holdout = ((tiles[:, 0]*73856093 ^ tiles[:, 1]*19349663 ^ tiles[:, 2]*83492791) % 5) == 0
    return center, normals, holdout


def corrected_floor_pose(pose, points, plane):
    measured = floor_plane(points)
    if measured is None:
        return pose.copy(), False
    normal, offset = measured
    target, target_offset = plane
    world_normal = pose[:3, :3] @ normal
    cross = np.cross(world_normal, target)
    angle = np.arctan2(np.linalg.norm(cross), np.dot(world_normal, target))
    rotation = Rotation.from_rotvec(cross / max(np.linalg.norm(cross), 1e-12)*angle).as_matrix()
    result = pose.copy()
    result[:3, :3] = rotation @ pose[:3, :3]
    # The floor normal at the lidar origin has signed distance offset.
    result[:3, 3] += target * (offset-target_offset-target @ pose[:3, 3])
    return result, True


def load_recording(prepared, record, yaw, initial, maximum=1400):
    with np.load(prepared/'recordings'/record['id']/'motion.npz') as data:
        motion = dict(data)
    motion['T_base_lidar'] = motion['T_base_lidar'] @ np.block([[Rotation.from_euler('z', yaw, degrees=True).as_matrix(), np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
    odom = _OdomInterpolator(motion)
    frames = []
    index = -1
    for source in record['source_files']:
        if _sha256(Path(source['path'])) != source['sha256']:
            raise ValueError('Source hash mismatch')
        with open(source['path'], 'rb') as stream:
            for _, _, msg in make_reader(stream, validate_crcs=True).iter_messages(topics=['/livox/lidar']):
                index += 1
                if not motion['full_odom_coverage'][index]:
                    continue
                x, _, _, line, offset, stamp = _custom_msg_cdr_arrays(msg.data)
                if stamp != int(motion['frame_time_ns'][index]):
                    raise ValueError('Timestamp mismatch')
                select = balanced_indices(x, line, offset, maximum, 7919+index)
                frames.append(dict(index=index, stamp=stamp, raw=x[select], offset=offset[select],
                                   lines=np.bincount(line[select], minlength=4),
                                   points=_deskew_to_start(x[select], offset[select], stamp, odom)))
    first = frames[0]['stamp']
    base = initial @ np.linalg.inv(odom.pose(first))
    return dict(id=record['id'], motion=motion, odom=odom, frames=frames,
                base=base, first=first, initial=initial)


def calibrate_yaw(run, offsets_deg=None):
    """Upper-wall scan pairs, disjoint alternating pairs for cross-checking."""
    frames = run['frames']
    scores = []
    angles = np.arange(-60, 61, 5) if offsets_deg is None else np.asarray(offsets_deg)
    for k in range(2, len(frames)-5, 12):
        a, b = frames[k], frames[k+5]
        rel = np.linalg.solve(run['odom'].pose(a['stamp']), run['odom'].pose(b['stamp']))
        if np.linalg.norm(rel[:2, 3]) < .14:
            continue
        upper = a['points'][a['points'][:, 2] > .5]
        source = b['points'][b['points'][:, 2] > .5]
        if min(len(upper), len(source)) < 80:
            continue
        tree = cKDTree(upper)
        costs = []
        for angle in angles:
            r = Rotation.from_euler('z', angle, degrees=True).as_matrix()
            transformed = (source @ r.T @ rel[:3, :3].T + rel[:3, 3]) @ r
            costs.append(np.minimum(tree.query(transformed)[0], .5).mean())
        scores.append(costs)
    scores = np.asarray(scores)
    if len(scores) < 4:
        raise ValueError('Too few motion pairs for calibration')
    return dict(pairs=len(scores), offsets_deg=angles.tolist(),
                mean_cost_m=scores.mean(axis=0).tolist(),
                best_all_odd_even=[int(angles[np.argmin(s.mean(axis=0))]) for s in (scores, scores[::2], scores[1::2])],
                status='automatic_motion_calibration_candidate')


def gyro_timing(motion):
    origin = motion['odom_time_ns'][0]
    wheel_time = (motion['odom_time_ns']-origin)*1e-9
    imu_time = (motion['imu_time_ns']-origin)*1e-9
    yaw = np.unwrap(Rotation.from_quat(motion['odom_quaternion_xyzw']).as_euler('xyz')[:, 2])
    rate = np.gradient(yaw, wheel_time)
    inside = (wheel_time > .3) & (wheel_time < wheel_time[-1]-.3)
    offsets = np.arange(-.15, .151, .01)
    errors = [float(np.sqrt(np.mean((rate[inside]-np.interp(wheel_time[inside]+offset, imu_time, motion['imu_angular_velocity_raw'][:, 2]))**2))) for offset in offsets]
    best = int(np.argmin(errors))
    return dict(offset_s=float(offsets[best]), rmse=errors[best], zero_rmse=errors[len(offsets)//2])


def interpolate(times, poses, query):
    seconds = (times-times[0])*1e-9
    q = np.clip((np.asarray(query)-times[0])*1e-9, seconds[0], seconds[-1])
    result = np.repeat(np.eye(4)[None], len(q), axis=0)
    result[:, :3, :3] = Slerp(seconds, Rotation.from_matrix(poses[:, :3, :3]))(q).as_matrix()
    result[:, :3, 3] = np.column_stack([np.interp(q, seconds, poses[:, j, 3]) for j in range(3)])
    return result


def trajectory_seed(previous, initial, index):
    """Retain drift corrections while honoring a replacement absolute anchor."""
    return initial @ np.linalg.inv(previous[0]) @ previous[index]


def directional_residuals(reference, points, origins, offsets):
    residuals = np.full(len(points), np.nan)
    bins = np.floor(offsets / .005).astype(int)
    for value in np.unique(bins):
        take = bins == value
        residuals[take] = query_directional_ranges(reference, points[take], origins[take].mean(axis=0))[1]
    return residuals


def solve(runs, centers, normals, holdout, floor, iterations):
    fit_tree = cKDTree(centers[~holdout])
    fit_centers, fit_normals = centers[~holdout], normals[~holdout]
    nodes = []
    for run in runs:
        selected = list(range(0, len(run['frames']), 3))
        if selected[-1] != len(run['frames'])-1:
            selected.append(len(run['frames'])-1)
        run['node_indices'] = selected
        run['node_start'] = len(nodes)
        for k in selected:
            frame = run['frames'][k]
            pose = trajectory_seed(run['old']['T_layout_lidar'], run['initial'], k)
            if run['yaw'] != 180.:
                # Compare calibrated motion with the previous map trajectory.
                # Keep a map-supported seed rather than discarding previous fixes.
                candidate = run['base'] @ run['odom'].pose(frame['stamp'])
                costs = []
                for trial in (pose, candidate):
                    world = transform_points(trial, frame['points'])
                    distance, near = fit_tree.query(world)
                    plane_error = abs(np.sum((world-fit_centers[near])*fit_normals[near], axis=1))
                    costs.append(np.mean(np.minimum(plane_error, .2) + np.minimum(distance, .5)))
                if costs[1] < costs[0]:
                    pose = candidate
            pose, found = corrected_floor_pose(pose, frame['points'], floor)
            nodes.append(dict(run=run, frame=frame, pose=pose, floor=found))
    poses = np.array([n['pose'] for n in nodes])
    edges = []
    for run in runs:
        begin = run['node_start']
        for k in range(len(run['node_indices'])-1):
            i, j = begin+k, begin+k+1
            a, b = nodes[i]['frame']['stamp'], nodes[j]['frame']['stamp']
            relative = np.linalg.solve(run['odom'].pose(a), run['odom'].pose(b))
            t = run['motion']['imu_time_ns']
            # Compare gyro(t + offset) with wheel(t), as measured in gyro_timing.
            seconds = (t-a)*1e-9-run['gyro_offset_s']
            duration = (b-a)*1e-9
            knots = np.r_[0, seconds[(seconds>0)&(seconds<duration)], duration]
            gyro = run['motion']['imu_angular_velocity_raw'][:, 2]
            turn = trapezoid(np.interp(knots, seconds, gyro), knots)
            edges.append((i, j, relative, turn, duration))
    # Joint endpoint constraint is requested only for the verified same-stop pair.
    if len(runs) == 2:
        i = runs[1]['node_start']-1
        edges.append((i, runs[1]['node_start'], np.eye(4), 0., 0.))
    history = []
    for iteration in range(iterations):
        associations = []
        for run in runs:
            begin = run['node_start']
            end = begin+len(run['node_indices'])
            stamps = np.array([n['frame']['stamp'] for n in nodes[begin:end]], dtype=np.int64)
            corrections = np.array([poses[i] @ np.linalg.inv(run['base'] @ run['odom'].pose(nodes[i]['frame']['stamp'])) for i in range(begin, end)])
            stamps = np.r_[stamps, stamps[-1]+200_000_000]
            corrections = np.concatenate([corrections, corrections[-1:]])
            for i in range(begin, end):
                frame = nodes[i]['frame']
                point_times = frame['stamp']+np.rint(frame['offset']*1e9).astype(np.int64)
                acquisition = interpolate(stamps, corrections, point_times) @ run['base'] @ run['odom'].poses(point_times)
                relative = np.linalg.inv(poses[i]) @ acquisition
                frame['fit_points'] = np.einsum('nij,nj->ni', relative[:, :3, :3], frame['raw'])+relative[:, :3, 3]
        for i, node in enumerate(nodes):
            p = node['frame']['fit_points']
            world = transform_points(poses[i], p)
            d, near = fit_tree.query(world)
            residual = np.abs(np.sum((world-fit_centers[near])*fit_normals[near], axis=1))
            keep = (d < .45) & (residual < .16)
            # Bound influence of each spatial patch, avoiding domination by floor.
            valid = np.flatnonzero(keep)
            tiles = np.floor(fit_centers[near[valid]]/.35).astype(int)
            _, unique = np.unique(tiles, axis=0, return_index=True)
            valid = valid[unique][:500]
            associations.append((p[valid], fit_centers[near[valid]], fit_normals[near[valid]]))
        count = sum(len(p) for p, _, _ in associations)+len(edges)*7+6
        sparsity = lil_matrix((count, len(nodes)*6), dtype=int)
        row = 0
        for i, (p, _, _) in enumerate(associations):
            sparsity[row:row+len(p), i*6:(i+1)*6] = 1
            row += len(p)
        for i, j, _, _, _ in edges:
            sparsity[row:row+7, i*6:(i+1)*6] = 1
            sparsity[row:row+7, j*6:(j+1)*6] = 1
            row += 7
        sparsity[row:, :6] = 1
        anchor = poses[0].copy()

        def unpack(delta):
            x = delta.reshape(-1, 6)
            current = poses.copy()
            current[:, :3, 3] += x[:, :3]
            current[:, :3, :3] = Rotation.from_rotvec(x[:, 3:]).as_matrix() @ poses[:, :3, :3]
            return current

        point_nodes = np.repeat(np.arange(len(nodes)), [len(p) for p, _, _ in associations])
        local_points = np.concatenate([p for p, _, _ in associations])
        target_centers = np.concatenate([c for _, c, _ in associations])
        target_normals = np.concatenate([n for _, _, n in associations])
        edge_i = np.array([e[0] for e in edges])
        edge_j = np.array([e[1] for e in edges])
        edge_motion = np.array([e[2] for e in edges])
        edge_turn = np.array([e[3] for e in edges])
        edge_duration = np.array([e[4] for e in edges])
        edge_sigma = np.where(edge_duration == 0, .005, .008+.02*edge_duration)
        translation_sigma = np.column_stack((edge_sigma, edge_sigma, np.full(len(edges), .05)))

        def residuals(delta):
            current = unpack(delta)
            world = np.einsum('nij,nj->ni', current[point_nodes, :3, :3], local_points)+current[point_nodes, :3, 3]
            error = np.sum((world-target_centers)*target_normals, axis=1)/.04
            magnitude = np.abs(error)
            map_error = np.sign(error)*np.sqrt(np.where(magnitude <= 1, error**2, 2*magnitude-1))
            inverse_rotation = current[edge_i, :3, :3].transpose(0, 2, 1)
            observed_rotation = inverse_rotation @ current[edge_j, :3, :3]
            observed_translation = np.einsum('nij,nj->ni', inverse_rotation, current[edge_j, :3, 3]-current[edge_i, :3, 3])
            translation_error = (observed_translation-edge_motion[:, :3, 3])/translation_sigma
            rotation_error = Rotation.from_matrix(edge_motion[:, :3, :3].transpose(0, 2, 1) @ observed_rotation).as_rotvec()/.035
            yaw = np.arctan2(observed_rotation[:, 1, 0], observed_rotation[:, 0, 0])
            turn_error = np.arctan2(np.sin(yaw-edge_turn), np.cos(yaw-edge_turn))/(.01+.03*edge_duration)
            motion_error = np.column_stack((translation_error, rotation_error, turn_error)).ravel()
            anchor_translation = (current[0, :3, 3]-anchor[:3, 3])/.12
            anchor_rotation = Rotation.from_matrix(anchor[:3, :3].T @ current[0, :3, :3]).as_rotvec()/.03
            return np.concatenate((map_error, motion_error, anchor_translation, anchor_rotation))

        print('Solving batch', iteration, 'with', len(local_points), 'surface returns', flush=True)
        fit = least_squares(residuals, np.zeros(len(nodes)*6), jac_sparsity=sparsity.tocsr(),
                            loss='linear', max_nfev=12, ftol=.002, xtol=.001,
                            tr_options={'maxiter': 50, 'atol': 1e-3, 'btol': 1e-3}, x_scale='jac',
                            bounds=(np.tile([-.18]*3+[-.05]*3, len(nodes)), np.tile([.18]*3+[.05]*3, len(nodes))))
        poses = unpack(fit.x)
        history.append(dict(iteration=iteration, cost=float(fit.cost), correspondences=count-len(edges)*7-6,
                            max_translation_update_m=float(np.max(np.linalg.norm(fit.x.reshape(-1, 6)[:, :3], axis=1)))))
        print(history[-1], flush=True)
        updates = fit.x.reshape(-1, 6)
        if (iteration >= 3 and np.max(np.linalg.norm(updates[:, :3], axis=1)) < .002
                and np.max(np.linalg.norm(updates[:, 3:], axis=1)) < .0005):
            break
    for i, (points, _, normal) in enumerate(associations):
        rotated = points @ poses[i, :3, :3].T
        jacobian = np.column_stack((normal, np.cross(rotated, normal)/5))
        eigenvalues = np.linalg.eigvalsh(jacobian.T @ jacobian)
        nodes[i]['surface_eigenvalues'] = eigenvalues.tolist()
        nodes[i]['weak_geometry'] = len(points) < 60 or eigenvalues[0] < max(eigenvalues[-1], 1e-9)*1e-4
        nodes[i]['last_translation_update_m'] = float(np.linalg.norm(updates[i, :3]))
        nodes[i]['last_rotation_update_rad'] = float(np.linalg.norm(updates[i, 3:]))
    return nodes, poses, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/raw_car/prepared/20261113_v1'))
    parser.add_argument('--reference', type=Path, default=Path('data/raw_car/prepared/references/hall_v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=60)
    parser.add_argument('--anchors', type=Path, help='JSON mapping recording IDs to reviewed first-frame 4x4 poses')
    parser.add_argument('--gyro-offset-s', type=float, help='Gyro(t+offset) matches wheel(t), estimated on clean drive 1 by default')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'solver_used.py').write_text(Path(__file__).read_text())
    if args.iterations < 1:
        raise ValueError('iterations must be positive')
    manifest = json.loads((args.prepared/'manifest.json').read_text())
    reference = np.load(args.reference/'reference.npz')['xyz']
    station = np.load(args.prepared/'recordings/hall_clean_stationary1/stationary.npz')['xyz']
    floor = floor_plane(station)
    if floor is None:
        raise ValueError('Stationary floor unavailable')
    centers, normals, holdout = surface_map(reference)
    review = json.loads((args.reference/'alignment_used.json').read_text())
    station_support = []
    station_origins = []
    station_ids = []
    for entry in review['references']:
        source = np.load(args.prepared/entry['cloud'])['xyz']
        transform = validate_transform(entry['T_layout_lidar'])
        transformed = transform_points(transform, source)
        station_support.append(cKDTree(transformed).query(centers)[0] < .12)
        station_origins.append(transform[:3, 3])
        station_ids.append(entry['id'])
    np.savez_compressed(args.output/'surface_map.npz', xyz=centers, normals=normals,
                        holdout=holdout, station_support=np.array(station_support).T,
                        station_origins=station_origins, station_ids=station_ids)
    map_tree = cKDTree(reference)
    held_tree = cKDTree(centers[holdout])
    report = dict(status='automatic_alignment_candidates_require_landmark_review',
                  reference_sha256=_sha256(args.reference/'reference.npz'),
                  floor_normal=floor[0].tolist(), floor_offset_m=floor[1],
                  training_patches=int((~holdout).sum()), holdout_patches=int(holdout.sum()), recordings={})
    anchors = json.loads(args.anchors.read_text()) if args.anchors else {}
    gyro_offset = args.gyro_offset_s
    for group in ('clean', 'smoke'):
        runs = []
        for number in (1, 2):
            rid = f'hall_{"clean_moving" if group=="clean" else "smoke"}{number}'
            record = next(r for r in manifest['recordings'] if r['id']==rid)
            if group == 'clean':
                baseline = args.prepared.parent/'registration/hall_yaw_validation_20261121'/rid
            else:
                baseline = args.prepared.parent/'registration'/(rid+'_auto_v1')
            old = dict(np.load(baseline/'trajectory.npz'))
            initial = validate_transform(anchors.get(rid, old['T_layout_lidar'][0]))
            run = load_recording(args.prepared, record, 180., initial)
            calibration = calibrate_yaw(run)
            timing = gyro_timing(run['motion'])
            if gyro_offset is None:
                gyro_offset = timing['offset_s'] if timing['rmse'] < .8*timing['zero_rmse'] else 0.
            # A shared candidate is tested across both smoky recordings, never estimated per frame.
            if group == 'smoke' and number == 1:
                session_yaw = 180. + calibration['best_all_odd_even'][0]
            yaw = session_yaw if group == 'smoke' else 180.
            if yaw != 180.:
                run = load_recording(args.prepared, record, yaw, initial)
            run.update(old=old, calibration=calibration, yaw=yaw, gyro_offset_s=gyro_offset, timing_check=timing, source_files=record['source_files'], baseline_sha256=_sha256(baseline/'trajectory.npz'))
            runs.append(run)
            print(rid, calibration, flush=True)
        nodes, poses, history = solve(runs, centers, normals, holdout, floor, args.iterations)
        report[group+'_optimization'] = history
        for run in runs:
            folder = args.output/run['id']
            folder.mkdir()
            begin = run['node_start']
            end = begin+len(run['node_indices'])
            (folder/'geometry_quality.json').write_text(json.dumps([
                dict(frame_index=n['frame']['index'], surface_eigenvalues=n['surface_eigenvalues'], weak_geometry=bool(n['weak_geometry']),
                     last_translation_update_m=n['last_translation_update_m'], last_rotation_update_rad=n['last_rotation_update_rad'])
                for n in nodes[begin:end]], indent=2)+'\n')
            node_times = np.array([n['frame']['stamp'] for n in nodes[begin:end]], dtype=np.int64)
            correction = np.array([poses[i] @ np.linalg.inv(run['base'] @ run['odom'].pose(nodes[i]['frame']['stamp'])) for i in range(begin, end)])
            frames = run['frames']
            # Extend last correction through the final scan; wheel data still cover all returns.
            times = np.r_[node_times, node_times[-1]+200_000_000]
            correction = np.concatenate([correction, correction[-1:]])
            trajectory = []
            rows = []
            baseline_motion = dict(run['motion'])
            baseline_motion['T_base_lidar'] = run['motion']['T_base_lidar'].copy()
            baseline_motion['T_base_lidar'][:3, :3] = run['motion']['T_base_lidar'][:3, :3] @ Rotation.from_euler('z', 180-run['yaw'], degrees=True).as_matrix()
            baseline_odom = _OdomInterpolator(baseline_motion)
            for k, frame in enumerate(frames):
                stamp = frame['stamp']
                c = interpolate(times, correction, [stamp])[0]
                pose = c @ run['base'] @ run['odom'].pose(stamp)
                trajectory.append(pose)
                point_times = stamp+np.rint(frame['offset']*1e9).astype(np.int64)
                point_poses = interpolate(times, correction, point_times) @ run['base'] @ run['odom'].poses(point_times)
                world = np.einsum('nij,nj->ni', point_poses[:, :3, :3], frame['raw'])+point_poses[:, :3, 3]
                previous = run['old']['T_layout_lidar'][k]
                old_points = _deskew_to_start(frame['raw'], frame['offset'], stamp, baseline_odom)
                old_point_poses = previous @ np.linalg.inv(baseline_odom.pose(stamp)) @ baseline_odom.poses(point_times)
                old_world = transform_points(previous, old_points)
                row = dict(frame_index=frame['index'], frame_time_ns=stamp, points=len(world), floor_detected=floor_plane(frame['points']) is not None)
                bracket = min(np.searchsorted(node_times, stamp, side='right')-1, end-begin-1)
                supporting_nodes = nodes[begin+bracket:min(begin+bracket+2, end)]
                row['local_fit_unsettled'] = any(n['last_translation_update_m'] > .01 or n['last_rotation_update_rad'] > .002 for n in supporting_nodes)
                for tag, xyz in [('baseline', old_world), ('joint', world)]:
                    d = map_tree.query(xyz)[0]
                    h, near = held_tree.query(xyz)
                    supported = h < .35
                    p = np.abs(np.sum((xyz-centers[holdout][near])*normals[holdout][near], axis=1))
                    row.update({tag+'_median_m':float(np.median(d)), tag+'_within20':float((d<.2).mean()), tag+'_held_support':float(supported.mean()), tag+'_held_p95_m':float(np.percentile(p[supported],95)) if supported.any() else None})
                if k%20==5:
                    for tag, xyz, origins in [('baseline',old_world,old_point_poses[:,:3,3]),('joint',world,point_poses[:,:3,3])]:
                        ray = directional_residuals(reference, xyz, origins, frame['offset'])
                        row[tag+'_ray_support'] = float(np.isfinite(ray).mean())
                        row[tag+'_ray_p95_m'] = float(np.nanpercentile(abs(ray),95)) if np.isfinite(ray).any() else None
                        row[tag+'_ray_early50'] = float((ray<-.5).sum()/max(1,np.isfinite(ray).sum()))
                rows.append(row)
                if k in (0,len(frames)//2,len(frames)-1):
                    _write_overlay(folder/f'overlay_{frame["index"]:06d}.ply', reference, world, map_tree.query(world)[0])
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (folder/'metrics.csv').open('w',newline='') as out:
                writer = csv.DictWriter(out, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            np.savez_compressed(folder/'trajectory.npz',frame_index=[f['index'] for f in frames],frame_time_ns=[f['stamp'] for f in frames],T_layout_lidar=trajectory, correction_time_ns=times, T_layout_odom_correction=correction, T_base_lidar=run['motion']['T_base_lidar'], T_layout_odom=run['base'])
            summary=dict(source_files=run['source_files'], baseline_sha256=run['baseline_sha256'], frames=len(frames), gyro_offset_s=run['gyro_offset_s'], timing_check=run['timing_check'], wheel_mount_yaw_deg=run['yaw'],calibration=run['calibration'], line_counts=np.sum([f['lines'] for f in frames],axis=0).tolist())
            for field in fields:
                if field.startswith(('baseline_', 'joint_')):
                    values=[r[field] for r in rows if r.get(field) is not None]
                    summary[field]=float(np.median(values)) if values else None
            report['recordings'][run['id']]=summary
            print(run['id'],summary,flush=True)
        a=poses[runs[1]['node_start']-1];b=poses[runs[1]['node_start']]
        report[group+'_junction']=dict(translation_m=float(np.linalg.norm(a[:3,3]-b[:3,3])),rotation_deg=float(np.degrees(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude())))
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'effective_config.json').write_text(json.dumps(vars(args),default=str,indent=2)+'\n')


if __name__ == '__main__':
    main()
