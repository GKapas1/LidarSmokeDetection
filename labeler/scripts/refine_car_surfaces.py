"""Build a versioned car surface-label pilot from registered scan archives.

Run from the repository root. The input archives are the same November review
artifacts used to create car_binary_v4. Class changes retain original point IDs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from smoke_labeler.car_joint_align import interpolate
from smoke_labeler.car_registration import _OdomInterpolator
from smoke_labeler.object_surfaces import (
    fit_panels, fit_trunk, floor_coordinates as coordinates,
    panel_evidence, revise_labels, trunk_evidence,
)
from smoke_labeler.unified_dataset import _sha256, _write_chunk, load_unified_chunk

SETTINGS = dict(
    version="car_surface_v2", time_block_s=1.0, panel_anchor_intensity_min=4,
    panel_roi_min=[-5.7, -3.5, .08], panel_roi_max=[-.4, .2, 1.6],
    smoke_roi_min=[-5.8, -3.6, .12], smoke_roi_max=[-.3, .4, 2.5],
    trunk_seed_min=[-1.95, -1.49, .10], trunk_seed_max=[-1.82, -1.31, .55],
    trunk_center_bounds=[[-1.97, -1.51], [-1.80, -1.29]],
    smoke_cluster_radius_m=.20, smoke_minimum_neighbors=3,
    temporal_window_s=.31, temporal_radius_m=.25,
    smoke_policy="Only tracks connected to user-reviewed smoke points become new label 1. Other tracks are review candidates.",
)
RECORDINGS = [
    "corridor_clean_moving1", "corridor_smoke1", "corridor_smoke2",
    "tree_clean_moving1", "tree_clean_moving2", "tree_smoke1",
]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def inside(points, low, high):
    return np.all((points >= low) & (points <= high), axis=1)


def scan_origins(folder, prepared, drive):
    with np.load(folder / drive / "trajectory.npz") as file:
        trajectory = dict(file)
    with np.load(prepared / "20261113_v1/recordings" / drive / "motion.npz") as file:
        motion = dict(file)
    motion["T_base_lidar"] = trajectory["T_base_lidar"]
    odom = _OdomInterpolator(motion)
    with np.load(folder / drive / "floor_corrections.npz") as file:
        floor_correction = dict(file)

    def origins(stamp, offsets):
        times = int(stamp) + np.rint(offsets * 1e9).astype(np.int64)
        poses = (interpolate(trajectory["correction_time_ns"],
                             trajectory["T_layout_odom_correction"], times)
                 @ trajectory["T_layout_odom"] @ odom.poses(times))
        correction = interpolate(floor_correction["correction_time_ns"],
                                 floor_correction["T_floor_layout_saved_layout"], times)
        return (correction @ poses)[:, :3, 3]

    return origins


def load_scans(prepared, drive, arrays):
    scene = drive.split("_")[0]
    folder = prepared / "diagnostics/other_scenes_v3" / scene
    floor = json.loads((folder / "reference_alignment_results.json").read_text())
    with np.load(prepared / "20261113_v1/recordings" / drive / "motion.npz") as file:
        stamps = file["frame_time_ns"]
    if "smoke" in drive:
        with np.load(prepared / "labels/car_smoke_v1" / (drive + ".npz")) as file:
            approved = file["pose_approved"]
    else:
        rows = json.loads((folder / drive / "label_results.json").read_text())["rows"]
        approved = np.zeros(len(stamps), bool)
        for row in rows:
            approved[row["frame"]] = row["accepted"]
    origin_at = scan_origins(folder, prepared, drive)
    dependencies = [folder / "reference_alignment_results.json",
                    folder / drive / "trajectory.npz",
                    folder / drive / "floor_corrections.npz",
                    prepared / "20261113_v1/recordings" / drive / "motion.npz",
                    folder / drive / "label_results.json"]
    if "smoke" in drive:
        dependencies.append(prepared / "labels/car_smoke_v1" / (drive + ".npz"))
    scans = []
    sources = [dict(path=str(path), sha256=_sha256(path)) for path in dependencies]
    for frame in np.flatnonzero(approved):
        path = folder / drive / "scans" / f"frame_{frame:06d}.npz"
        with np.load(path) as file:
            ids, world, raw = file["ids"], file["world"], file["raw_xyz"]
            start, stop = arrays["frame_ptr"][frame:frame + 2]
            if (int(file["stamp"]) != int(stamps[frame])
                    or not np.array_equal(raw, arrays["xyz"][start:stop])
                    or np.any(np.diff(ids) <= 0)):
                raise ValueError(f"{path}: raw point identity mismatch")
            offsets = arrays["point_offset_s"][start + ids]
            origin = origin_at(stamps[frame], offsets)
            mapped = coordinates(world.astype(float), floor)
            model_origin = coordinates(origin, floor)
            # Archive coordinates and continuous acquisition poses must agree.
            measured_range = np.linalg.norm(mapped - model_origin, axis=1)
            raw_range = np.linalg.norm(raw[ids], axis=1)
            if np.quantile(np.abs(measured_range - raw_range), .99) > .04:
                raise ValueError(f"{path}: mapped scan and acquisition pose disagree")
        seconds = float(arrays["frame_time_s"][frame])
        scans.append(dict(frame=int(frame), stamp=int(stamps[frame]), seconds=seconds,
                          block=int(seconds / SETTINGS["time_block_s"]), ids=ids,
                          points=mapped, origins=model_origin,
                          intensity=arrays["intensity"][start + ids], ranges=raw_range))
        sources.append(dict(path=str(path), sha256=_sha256(path)))
    return scans, sources, floor


def panel_models(scans):
    points, blocks = [], []
    # Fit once per recording. Use each time block evenly.
    previous_block = -1
    for scan in scans:
        if scan["block"] == previous_block:
            continue
        previous_block = scan["block"]
        keep = inside(scan["points"], SETTINGS["panel_roi_min"], SETTINGS["panel_roi_max"])
        keep &= scan["intensity"] >= SETTINGS["panel_anchor_intensity_min"]
        cloud = scan["points"][keep]
        _, index = np.unique(np.floor(cloud / .025).astype(int), axis=0, return_index=True)
        points.append(cloud[index])
        blocks.append(np.full(len(index), scan["block"], dtype=int))
    return fit_panels(np.concatenate(points), np.concatenate(blocks))


def trunk_model(scans):
    points, blocks = [], []
    for scan in scans:
        keep = inside(scan["points"], SETTINGS["trunk_seed_min"], SETTINGS["trunk_seed_max"])
        cloud = scan["points"][keep]
        points.append(cloud)
        blocks.append(np.full(len(cloud), scan["block"], int))
    points, blocks = np.concatenate(points), np.concatenate(blocks)
    if len(points) < 100:
        return dict(accepted=False, reason="Insufficient measured trunk support")
    keys = np.column_stack((np.floor(points / .012).astype(int), blocks))
    _, index = np.unique(keys, axis=0, return_index=True)
    return fit_trunk(points[index], blocks[index], SETTINGS["trunk_center_bounds"])


def clean_stations(prepared, scene, floor, sources):
    folder = prepared / "diagnostics/other_scenes_v3" / scene
    result = []
    for path in sorted(folder.glob("*stationary*_rays.npz")):
        if scene == "corridor" and "stationary1" in path.name:
            path = prepared / "diagnostics/near_object_pilot_v2/rebuilt_corridor_station1_rays.npz"
        with np.load(path) as file:
            end = file["origin"] + file["direction"] * file["depth"][:, None]
            endpoint = coordinates(end, floor)
            origin = coordinates(file["origin"][None], floor)[0]
        vector = endpoint - origin
        depth = np.linalg.norm(vector, axis=1)
        result.append((origin, cKDTree(vector / depth[:, None]), depth))
        sources.append(dict(path=str(path), sha256=_sha256(path)))
    return result


def free_votes(points, stations):
    votes = np.zeros(len(points), np.uint8)
    for origin, tree, depth in stations:
        vector = points - origin
        radius = np.linalg.norm(vector, axis=1)
        distance, index = tree.query(vector / np.maximum(radius[:, None], 1e-9), k=8,
                                     distance_upper_bound=np.deg2rad(.35))
        found = distance * radius[:, None] <= .05
        gap = depth[np.minimum(index, len(depth) - 1)] - radius[:, None]
        votes += (found.sum(axis=1) >= 4) & np.all(~found | (gap >= .50), axis=1)
    return votes


def classify(scans, drive, model, stations):
    for scan in scans:
        if scan["frame"] % 100 == 0:
            print(drive, "classifying", scan["frame"], flush=True)
        points = scan["points"]
        if drive.startswith("corridor"):
            surface, ambiguity, early = panel_evidence(points, scan["origins"], model, scan["block"])
            region = inside(points, SETTINGS["smoke_roi_min"], SETTINGS["smoke_roi_max"])
        else:
            if model.get("accepted"):
                surface, ambiguity = trunk_evidence(points, model, [.10, .55])
            else:
                surface, ambiguity = np.zeros((2, len(points)), bool)
            early = np.zeros(len(points), bool)
            region = inside(points, [-2.5, -2.6, .12], [-.3, -.3, 2.3])
        region &= ~surface & ~ambiguity & (scan["ranges"] >= .3) & (scan["ranges"] <= 25)
        selected = np.flatnonzero(region)
        votes = free_votes(points[selected], stations)
        proposed = selected[(votes >= 2) | ((votes >= 1) & early[selected])]
        if len(proposed):
            tree = cKDTree(points[proposed])
            count = tree.query_ball_point(points[proposed], SETTINGS["smoke_cluster_radius_m"],
                                           return_length=True)
            proposed = proposed[count >= SETTINGS["smoke_minimum_neighbors"]]
        scan.update(surface=surface, ambiguity=ambiguity, candidate=proposed,
                    candidate_tree=cKDTree(points[proposed]))
    # Require support on both sides. This excludes endpoints and gaps explicitly.
    for index, scan in enumerate(scans):
        points = scan["points"][scan["candidate"]]
        past, future = np.zeros((2, len(points)), bool)
        for other in scans[max(0, index - 4):index + 5]:
            delta = other["seconds"] - scan["seconds"]
            if delta == 0 or abs(delta) > SETTINGS["temporal_window_s"]:
                continue
            near = np.isfinite(other["candidate_tree"].query(
                points, distance_upper_bound=SETTINGS["temporal_radius_m"])[0])
            if delta < 0:
                past |= near
            else:
                future |= near
        scan["supported"] = scan["candidate"][past & future]


def reviewed_points(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        key = (row["recording"], int(row["frame"]))
        result.setdefault(key, []).append(row)
    return result


def propagate_reviewed_smoke(scans, drive, reviews):
    """Grow reviewed smoke through supported candidates, keeping each temporal step bounded."""
    reached = [np.zeros(len(scan["supported"]), bool) for scan in scans]
    for index, scan in enumerate(scans):
        ids = scan["ids"][scan["supported"]]
        reviewed = [int(r["source_point_index"]) for r in reviews.get((drive, scan["frame"]), [])
                    if int(r["review_label"]) == 1]
        reached[index] |= np.isin(ids, reviewed)
    changed = True
    while changed:
        changed = False
        for index, scan in enumerate(scans):
            if not len(scan["supported"]):
                continue
            points = scan["points"][scan["supported"]]
            for other_index in range(max(0, index - 3), min(len(scans), index + 4)):
                other = scans[other_index]
                if (other_index == index or not reached[other_index].any()
                        or abs(scan["seconds"] - other["seconds"]) > .31):
                    continue
                tree = cKDTree(other["points"][other["supported"]][reached[other_index]])
                near = np.isfinite(tree.query(points, distance_upper_bound=.20)[0])
                if np.any(near & ~reached[index]):
                    reached[index] |= near
                    changed = True
    return reached


def candidate_tracks(scans, output, drive, arrays):
    """Group supported candidates in space and time for sequence-level review."""
    parent = []

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, scan in enumerate(scans):
        points = scan["points"][scan["supported"]]
        if not len(points):
            scan["track_nodes"] = np.empty(0, int)
            continue
        tree = cKDTree(points)
        pairs = tree.query_pairs(.20, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                           shape=(len(points), len(points)))
        count, component = connected_components(graph, directed=False)
        offset = len(parent)
        parent.extend(range(offset, offset + count))
        nodes = component + offset
        scan["track_nodes"] = nodes
        for previous in scans[max(0, index - 3):index]:
            if not len(previous["supported"]) or scan["seconds"] - previous["seconds"] > .31:
                continue
            previous_points = previous["points"][previous["supported"]]
            distance, nearest = cKDTree(previous_points).query(points, distance_upper_bound=.20)
            hit = np.flatnonzero(np.isfinite(distance))
            for node, other in np.unique(np.column_stack(
                    (nodes[hit], previous["track_nodes"][nearest[hit]])), axis=0):
                a, b = root(int(node)), root(int(other))
                parent[max(a, b)] = min(a, b)
    groups = {}
    for scan in scans:
        for position, node in zip(scan["supported"], scan["track_nodes"]):
            track = root(int(node))
            row = groups.setdefault(track, dict(frames=set(), points=0, minimum_range_m=1e9,
                                                maximum_range_m=0., original_background=0,
                                                original_smoke=0, original_unknown=0))
            row["frames"].add(scan["frame"])
            row["points"] += 1
            radius = float(scan["ranges"][position])
            row["minimum_range_m"] = min(row["minimum_range_m"], radius)
            row["maximum_range_m"] = max(row["maximum_range_m"], radius)
            label = arrays["label"][arrays["frame_ptr"][scan["frame"]] + scan["ids"][position]]
            key = {0: "original_background", 1: "original_smoke", 255: "original_unknown"}[int(label)]
            row[key] += 1
    rows = []
    for track, row in groups.items():
        row = dict(track_id=track, **row)
        row["frames"] = sorted(row["frames"])
        rows.append(row)
    write_json(output / (drive + "_candidate_tracks.json"), sorted(rows, key=lambda row: -row["points"]))
    return len(groups)


def write_ply(path, points, original, revised, ids, frame, reasons):
    dtype = [(axis, "<f4") for axis in "xyz"]
    dtype += [(c, "u1") for c in ("red", "green", "blue")]
    dtype += [("source_point_index", "<i4"), ("frame", "<i4"),
              ("v4_label", "u1"), ("label", "u1"), ("reason", "u1")]
    cloud = np.empty(len(ids), dtype=dtype)
    for axis, values in zip("xyz", points.T):
        cloud[axis] = values
    colors = np.tile([152, 164, 179], (len(ids), 1))
    colors[revised == 0] = [21, 145, 125]
    colors[revised == 1] = [230, 106, 38]
    for color, values in zip(("red", "green", "blue"), colors.T):
        cloud[color] = values
    cloud["source_point_index"], cloud["frame"] = ids, frame
    cloud["v4_label"], cloud["label"], cloud["reason"] = original, revised, reasons
    names = {"f": "float", "i": "int", "u": "uchar"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(ids)}"]
    header += [f"property {names[cloud[k].dtype.kind]} {k}" for k in cloud.dtype.names]
    with path.open("wb") as stream:
        stream.write(("\n".join(header + ["end_header", ""])).encode("ascii"))
        cloud.tofile(stream)


def process_recording(prepared, arrays, drive, output, reviews):
    scans, sources, floor = load_scans(prepared, drive, arrays)
    model = panel_models(scans) if drive.startswith("corridor") else trunk_model(scans)
    if isinstance(model, list):
        description = [dict(center=p.center.tolist(), normal=p.normal.tolist(),
                            tangent=p.tangent.tolist(), vertical=p.vertical.tolist(),
                            low=p.low.tolist(), high=p.high.tolist(),
                            anchor_count=len(p.anchors), blocks=len(np.unique(p.blocks))) for p in model]
        np.savez_compressed(output / (drive + "_panel_anchors.npz"),
                            **{f"panel_{i}_anchors": p.anchors for i, p in enumerate(model)},
                            **{f"panel_{i}_blocks": p.blocks for i, p in enumerate(model)})
    else:
        description = model
    write_json(output / (drive + "_model.json"), description)
    print(drive, "approved scans", len(scans), "model", len(model) if isinstance(model, list) else model, flush=True)
    stations = clean_stations(prepared, drive.split("_")[0], floor, sources)
    classify(scans, drive, model, stations)
    track_count = candidate_tracks(scans, output, drive, arrays)
    reached = propagate_reviewed_smoke(scans, drive, reviews)
    rows, changes, proposals = [], [], []
    snapshots = set(np.linspace(0, max(0, len(scans) - 1), 8, dtype=int))
    for index, scan in enumerate(scans):
        frame, ids = scan["frame"], scan["ids"]
        original = arrays["label"][arrays["frame_ptr"][frame] + ids].copy()
        smoke = np.zeros(len(ids), bool)
        smoke[scan["supported"][reached[index]]] = True
        revised = revise_labels(original, scan["surface"], scan["ambiguity"], smoke)
        reason = np.zeros(len(ids), np.uint8)
        reason[scan["surface"]] = 1
        # Preserve previously recovered smoke inside the surface ambiguity band.
        reason[smoke] = 4
        for decision in reviews.get((drive, frame), []):
            raw_index = int(decision["source_point_index"])
            position = int(np.searchsorted(ids, raw_index))
            if position == len(ids) or ids[position] != raw_index:
                raise ValueError(f"{drive} {frame}: reviewed point absent from registered scan")
            if original[position] != int(decision["v4_label"]):
                raise ValueError(f"{drive} {frame}: review belongs to another label version")
            revised[position] = int(decision["review_label"])
            reason[position] = 8
        # Known clean controls measure proposed smoke errors without turning them into training positives.
        if "clean" in drive:
            revised = original.copy()
        changed = original != revised
        source_ids = ids[changed]
        changes.extend(zip([frame] * len(source_ids), source_ids.tolist(),
                           original[changed].tolist(), revised[changed].tolist(), reason[changed].tolist()))
        supported = scan["supported"]
        proposals.extend(zip([frame] * len(supported), ids[supported].tolist(),
                             scan["ranges"][supported].tolist(),
                             original[supported].tolist(), reached[index].tolist()))
        arrays["label"][arrays["frame_ptr"][frame] + ids] = revised
        rows.append(dict(frame=frame, surface_points=int(scan["surface"].sum()),
                         supported_smoke_candidates=len(supported),
                         changed=int(changed.sum()),
                         transitions={f"{a}->{b}": int(np.sum((original == a) & (revised == b)))
                                      for a, b in [(255, 0), (1, 0), (1, 255), (255, 1)]}))
        if index in snapshots or (drive, frame) in reviews:
            write_ply(output / f"{drive}_{frame:04d}.ply", scan["points"],
                      original, revised, ids, frame, reason)
    for name, data, header in [
        ("changes", changes, ["frame", "source_point_index", "old_label", "label", "reason"]),
        ("smoke_candidates", proposals, ["frame", "source_point_index", "range_m", "old_label", "review_linked"]),
    ]:
        with (output / f"{drive}_{name}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(data)
    summary = dict(recording=drive, approved_frames=len(scans), total_frames=len(arrays["frame_time_s"]),
                   changed=sum(r["changed"] for r in rows),
                   supported_smoke_candidates=len(proposals), candidate_tracks=track_count,
                   candidate_frames=sum(r["supported_smoke_candidates"] > 0 for r in rows),
                   transitions={key: sum(r["transitions"][key] for r in rows) for key in rows[0]["transitions"]},
                   rows=rows, scan_sources=sources)
    write_json(output / (drive + "_report.json"), summary)
    print(drive, {k: v for k, v in summary.items() if k not in {"rows", "scan_sources"}}, flush=True)
    return summary


def export_split(source, manifest, manifest_path, output, base):
    plan = json.loads(source.read_text())
    plan["protocol"] = "car_whole_recording_surface_v2"
    plan["manifest_sha256"] = _sha256(manifest_path)
    for row, chunk in zip(plan["chunks"], manifest["chunks"], strict=True):
        row["sha256"] = chunk["sha256"]
    counts = {role: dict(frames=0, points=0, unimpacted=0, smoke_impacted=0, ignore=0)
              for role in plan["roles"]}
    for chunk_index, chunk in enumerate(manifest["chunks"]):
        arrays = load_unified_chunk(base / chunk["path"])
        for role, rows in plan["roles"].items():
            for row in rows:
                if row["chunk_index"] != chunk_index:
                    continue
                for start, stop in row["ranges"]:
                    labels = arrays["label"][arrays["frame_ptr"][start]:arrays["frame_ptr"][stop]]
                    counts[role]["frames"] += stop - start
                    counts[role]["points"] += len(labels)
                    for key, value in [("unimpacted", 0), ("smoke_impacted", 1), ("ignore", 255)]:
                        counts[role][key] += int(np.sum(labels == value))
    plan["summary"] = counts
    plan.pop("split_sha256", None)
    import hashlib
    plan["split_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(output, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_manifest = args.base / "dataset_manifest.json"
    manifest = json.loads(source_manifest.read_text())
    reviews = reviewed_points(args.reviews)
    source_hashes = {str(source_manifest): _sha256(source_manifest), str(args.reviews): _sha256(args.reviews)}
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        audit = stage / "audit"
        audit.mkdir(parents=True)
        shutil.copyfile(args.reviews, audit / "reviewed_input_points.csv")
        summaries = {}
        for row in manifest["chunks"]:
            source = args.base / row["path"]
            if _sha256(source) != row["sha256"]:
                raise ValueError(f"{source}: checksum differs from input manifest")
            drive = row["recording_id"]
            if drive in RECORDINGS:
                arrays = load_unified_chunk(source)
                before = arrays["label"].copy()
                summaries[drive] = process_recording(args.prepared, arrays, drive, audit, reviews)
                if np.any(before != arrays["label"]):
                    written = _write_chunk(stage / row["path"], arrays)
                    row.update({k: v for k, v in written.items() if k != "path"})
                    row["label_policy"] = "object_surface_review_pilot_v2_preserve_ambiguous_smoke"
                else:
                    (stage / row["path"]).parent.mkdir(exist_ok=True)
                    shutil.copyfile(source, stage / row["path"])
                del arrays, before
            else:
                (stage / row["path"]).parent.mkdir(exist_ok=True)
                shutil.copyfile(source, stage / row["path"])
        schema = json.loads((args.base / "dataset_schema.json").read_text())
        schema["surface_review_policy"] = SETTINGS
        write_json(stage / "dataset_schema.json", schema)
        manifest["dataset_version"] = SETTINGS["version"]
        manifest["label_review_status"] = "Automatic full-recording pilot with user-reviewed clip anchors and preserved ambiguous smoke"
        manifest["labels"] = {key: sum(row["labels"][key] for row in manifest["chunks"])
                              for key in ["unimpacted", "smoke_impacted", "ignore"]}
        manifest["surface_review"] = dict(settings=SETTINGS, source_hashes=source_hashes,
                                        report="audit/summary.json")
        write_json(stage / "dataset_manifest.json", manifest)
        export_split(args.split, manifest, stage / "dataset_manifest.json",
                     stage / "split.json", stage)
        report = dict(settings=SETTINGS, source_hashes=source_hashes,
                      implementation_sha256={str(Path(__file__)): _sha256(Path(__file__)),
                                             "object_surfaces.py": _sha256(Path(__file__).parents[1] / "src/smoke_labeler/object_surfaces.py")},
                      records={k: {x: v for x, v in value.items() if x not in {"rows", "scan_sources"}}
                               for k, value in summaries.items()},
                      limitations=["Surface tolerances and registration are empirical.",
                                   "Clean recordings have also been inspected during earlier label development.",
                                   "Label correctness across unreviewed scans remains unmeasured.",
                                   "Tree test labels have changed. Report old- and new-label metrics separately.",
                                   "Unlinked smoke candidates require review before class-1 export."])
        write_json(audit / "summary.json", report)
        stage.rename(args.output)
    print("Exported", args.output, flush=True)


if __name__ == "__main__":
    main()
