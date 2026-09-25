from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterator, Sequence

import numpy as np

from .schema import SCHEMA_VERSION, load_chunk, sha256_file


ROLES = ("train", "selection", "calibration", "test")
DEFAULT_FRACTIONS = (0.55, 0.15, 0.15, 0.15)


@dataclass(frozen=True)
class ChunkInfo:
    index: int
    path: Path
    source_domain: str
    session_id: str
    recording_id: str
    condition: str
    frames: int
    points: int
    sha256: str
    source_start_s: float | None = None

    @property
    def stream_id(self) -> str:
        if self.source_domain == "grandtour_ros1":
            return f"{self.source_domain}/{self.session_id}"
        return f"{self.source_domain}/{self.session_id}/{self.recording_id}"


@dataclass(frozen=True, order=True)
class FrameRef:
    chunk_index: int
    frame_index: int


@dataclass(frozen=True)
class Frame:
    ref: FrameRef
    source_domain: str
    session_id: str
    recording_id: str
    condition: str
    timestamp_s: float
    xyz: np.ndarray
    intensity: np.ndarray
    label: np.ndarray


class UnifiedDataset:
    def __init__(self, manifest_path: str | Path, *, verify_checksums: bool = False):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{self.manifest_path}: unsupported schema version")
        if raw.get("status") != "complete":
            raise ValueError(f"{self.manifest_path}: dataset is not complete")
        rows = raw.get("chunks")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{self.manifest_path}: chunks must be a nonempty list")

        chunks: list[ChunkInfo] = []
        for index, row in enumerate(rows):
            path = Path(str(row["path"]))
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_hash = str(row["sha256"])
            if verify_checksums and sha256_file(path) != expected_hash:
                raise ValueError(f"{path}: checksum does not match manifest")
            selection = row.get("source_time_selection_s") or {}
            chunks.append(
                ChunkInfo(
                    index=index,
                    path=path,
                    source_domain=str(row["source_domain"]),
                    session_id=str(row["session_id"]),
                    recording_id=str(row["recording_id"]),
                    condition=str(row["condition"]),
                    frames=int(row["frames"]),
                    points=int(row["points"]),
                    sha256=expected_hash,
                    source_start_s=(
                        float(selection["start"]) if "start" in selection else None
                    ),
                )
            )
        if sum(chunk.frames for chunk in chunks) != int(raw["frames"]):
            raise ValueError(f"{self.manifest_path}: frame total does not match chunks")
        if sum(chunk.points for chunk in chunks) != int(raw["points"]):
            raise ValueError(f"{self.manifest_path}: point total does not match chunks")
        self.raw_manifest = raw
        self.chunks = tuple(chunks)
        self.manifest_sha256 = sha256_file(self.manifest_path)

    def load_chunk(self, index: int, *, validate: bool = True) -> dict[str, np.ndarray]:
        chunk = self.chunks[index]
        arrays = load_chunk(chunk.path, validate=validate)
        if len(arrays["frame_time_s"]) != chunk.frames or len(arrays["xyz"]) != chunk.points:
            raise ValueError(f"{chunk.path}: array counts do not match manifest")
        metadata = {
            "source_domain": chunk.source_domain,
            "session_id": chunk.session_id,
            "recording_id": chunk.recording_id,
            "condition": chunk.condition,
        }
        for name, expected in metadata.items():
            if str(arrays[name].item()) != expected:
                raise ValueError(f"{chunk.path}: {name} does not match manifest")
        return arrays

    def frame(self, chunk: ChunkInfo, arrays: dict[str, np.ndarray], index: int) -> Frame:
        if not 0 <= index < chunk.frames:
            raise IndexError(index)
        start = int(arrays["frame_ptr"][index])
        end = int(arrays["frame_ptr"][index + 1])
        return Frame(
            ref=FrameRef(chunk.index, index),
            source_domain=chunk.source_domain,
            session_id=chunk.session_id,
            recording_id=chunk.recording_id,
            condition=chunk.condition,
            timestamp_s=float(arrays["frame_time_s"][index]),
            xyz=arrays["xyz"][start:end],
            intensity=arrays["intensity"][start:end],
            label=arrays["label"][start:end],
        )


def _stream_chunks(dataset: UnifiedDataset) -> dict[str, list[ChunkInfo]]:
    streams: dict[str, list[ChunkInfo]] = defaultdict(list)
    for chunk in dataset.chunks:
        streams[chunk.stream_id].append(chunk)
    for chunks in streams.values():
        if chunks[0].source_domain == "grandtour_ros1":
            chunks.sort(
                key=lambda value: (
                    value.source_start_s is None,
                    value.source_start_s if value.source_start_s is not None else value.index,
                )
            )
        else:
            chunks.sort(key=lambda value: value.index)
    return dict(streams)


def _role_for_global_frame(
    index: int, boundaries: Sequence[int], purge_frames: int
) -> str | None:
    starts = (0, boundaries[0] + purge_frames, boundaries[1] + purge_frames, boundaries[2] + purge_frames)
    ends = (boundaries[0] - purge_frames, boundaries[1] - purge_frames, boundaries[2] - purge_frames, boundaries[3])
    for role, start, end in zip(ROLES, starts, ends, strict=True):
        if start <= index < end:
            return role
    return None


def _compact_ranges(indices: Sequence[int]) -> list[list[int]]:
    if not indices:
        return []
    values = sorted(indices)
    ranges: list[list[int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            ranges.append([start, previous + 1])
            start = value
        previous = value
    ranges.append([start, previous + 1])
    return ranges


def create_split_plan(
    dataset: UnifiedDataset,
    output_path: str | Path,
    *,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    purge_frames: int = 20,
) -> dict:
    if len(fractions) != len(ROLES) or any(value <= 0 for value in fractions):
        raise ValueError("fractions must contain four positive values")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to one")
    if purge_frames < 0:
        raise ValueError("purge_frames must be nonnegative")

    members: dict[str, dict[int, list[int]]] = {
        role: defaultdict(list) for role in ROLES
    }
    excluded: dict[int, list[int]] = defaultdict(list)
    stream_rows: list[dict] = []

    for stream_id, chunks in sorted(_stream_chunks(dataset).items()):
        frame_count = sum(chunk.frames for chunk in chunks)
        cumulative = np.cumsum(np.asarray(fractions, dtype=np.float64))
        boundaries = [int(np.floor(value * frame_count)) for value in cumulative[:-1]]
        boundaries.append(frame_count)
        if any(boundaries[index] - boundaries[index - 1] <= 2 * purge_frames for index in range(1, 3)):
            raise ValueError(f"{stream_id}: partition is too short for purge={purge_frames}")

        global_index = 0
        for chunk in chunks:
            for local_index in range(chunk.frames):
                role = _role_for_global_frame(global_index, boundaries, purge_frames)
                if role is None:
                    excluded[chunk.index].append(local_index)
                else:
                    members[role][chunk.index].append(local_index)
                global_index += 1
        stream_rows.append(
            {
                "stream_id": stream_id,
                "chunk_indices": [chunk.index for chunk in chunks],
                "frames": frame_count,
                "boundaries": boundaries,
            }
        )

    role_rows: dict[str, list[dict]] = {}
    for role in ROLES:
        role_rows[role] = [
            {"chunk_index": chunk_index, "ranges": _compact_ranges(indices)}
            for chunk_index, indices in sorted(members[role].items())
            if indices
        ]

    plan = {
        "version": 1,
        "protocol": "pilot_temporal_v1",
        "manifest": dataset.manifest_path.name,
        "manifest_sha256": dataset.manifest_sha256,
        "fractions": dict(zip(ROLES, fractions, strict=True)),
        "purge_frames_each_side": purge_frames,
        "chunks": [
            {
                "index": chunk.index,
                "path": str(chunk.path.relative_to(dataset.manifest_path.parent)),
                "sha256": chunk.sha256,
                "frames": chunk.frames,
                "stream_id": chunk.stream_id,
            }
            for chunk in dataset.chunks
        ],
        "streams": stream_rows,
        "roles": role_rows,
        "excluded": [
            {"chunk_index": chunk_index, "ranges": _compact_ranges(indices)}
            for chunk_index, indices in sorted(excluded.items())
            if indices
        ],
    }
    plan["summary"] = split_summary(dataset, plan)
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    plan["split_sha256"] = hashlib.sha256(canonical).hexdigest()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def load_split_plan(dataset: UnifiedDataset, path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    plan = json.loads(source.read_text(encoding="utf-8"))
    claimed_hash = plan.get("split_sha256")
    unsigned = dict(plan)
    unsigned.pop("split_sha256", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    actual_hash = hashlib.sha256(canonical).hexdigest()
    if claimed_hash != actual_hash:
        raise ValueError(f"{source}: split content hash is invalid")
    if plan.get("manifest_sha256") != dataset.manifest_sha256:
        raise ValueError(f"{source}: split was created for another manifest")
    if set(plan.get("roles", {})) != set(ROLES):
        raise ValueError(f"{source}: split must define {ROLES}")
    expected = {chunk.index: chunk for chunk in dataset.chunks}
    for row in plan.get("chunks", []):
        index = int(row["index"])
        if index not in expected or row["sha256"] != expected[index].sha256:
            raise ValueError(f"{source}: chunk {index} differs from dataset")
    seen: set[FrameRef] = set()
    for role in ROLES:
        for ref in frame_refs(plan, role):
            chunk = expected.get(ref.chunk_index)
            if chunk is None or not 0 <= ref.frame_index < chunk.frames:
                raise ValueError(f"{source}: invalid frame reference {ref}")
            if ref in seen:
                raise ValueError(f"{source}: frame appears in multiple roles: {ref}")
            seen.add(ref)
    return plan


def frame_refs(plan: dict, role: str) -> list[FrameRef]:
    if role not in ROLES:
        raise ValueError(f"unknown role: {role}")
    refs: list[FrameRef] = []
    for row in plan["roles"][role]:
        chunk_index = int(row["chunk_index"])
        for start, end in row["ranges"]:
            refs.extend(FrameRef(chunk_index, index) for index in range(int(start), int(end)))
    return refs


def iter_frames(
    dataset: UnifiedDataset,
    plan: dict,
    role: str,
    *,
    shuffle: bool = False,
    seed: int = 0,
    max_frames: int | None = None,
) -> Iterator[Frame]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for ref in frame_refs(plan, role):
        grouped[ref.chunk_index].append(ref.frame_index)
    chunk_indices = list(grouped)
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(chunk_indices)
    emitted = 0
    for chunk_index in chunk_indices:
        local_indices = grouped[chunk_index]
        if shuffle:
            rng.shuffle(local_indices)
        chunk = dataset.chunks[chunk_index]
        arrays = dataset.load_chunk(chunk_index)
        for frame_index in local_indices:
            if max_frames is not None and emitted >= max_frames:
                return
            yield dataset.frame(chunk, arrays, frame_index)
            emitted += 1


def split_summary(dataset: UnifiedDataset, plan: dict) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    refs_by_role = {role: frame_refs(plan, role) for role in ROLES}
    refs_by_chunk: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for role, refs in refs_by_role.items():
        for ref in refs:
            refs_by_chunk[ref.chunk_index][role].append(ref.frame_index)

    counts = {
        role: {"frames": 0, "points": 0, "unimpacted": 0, "smoke_impacted": 0, "ignore": 0}
        for role in ROLES
    }
    for chunk_index, roles in refs_by_chunk.items():
        arrays = dataset.load_chunk(chunk_index)
        ptr = arrays["frame_ptr"]
        labels = arrays["label"]
        for role, frame_indices in roles.items():
            for frame_index in frame_indices:
                start, end = int(ptr[frame_index]), int(ptr[frame_index + 1])
                frame_labels = labels[start:end]
                counts[role]["frames"] += 1
                counts[role]["points"] += len(frame_labels)
                counts[role]["unimpacted"] += int(np.count_nonzero(frame_labels == 0))
                counts[role]["smoke_impacted"] += int(np.count_nonzero(frame_labels == 1))
                counts[role]["ignore"] += int(np.count_nonzero(frame_labels == 255))
    summary.update(counts)
    return summary
