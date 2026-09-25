from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from smoke_trainer.data import (
    ROLES,
    UnifiedDataset,
    create_split_plan,
    frame_refs,
    load_split_plan,
)

from helpers import make_dataset


class DatasetTests(unittest.TestCase):
    def test_split_is_disjoint_purged_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = UnifiedDataset(make_dataset(root), verify_checksums=True)
            first = create_split_plan(dataset, root / "first.json", purge_frames=1)
            second = create_split_plan(dataset, root / "second.json", purge_frames=1)

            self.assertEqual(first["split_sha256"], second["split_sha256"])
            self.assertEqual(first["summary"]["train"]["frames"], 21)
            self.assertEqual(first["summary"]["selection"]["frames"], 4)
            self.assertEqual(first["summary"]["calibration"]["frames"], 4)
            self.assertEqual(first["summary"]["test"]["frames"], 5)
            refs = {role: set(frame_refs(first, role)) for role in ROLES}
            for index, role in enumerate(ROLES):
                for other in ROLES[index + 1 :]:
                    self.assertFalse(refs[role] & refs[other])
            self.assertEqual(sum(len(values) for values in refs.values()), 34)
            self.assertEqual(load_split_plan(dataset, root / "first.json"), first)

            tampered = json.loads((root / "first.json").read_text())
            tampered["purge_frames_each_side"] = 99
            (root / "first.json").write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "content hash"):
                load_split_plan(dataset, root / "first.json")

    def test_manifest_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = make_dataset(root)
            text = manifest.read_text().replace('"frames": 40', '"frames": 41', 1)
            manifest.write_text(text)
            with self.assertRaisesRegex(ValueError, "frame total"):
                UnifiedDataset(manifest)


if __name__ == "__main__":
    unittest.main()
