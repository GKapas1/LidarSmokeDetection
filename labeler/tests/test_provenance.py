import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smoke_labeler.provenance import write_run_provenance


class ProvenanceTests(unittest.TestCase):
    def test_missing_git_still_saves_exact_config_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = {"sampling": {"target_point_stride": 3}}
            with patch("smoke_labeler.provenance.subprocess.run", side_effect=FileNotFoundError):
                result = write_run_provenance(output, config, {"session_id": "example"})
            saved = (output / "effective_config.json").read_bytes()
            self.assertEqual(json.loads(saved), config)
            self.assertEqual(result["effective_config_sha256"], hashlib.sha256(saved).hexdigest())
            self.assertIsNone(result["git_commit"])
            self.assertIsNone(result["git_labeler_dirty"])
            import smoke_labeler.core
            core = Path(smoke_labeler.core.__file__).read_bytes()
            self.assertEqual(result["source_sha256"]["core.py"], hashlib.sha256(core).hexdigest())
            self.assertEqual(json.loads((output / "run_provenance.json").read_text()), result)
