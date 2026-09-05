"""Record the configuration and implementation used for an offline run."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__


def collect_code_version() -> dict:
    package_dir = Path(__file__).resolve().parent
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_dir.glob("*.py"))
    }
    result = {
        "labeler_version": __version__,
        "python_version": platform.python_version(),
        "dependencies": {},
        "source_sha256": hashes,
        "git_commit": None,
        "git_labeler_dirty": None,
    }
    for name in ("numpy", "mcap"):
        try:
            result["dependencies"][name] = version(name)
        except PackageNotFoundError:
            result["dependencies"][name] = None

    # Only inspect Git for a source checkout, not an unrelated repository
    # containing an installed virtual environment.
    project_dir = package_dir.parent.parent
    if (project_dir / "pyproject.toml").is_file():
        try:
            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(project_dir), *args],
                    check=True, capture_output=True, text=True, timeout=5,
                ).stdout.strip()

            commit = git("rev-parse", "HEAD")
            dirty = bool(git("status", "--porcelain", "--untracked-files=all", "--", "."))
            result.update(git_commit=commit, git_labeler_dirty=dirty)
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def write_run_provenance(output_dir: Path, config: dict, invocation: dict) -> dict:
    """Write before processing, so metadata describes the run's starting state."""
    config_text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    (output_dir / "effective_config.json").write_text(config_text, encoding="utf-8")
    provenance = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "effective_config_file": "effective_config.json",
        "effective_config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "invocation": invocation,
        **collect_code_version(),
    }
    (output_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )
    return provenance
