"""Provenance: ``record(model, revision, inputs)`` returns a dict with code SHA,
platform, UTC timestamp; intended for inclusion in eval-report JSON sidecars.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Provenance:
    model_name: str
    model_revision: str | None
    inputs: list[dict[str, Any]] = field(default_factory=list)
    code_sha: str | None = None
    platform: str = ""
    captured_at_utc: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def code_sha(repo_dir: str | None = None) -> str | None:
    repo_dir = repo_dir or os.getcwd()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True, timeout=2
        ).strip()
    except Exception:
        return None


def record(model_name: str, model_revision: str | None, inputs: list[dict]) -> Provenance:
    return Provenance(
        model_name=model_name,
        model_revision=model_revision,
        inputs=list(inputs),
        code_sha=code_sha(),
        platform=f"{platform.system()} {platform.machine()} py{platform.python_version()}",
        captured_at_utc=datetime.now(UTC).isoformat(),
    )


def fingerprint_bytes(b: bytes) -> str:
    """Stable short hash for fixture identification."""
    return hashlib.sha256(b).hexdigest()[:16]


def write_sidecar(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
