from __future__ import annotations

"""Project identity helpers for multi-project capture."""

import hashlib
import re
import subprocess
from pathlib import Path


def _safe_path(path_str: str) -> Path:
    if path_str:
        return Path(path_str).expanduser()
    return Path.cwd()


def normalize_cwd(path_str: str) -> str:
    """Return a stable absolute cwd string."""
    try:
        return str(_safe_path(path_str).resolve())
    except OSError:
        return str(_safe_path(path_str).absolute())


def detect_project_root(cwd: str) -> str:
    """Resolve project root as git top-level, falling back to cwd."""
    normalized_cwd = normalize_cwd(cwd)
    cwd_path = Path(normalized_cwd)
    if not cwd_path.exists():
        return normalized_cwd

    try:
        result = subprocess.run(
            ["git", "-C", normalized_cwd, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return normalized_cwd

    if result.returncode == 0 and result.stdout.strip():
        return normalize_cwd(result.stdout.strip())

    return normalized_cwd


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "project"


def project_metadata(cwd: str, source_platform: str = "unknown") -> dict[str, str]:
    """Build stable project identity fields from cwd."""
    normalized_cwd = normalize_cwd(cwd)
    root = detect_project_root(normalized_cwd)
    root_path = Path(root)
    name = root_path.name or "project"
    digest = hashlib.sha1(root.encode("utf-8")).hexdigest()[:8]
    slug = f"{_slugify(name)}-{digest}"
    return {
        "cwd": normalized_cwd,
        "project_root": root,
        "project_name": name,
        "project_slug": slug,
        "source_platform": source_platform or "unknown",
    }


def safe_session_token(value: str) -> str:
    """Return a filename-safe session token."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return token or "unknown"
