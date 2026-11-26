"""Repository-local Git receipts for research source freezes.

All Git commands read the supplied checkout. Source-only deployments use an
explicit receipt, validated against the deployment's repository identity.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping
from urllib.parse import urlsplit


DEFAULT_REPOSITORY = "YuhuaJiang2002/HarnessMM"
STATE_FIELDS = {"captured_at_utc", "repository", "head", "branch", "tracked_status_sha256"}
REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class SourceProvenanceError(ValueError):
    """The checkout or source receipt cannot establish its Git identity."""


def git_output(root: Path, *arguments: str) -> str:
    """Read local Git metadata without fetching or consulting another checkout."""
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True,
            text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SourceProvenanceError(
            "Cannot read source Git metadata; supply --source-git-state for a source-only deployment"
        ) from error
    return result.stdout


def _require_checkout(root: Path) -> None:
    # Git searches parent directories; never record an enclosing project's HEAD.
    top = Path(git_output(root, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != root.resolve():
        raise SourceProvenanceError("Source directory must be the root of its own Git checkout")


def repository_from_remote(remote: str) -> str:
    """Normalize HTTPS, ssh:// and SCP-style Git remotes to owner/repository."""
    if "://" in remote:
        parsed = urlsplit(remote)
        if parsed.scheme not in {"https", "http", "ssh", "git"} or not parsed.hostname:
            raise SourceProvenanceError("Source remote must identify a hosted Git repository")
        path = parsed.path.lstrip("/")
    else:
        match = re.fullmatch(r"(?:[^/@:\s]+@)?[^/:\s]+:(.+)", remote)
        if match is None:
            raise SourceProvenanceError("Source remote must identify a hosted Git repository")
        path = match.group(1)
    repository = path.rstrip("/").removesuffix(".git")
    if not REPOSITORY_PATTERN.fullmatch(repository):
        raise SourceProvenanceError("Source remote must identify owner/repository")
    return repository


def repository_identity(root: Path) -> str:
    """Use this checkout's origin, or the release identity for a source archive."""
    try:
        _require_checkout(root)
    except SourceProvenanceError:
        return DEFAULT_REPOSITORY
    try:
        remote = git_output(root, "remote", "get-url", "origin").strip()
    except SourceProvenanceError:
        return DEFAULT_REPOSITORY
    return repository_from_remote(remote)


def validate_git_state(
    state: Any, *, expected_repository: str = DEFAULT_REPOSITORY,
) -> dict[str, Any]:
    """Validate the complete receipt before accepting it into a source freeze."""
    if not isinstance(state, Mapping) or set(state) != STATE_FIELDS:
        raise SourceProvenanceError("Source Git state has invalid fields")
    if state["repository"] != expected_repository:
        raise SourceProvenanceError("Source Git state repository differs from the current repository")
    if not isinstance(state["branch"], str) or not state["branch"].strip():
        raise SourceProvenanceError("Source Git state requires a branch (HEAD for a detached checkout)")
    for field, length in (("head", 40), ("tracked_status_sha256", 64)):
        value = state[field]
        if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
            raise SourceProvenanceError(f"Source Git state has an invalid {field}")
    try:
        captured = datetime.fromisoformat(state["captured_at_utc"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as error:
        raise SourceProvenanceError("Source Git state has an invalid capture time") from error
    if captured.tzinfo is None:
        raise SourceProvenanceError("Source Git state requires an explicit timezone")
    return dict(state)


def capture_git_state(root: Path) -> dict[str, Any]:
    """Capture a receipt from this checkout, including a tracked-status digest."""
    _require_checkout(root)
    repository = repository_identity(root)
    state = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repository": repository,
        "head": git_output(root, "rev-parse", "HEAD").strip(),
        "branch": git_output(root, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "tracked_status_sha256": hashlib.sha256(
            git_output(root, "status", "--porcelain", "--untracked-files=no").strip().encode()
        ).hexdigest(),
    }
    return validate_git_state(state, expected_repository=repository)


def main() -> None:
    """Print a source receipt for deployment without Git metadata."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="source checkout root")
    args = parser.parse_args()
    print(json.dumps(capture_git_state(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
