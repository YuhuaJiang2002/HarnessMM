"""Check that released project code resolves within a single checkout.

Third-party packages, benchmark runtimes and license attribution are separate
from dependencies on another HarnessMM or research-project checkout.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = {"harnessmm", "phiagent", "integrations", "scripts", "chisel"}
GENERATED = {".git", ".venv", "node_modules", "__pycache__", "build", "dist"}
PROJECT_LINK = re.compile(
    r"(?:https?://github\.com/|git@github\.com:|git:github\.com/)"
    r"(?:YuhuaJiang2002/(?:HarnessMM(?:-R)?|HarnessMinusMinus|PhiAgent(?:-0)?)"
    r"|microsoft/HarnessMinusMinus|jlon/pi-harnessmm|DSH-Extensions-list/dsh-harnessmm)(?=[/.@\s\"'):#]|$)"
)


def git(root: Path, *arguments: str) -> str | None:
    """Source archives can be checked without Git metadata or a Git executable."""
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=root, stderr=subprocess.DEVNULL,
        ).decode().strip()
        if Path(top).resolve() != root.resolve():
            return None
        return subprocess.check_output(
            ["git", *arguments], cwd=root, stderr=subprocess.DEVNULL,
        ).decode()
    except (OSError, subprocess.CalledProcessError):
        return None


def files(root: Path) -> list[str]:
    """Include new source files during local checks; ignore build output."""
    tracked = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if tracked is not None:
        return sorted(set(tracked.split("\0")) - {""})
    return sorted(
        str(path.relative_to(root)) for path in root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and not GENERATED.intersection(path.relative_to(root).parts)
    )


def inside(root: Path, path: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def check(root: Path) -> tuple[list[str], int]:
    errors = []
    source = files(root)
    repository = re.search(r'^Repository\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.MULTILINE)[1]
    identity = repository.removeprefix("https://github.com/")
    for line in (git(root, "ls-files", "--stage") or "").splitlines():
        if line.startswith("160000 "):
            errors.append(f"{line.split(chr(9))[-1]}: Git submodule")
    for name in source:
        path = root / name
        if path.is_symlink():
            if not inside(root, path) or not path.exists():
                errors.append(f"{name}: external or broken symlink")
            continue
        try:
            text = path.read_text()
        except (UnicodeError, OSError):
            continue
        # Notices retain original attribution. Test fixtures can name synthetic
        # or historical identities without creating a deployment dependency.
        if path.name != "NOTICE" and "tests" not in path.parts:
            for match in PROJECT_LINK.finditer(text):
                linked = re.split(r"github\.com[:/]", match[0], maxsplit=1)[-1]
                if linked != identity:
                    errors.append(f"{name}: points to another project: {linked}")
        if path.suffix == ".py":
            tree = ast.parse(text, filename=name)
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module]
                for module in modules:
                    if module.split(".")[0] not in LOCAL_PACKAGES:
                        continue
                    module = re.sub(r"^phiagent(?=\.|$)", "harnessmm", module)
                    base = root / "legacy/chisel" if module.split(".")[0] == "chisel" else root
                    target = base / module.replace(".", "/")
                    if not target.is_dir() and not target.with_suffix(".py").is_file():
                        errors.append(f"{name}:{node.lineno}: missing project module {module}")
        if path.suffix in {".mjs", ".js", ".ts"} and "tests" not in path.parts and "test" not in path.parts:
            for match in re.finditer(r"(?:\bfrom\s*|\bimport\s*\(?\s*)['\"](\.[^'\"]+)['\"]", text):
                target = path.parent / match[1]
                if not inside(root, target) or not target.exists():
                    errors.append(f"{name}: missing or external JavaScript import {match[1]}")
        if path.name in {"package.json", "package-lock.json"}:
            data = json.loads(text)
            for key, entry in data.get("packages", {}).items():
                if key.startswith(("..", "/")) or entry.get("link") or str(entry.get("resolved", "")).startswith(("file:", "link:", "../", "/")):
                    errors.append(f"{name}: machine-local dependency {key}")
            for group in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
                for package, version in data.get(group, {}).items():
                    if str(version).startswith(("file:", "link:", "../", "/")):
                        errors.append(f"{name}: machine-local dependency {package}")
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    for item in manifest["files"]:
        target = root / item["path"]
        if not inside(root, target) or not target.is_file():
            errors.append(f"SOURCE_MANIFEST.json: missing or external file {item['path']}")
    return errors, len(source)


def main() -> None:
    errors, count = check(ROOT)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Self-contained source check passed: {count} files; no missing project imports, external checkout links, submodules or machine-local npm dependencies.")


if __name__ == "__main__":
    main()
