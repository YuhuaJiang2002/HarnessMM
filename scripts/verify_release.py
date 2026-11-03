"""Verify the published bytes of extracted source files without importing them."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    failures = []
    for item in manifest['files']:
        path = root / item['path']
        if not path.is_file():
            failures.append(f"Missing: {item['path']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item['published_sha256']:
            failures.append(f"Changed: {item['path']}")
    reference = root / 'reproduction' / 'humaneval_v23'
    reference_manifest = json.loads((reference / 'SOURCE_MANIFEST.json').read_text())
    for item in reference_manifest['files']:
        path = reference / item['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            failures.append(f"Changed archived reference: {item['path']}")
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f"Verified {len(manifest['files'])} extracted source files.")
    print(f"Verified {len(reference_manifest['files'])} archived HumanEval modules.")


if __name__ == '__main__':
    main()
