"""Run the four offline cases in isolated processes and check their digests."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "reproduction/CASES.json").read_text())
for item in manifest["files"]:
    path = ROOT / item["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise SystemExit(f"Case file digest mismatch: {item['path']}")
if shutil.which("node") is None:
    raise SystemExit("Install Node.js 22 or later and add node to PATH for the Codex case.")
for case in manifest["cases"]:
    result = subprocess.run([sys.executable, str(ROOT / case["entrypoint"])],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    if result.returncode:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"Case failed: {case['name']}")
    actual = json.loads(result.stdout)
    expected = json.loads((ROOT / case["expected"]).read_text())
    if actual != expected:
        raise SystemExit(f"Unexpected output: {case['name']}")
    print(f"PASS {case['name']} ({case['scope']})")
