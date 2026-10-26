#!/usr/bin/env python3
"""Run one frozen GAIA2 scenario with official Codex CLI and native ARE scoring.

See phiagent.harness.codex_gaia2 for the freeze, MCP, provider, isolation and
accounting contracts. Run only on authorized A800-4 in a new campaign directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phiagent.harness.codex_gaia2 import RunConfig, run_episode  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario", type=Path, required=True)
    result.add_argument("--filesystem", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True, help="New, nonexisting run directory")
    result.add_argument("--codex-bin", type=Path, default=Path("/usr/local/bin/codex"))
    result.add_argument("--provider-url", required=True, help="Campaign-local metering root URL")
    result.add_argument("--relay-secret-file", type=Path, required=True,
                        help="Owner-private HMAC relay secret; never mounted into Codex")
    result.add_argument("--trial-id", help="Unique trial prefix; actor/judge suffixes are added")
    result.add_argument("--model", default="deepseek-flash")
    result.add_argument("--residual-file", type=Path)
    result.add_argument("--partition-manifest", type=Path)
    result.add_argument("--freeze-manifest", type=Path, required=True)
    result.add_argument("--judge-url", help="Optional separate evaluator metering root URL")
    result.add_argument("--native-bank-policy", type=json.loads,
                        help="Source-frozen native CoT/Jev policy JSON; omit for the original harness")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = ("scenario", "filesystem", "output", "codex_bin", "residual_file",
             "partition_manifest", "freeze_manifest", "relay_secret_file")
    for key in paths:
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    if args.partition_manifest is None:
        args.partition_manifest = args.scenario.parent.parent / "SPLIT_MANIFEST.json"
    args.provider_url = args.provider_url.rstrip("/")
    if args.judge_url:
        args.judge_url = args.judge_url.rstrip("/")
    try:
        result = run_episode(RunConfig(**vars(args)))
    except Exception as error:
        result = {
            "status": "BLOCKED", "native_score": None, "provider_usage": None,
            "error_type": type(error).__name__, "blocker": str(error),
        }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
