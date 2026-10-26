#!/usr/bin/env python3
"""Freeze a common request-bound compiler continuation for an existing core run."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_chisel_core_ablation as study  # noqa: E402


def prepare(old_prepared: Path, output: Path, run_root: Path,
            gpus: list[int], ports: list[int], inherit_selected: list[str] | None = None) -> dict:
    old = study.native.read(old_prepared / "FROZEN.json")
    study.native.verify_signed(old, "freeze_sha256")
    spec = deepcopy(old["spec"])
    spec.update(
        campaign=run_root.name, run_root=str(run_root),
        predecessor_campaign=old["spec"]["run_root"],
        proposal_compilation=study.NORMALIZED_COMPILATION, resume_development=True,
        gpu_indices=gpus, ports=ports,
        scope="Prospective compiler recovery with complete native phases and all predecessor costs preserved.",
    )
    spec["inherit_selected_methods"] = inherit_selected or []
    study.native.require(set(spec["inherit_selected_methods"]) <= set(study.METHODS),
                         "Unknown inherited method")
    spec["design"]["compiler_recovery"] = (
        "Bind only an omitted parent hash from the exact original request; an explicit mismatch "
        "still fails. Remove byte-identical no-op edit entries without rewriting any changed "
        "replacement. Effective edits retain all cardinality, support and content guards. "
        "Reuse raw proposal responses only for byte-identical messages, parent, seed and budgets. "
        "Keep rejected legacy proposals and all acquisition costs. Replay the same two rounds "
        "under this prospectively frozen compiler, using complete audited phase caches only. "
        "No test outcome is used."
    )
    study.native.require(all(spec[key] == old["spec"][key] for key in study.SCIENTIFIC_KEYS),
                         "Recovery may not tune the scientific protocol")
    path = output.parent / "RECOVERY_SPEC.json"
    study.native.require(not path.exists(), "Use a new recovery identity")
    study.native.write(path, spec)
    return study.prepare(path, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=3, required=True)
    parser.add_argument("--ports", type=int, nargs=3, required=True)
    parser.add_argument("--inherit-selected", nargs="*", default=[])
    args = parser.parse_args()
    print(json.dumps(prepare(args.old_prepared, args.output, args.run_root, args.gpus, args.ports,
                             args.inherit_selected), indent=2))
