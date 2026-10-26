"""Read-only accounting reconciliation for settled native EdgeBench executions."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from phiagent.harness.deepseek_messages_meter import recover_meter_snapshot
from phiagent.harness.edgebench_protocol import actor_usage


def reconcile_native_result(result: Mapping, sidecar_directory: Path) -> dict:
    if result.get("status") != "SCORED":
        raise ValueError("Only settled native scoring can enter the method comparison")
    native_root = Path(result["run_dir"]).resolve()
    original_path = native_root / "RESULT.json"
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    for field in ("status", "request_id", "task_id", "phase", "arm", "seed", "sdk_run_dir"):
        if result.get(field) != original.get(field):
            raise ValueError("Controller receipt differs from the original native execution")
    planned = json.loads((native_root / "SDK_JOB_PLANNED.json").read_text())
    sdk_root = Path(original["sdk_run_dir"]).resolve()
    if Path(planned["run_dir"]).resolve() != sdk_root:
        raise ValueError("Provider session differs from the frozen native dispatch")
    provider = sdk_root / "provider"
    destination = sidecar_directory.resolve()
    if destination == native_root or native_root in destination.parents:
        raise ValueError("Reconciliation sidecars must stay outside the frozen native trial")
    if destination == sdk_root or sdk_root in destination.parents:
        raise ValueError("Reconciliation sidecars must stay outside the frozen SDK session")
    manifest = json.loads((provider / "INSTANCE.json").read_text())
    if manifest.get("state") != "CLOSED":
        raise ValueError("A settled trial requires sealed and drained provider accounting")
    sidecar = destination / f"{original['request_id']}-{uuid4().hex}.json"
    recovery = recover_meter_snapshot(provider, sidecar=sidecar)
    if not recovery["complete"] or recovery["source_meter_state"] != "CLOSED":
        raise ValueError(f"Provider accounting remains unresolved; preserved recovery: {sidecar}")
    measured = actor_usage(recovery["usage"], complete=True)
    if not measured["usage_complete"]:
        raise ValueError(f"Recovered provider usage does not reconcile: {sidecar}")
    resolved = {**result, **measured}
    resolved["usage_reconciliation"] = {
        "source_native_result": str(original_path),
        "source_native_result_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "source_provider_root": str(provider),
        "sidecar": str(sidecar),
        "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "physical_request_count": recovery["request_count"],
        "recovered_request_count": recovery["recovered_request_count"],
        "unknown_request_count": recovery["unknown_request_count"],
        "original_usage_complete": original.get("usage_complete"),
        "original_total_tokens": original.get("total_tokens"),
        "replacement_aggregate": True,
        "new_provider_calls": 0,
        "native_score_changed": False,
    }
    return resolved
