"""TRAIN-only exact native-history references for redundant controller receipts."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

from phiagent.harness.gaia2_jev import action_object


PREFIX = "[Public receipt-consistent action graph]\n"
OUTPUT = re.compile(r"^\[OUTPUT OF STEP (\d+)\] Observation:\n\*\*\*\n(.*)\n\*\*\*\n?$", re.S)
SEMANTICS = (
    "native_step refers to the unchanged preceding assistant action and OUTPUT OF STEP "
    "in native history. Each receipt is evidence requiring interpretation, not proof "
    "of goal completion. All arguments and full observations remain there."
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def project_ledger(messages):
    """Fail open unless a duplicate is bound to an exact action and full output."""
    audit = {
        "applied": False, "referenced_receipts": 0, "request_duplicate_removed": False,
        "native_history_modified": False, "native_tools_removed": 0,
        "unique_observations_removed": 0, "new_model_calls": 0,
    }
    if not messages or not isinstance(messages[-1].get("content"), str):
        return messages, audit
    content = messages[-1]["content"]
    if not content.startswith(PREFIX):
        return messages, audit
    try:
        ledger = json.loads(content[len(PREFIX):])
    except (ValueError, TypeError):
        return messages, audit
    if not isinstance(ledger, dict) or not isinstance(ledger.get("action_receipts"), list):
        return messages, audit
    native = {}
    preceding = None
    for message in messages[:-1]:
        text = message.get("content", "")
        if not isinstance(text, str):
            continue
        if message.get("role") == "assistant":
            preceding = action_object(text)
            if preceding is not None:
                preceding = {key: preceding[key] for key in ("action", "action_input")}
        match = OUTPUT.fullmatch(text) if message.get("role") in ("tool-response", "tool", "user") else None
        if match and preceding is not None:
            native.setdefault((digest(preceding), digest(match[2])), []).append(int(match[1]))
            preceding = None
    projected = deepcopy(ledger)
    for index, receipt in enumerate(ledger["action_receipts"]):
        if not isinstance(receipt, dict) or receipt.get("status") != "observed":
            continue
        matches = native.get((digest(receipt.get("executed_action")),
                              receipt.get("observation_sha256")), [])
        if len(matches) != 1:
            continue
        if set(receipt) != {
            "receipt_id", "executed_action", "status", "observation_sha256",
            "effect_completion", "observation_excerpt", "excerpt_truncated",
            "full_observation_in_native_history",
        } or receipt["effect_completion"] != "Requires interpretation of this public receipt":
            continue
        projected["action_receipts"][index] = {
            "receipt_id": receipt["receipt_id"], "status": receipt["status"],
            "native_step": matches[0],
        }
        audit["referenced_receipts"] += 1
    request = ledger.get("request")
    if isinstance(request, str) and request and any(
        request in str(message.get("content", ""))
        for message in messages[:-1] if message.get("role") == "user"
    ):
        projected.pop("request")
        projected["request_in_unchanged_native_task_history"] = True
        audit["request_duplicate_removed"] = True
    if audit["referenced_receipts"]:
        projected["receipt_reference_semantics"] = SEMANTICS
    if not audit["referenced_receipts"] and not audit["request_duplicate_removed"]:
        return messages, audit
    replacement = PREFIX + json.dumps(projected, ensure_ascii=False)
    if len(replacement) >= len(content):
        return messages, {**audit, "referenced_receipts": 0, "request_duplicate_removed": False}
    result = list(messages)
    result[-1] = {**messages[-1], "content": replacement}
    audit.update(applied=True, ledger_bytes_before=len(content.encode()),
                 ledger_bytes_after=len(replacement.encode()),
                 original_ledger_sha256=hashlib.sha256(content.encode()).hexdigest(),
                 projected_ledger_sha256=hashlib.sha256(replacement.encode()).hexdigest(),
                 entity_sets_and_enumeration_unchanged=True)
    return result, audit


def install_adapter(config, directory, role, mode, skill):
    if (config.get("stage") != "train-candidate" or config.get("evaluation_split") != "train"
            or config.get("method") != "chisel" or mode != "jev"):
        raise RuntimeError("State-ledger development is TRAIN-only")
    variant = config.get("state_ledger_variant")
    if variant not in ("parent", "native_references"):
        raise ValueError("Frozen parent or native-reference ledger required")
    from phiagent.harness import gaia2_202060_progress as progress

    parent_engine = progress.ReceiptEngine

    class ReferencedEngine(parent_engine):
        def __init__(self, agent, provider, *, skill, record, execute_followups):
            def projected_provider(messages, **kwargs):
                projected, audit = project_ledger(messages)
                record({"source": "native_state_ledger_projection",
                        "agent_id": agent.agent_id, **audit})
                return provider(projected, **kwargs)

            super().__init__(agent, projected_provider, skill=skill, record=record,
                             execute_followups=execute_followups)

    if variant == "native_references":
        progress.ReceiptEngine = ReferencedEngine
    progress.install_adapter(config, directory, role, mode, skill)
