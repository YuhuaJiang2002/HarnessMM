"""Keep held-out scalar selection inside the engine, outside patch-model evidence."""
from __future__ import annotations
import hashlib
import json


def strip_selection_fields(value):
    if isinstance(value, dict):
        return {key: strip_selection_fields(child) for key, child in value.items()
                if not any(marker in key.lower() for marker in
                           ("development", "validation", "held_out", "heldout", "test_score"))}
    if isinstance(value, list):
        return [strip_selection_fields(child) for child in value]
    return value


def optimizer_workspace(files, forbidden_ids):
    result = {}
    removed = []
    manifest_name = ".autosaddler/history/manifest.json"
    for name, text in files.items():
        if name == manifest_name:
            continue
        if name.endswith(".json"):
            original = json.loads(text)
            filtered = strip_selection_fields(original)
            if filtered != original:
                removed.append(name)
            text = json.dumps(filtered, ensure_ascii=False, sort_keys=True) + "\n"
        if any(ident in text for ident in forbidden_ids):
            raise ValueError("Held-out validation identity entered optimizer workspace")
        result[name] = text
    if manifest_name in files:
        manifest = strip_selection_fields(json.loads(files[manifest_name]))
        manifest["files"] = {
            name: {"sha256": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
                   "bytes": len(text.encode())}
            for name, text in result.items() if name.startswith(".autosaddler/history/")
        }
        manifest["selection_feedback_removed"] = True
        result[manifest_name] = json.dumps(manifest, sort_keys=True) + "\n"
    return result, {"selection_fields_removed_from": removed,
                    "validation_ids_present": False, "training_evidence_preserved": True}
