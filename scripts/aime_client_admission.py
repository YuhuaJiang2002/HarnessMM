"""CPU-only transport policy and exact-once, hash-bound prediction reuse."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import threading


def policy(config):
    resource, budget = config.get("resource", {}), config.get("budget", {})
    limit = resource.get("client_max_inflight")
    workers = resource.get("client_batch_workers", min(4, limit) if limit else 4)
    analysts = resource.get("analyst_workers", min(4, limit) if limit else 4)
    provider = budget.get("request_timeout_seconds", 1800)
    client = budget.get("client_timeout_seconds", provider + 30)
    for name, value in (("client_batch_workers", workers), ("analyst_workers", analysts)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("client_max_inflight must be a positive integer or absent")
    if limit is not None and (workers > limit or analysts > limit):
        raise ValueError("Client workers must not exceed the explicit admission limit")
    if any(type(value) not in (int, float) or value <= 0 for value in (provider, client)):
        raise ValueError("Transport timeouts must be positive")
    if client < provider + 30:
        raise ValueError("Client timeout must exceed provider timeout by at least 30 seconds")
    return {"limit": limit, "workers": workers, "analysts": analysts,
            "provider_timeout": provider, "client_timeout": client}


def slots(config):
    limit = policy(config)["limit"]
    return threading.BoundedSemaphore(limit) if limit else nullcontext()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial_configuration(config):
    """Prospective settings; per-call output capacity and cumulative limits are unchanged."""
    config["resource"].update(client_max_inflight=1, client_batch_workers=1, analyst_workers=1)
    config["budget"].update(request_timeout_seconds=7200, client_timeout_seconds=7500)
    policy(config)
    return config


def reuse_prediction(root, config, destination, item, skill, user_prompt, role, maximum):
    binding = config.get("prediction_reuse")
    if not binding:
        return None
    manifest_path = Path(binding["path"])
    if sha(manifest_path) != binding["sha256"]:
        raise RuntimeError("Prediction reuse manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if sha(root / "DATA.json") != manifest["data_sha256"]:
        raise RuntimeError("Prediction reuse dataset changed")
    relative = str(destination.relative_to(root))
    entry = manifest["entries"].get(relative)
    if entry is None:
        return None
    if entry["role"] != role or entry["maximum"] != maximum:
        raise RuntimeError("Prediction reuse role/output capacity differs")
    for name, expected in entry["files_sha256"].items():
        if sha(destination / name) != expected:
            raise RuntimeError("Completed prediction changed: " + name)
    for key in ("provider_request", "provider_response"):
        if sha(Path(entry[key]["path"])) != entry[key]["sha256"]:
            raise RuntimeError("Original paid provider evidence changed")
    row = json.loads((destination / "result.json").read_text())
    request = json.loads(Path(entry["provider_request"]["path"]).read_text())["json"]
    messages = [{"role": "user", "content": user_prompt}]
    if skill:
        messages.insert(0, {"role": "system", "content": skill})
    if (
        request["messages"] != messages or request["max_tokens"] != maximum
        or request["model"] != config["model"]["served_name"]
        or request["temperature"] != 0 or request["seed"] != config["seed"]
        or request["chat_template_kwargs"] != {"enable_thinking": False}
        or row["id"] != item["id"] or row["question"] != item["question"]
        or (destination / "target_system_prompt.txt").read_text() != skill
        or (destination / "target_user_prompt.txt").read_text() != user_prompt
        or row["raw_response"] != json.loads(Path(entry["provider_response"]["path"]).read_text())
    ):
        raise RuntimeError("Completed prediction input/response binding differs")
    receipts = root / "reused_predictions"
    receipts.mkdir(exist_ok=True)
    with (receipts / f"call-{entry['prior_sequence']:06d}.json").open("x") as stream:
        json.dump({"prior_sequence": entry["prior_sequence"], "destination": relative,
                   "entry": entry, "new_model_calls": 0, "cost_in_historical_lineage": True}, stream)
    return row
