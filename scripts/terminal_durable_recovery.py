"""Read-only recovery of one previously submitted, durably captured model response."""
from __future__ import annotations

import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request


CAPTURE_SOURCE_SHA256 = "2e78ea85550ef628e13368edf346aea75f55845a1e18c09d1f084f7c80d3d3bd"


def settle_pre_capture_rejections(usage, proofs):
    if (usage["inflight"] != 0 or type(usage["unknown_usage_attempts"]) is not int
            or usage["unknown_usage_attempts"] != len(proofs) or not proofs):
        raise ValueError("Exactly the closed unresolved attempts require admission proofs")
    ids = []
    for proof in proofs:
        if (proof.get("schema") != "sealed-pre-capture-rejection/v1"
                or proof.get("capture_source_sha256") != CAPTURE_SOURCE_SHA256
                or proof.get("service_retired") is not True
                or proof.get("all_accepted_captures_terminal") is not True
                or proof.get("capture_present") is not False
                or proof.get("client_delivery_closed") is not True
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", proof.get("request_id", ""))
                or not re.fullmatch(r"[a-f0-9]{64}", proof.get("client_request_sha256", ""))):
            raise ValueError("An absent response alone is not evidence of zero model usage")
        ids.append(proof["request_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate pre-capture admission evidence")
    return {
        **usage, "complete": True, "budget_accounting_complete": True,
        "unknown_usage_attempts": 0, "unbounded_unknown_usage_attempts": 0,
        "all_attempts_returned_native_usage": False,
        "verified_pre_capture_rejections": usage.get("verified_pre_capture_rejections", 0) + len(ids),
        "pre_capture_rejection_request_ids": [
            *usage.get("pre_capture_rejection_request_ids", []), *ids,
        ],
    }


def recover_captured_response(opener, upstream, request_id, model, *, deadline):
    endpoint = upstream.rstrip("/") + "/captures/" + request_id
    last_error = None
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            request = Request(endpoint, method="GET")
            with opener.open(request, timeout=min(5.0, remaining)) as response:
                record = json.load(response)
            if record.get("request_id") != request_id:
                raise ValueError("Captured response belongs to a different model request")
            if record.get("state") == "COMPLETE":
                body, status = record["response"], record["upstream_status"]
                if type(status) is not int or not 100 <= status <= 599:
                    raise ValueError("Invalid captured upstream status")
                if status == 200:
                    if body.get("model") != model:
                        raise ValueError("Recovered native response has the wrong model identity")
                    usage = body.get("usage", {})
                    counts = [usage.get(key) for key in
                              ("prompt_tokens", "completion_tokens", "total_tokens")]
                    if (any(type(value) is not int or value < 0 for value in counts)
                            or counts[0] + counts[1] != counts[2]):
                        raise ValueError("Recovered native response has invalid usage")
                return status, body, {
                    "request_id": request_id, "server_response_sha256": record["response_sha256"],
                    "recovery_endpoint": endpoint, "model_post_attempts": 1,
                    "native_json_reencoded_for_delivery": True,
                }
            if record.get("state") != "PENDING":
                raise ValueError("Unknown durable capture state")
        except HTTPError as error:
            if error.code != 404:
                raise
            last_error = f"capture not yet present: HTTP {error.code}"
            error.close()
        except (URLError, TimeoutError, ConnectionError) as error:
            last_error = str(error)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(2.0, remaining))
    raise TimeoutError(f"Original model response remains unresolved; no POST replay: {last_error}")
