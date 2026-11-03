"""Callback adapter for the screenshot-only B0 decision loop.

The host owns episode limits, public screenshots, native execution, scoring
and physical provider accounting. The action contract is extracted from the
visual research implementation; this module adds no provider or browser.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json

from ._visual_contract import NATIVE, action_check, compile_coordinates, parse

EFFICIENCY_OBJECTIVE = (
    " Preserve task performance while reducing complete-task input plus output "
    "token consumption. Use concise reasoning and avoid redundant model calls "
    "and repeated action checks."
)
NATIVE_INSTRUCTION = NATIVE + EFFICIENCY_OBJECTIVE


class VisualBaseline:
    """Make a fresh metered decision for each current screenshot.

    actor(messages, purpose, callback_id) returns the actual model text and
    records usage outside this adapter, including unknown usage on errors.
    audit(event) must persist each event. No failed request is retried here.
    A measured malformed response receives the reference loop's one additional
    paid parsing fallback; both calls remain in the host's usage ledger.
    """

    def __init__(self, *, actor, audit, instruction=NATIVE_INSTRUCTION):
        self.actor = actor
        self.audit = audit
        self.instruction = instruction

    def decide(self, *, png: bytes, image_size: tuple[int, int], history, callback_id: str):
        """Return one literal pixel action for the host's original dispatcher."""
        if not isinstance(png, bytes) or not png:
            raise ValueError("Supply the complete current PNG bytes")
        if (len(image_size) != 2
                or any(type(v) is not int or v <= 0 for v in image_size)):
            raise ValueError("Supply the original screenshot dimensions")
        if not isinstance(callback_id, str) or not callback_id:
            raise ValueError("A registered slow callback identifier is required")
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": [
                {"type": "text", "text": "Public executed actions: " + json.dumps(history)},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(png).decode(),
                }},
            ]},
        ]
        for purpose, ident in (("native", callback_id),
                               ("paid_parse_fallback", callback_id + "-fallback")):
            # Provider/transport failures propagate: settlement is a host duty.
            text = self.actor(deepcopy(messages), purpose, ident)
            try:
                payload = compile_coordinates(parse(text), image_size)
                action = action_check(payload["action"], image_size)
            except (ValueError, TypeError, KeyError) as error:
                self.audit({"kind": "baseline/invalid-action", "callback_id": ident,
                            "purpose": purpose, "error": str(error), "raw_output": text})
                if purpose == "paid_parse_fallback":
                    raise
                continue
            self.audit({"kind": "baseline/decision", "callback_id": ident,
                        "purpose": purpose, "action": deepcopy(action),
                        "reasoning": payload.get("reasoning"),
                        "executed_substitutions": 0})
            return action
        raise AssertionError("Unreachable decision state")
