"""Frozen public app-task context shared by native GAIA2 harness arms."""

from __future__ import annotations

import hashlib
from copy import deepcopy

PROMPT_VERSION = "industrial-gaia2-native-apps/v3"
SHARED_PROMPT = (
    '<native-app-task version="3">\n'
    "The current task takes place in a stateful, time-driven app environment, "
    "not in the coding harness's host workspace. Use only the registered native "
    "app tools. The harness working-directory path is not a path in the Files "
    "app, and coding/background-job instructions do not create extra app tools.\n"
    "Complete every requested operation through the actual tools before reporting "
    "completion. Keep a checklist of the user's requirements, including exact "
    "recipients, content, dates, time zones, amounts and subsequent updates. "
    "Resolve references from public app state; copy identifiers from real tool "
    "observations rather than guessing them. Check tool results before using "
    "their identifiers or depending on a preceding mutation. Do not repeat a "
    "successful mutation merely to verify it; use an appropriate read tool.\n"
    "Use the native current-time tool when interpreting relative dates or "
    "time-sensitive requests. Observe notifications and changed state. Use "
    "SystemApp__wait_for_notification when a task requires a later event; this "
    "ends the current turn, not the whole scenario. Ask the user only when "
    "essential information remains unavailable after relevant public-state "
    "checks. A plan, an acknowledgement or a request for unnecessary confirmation "
    "does not complete the task.\n"
    "Track each requested obligation as pending, completed or conditional, with "
    "its public trigger and due time. A user-facing progress message completes "
    "only the notification obligation; it does not complete requested later "
    "replies, rescheduling or conditional actions. After sending that message, "
    "keep pending obligations active and wait through the native notification "
    "tool. On a new notification, read the indicated app conversation or changed "
    "object, update the obligation state, and perform only the actions whose "
    "conditions are now supported. Do not treat a current empty inbox or lack "
    "of an immediate reply as proof that a future condition cannot occur. "
    "For an exact delay or interval, anchor an absolute deadline to the request's "
    "native timestamp, re-read native time after tool work, and wait only the "
    "remaining interval. Avoid repeatedly adding the full original delay. "
    "End the scenario only when all applicable obligations have been discharged "
    "or the unchanged native environment limit has been reached.\n"
    "Before each app mutation, write a compact operation contract in your "
    "reasoning: the user requirement, the target app and tool, the exact argument "
    "keys, and the public observation that supplies each entity or value. "
    "An identifier resolved in Chats is not automatically a Messages identifier; "
    "resolve the destination app's own account or conversation before writing. "
    "A phone number, display name or email address is not a substitute for a "
    "tool's required user or conversation ID. Use the tool's declared parameter "
    "names and types, without inventing aliases. For Contacts and conversation "
    "enumeration, use small pages of at most five items unless the tool states "
    "a smaller cap, and advance offsets only from observed page bounds or counts. "
    "Keep recipients and group membership distinct from search hits. Apply "
    "filters, counts, dates and numeric comparisons to the complete observed "
    "population before committing the requested write. Verify the resulting "
    "state against the same operation contract, then mark that operation done.\n"
    "Respect the tool descriptions, defaults and pagination. If a tool reports "
    "a view limit or an invalid argument, correct the diagnosed argument instead "
    "of repeating the failing call. Preserve all original user requests and "
    "native tool observations. Report observed results and unresolved requirements "
    "accurately; never infer success from missing evidence.\n"
    "</native-app-task>"
)
PROMPT_SHA256 = hashlib.sha256(SHARED_PROMPT.encode("utf-8")).hexdigest()


def prompt_identity() -> dict[str, str]:
    return {"version": PROMPT_VERSION, "sha256": PROMPT_SHA256}


def validate_prompt_identity(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or value != prompt_identity():
        raise ValueError("Shared GAIA2 public prompt version/hash differs from its frozen bytes")
    return prompt_identity()


def shared_actor_request(payload: dict, role: str, identity: object = None) -> tuple[dict, dict | None]:
    """Give ARE actors the same frozen public brief as the industrial adapter.

    The provider gateway calls this before tokenization, so the common brief is
    charged even when an actor-specific projection omits it. Judge and optimizer
    requests retain their original inputs. Legacy configs remain unchanged.
    """
    if identity is None or role.rsplit("/", 1)[-1] != "actor":
        return payload, None
    frozen = validate_prompt_identity(identity)
    value = deepcopy(payload)
    messages = value.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Common GAIA2 context requires actor messages")
    system = [item for item in messages if item.get("role") == "system"]
    if not system or any(not isinstance(item.get("content"), str) for item in system):
        raise ValueError("Common GAIA2 context requires textual native system messages")
    count = sum(item["content"].count(SHARED_PROMPT) for item in system)
    if count > 1:
        raise ValueError("Common GAIA2 public brief was duplicated")
    if not count:
        system[0]["content"] += "\n\n" + SHARED_PROMPT
    return value, {
        **frozen, "role": role, "placement": "additive_public_benchmark_system_context",
        "same_original_and_chisel_parent": True, "native_system_messages_preserved": True,
        "charged_before_provider_forward": True, "private_or_heldout_feedback": False,
    }


def public_context(
    current_time: str,
    notification_prompt: str,
    additional_prompt: str | None,
    shared_prompt: object = None,
) -> tuple[str, dict | None]:
    parts = [
        f"Current native simulation time (UTC): {current_time}",
        notification_prompt,
        additional_prompt,
    ]
    receipt = None
    if shared_prompt is not None:
        identity = validate_prompt_identity(shared_prompt)
        parts.append(SHARED_PROMPT)
        receipt = {
            **identity,
            "placement": "additive_public_benchmark_system_context",
            "same_original_and_chisel_parent": True,
            "native_system_loop_tools_and_scoring": "unchanged",
            "private_oracle_or_heldout_feedback": False,
        }
    return "\n\n".join(part for part in parts if part), receipt
