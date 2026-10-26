"""Public, episode-scoped native receipts for CoT bank/planner progress discipline."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json


PROGRESS_INSTRUCTION = """
Use the public progress index below alongside the unchanged native history.
Its action records come from actual native tool-call logs, not your proposals.
Each returned receipt preserves the original native observation bytes. A scalar
ID establishes an observed return only: it does not supply invented domain
fields, prove a requested effect, or establish completion of the user's goal.
Earlier receipts are historical evidence; later native actions can supersede them.

Reconcile all source-backed task clauses, filters, timing, exceptions and channels
before choosing the current action. Do not keep repeating a write merely because
it was in an earlier plan. An identical call is NOT automatically redundant:
new requests, explicitly requested multiplicity, retries and verification reads
can justify repeats. Explain that justification from the public sources.
When a requested effect needs verification, choose an appropriate available
native read with public arguments; do not invent a tool, argument or result.
Your CURRENT CoT action may use the full native interface, including appropriate
waiting or communication with the user. Restrictions on future fast-bank
candidates do not forbid those current slow actions.

Keep the same response envelope and native action schema. Encode jev_bank.note
as a compact JSON string:
{"unresolved":[{"source_id":"a listed source ID","quote":"an exact source clause",
"why":"remaining work or verification needed","action_ids":[],"receipt_ids":[]}],
"plan":"brief next-step rationale"}.
Only cite IDs in this episode. Clauses must be copied exactly from their listed
public source. List the obligations that remain unresolved or need verification;
omitting an earlier clause is a planner judgment, never machine-verified success.
The pending list and plan remain hypotheses. Do not force a fast decision or
bypass UNKNOWN to obtain activation.
"""


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def public_payload(row):
    value = getattr(row, "content", None)
    if isinstance(value, str):
        return value
    raise ValueError("Public native log content must retain its original string bytes")


class PlannerProgress:
    def __init__(self, runtime_agent_id):
        self.runtime_agent_id = str(runtime_agent_id)
        self.epoch = 0
        self.first_task_object = None
        self.previous_log_count = 0
        self.obligations = []
        self.sources = {}
        self.action_ids = set()
        self.receipt_ids = set()

    def snapshot(self, logs):
        first_task = next((row for row in logs if row.get_type() == "task"), None)
        if (
            self.epoch == 0 or first_task is not self.first_task_object
            or len(logs) < self.previous_log_count
        ):
            self.epoch += 1
            self.obligations = []
        self.first_task_object = first_task
        self.previous_log_count = len(logs)
        scope = f"{self.runtime_agent_id}/native-log-epoch-{self.epoch}"
        sources, actions, receipts = [], [], []
        latest_action = None
        fingerprints = Counter()
        for index, row in enumerate(logs):
            kind = row.get_type()
            if kind in ("task", "agent_user_interface"):
                text = public_payload(row)
                sources.append({
                    "source_id": f"T{len(sources) + 1:04d}", "kind": kind,
                    "native_log_index": index, "text": text,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                })
            elif kind == "tool_call":
                name = getattr(row, "tool_name", None)
                arguments = getattr(row, "tool_arguments", None)
                if not isinstance(name, str) or not isinstance(arguments, (dict, str)):
                    raise ValueError("Native tool-call name and arguments are required")
                normalized = deepcopy(arguments)
                if isinstance(arguments, str):
                    try:
                        normalized = json.loads(arguments)
                    except ValueError:
                        pass
                fingerprint = digest({"tool_name": name, "arguments": normalized})
                fingerprints[fingerprint] += 1
                latest_action = {
                    "action_id": f"A{len(actions) + 1:04d}",
                    "native_log_index": index,
                    "tool_name": name,
                    "exact_native_arguments": deepcopy(arguments),
                    "argument_identity_sha256": fingerprint,
                    "same_argument_occurrence_in_this_epoch": fingerprints[fingerprint],
                    "receipt_ids": [],
                    "domain_effect_verified": None,
                }
                actions.append(latest_action)
            elif kind in ("observation", "error"):
                text = public_payload(row)
                receipt = {
                    "receipt_id": f"R{len(receipts) + 1:04d}",
                    "native_log_index": index, "kind": kind,
                    "native_action_id": latest_action["action_id"] if latest_action else None,
                    "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "native_return_observed": kind == "observation",
                    "domain_fields_inferred": False,
                    "goal_completion_inferred": False,
                }
                receipts.append(receipt)
                if latest_action is not None:
                    latest_action["receipt_ids"].append(receipt["receipt_id"])
        self.sources = {row["source_id"]: row["text"] for row in sources}
        self.sources.update({row["receipt_id"]: row["text"] for row in receipts})
        self.action_ids = {row["action_id"] for row in actions}
        self.receipt_ids = {row["receipt_id"] for row in receipts}
        return {
            "episode_scope": scope,
            "scope_basis": "Runtime agent identity and actual native-log lifetime, not task-text equality",
            "task_sources": sources,
            "actual_native_actions": actions,
            "original_public_native_receipts": receipts,
            "planner_declared_unresolved_obligations": deepcopy(self.obligations),
            "obligation_coverage_or_completion_machine_verified": False,
            "repeat_counts_do_not_classify_redundancy": True,
            "native_observations_rewritten": False,
            "public_history_silently_truncated": False,
        }

    def update_note(self, note):
        try:
            value = json.loads(note)
        except (TypeError, ValueError):
            return {"status": "UNKNOWN", "reason": "Planner did not declare source-backed obligations"}
        if (
            not isinstance(value, dict) or set(value) != {"unresolved", "plan"}
            or not isinstance(value["unresolved"], list)
            or not isinstance(value["plan"], str)
        ):
            return {"status": "UNKNOWN", "reason": "Malformed planner obligation declaration"}
        checked = []
        for obligation in value["unresolved"]:
            if (
                not isinstance(obligation, dict)
                or set(obligation) != {"source_id", "quote", "why", "action_ids", "receipt_ids"}
                or not isinstance(obligation["source_id"], str)
                or obligation["source_id"] not in self.sources
                or not isinstance(obligation["quote"], str) or not obligation["quote"].strip()
                or obligation["quote"] not in self.sources[obligation["source_id"]]
                or not isinstance(obligation["why"], str)
                or not isinstance(obligation["action_ids"], list)
                or not isinstance(obligation["receipt_ids"], list)
                or any(not isinstance(ref, str) or ref not in self.action_ids
                       for ref in obligation["action_ids"])
                or any(not isinstance(ref, str) or ref not in self.receipt_ids
                       for ref in obligation["receipt_ids"])
            ):
                return {"status": "UNKNOWN", "reason": "Unbound public quote or native record ID"}
            checked.append(deepcopy(obligation))
        omitted = [item for item in self.obligations if item not in checked]
        self.obligations = checked
        return {
            "status": "SOURCE_BOUND_PLANNER_HYPOTHESES",
            "unresolved": checked,
            "previous_obligations_omitted_by_planner": omitted,
            "omission_implies_goal_completion": False,
        }

    def messages(self, prompt, logs):
        messages = deepcopy(prompt)
        if not messages or messages[0].get("role") != "system":
            raise ValueError("Preserve the complete native CoT system prompt")
        state = self.snapshot(logs)
        serialized = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        messages[0]["content"] += (
            "\n" + PROGRESS_INSTRUCTION + "\nPUBLIC_NATIVE_PROGRESS_INDEX:\n" + serialized
        )
        return messages, {
            "source": "cot_jev_public_planner_progress",
            "episode_scope": state["episode_scope"],
            "actual_native_actions": len(state["actual_native_actions"]),
            "public_receipts": len(state["original_public_native_receipts"]),
            "source_bound_obligations": len(self.obligations),
            "added_input_utf8_bytes": len((PROGRESS_INSTRUCTION + serialized).encode()),
            "native_observations_rewritten": False,
            "original_native_prompt_messages_removed": 0,
            "source_ids_sha256": digest(sorted(self.sources)),
        }
