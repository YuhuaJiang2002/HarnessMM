"""Opt-in, first-exposure-preserving recall for native GAIA2 public observations."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import re

from phiagent.harness.deferred_observation_pack import (
    DeferredObservationPack, RECALL_PREFIX, RECALL_TOOL,
)


OBSERVATION = re.compile(r"^\[OUTPUT OF STEP ([0-9]+)\] Observation:\n\*\*\*\n", re.DOTALL)
READ_INSTRUCTION = (
    "\nSome older large observations may be represented by exact-archive receipts. "
    "They were previously supplied in full, and the native history remains unchanged. "
    "A receipt's preview is incomplete. If an omitted value is needed, read its exact "
    "archived page with CHISELMemory__recall; use next_offset to continue. "
    "This read-only harness tool adds no new environment facts and does not establish "
    "task completion. All original task constraints and action limits still apply.\n"
)


def prepare_native_observations(pack, messages, **routing_context):
    compatible = []
    for index, message in enumerate(messages):
        item = dict(message)
        text = item.get("content")
        role = item.get("role")
        match = OBSERVATION.match(text) if isinstance(text, str) else None
        if role in ("tool-response", "tool") and match is not None:
            item.update(
                role="tool", name=RECALL_TOOL if RECALL_PREFIX in text else "native_observation",
                tool_call_id="native-step-" + match[1],
                is_error=bool(re.search(r'(?i)(?:error:|traceback \(|"error"\s*:)', text)),
            )
        elif role in ("tool-response", "tool"):
            item["is_error"] = True
        compatible.append(item)
    projection = pack.prepare(compatible, **routing_context)
    native = [{**original, "content": projected.get("content")}
              for original, projected in zip(messages, projection.messages, strict=True)]
    if [message.get("role") for message in native] != [message.get("role") for message in messages]:
        raise ValueError("Observation packing cannot change native message roles")
    return native, projection


def install_adapter(config, directory, role, mode, skill):
    if (config.get("stage") != "train-candidate" or config.get("evaluation_split") != "train"
            or config.get("method") != "chisel" or mode != "jev"):
        raise RuntimeError("This literature-derived mechanism is admitted to TRAIN only")
    variant = config.get("deferred_recall_variant")
    if variant not in ("parent", "deferred_exact_recall"):
        raise ValueError("A frozen matched parent/recall variant is required")
    from are.simulation.tool_box import get_tool_description_with_args
    from are.simulation.tools import Tool
    from phiagent.harness import gaia2_202060_progress as progress

    parent_engine = progress.ReceiptEngine

    class RecallEngine(parent_engine):
        def __init__(self, agent, provider, *, skill, record, execute_followups):
            archive_id = hashlib.sha256(str(agent.agent_id).encode()).hexdigest()
            archive_root = Path(directory) / "exact-observations" / archive_id
            pack = DeferredObservationPack(archive_root / "episode-0")
            first_log = agent.logs[0] if agent.logs else None
            episode_index = 0

            class RecallTool(Tool):
                name = RECALL_TOOL
                description = (
                    "Read exact text previously returned to this agent by a native tool. "
                    "Only session-local archived observation IDs are accepted. "
                    "offset and limit count Unicode characters. Follow next_offset for remaining pages. "
                    "This does not modify the simulated environment."
                )
                inputs = {
                    "observation_id": {"type": "string", "description": "ID in an archive receipt"},
                    "offset": {"type": "integer", "description": "Character offset", "default": 0},
                    "limit": {"type": "integer", "description": "Characters to read, at most8000",
                              "default": 8000},
                }
                output_type = "string"

                def forward(self, observation_id, offset=0, limit=8000):
                    result = pack.recall(observation_id, offset, limit)
                    record({"source": "exact_observation_recall", "agent_id": agent.agent_id,
                            "observation_id": observation_id, "offset": offset, "limit": limit,
                            "characters_returned": len(result), "new_model_calls": 0,
                            "native_action_iteration_consumed": True})
                    return result

            if RECALL_TOOL in agent.action_executor.tools:
                raise ValueError("The private recall tool name is already registered")
            recall = RecallTool()
            recall.validate_arguments()
            agent.action_executor.tools[RECALL_TOOL] = recall
            declaration = get_tool_description_with_args(recall)

            def projected_provider(messages, **kwargs):
                nonlocal pack, first_log, episode_index
                current_first = agent.logs[0] if agent.logs else None
                if current_first is not first_log:
                    episode_index += 1
                    pack = DeferredObservationPack(archive_root / f"episode-{episode_index}")
                    first_log = current_first
                original = deepcopy(messages)
                native, prepared = prepare_native_observations(pack, messages)
                if prepared.audit["projected_observations"]:
                    if not native or native[0].get("role") != "system":
                        pack.abort(prepared)
                        raise ValueError("Native actor system message is required")
                    native[0]["content"] += READ_INSTRUCTION + declaration
                record({"source": "deferred_observation_projection", "agent_id": agent.agent_id,
                        **prepared.audit, "instruction_bytes_preserved": True,
                        "native_roles_preserved": True,
                        "private_recall_tool_declared": prepared.audit["projected_observations"] > 0})
                try:
                    response = provider(native, **kwargs)
                    if not isinstance(response, tuple) or len(response) != 2:
                        raise ValueError("Native provider must return text and exact usage metadata")
                    pack.commit(prepared, response[1])
                except Exception:
                    if prepared.request_index in pack._pending:
                        pack.abort(prepared)
                    raise
                if messages != original:
                    raise ValueError("The original native history was modified")
                record({"source": "deferred_observation_paid_exposure",
                        "agent_id": agent.agent_id, "usage": response[1],
                        "full_events_committed": len(prepared.full_events),
                        "archive_report": pack.report(), "new_model_calls": 1})
                return response

            super().__init__(agent, projected_provider, skill=skill, record=record,
                             execute_followups=execute_followups)

    if variant == "deferred_exact_recall":
        progress.ReceiptEngine = RecallEngine
    progress.install_adapter(config, directory, role, mode, skill)
