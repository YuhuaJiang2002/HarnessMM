"""TRAIN-only recovery-aware routing at the native GAIA2 observation boundary."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from phiagent.harness.deferred_observation_pack import RECALL_PREFIX, RECALL_TOOL
from phiagent.harness.gaia2_deferred_recall import READ_INSTRUCTION, prepare_native_observations
from phiagent.harness.gaia2_jev import action_object
from phiagent.harness.recovery_aware_context import (
    RecoveryAwareObservationPack, RecoveryRoutingPolicy,
)


def public_routing_context(messages, logs):
    tasks = [row.content for row in logs if row.get_type() == "task"
             and isinstance(getattr(row, "content", None), str)]
    action = None
    for message in reversed(messages):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            action = action_object(message["content"])
            if action is not None:
                break
    return {
        "task_text": "\n".join(tasks),
        "action_text": json.dumps(action, ensure_ascii=False) if action is not None else "",
    }


def install_adapter(config, directory, role, mode, skill):
    if (config.get("stage") != "train-candidate" or config.get("evaluation_split") != "train"
            or config.get("method") != "chisel" or mode != "jev"):
        raise RuntimeError("Recovery-routing mechanism validation is TRAIN only")
    settings = config["recovery_routing"]
    if set(settings) != {"variant", "policy"}:
        raise ValueError("A frozen variant and complete routing policy are required")
    variant = settings["variant"]
    policy = RecoveryRoutingPolicy(**settings["policy"])
    if variant not in ("parent", "fixed", "relevance_only", "recovery_aware"):
        raise ValueError("Unknown recovery-routing experiment identity")
    if variant != "parent" and policy.mode != variant:
        raise ValueError("The mechanism variant and policy disagree")
    from phiagent.harness import gaia2_202060_progress as progress
    if variant == "parent":
        progress.install_adapter(config, directory, role, mode, skill)
        return
    from are.simulation.tool_box import get_tool_description_with_args
    from are.simulation.tools import Tool
    from scripts import run_gaia2_jev as native

    parent_engine = progress.ReceiptEngine

    class RecoveryEngine(parent_engine):
        def __init__(self, agent, provider, *, skill, record, execute_followups):
            archive_id = hashlib.sha256(str(agent.agent_id).encode()).hexdigest()
            archive_root = Path(directory) / "recovery-observations" / archive_id
            first_log = agent.logs[0] if agent.logs else None
            episode_index = 0

            def count_tokens(text):
                result = native.get_json(
                    f"http://127.0.0.1:{config['resource']['model_port']}/tokenize",
                    {"model": config["model"]["served_name"], "prompt": text,
                     "add_special_tokens": False},
                )
                count = result["count"]
                if type(count) is not int or count < 0:
                    raise ValueError("The pinned serving tokenizer returned invalid counts")
                return count

            def new_pack():
                return RecoveryAwareObservationPack(
                    archive_root / f"episode-{episode_index}", policy=policy,
                    count_tokens=count_tokens, recall_tool=RECALL_TOOL,
                )

            pack = new_pack()

            class RecallTool(Tool):
                name = RECALL_TOOL
                description = (
                    "Read exact pages of this agent's archived observations. These are original "
                    "snapshots, not current state. This does not repeat the original tool action."
                )
                inputs = {
                    "observation_id": {"type": "string", "description": "ID in an archive receipt"},
                    "offset": {"type": "integer", "description": "Unicode character offset",
                               "default": 0},
                    "limit": {"type": "integer", "description": "Characters to read, at most 8000",
                              "default": 8000},
                }
                output_type = "string"

                def forward(self, observation_id, offset=0, limit=8000):
                    result = pack.recall(observation_id, offset, limit)
                    record({
                        "source": "recovery_routing_recall", "agent_id": agent.agent_id,
                        "observation_id": observation_id, "offset": offset, "limit": limit,
                        "characters_returned": len(json.loads(result)["text"]),
                        "new_model_calls": 0, "native_action_iteration_consumed": True,
                        "original_environment_action_replayed": False,
                    })
                    return RECALL_PREFIX + "\n" + result

            if RECALL_TOOL in agent.action_executor.tools:
                raise ValueError("The exact-recall tool already exists in the native actor")
            recall = RecallTool()
            recall.validate_arguments()
            agent.action_executor.tools[RECALL_TOOL] = recall
            declaration = READ_INSTRUCTION + get_tool_description_with_args(recall)
            overhead = count_tokens(declaration)

            def projected_provider(messages, **kwargs):
                nonlocal pack, first_log, episode_index
                current_first = agent.logs[0] if agent.logs else None
                if current_first is not first_log:
                    episode_index += 1
                    pack = new_pack()
                    first_log = current_first
                original = deepcopy(messages)
                projected, prepared = prepare_native_observations(
                    pack, messages, **public_routing_context(messages, agent.logs),
                    request_overhead_tokens=overhead,
                )
                if prepared.audit["projected_observations"]:
                    if not projected or projected[0].get("role") != "system":
                        pack.abort(prepared)
                        raise ValueError("The native system message is required")
                    projected[0]["content"] += declaration
                record({
                    "source": "recovery_routing_projection", "agent_id": agent.agent_id,
                    **prepared.audit, "full_callable_contracts_preserved": True,
                    "conditioning": "public task and most recent observed model action; no future oracle",
                })
                try:
                    response = provider(projected, **kwargs)
                    if not isinstance(response, tuple) or len(response) != 2:
                        raise ValueError("Native provider response and complete usage are required")
                    pack.commit(prepared, response[1])
                except Exception:
                    if prepared.request_index in pack._pending:
                        pack.abort(prepared)
                    raise
                if messages != original:
                    raise ValueError("Recovery routing changed the original native history")
                record({
                    "source": "recovery_routing_paid_request", "agent_id": agent.agent_id,
                    "usage": response[1], "archive_report": pack.report(),
                    "projected_observations": prepared.audit["projected_observations"],
                    "new_model_calls": 1,
                })
                return response

            super().__init__(agent, projected_provider, skill=skill, record=record,
                             execute_followups=execute_followups)

    progress.ReceiptEngine = RecoveryEngine
    progress.install_adapter(config, directory, role, mode, skill)
