"""TRAIN-only deferred read-tool schemas with exact metadata lookup."""
from __future__ import annotations

from copy import deepcopy
import hashlib


LOOKUP = "CHISELMemory__tool_schema"
READ_PREFIXES = ("get_", "list_", "read_", "search_", "find_", "retrieve_", "count_")
INDEX_HEADER = (
    "\n[Native tool index]\n"
    "Every original tool remains available. Entries marked SCHEMA ON DEMAND show only "
    "a verbatim description prefix, not their full contract. Before using unfamiliar "
    "arguments or defaults, obtain the exact native description with "
    "CHISELMemory__tool_schema(tool_name). State-changing and previously used tool "
    "contracts remain fully displayed. Task instructions and observations are unchanged.\n"
)


def read_schema_can_be_deferred(name, *, write_operation=None):
    operation = name.split("__")[-1]
    return write_operation is False and operation.startswith(READ_PREFIXES) and name != LOOKUP


def project_catalogue(messages, descriptions, summaries, hot, read_only_names):
    if not messages or messages[0].get("role") != "system":
        raise ValueError("Native system message is required")
    catalogue = "\n".join(descriptions.values())
    system = messages[0].get("content")
    if not isinstance(system, str):
        raise ValueError("Only native text tool catalogues are supported")
    if system.count(catalogue) != 1:
        return messages, {"applied": False, "reason": "native_catalogue_not_uniquely_bound"}
    cold = [name for name in descriptions if name not in hot and name in read_only_names]
    if not cold:
        return messages, {"applied": False, "reason": "all_tool_contracts_are_protected"}
    lines = []
    for name, text in descriptions.items():
        if name in cold:
            lines.append("- " + name + ": " + summaries[name][:180]
                         + " [SCHEMA ON DEMAND; description prefix only]")
        else:
            lines.append(text)
    replacement = INDEX_HEADER + "\n".join(lines)
    if len(replacement) >= len(catalogue):
        return messages, {"applied": False, "reason": "catalogue_index_not_shorter"}
    result = [dict(message) for message in messages]
    result[0]["content"] = system.replace(catalogue, replacement, 1)
    return result, {"applied": True, "deferred_names": cold,
                    "source_characters": len(catalogue), "index_characters": len(replacement),
                    "all_original_names_retained": True,
                    "state_changing_and_used_contracts_protected": True,
                    "unique_observations_removed": 0, "new_model_calls_for_projection": 0}


def install_adapter(config, directory, role, mode, skill):
    if (config.get("stage") != "train-candidate" or config.get("evaluation_split") != "train"
            or config.get("method") != "chisel" or mode != "jev"):
        raise RuntimeError("Lazy native schemas are a prospectively frozen TRAIN candidate only")
    variant = config.get("lazy_schema_variant")
    if variant not in ("parent", "lazy_read_schemas"):
        raise ValueError("Declare the matched parent or lazy-read-schema variant")
    from are.simulation.tools import Tool
    from are.simulation.tool_box import get_tool_description_with_args
    from phiagent.harness import gaia2_202060_progress as progress

    parent = progress.ReceiptEngine

    class SchemaEngine(parent):
        def __init__(self, agent, provider, *, skill, record, execute_followups):
            original_tools = dict(agent.action_executor.tools)
            descriptions = {name: get_tool_description_with_args(tool)
                            for name, tool in original_tools.items()}
            summaries = {name: tool.description for name, tool in original_tools.items()}
            read_only = {
                name for name, tool in original_tools.items()
                if read_schema_can_be_deferred(
                    name, write_operation=getattr(getattr(tool, "app_tool", None), "write_operation", None),
                )
            }
            hot, full_exposures = set(), 0
            first_log = agent.logs[0] if agent.logs else None

            class SchemaLookup(Tool):
                name = LOOKUP
                description = (
                    "Return the exact original native description and argument schema for one "
                    "available tool. Reads metadata only; never reads private scenario state."
                )
                inputs = {"tool_name": {"type": "string", "description": "Exact name in the native tool index"}}
                output_type = "string"

                def forward(self, tool_name):
                    if not isinstance(tool_name, str) or tool_name not in descriptions:
                        raise ValueError("Unknown native tool name")
                    hot.add(tool_name)
                    text = descriptions[tool_name]
                    record({"source": "exact_native_schema_lookup", "tool_name": tool_name,
                            "schema_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "new_model_calls": 0, "native_action_iteration_consumed": True})
                    return text

            if LOOKUP in agent.action_executor.tools:
                raise ValueError("The schema lookup name is already registered")
            lookup = SchemaLookup()
            lookup.validate_arguments()
            agent.action_executor.tools[LOOKUP] = lookup
            declaration = get_tool_description_with_args(lookup)

            def projected_provider(messages, **kwargs):
                nonlocal first_log, full_exposures
                current_first = agent.logs[0] if agent.logs else None
                if current_first is not first_log:
                    first_log, full_exposures = current_first, 0
                    hot.clear()
                for entry in agent.logs:
                    if entry.get_type() == "tool_call":
                        name = getattr(entry, "tool_name", None)
                        if name in original_tools:
                            hot.add(name)
                original = deepcopy(messages)
                if full_exposures < 2:
                    projected, audit = messages, {"applied": False, "reason": "first_two_full_catalogue_exposures"}
                else:
                    projected, audit = project_catalogue(messages, descriptions, summaries, hot, read_only)
                    if audit["applied"]:
                        projected[0]["content"] += "\n" + declaration
                record({"source": "native_schema_projection", "agent_id": agent.agent_id,
                        "completed_full_exposures": full_exposures, **audit})
                response = provider(projected, **kwargs)
                if not isinstance(response, tuple) or len(response) != 2:
                    raise ValueError("Provider text and exact usage are required")
                usage = response[1]
                fields = [usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
                if (any(type(value) is not int or value < 0 for value in fields)
                        or fields[0] + fields[1] != fields[2] or fields[2] <= 0):
                    raise ValueError("A successful catalogue exposure needs complete native usage")
                if not audit["applied"]:
                    full_exposures += 1
                if messages != original:
                    raise ValueError("Native history was modified by schema projection")
                return response

            super().__init__(agent, projected_provider, skill=skill, record=record,
                             execute_followups=execute_followups)

    if variant == "lazy_read_schemas":
        progress.ReceiptEngine = SchemaEngine
    progress.install_adapter(config, directory, role, mode, skill)
