"""Native ARE baseline with provider metering and no agent-policy wrapper."""
from __future__ import annotations


def validate_identity(config, mode, skill):
    if (config.get("baseline_identity") != "no_harness"
            or config.get("method") != "native" or mode != "native" or skill != ""
            or config.get("frozen_skill", "") != ""
            or config.get("receipt_execution") is not False
            or config.get("chisel_policy")):
        raise ValueError("No Harness requires the native actor without skills or CHISEL controls")
    if (config["budget"].get("maximum_tokens") is not None
            or config["budget"].get("maximum_requests") is not None
            or config["budget"].get("token_usage_mode") != "unlimited"):
        raise ValueError("No Harness must retain unlimited cumulative accounting")


def install_adapter(config, directory, role, mode, skill):
    validate_identity(config, mode, skill)
    from are.simulation.agents.default_agent.base_agent import BaseAgent
    from phiagent.harness.gaia2_202060_progress import install_metered_provider

    native_step = BaseAgent.step
    if native_step.__module__ != "are.simulation.agents.default_agent.base_agent":
        raise RuntimeError("No Harness cannot inherit an already wrapped actor step")
    record = install_metered_provider(config, directory, role)
    if BaseAgent.step is not native_step:
        raise RuntimeError("Provider metering unexpectedly changed the native actor policy")
    record({
        "source": "no_harness_identity",
        "native_step_module": native_step.__module__,
        "native_step_name": native_step.__qualname__,
        "agent_step_wrapper_installed": False,
        "skills_added": False,
        "context_projection": False,
        "compiled_followups": False,
        "native_task_history_tools_preserved": True,
        "provider_metering_only": True,
        "new_model_calls": 0,
    })
