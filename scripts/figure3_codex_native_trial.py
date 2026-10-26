#!/usr/bin/env python3
"""Task-local installation adapter; inherit the pinned official Codex execution loop."""

from pathlib import Path
import shlex

from phiagent.data_engine.provenance import write_json_atomic
from scripts import industrial_terminal_agents
from scripts import run_codex_terminal


class TaskLocalCodex(industrial_terminal_agents.SharedPromptCodex):
    _REMOTE_CODEX_HOME = Path("/installed-agent/phiagent-codex-home")
    _REMOTE_CODEX_SECRETS_DIR = Path("/installed-agent/phiagent-codex-secrets")

    async def install(self, environment):
        await environment.upload_file(self._codex_binary, "/installed-agent/codex")
        result = await self.exec_as_root(
            environment, command="install -m 755 /installed-agent/codex /usr/local/bin/codex",
        )
        if result.return_code or not await self._installed_codex_satisfies_version(environment):
            raise RuntimeError("Pinned official Codex installation failed")
        home = self._REMOTE_CODEX_HOME.as_posix()
        result = await self.exec_as_agent(environment, command="mkdir -p " + shlex.quote(home))
        if result.return_code:
            raise RuntimeError("Task-local Codex home creation failed")
        config = self.logs_dir / "provider-config.toml"
        config.write_text(self._provider_config)
        remote = home + "/config.toml"
        await environment.upload_file(config, remote)
        if environment.default_user is not None:
            result = await self.exec_as_root(
                environment,
                command=f"chown {shlex.quote(environment.default_user)} {shlex.quote(remote)}",
            )
            if result.return_code:
                raise RuntimeError("Task-local provider configuration ownership failed")
        write_json_atomic(self.logs_dir / "CONTEXT_POLICY.json", {
            "mechanism": "official_codex_automatic_compaction",
            "active_context_trigger_tokens": self._context_compaction_threshold,
            "model_context_window": 1048576,
            "cumulative_token_limit": None,
            "compaction_calls_in_provider_ledger": True,
            "provider_requests_rewritten": False,
            "codex_home": home,
            "native_execution_loop_inherited": True,
        })


if __name__ == "__main__":
    industrial_terminal_agents.SharedPromptCodex = TaskLocalCodex
    run_codex_terminal.main()
