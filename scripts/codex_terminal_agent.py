"""Installation/provider adapter around Harbor 0.18.0's unchanged Codex loop."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shlex

from harbor.agents.installed.codex import Codex

from phiagent.data_engine.provenance import write_json_atomic
from phiagent.harness.codex_runtime import CODEX_SHA256, codex_config
from scripts.serve_codex_deepseek import read_credential, trial_token


class MeteredCodex(Codex):
    """Use the pinned official binary and native provider; alter no tool semantics."""

    def __init__(
        self, *args, provider_url: str, trial_id: str, relay_secret_file: str,
        codex_binary: str, residual: str = "", context_compaction_threshold: int = 996147,
        **kwargs,
    ):
        if kwargs.get("version") != "0.142.5":
            raise ValueError("The Codex version must be pinned to 0.142.5")
        self._provider_url = provider_url.rstrip("/") + "/" + trial_id
        self._provider_config = codex_config(
            self._provider_url, context_compaction_threshold=context_compaction_threshold,
        )
        self._context_compaction_threshold = context_compaction_threshold
        self._relay_token = trial_token(read_credential(Path(relay_secret_file)), trial_id)
        self._codex_binary = Path(codex_binary)
        if hashlib.sha256(self._codex_binary.read_bytes()).hexdigest() != CODEX_SHA256:
            raise ValueError("Codex executable does not match the pinned official release")
        if not isinstance(residual, str):
            raise ValueError("A single text residual is required")
        self._residual = residual
        super().__init__(*args, **kwargs)

    def _get_env(self, name: str):
        if name == "OPENAI_API_KEY":
            return self._relay_token
        if name in ("OPENAI_BASE_URL", "CODEX_AUTH_JSON_PATH", "CODEX_FORCE_AUTH_JSON"):
            return None
        return super()._get_env(name)

    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        return await super().exec_as_agent(
            environment, command,
            env={**(env or {}), "CODEX_PROVIDER_TOKEN": self._relay_token},
            cwd=cwd, timeout_sec=timeout_sec,
        )

    async def install(self, environment):
        await environment.upload_file(self._codex_binary, "/installed-agent/codex")
        result = await self.exec_as_root(
            environment,
            command="install -m 755 /installed-agent/codex /usr/local/bin/codex",
        )
        if result.return_code != 0:
            raise RuntimeError("Installing the verified native Codex executable failed")
        if not await self._installed_codex_satisfies_version(environment):
            raise RuntimeError("Installed native Codex version failed verification")
        result = await self.exec_as_agent(
            environment, command="mkdir -p /tmp/codex-home"
        )
        if result.return_code != 0:
            raise RuntimeError("Creating the per-trial Codex home failed")
        config = self.logs_dir / "provider-config.toml"
        config.write_text(self._provider_config)
        write_json_atomic(self.logs_dir / "CONTEXT_POLICY.json", {
            "mechanism": "official_codex_automatic_compaction",
            "active_context_trigger_tokens": self._context_compaction_threshold,
            "model_context_window": 1048576,
            "cumulative_token_limit": None,
            "compaction_calls_in_provider_ledger": True,
            "provider_requests_rewritten": False,
        })
        remote = "/tmp/codex-home/config.toml"
        await environment.upload_file(config, remote)
        if environment.default_user is not None:
            result = await self.exec_as_root(
                environment,
                command=f"chown {shlex.quote(environment.default_user)} {shlex.quote(remote)}",
            )
            if result.return_code != 0:
                raise RuntimeError("Setting per-trial Codex configuration ownership failed")

    async def run(self, instruction, environment, context):
        if self._residual:
            instruction = instruction + "\n\n" + self._residual
        await super().run(instruction, environment, context)
