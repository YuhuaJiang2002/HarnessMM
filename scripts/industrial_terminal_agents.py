"""Shared-prompt entry points; native Codex/SDK implementations remain inherited."""

from phiagent.harness.industrial_common_prompt import CommonPromptMixin
from scripts.codex_terminal_agent import MeteredCodex
from scripts.run_deepseek_terminal_campaign import NativeCompletionAgent


class SharedPromptCodex(CommonPromptMixin, MeteredCodex):
    pass


class SharedPromptDeepSeek(CommonPromptMixin, NativeCompletionAgent):
    pass
