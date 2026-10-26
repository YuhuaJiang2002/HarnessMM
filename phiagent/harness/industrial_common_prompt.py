"""Versioned public task brief shared by each native industrial parent and CHISEL."""

from __future__ import annotations

import hashlib
from pathlib import Path

from phiagent.data_engine.provenance import write_json_atomic

BENCHMARK = "terminal-bench-2.0"
PROMPT_VERSION = "industrial-coding-completion/v1"
PROMPT_OPEN = '<industrial-coding-completion version="1">'
PROMPT_CLOSE = "</industrial-coding-completion>"
SHARED_PROMPT = (
    PROMPT_OPEN + "\n"
    "Complete the supplied coding task in its workspace. Inspect the full task "
    "instructions and relevant files, then implement the required changes rather "
    "than stopping at a plan. Match the requested artifacts, paths, interfaces, "
    "and output formats exactly. Use available task-relevant checks to verify "
    "your work, fix diagnosed errors, and recheck changed behavior. Report only "
    "checks you actually ran and their observed results. Do not modify the "
    "task's scoring or verification machinery.\n"
    + PROMPT_CLOSE
)
PROMPT_SHA256 = hashlib.sha256(SHARED_PROMPT.encode("utf-8")).hexdigest()


def prompt_manifest(
    benchmark: str = BENCHMARK,
    version: str = PROMPT_VERSION,
    sha256: str = PROMPT_SHA256,
) -> dict:
    if benchmark != BENCHMARK:
        raise ValueError("The shared coding prompt supports only terminal-bench-2.0")
    if version != PROMPT_VERSION or sha256 != PROMPT_SHA256:
        raise ValueError("Shared prompt version/hash differs from the frozen public brief")
    return {
        "benchmark": benchmark,
        "version": version,
        "sha256": sha256,
        "text": SHARED_PROMPT,
        "placement": "original_task_instruction_suffix",
        "native_system_and_loop": "unchanged",
        "chisel_residual": "separate_from_shared_prompt",
    }


def append_common_prompt(
    instruction: str, *, benchmark: str = BENCHMARK,
    version: str = PROMPT_VERSION, sha256: str = PROMPT_SHA256,
) -> str:
    prompt_manifest(benchmark, version, sha256)
    if not isinstance(instruction, str) or not instruction:
        raise ValueError("Preserve a nonempty original task instruction")
    if "<industrial-coding-completion" in instruction or PROMPT_CLOSE in instruction:
        if (
            instruction.count("<industrial-coding-completion") != 1
            or instruction.count(PROMPT_CLOSE) != 1
            or not instruction.endswith("\n\n" + SHARED_PROMPT)
        ):
            raise ValueError("Duplicate, modified, or misplaced shared completion block")
        return instruction
    return instruction + "\n\n" + SHARED_PROMPT


def prompt_receipt(instruction: str, **identity: str) -> tuple[str, dict]:
    rendered = append_common_prompt(instruction, **identity)
    original = rendered[: -len("\n\n" + SHARED_PROMPT)]
    return rendered, {
        **prompt_manifest(**identity),
        "original_task_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
        "original_task_bytes": len(original.encode("utf-8")),
        "rendered_task_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "shared_block_count": 1,
        "original_task_preserved": True,
    }


class CommonPromptMixin:
    """Add the same task suffix before the existing adapter applies any residual."""

    def __init__(
        self, *args, shared_prompt_version: str, shared_prompt_sha256: str,
        benchmark: str = BENCHMARK, **kwargs,
    ):
        prompt_manifest(benchmark, shared_prompt_version, shared_prompt_sha256)
        self._common_identity = {
            "benchmark": benchmark, "version": shared_prompt_version,
            "sha256": shared_prompt_sha256,
        }
        super().__init__(*args, **kwargs)

    async def run(self, instruction, environment, context):
        rendered, receipt = prompt_receipt(instruction, **self._common_identity)
        write_json_atomic(Path(self.logs_dir) / "SHARED_PROMPT.json", receipt)
        await super().run(rendered, environment, context)
