"""Frozen Terminal-Bench component-edit sequences for constrained GSPO DEV."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class TerminalHarnessCandidate:
    action_id: str
    operations: tuple[str, ...]
    max_turns: int
    components: tuple[tuple[str, str], ...]

    @property
    def component_map(self) -> dict[str, str]:
        return dict(self.components)

    @property
    def component_sha256(self) -> str:
        payload = "\n".join(f"{key}\0{value}" for key, value in self.components)
        return hashlib.sha256(payload.encode()).hexdigest()

    def render(self, base_prompt: str) -> str:
        sections = "\n\n".join(
            f"### {name.replace('_', ' ').title()}\n{text}"
            for name, text in self.components
        )
        return (
            base_prompt.rstrip()
            + "\n\n## Ours: constrained component policy\n"
            + sections
            + "\n\nThe task instruction remains authoritative.\n"
        )


def terminal_candidates() -> tuple[TerminalHarnessCandidate, ...]:
    definitions = (
        TerminalHarnessCandidate(
            action_id="adm_compact_12",
            operations=(
                "ADD:inspect_before_action",
                "DELETE:repeated_dependency_retries",
                "MODIFY:max_turns_30_to_12",
            ),
            max_turns=12,
            components=(
                (
                    "execution_component",
                    "Inspect the working directory and relevant source before editing. "
                    "Translate the request into the smallest concrete artifact change.",
                ),
                (
                    "efficiency_component",
                    "Never repeat a failed dependency or network command. Pivot once to "
                    "installed tools, direct file manipulation, or a simpler implementation.",
                ),
                (
                    "validation_component",
                    "Run the task-named compiler, checker, or executable once. Fix a concrete "
                    "failure if time remains, then finish immediately after a passing check.",
                ),
            ),
        ),
        TerminalHarnessCandidate(
            action_id="adm_guarded_16",
            operations=(
                "ADD:failure_taxonomy",
                "DELETE:blind_package_installation",
                "MODIFY:max_turns_30_to_16",
            ),
            max_turns=16,
            components=(
                (
                    "execution_component",
                    "First classify the task as recovery, proof, transformation, build, or "
                    "data processing. Inspect exact inputs and preserve required formats.",
                ),
                (
                    "efficiency_component",
                    "Use existing domain tools and files before installing anything. A failed "
                    "network install gets no identical retry; choose an offline path.",
                ),
                (
                    "validation_component",
                    "Validate observable behavior, not just file existence. Keep a known-good "
                    "state, make one targeted correction per failed check, and stop on success.",
                ),
            ),
        ),
        TerminalHarnessCandidate(
            action_id="adm_validation_20",
            operations=(
                "ADD:state_checkpoint_and_validation",
                "DELETE:verbose_reinspection",
                "MODIFY:max_turns_30_to_20",
            ),
            max_turns=20,
            components=(
                (
                    "execution_component",
                    "Inspect inputs, infer formats and invariants, then implement a direct "
                    "solution in small executable steps. Save a checkpoint before destructive edits.",
                ),
                (
                    "efficiency_component",
                    "Do not re-read unchanged large files or retry an unchanged failing command. "
                    "Prefer concise probes that decide the next edit.",
                ),
                (
                    "validation_component",
                    "Exercise the exact requested entry point with representative provided data. "
                    "Compare outputs or compiler diagnostics and repair discrepancies before finishing.",
                ),
            ),
        ),
        TerminalHarnessCandidate(
            action_id="adm_conservative_24",
            operations=(
                "ADD:task_decomposition",
                "DELETE:duplicate_commands",
                "MODIFY:max_turns_30_to_24",
            ),
            max_turns=24,
            components=(
                (
                    "execution_component",
                    "Decompose the task into inspect, implement, and verify phases. Read all "
                    "task-relevant local specifications before choosing an implementation.",
                ),
                (
                    "efficiency_component",
                    "Track completed subgoals and never issue an identical ineffective command. "
                    "Use package installation only when no present tool can perform the work.",
                ),
                (
                    "validation_component",
                    "Run the requested compiler or program and inspect its real output. Preserve "
                    "working artifacts while fixing one diagnosed issue at a time; stop when complete.",
                ),
            ),
        ),
    )
    candidates = tuple(sorted(definitions, key=lambda candidate: candidate.action_id))
    if [candidate.action_id for candidate in candidates] != sorted(
        candidate.action_id for candidate in candidates
    ):
        raise RuntimeError("Terminal candidate IDs must be sorted")
    if len({candidate.component_sha256 for candidate in candidates}) != len(candidates):
        raise RuntimeError("Terminal component candidates must be unique")
    return candidates


__all__ = ["TerminalHarnessCandidate", "terminal_candidates"]
