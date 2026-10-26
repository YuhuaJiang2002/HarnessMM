"""Prompt construction for compute-matched terminal-agent scaffold deletion."""

from __future__ import annotations


def delete_terminal_scaffold_fields(
    template: str,
    *,
    include_analysis: bool,
    include_plan: bool,
) -> str:
    """Remove selected reasoning fields without rewriting the remaining prompt."""

    if include_analysis and include_plan:
        return template

    omitted = []
    if not include_analysis:
        omitted.append("analysis")
    if not include_plan:
        omitted.append("plan")

    filtered = []
    for line in template.splitlines():
        stripped = line.strip()
        if not include_analysis and (
            stripped.startswith('"analysis":')
            or stripped.startswith('- "analysis":')
        ):
            continue
        if not include_plan and (
            stripped.startswith('"plan":') or stripped.startswith('- "plan":')
        ):
            continue
        filtered.append(line)

    marker = "Required fields:"
    index = filtered.index(marker) + 1
    filtered.insert(index, f'- Do not emit omitted fields: {", ".join(omitted)}.')
    return "\n".join(filtered) + ("\n" if template.endswith("\n") else "")


def build_terminal_scaffold_template(*, include_analysis: bool, include_plan: bool) -> str:
    """Build a Terminus-compatible prompt with selected reasoning scaffold fields."""

    fields: list[tuple[str, str]] = []
    if include_analysis:
        fields.append(
            (
                "analysis",
                "Briefly diagnose the current terminal state and remaining work.",
            )
        )
    if include_plan:
        fields.append(
            (
                "plan",
                "Briefly state what the next command batch should accomplish.",
            )
        )
    fields.append(
        (
            "commands",
            '[{{"keystrokes": "ls -la\\n", "duration": 0.1}}]',
        )
    )

    example_lines = ["{{"]
    for index, (name, value) in enumerate(fields):
        comma = "," if index < len(fields) - 1 else ""
        if name == "commands":
            example_lines.append(f'  "{name}": {value}{comma}')
        else:
            example_lines.append(f'  "{name}": "{value}"{comma}')
    example_lines.append('  ,"task_complete": false')
    example_lines.append("}}")

    required = [name for name, _ in fields]
    omitted = [
        name
        for name, enabled in (
            ("analysis", include_analysis),
            ("plan", include_plan),
        )
        if not enabled
    ]
    omitted_instruction = (
        "Do not emit omitted fields: " + ", ".join(omitted) + "."
        if omitted
        else "Emit every scaffold field shown below."
    )

    return f"""You are an AI assistant solving command-line tasks in a Linux environment.
You will receive a task description and output from previously executed commands.
Respond with exactly one JSON object and no surrounding prose.

Required top-level fields: {", ".join(required)}.
Optional top-level field: task_complete (boolean, default false).
{omitted_instruction}

Response schema example:
{chr(10).join(example_lines)}

Each command object must contain:
- keystrokes: exact terminal keystrokes, ending in a newline when the command should run.
- duration: seconds to wait before the next command, never more than 60.

Use short waits and poll long-running commands with an empty keystrokes string.
The commands interface, command budget, terminal observations, and completion protocol
are identical across scaffold variants.

Task Description:
{{instruction}}

Current terminal state:
{{terminal_state}}
"""
