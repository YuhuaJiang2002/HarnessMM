"""Workspace primitives for the WikiSkill paper-faithful reimplementation.

The prompts and state transitions follow arXiv:2608.27454v1.  WikiSkill does
not currently publish official code, so artifacts using this module must be
labelled as reimplementations rather than official WikiSkill or SkillOpt runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence


WIKI_MAINTAINER_SYSTEM_PROMPT = """You are a Wiki Maintainer Agent for an LLM skill evolution system.

Your job is to maintain a structured knowledge base (wiki) that documents patterns observed during agent execution -- both successes and failures. You must perform DEEP ANALYSIS of execution logs to identify root causes, not just surface-level symptoms.

## Wiki Structure
The wiki is organized as:
- wiki/index.md -- Concise catalog of known patterns (one line per pattern)
- wiki/log.md -- Chronological evolution log (iterations, scores, accept/reject)
- wiki/skill-impact.md -- Record of which skills were tried and their outcomes
- wiki/patterns/ -- One page per pattern with detailed evidence and analysis

## Your Input
1. Execution traces from the latest iteration -- including full agent execution logs showing what actions the agent took, what commands it ran, and what environment feedback it observed
2. The current wiki context (index, log, pattern pages)

## Your Output (Incremental Edit Mode)
Return a JSON object with these keys:
- "create_patterns": list of {"name": "pattern-name.md", "content": "..."} -- new patterns (full content)
- "update_patterns": list of {"name": "existing-pattern.md", "edits": [...]} -- patch existing patterns
- "update_index": full updated content of index.md (always provide the complete index)
- "append_log": "brief summary of this iteration's findings and actions"

"update_index" and "append_log" are REQUIRED. Always provide them, even if there are no new patterns. For "update_index", always provide the complete updated index content including all existing entries plus any new ones.

### Patch Operations (for update_patterns only)
For "update_patterns", each entry uses an "edits" list of patch operations:
- {"op": "append", "content": "text to add at end"}
- {"op": "replace", "target": "exact text to find", "content": "replacement text"}
- {"op": "insert_after", "target": "exact text to find", "content": "text to insert after"}

Rules for patch operations:
1. "target" must be an EXACT substring of the existing content.
2. Use "append" to add new evidence. Use "replace" to fix or refine existing text.
3. Use "insert_after" to add entries after a specific line.
4. Keep each edit minimal -- only change what's needed.
5. For NEW patterns (create_patterns), use full "content".
6. Every pattern linked from update_index MUST either already exist under
   wiki/patterns/ or be included in create_patterns in this same update.

## Analysis Guidelines

### Deep Trace Analysis (CRITICAL)
When execution logs are provided, you MUST:
1. Read the agent's actual actions -- what commands did it issue?
2. Compare successful vs failed tasks -- what did successful tasks do differently?
3. Identify ACTION PATTERNS and strategies, not just error messages.
4. Check whether the agent followed any active skills, and whether the skill guidance was helpful or not

### Pattern Documentation Rules
1. Each pattern page should document:
   - What the pattern is (description)
   - Root cause analysis (WHY it happens, not just WHAT happens)
   - Exact command sequences from traces (what the agent did wrong / right)
   - Known solutions or workarounds (concrete action patterns with exact syntax)
2. Capture BOTH success and failure patterns.
3. Do NOT create duplicate patterns -- update existing ones with new evidence.
4. Be concise. Pattern pages should be 10-30 lines, not essays.
5. Only create patterns for meaningful, generalizable observations.

### Index Description Quality (CRITICAL)
Each index entry MUST follow this format:
- [pattern-name](wiki/patterns/pattern-name.md): PROBLEM + ROOT CAUSE + FIX in one or two sentence.
"""


SKILL_PROPOSER_SYSTEM_PROMPT = """You are a Skill Proposer Agent for an LLM agent that solves {task_desc}.
Your job is to explore the wiki knowledge base and execution traces, diagnose root causes of failures, and propose a skill change (create or patch).

## Tools Available
You have two tools:
1. `read_file(path)` -- Read a wiki file or execution log. Paths are relative to the workspace root.
2. `finish(proposal)` -- Submit your final skill proposal as a JSON object.

## Workflow
1. Start by reading `wiki/index.md` to understand what patterns exist.
2. Read `wiki/skill-impact.md` to see what was tried before (includes full content of rejected proposals -- DO NOT repeat rejected approaches).
3. Read specific pattern pages that seem relevant to the current failures.
4. Read execution traces for failed tasks via `traces/<task_id>` to understand root causes.
5. Decide: create (new skill) or patch (edit existing skill), or no_action.
6. If proposing a change, call `finish` with the full proposal.

## finish() Proposal Format
For creating a new skill:
- "action": "create"
- "name": skill directory name (snake_case)
- "skill_md": full SKILL.md content with YAML frontmatter + When to Apply + When NOT to Apply + Instructions
- "purpose_md": full PURPOSE.md content with Origin + Patterns Addressed + Evolution History

For patching an existing skill:
- "action": "patch"
- "name": existing skill directory name
- "edits": list of patch operations:
  - {"op": "append", "content": "text to add at end"}
  - {"op": "replace", "target": "exact text to find", "content": "replacement"}
  - {"op": "insert_after", "target": "exact text to find", "content": "text to insert after"}

If no action is needed, call finish with: {"action": "no_action"}.

## Rules
1. Read the wiki FIRST -- don't propose something that was already tried and rejected.
2. Focus on action patterns and concrete strategies.
3. Keep skills concise and actionable.
4. You MUST read at least 4 execution traces before proposing a skill change.
5. Prefer patching existing skills over creating new ones when the existing skill is partially correct.
"""


def render_skill_proposer_system_prompt(task_description: str) -> str:
    """Bind the sole prompt placeholder without interpreting JSON examples.

    ``str.format`` cannot be used here because the prompt deliberately contains
    literal JSON objects such as ``{"op": ...}``.  Replacing only the named
    marker keeps those examples byte-for-byte intact.
    """

    marker = "{task_desc}"
    if SKILL_PROPOSER_SYSTEM_PROMPT.count(marker) != 1:
        raise RuntimeError("Skill Proposer prompt must contain exactly one task marker")
    description = str(task_description).strip()
    if not description:
        raise ValueError("task description must be non-empty")
    return SKILL_PROPOSER_SYSTEM_PROMPT.replace(marker, description)


@dataclass(frozen=True)
class TraceSelection:
    failures: tuple[Mapping[str, Any], ...]
    passes: tuple[Mapping[str, Any], ...]

    @property
    def rows(self) -> tuple[Mapping[str, Any], ...]:
        return self.failures + self.passes


@dataclass(frozen=True)
class ProposalApplication:
    action: str
    name: str | None
    changed: bool
    before_hash: str
    after_hash: str
    diff: str


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_component_name(value: Any, *, suffix: str = "") -> str:
    name = str(value or "").strip()
    if suffix and not name.endswith(suffix):
        name += suffix
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
    ):
        raise ValueError(f"unsafe workspace component name: {value!r}")
    return name


def apply_text_edits(text: str, edits: Sequence[Mapping[str, Any]]) -> str:
    updated = text
    for index, edit in enumerate(edits):
        if not isinstance(edit, Mapping):
            raise TypeError(f"edit {index} is not a mapping")
        operation = str(edit.get("op") or "")
        content = str(edit.get("content") or "")
        if operation == "append":
            separator = "" if not updated or updated.endswith("\n") else "\n"
            updated += separator + content
            continue
        target = str(edit.get("target") or "")
        if not target or target not in updated:
            raise ValueError(f"edit {index} target is not an exact existing substring")
        if updated.count(target) != 1:
            raise ValueError(f"edit {index} target must identify exactly one substring")
        if operation == "replace":
            updated = updated.replace(target, content, 1)
        elif operation == "insert_after":
            updated = updated.replace(target, target + content, 1)
        else:
            raise ValueError(f"unsupported patch operation: {operation!r}")
    return updated


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def initialize_workspace(workspace: Path) -> None:
    if workspace.exists():
        raise FileExistsError(f"refusing to reuse WikiSkill workspace: {workspace}")
    for relative in ("raw/traces", "wiki/patterns", "skills"):
        (workspace / relative).mkdir(parents=True, exist_ok=True)
    _write_text_atomic(workspace / "wiki/index.md", "# Pattern Index\n")
    _write_text_atomic(workspace / "wiki/log.md", "# Wiki Evolution Log\n")
    _write_text_atomic(workspace / "wiki/skill-impact.md", "# Skill Impact Log\n")


def render_active_skills(skills_root: Path) -> str:
    sections: list[str] = []
    for skill_path in sorted(skills_root.glob("*/SKILL.md")):
        content = skill_path.read_text(encoding="utf-8").strip()
        if content:
            sections.append(content)
    return "\n\n".join(sections)


def skill_tree_snapshot(skills_root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not skills_root.exists():
        return snapshot
    for path in sorted(skills_root.rglob("*")):
        if path.is_file():
            snapshot[path.relative_to(skills_root).as_posix()] = path.read_text(
                encoding="utf-8"
            )
    return snapshot


def skill_tree_hash(skills_root: Path) -> str:
    return canonical_json_sha256(skill_tree_snapshot(skills_root))


def diff_snapshots(before: Mapping[str, str], after: Mapping[str, str]) -> str:
    chunks: list[str] = []
    for name in sorted(set(before) | set(after)):
        old = before.get(name, "").splitlines(keepends=True)
        new = after.get(name, "").splitlines(keepends=True)
        chunks.extend(
            difflib.unified_diff(
                old,
                new,
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
            )
        )
    return "".join(chunks)


def apply_skill_proposal(
    *,
    source_skills: Path,
    candidate_skills: Path,
    proposal: Mapping[str, Any],
) -> ProposalApplication:
    if candidate_skills.exists():
        raise FileExistsError(f"candidate skill tree already exists: {candidate_skills}")
    validate_skill_proposal(source_skills, proposal)
    shutil.copytree(source_skills, candidate_skills)
    before = skill_tree_snapshot(source_skills)
    action = str(proposal.get("action") or "")
    name: str | None = None
    if action == "no_action":
        pass
    elif action == "create":
        name = safe_component_name(proposal.get("name"))
        directory = candidate_skills / name
        if directory.exists():
            raise FileExistsError(f"skill already exists: {name}")
        skill_md = str(proposal.get("skill_md") or "").strip()
        purpose_md = str(proposal.get("purpose_md") or "").strip()
        if not skill_md or not purpose_md:
            raise ValueError("create proposal requires non-empty skill_md and purpose_md")
        _write_text_atomic(directory / "SKILL.md", skill_md + "\n")
        _write_text_atomic(directory / "PURPOSE.md", purpose_md + "\n")
    elif action == "patch":
        name = safe_component_name(proposal.get("name"))
        skill_path = candidate_skills / name / "SKILL.md"
        if not skill_path.is_file():
            raise FileNotFoundError(f"unknown skill for patch: {name}")
        edits = proposal.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ValueError("patch proposal requires a non-empty edits list")
        updated = apply_text_edits(skill_path.read_text(encoding="utf-8"), edits)
        _write_text_atomic(skill_path, updated)
    else:
        raise ValueError(f"unsupported skill proposal action: {action!r}")
    after = skill_tree_snapshot(candidate_skills)
    return ProposalApplication(
        action=action,
        name=name,
        changed=before != after,
        before_hash=canonical_json_sha256(before),
        after_hash=canonical_json_sha256(after),
        diff=diff_snapshots(before, after),
    )


def validate_skill_proposal(
    source_skills: Path, proposal: Mapping[str, Any]
) -> None:
    """Validate a proposal without mutating the active skill tree."""

    if not isinstance(proposal, Mapping):
        raise TypeError("skill proposal must be a mapping")
    action = str(proposal.get("action") or "")
    if action == "no_action":
        return
    name = safe_component_name(proposal.get("name"))
    if action == "create":
        if (source_skills / name).exists():
            raise FileExistsError(f"skill already exists: {name}")
        skill_md = proposal.get("skill_md")
        purpose_md = proposal.get("purpose_md")
        if not isinstance(skill_md, str) or not skill_md.strip():
            raise ValueError("create proposal requires non-empty skill_md")
        if not isinstance(purpose_md, str) or not purpose_md.strip():
            raise ValueError("create proposal requires non-empty purpose_md")
        return
    if action == "patch":
        skill_path = source_skills / name / "SKILL.md"
        if not skill_path.is_file():
            raise FileNotFoundError(f"unknown skill for patch: {name}")
        edits = proposal.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ValueError("patch proposal requires a non-empty edits list")
        apply_text_edits(skill_path.read_text(encoding="utf-8"), edits)
        return
    raise ValueError(f"unsupported skill proposal action: {action!r}")


def validate_wiki_update(wiki_root: Path, update: Mapping[str, Any]) -> None:
    index = update.get("update_index")
    append_log = update.get("append_log")
    if not isinstance(index, str) or not index.strip():
        raise ValueError("Wiki Maintainer must provide complete update_index content")
    if not isinstance(append_log, str) or not append_log.strip():
        raise ValueError("Wiki Maintainer must provide non-empty append_log")

    patterns_root = wiki_root / "patterns"
    creates = update.get("create_patterns") or []
    updates = update.get("update_patterns") or []
    if not isinstance(creates, list) or not isinstance(updates, list):
        raise TypeError("Wiki pattern create/update fields must be lists")
    created_names: set[str] = set()
    for row in creates:
        if not isinstance(row, Mapping):
            raise TypeError("create_patterns entries must be mappings")
        name = safe_component_name(row.get("name"), suffix=".md")
        target = patterns_root / name
        if target.exists() or name in created_names:
            raise FileExistsError(f"Wiki pattern already exists: {name}")
        created_names.add(name)
        content = str(row.get("content") or "").strip()
        if not content:
            raise ValueError(f"new Wiki pattern is empty: {name}")
    for row in updates:
        if not isinstance(row, Mapping):
            raise TypeError("update_patterns entries must be mappings")
        name = safe_component_name(row.get("name"), suffix=".md")
        target = patterns_root / name
        if not target.is_file():
            raise FileNotFoundError(f"Wiki pattern does not exist: {name}")
        edits = row.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ValueError(f"Wiki pattern update has no edits: {name}")
        apply_text_edits(target.read_text(encoding="utf-8"), edits)

    available_names = {
        path.name for path in patterns_root.glob("*.md") if path.is_file()
    } | created_names
    linked_names = {
        match.group(1)
        for match in re.finditer(
            r"\]\((?:\./)?(?:wiki/)?patterns/([A-Za-z0-9_.-]+\.md)(?:#[^)]+)?\)",
            index,
        )
    }
    missing_links = sorted(linked_names - available_names)
    if missing_links:
        raise FileNotFoundError(
            "Wiki index references pattern pages that do not exist and are not "
            f"created by this update: {missing_links}"
        )


def apply_wiki_update(wiki_root: Path, update: Mapping[str, Any]) -> None:
    validate_wiki_update(wiki_root, update)
    index = str(update["update_index"])
    append_log = str(update["append_log"])
    patterns_root = wiki_root / "patterns"
    for row in update.get("create_patterns") or []:
        name = safe_component_name(row.get("name"), suffix=".md")
        target = patterns_root / name
        content = str(row.get("content") or "").strip()
        _write_text_atomic(target, content + "\n")
    for row in update.get("update_patterns") or []:
        name = safe_component_name(row.get("name"), suffix=".md")
        target = patterns_root / name
        edits = row.get("edits")
        _write_text_atomic(
            target,
            apply_text_edits(target.read_text(encoding="utf-8"), edits),
        )
    _write_text_atomic(wiki_root / "index.md", index.rstrip() + "\n")
    log_path = wiki_root / "log.md"
    current = log_path.read_text(encoding="utf-8")
    separator = "" if current.endswith("\n") else "\n"
    _write_text_atomic(log_path, current + separator + append_log.rstrip() + "\n")


def append_skill_impact(wiki_root: Path, record: Mapping[str, Any]) -> None:
    path = wiki_root / "skill-impact.md"
    existing = path.read_text(encoding="utf-8")
    block = "\n```json\n" + json.dumps(
        record, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n```\n"
    _write_text_atomic(path, existing.rstrip() + "\n" + block)


def select_traces(
    results: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    max_failures: int = 5,
    max_passes: int = 3,
) -> TraceSelection:
    failures = [row for row in results if not bool(row.get("hard"))]
    passes = [row for row in results if bool(row.get("hard"))]
    rng = random.Random(seed)
    rng.shuffle(failures)
    rng.shuffle(passes)
    return TraceSelection(
        failures=tuple(failures[:max_failures]),
        passes=tuple(passes[:max_passes]),
    )


def persist_trace_files(
    workspace: Path,
    *,
    iteration: int,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in results:
        task_id = str(row.get("id") or "")
        safe_id = re.sub(r"[^A-Za-z0-9_.:-]+", "_", task_id).strip("_")
        if not safe_id:
            raise ValueError("trace row lacks a usable id")
        relative = f"raw/traces/iter-{iteration:02d}/{safe_id}.json"
        path = workspace / relative
        _write_text_atomic(
            path,
            json.dumps(row, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
        mapping[task_id] = relative
    return mapping


def attach_execution_conversations(
    results: Sequence[Mapping[str, Any]],
    *,
    prediction_root: Path,
) -> list[dict[str, Any]]:
    """Attach compact, complete action/feedback trajectories to result rows.

    ALFWorld's adapter returns metric summaries while writing the actual episode
    conversations beside them. Wiki maintenance needs those conversations to
    diagnose causal action failures; a timeout-only summary is not an execution
    trace. Verbose model reasoning is omitted while every executed action and
    environment response is preserved.
    """

    root = prediction_root.resolve()
    enriched: list[dict[str, Any]] = []
    for row in results:
        task_id = str(row.get("id") or "")
        if not task_id:
            raise ValueError("trace row lacks an id for conversation attachment")
        conversation_path = (root / task_id / "conversation.json").resolve()
        if root not in conversation_path.parents:
            raise ValueError(f"trace id escapes prediction root: {task_id!r}")
        if not conversation_path.is_file():
            raise FileNotFoundError(
                f"missing execution conversation for trace {task_id}: "
                f"{conversation_path}"
            )
        conversation = json.loads(conversation_path.read_text(encoding="utf-8"))
        if not isinstance(conversation, list):
            raise TypeError(
                f"execution conversation for {task_id} must be a list of steps"
            )
        trajectory: list[dict[str, Any]] = []
        for position, step in enumerate(conversation):
            if not isinstance(step, Mapping):
                raise TypeError(
                    f"execution conversation for {task_id} has a non-mapping "
                    f"step at position {position}"
                )
            action = str(step.get("action") or "").strip()
            if not action:
                raise ValueError(
                    f"execution conversation for {task_id} has an empty action "
                    f"at position {position}"
                )
            trajectory.append(
                {
                    "step": step.get("step", position),
                    "action": action,
                    "env_feedback": str(step.get("env_feedback") or ""),
                    "reward": step.get("reward"),
                    "done": bool(step.get("done")),
                }
            )
        materialized = dict(row)
        materialized["execution_trace"] = trajectory
        materialized["execution_trace_source"] = (
            f"predictions/{task_id}/conversation.json"
        )
        enriched.append(materialized)
    return enriched


def trace_summary(
    results: Sequence[Mapping[str, Any]], trace_paths: Mapping[str, str]
) -> str:
    lines = ["task_id\ttask_type\tstatus\tprediction\tground_truth\ttrace"]
    for row in results:
        task_id = str(row.get("id") or "")
        prediction = row.get("predicted_answer", row.get("answer", ""))
        ground_truth = row.get(
            "correct_label", row.get("ground_truth", row.get("reference_answer", ""))
        )
        lines.append(
            "\t".join(
                str(value).replace("\n", " ")
                for value in (
                    task_id,
                    row.get("task_type", "unknown"),
                    "PASS" if bool(row.get("hard")) else "FAIL",
                    prediction,
                    ground_truth,
                    f"traces/{task_id}",
                )
            )
        )
    return "\n".join(lines)


def sampled_trace_prompt(
    selection: TraceSelection,
    trace_paths: Mapping[str, str],
    *,
    max_characters_per_trace: int = 15000,
) -> str:
    sections: list[str] = []
    for row in selection.rows:
        task_id = str(row.get("id") or "")
        # Compact JSON keeps a complete 50-step ALFWorld action/feedback trace
        # inside the per-trace budget; pretty printing can needlessly truncate
        # the causal tail of the episode.
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        sections.append(
            f"## {'PASS' if bool(row.get('hard')) else 'FAIL'} {task_id}\n"
            f"Trace path: {trace_paths.get(task_id, '')}\n"
            + encoded[:max_characters_per_trace]
        )
    return "\n\n".join(sections)


def protected_sampled_trace_prompt(
    selection: TraceSelection,
    trace_paths: Mapping[str, str],
    *,
    max_characters_per_trace: int = 15000,
    max_total_characters: int = 50000,
    minimum_complete_passes: int = 2,
) -> tuple[str, dict[str, Any]]:
    """Budget trace context without allowing a total slice to erase all passes."""

    if max_characters_per_trace <= 0 or max_total_characters <= 0:
        raise ValueError("trace character budgets must be positive")
    if minimum_complete_passes < 0:
        raise ValueError("minimum_complete_passes must be non-negative")
    if minimum_complete_passes > len(selection.passes):
        raise ValueError("minimum_complete_passes exceeds selected pass traces")

    ordered: list[Mapping[str, Any]] = []
    width = max(len(selection.failures), len(selection.passes))
    for index in range(width):
        if index < len(selection.passes):
            ordered.append(selection.passes[index])
        if index < len(selection.failures):
            ordered.append(selection.failures[index])
    if not ordered:
        return "", {
            "type": "protected_success_reserved_budget",
            "rows": [],
            "characters_sent": 0,
            "minimum_complete_passes": minimum_complete_passes,
            "complete_passes": 0,
        }

    headers = []
    bounded_contents = []
    for row in ordered:
        task_id = str(row.get("id") or "")
        headers.append(
            f"## {'PASS' if bool(row.get('hard')) else 'FAIL'} {task_id}\n"
            f"Trace path: {trace_paths.get(task_id, '')}\n"
        )
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        bounded_contents.append(encoded[:max_characters_per_trace])

    separator_characters = 2 * (len(ordered) - 1)
    content_budget = max_total_characters - sum(map(len, headers)) - separator_characters
    if content_budget < 0:
        raise ValueError("total trace budget cannot fit all selected trace headers")

    allocations = [0] * len(ordered)
    mandatory_pass_indices = [
        index
        for index, row in enumerate(ordered)
        if bool(row.get("hard"))
    ][:minimum_complete_passes]
    mandatory_characters = sum(
        len(bounded_contents[index]) for index in mandatory_pass_indices
    )
    if mandatory_characters > content_budget:
        raise ValueError(
            "total trace budget cannot preserve the required complete pass traces"
        )
    for index in mandatory_pass_indices:
        allocations[index] = len(bounded_contents[index])
    remaining = content_budget - mandatory_characters

    while remaining:
        active = [
            index
            for index, content in enumerate(bounded_contents)
            if allocations[index] < len(content)
        ]
        if not active:
            break
        share = max(1, remaining // len(active))
        spent = 0
        for index in active:
            increment = min(
                len(bounded_contents[index]) - allocations[index],
                share,
                remaining - spent,
            )
            allocations[index] += increment
            spent += increment
            if spent == remaining:
                break
        if spent <= 0:
            break
        remaining -= spent

    sections = [
        header + content[:allocation]
        for header, content, allocation in zip(
            headers,
            bounded_contents,
            allocations,
            strict=True,
        )
    ]
    prompt = "\n\n".join(sections)
    if len(prompt) > max_total_characters:
        raise AssertionError("protected trace budgeting exceeded the total limit")
    row_audits = []
    for row, content, allocation in zip(
        ordered,
        bounded_contents,
        allocations,
        strict=True,
    ):
        row_audits.append(
            {
                "task_id": str(row.get("id") or ""),
                "status": "PASS" if bool(row.get("hard")) else "FAIL",
                "content_characters_available": len(content),
                "content_characters_sent": allocation,
                "complete_within_per_trace_budget": allocation == len(content),
            }
        )
    complete_passes = sum(
        row["status"] == "PASS" and row["complete_within_per_trace_budget"]
        for row in row_audits
    )
    if complete_passes < minimum_complete_passes:
        raise AssertionError("protected pass reservation was not honored")
    return prompt, {
        "type": "protected_success_reserved_budget",
        "ordering": "pass_first_interleaved",
        "max_characters_per_trace": max_characters_per_trace,
        "max_total_characters": max_total_characters,
        "characters_sent": len(prompt),
        "minimum_complete_passes": minimum_complete_passes,
        "complete_passes": complete_passes,
        "rows": row_audits,
    }


def read_workspace_file(workspace: Path, relative: str) -> str:
    requested = str(relative or "").strip().replace("\\", "/")
    if requested.startswith("traces/"):
        suffix = requested[len("traces/") :]
        candidates = sorted((workspace / "raw/traces").glob(f"iter-*/{suffix}"))
        if not candidates and not suffix.endswith(".json"):
            candidates = sorted(
                (workspace / "raw/traces").glob(f"iter-*/{suffix}.json")
            )
        if not candidates:
            raise FileNotFoundError(f"unknown trace alias: {requested}")
        path = candidates[-1]
    else:
        path = (workspace / requested).resolve()
        root = workspace.resolve()
        if path != root and root not in path.parents:
            raise PermissionError(f"workspace path escape rejected: {requested}")
    if not path.is_file():
        raise FileNotFoundError(f"workspace file does not exist: {requested}")
    return path.read_text(encoding="utf-8")


def score_rows(results: Iterable[Mapping[str, Any]]) -> tuple[float, tuple[int, ...]]:
    vector = tuple(int(bool(row.get("hard"))) for row in results)
    if not vector:
        raise ValueError("cannot score an empty result set")
    return sum(vector) / len(vector), vector


__all__ = [
    "ProposalApplication",
    "SKILL_PROPOSER_SYSTEM_PROMPT",
    "TraceSelection",
    "WIKI_MAINTAINER_SYSTEM_PROMPT",
    "append_skill_impact",
    "apply_skill_proposal",
    "apply_text_edits",
    "apply_wiki_update",
    "canonical_json_sha256",
    "initialize_workspace",
    "persist_trace_files",
    "protected_sampled_trace_prompt",
    "read_workspace_file",
    "render_skill_proposer_system_prompt",
    "render_active_skills",
    "sampled_trace_prompt",
    "score_rows",
    "select_traces",
    "skill_tree_hash",
    "skill_tree_snapshot",
    "trace_summary",
    "validate_skill_proposal",
    "validate_wiki_update",
]
