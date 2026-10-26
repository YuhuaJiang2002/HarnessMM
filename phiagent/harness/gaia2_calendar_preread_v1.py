"""Isolated public-availability read guard; not registered with a running campaign."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import math
import re

from phiagent.harness.gaia2_jev import action_object
from phiagent.harness.gaia2_202060_ranges import coverage, top_literal
from phiagent.harness.gaia2_native_fallback import NativeFallbackEngine


READ = "Calendar__get_calendar_events_from_to"
WRITE = "Calendar__add_calendar_event"
FORMAT = "%Y-%m-%d %H:%M:%S"
FORMAT_DECLARATION = "YYYY-MM-DD HH:MM:SS"
PUBLIC_INTERRUPTS = {
    "agent_user_interface", "environment_notifications", "hint", "task_reminder",
}
QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
ZONE = re.compile(r"\b(?:UTC|GMT|PST|PDT|EST|EDT|CST|CDT|MST|MDT|CET|CEST|BST|IST|time\s*zone)\b|"
                  r"\b(?:America|Europe|Asia|Africa|Australia)/|\d[+-]\d\d:\d\d|"
                  r"\b(?!first\b|earliest\b|current\b|same\b|next\b)[A-Za-z]+\s+time\b", re.I)
CONFLICT_PERMISSION = re.compile(
    r"\b(?:allow|ignore|permit)\b.{0,45}\b(?:conflicts?|overlaps?|double.book)"
    r"|\b(?:conflicts?|overlaps?)\b.{0,25}\b(?:allowed|fine|okay|ok)\b"
    r"|\beven if\b.{0,45}\b(?:busy|occupied|conflicts?|overlaps?)\b", re.I,
)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def native_datetime(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", value):
        raise ValueError("Datetime is not in the public native calendar format")
    return datetime.strptime(value, FORMAT)


def public_task(text):
    received = re.search(r"^\s*Received at: (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", text)
    anchor = native_datetime(received[1]) if received else None
    if "\nMessage: " in text:
        text = text.split("\nMessage: ", 1)[1]
        text = re.sub(
            r"\nAlready read: (?:True|False)(?:\nRead at: (?:None|\d{4}-\d\d-\d\d \d\d:\d\d:\d\d))?\s*$",
            "", text,
        )
    return text.translate(QUOTES), anchor


def read_contract(tools):
    read = tools.get(READ)
    write = tools.get(WRITE)
    if read is None or write is None:
        return None
    inputs = getattr(read, "inputs", None)
    write_inputs = getattr(write, "inputs", None)
    description = getattr(read, "description", "")
    if (not isinstance(inputs, dict) or set(inputs) != {
        "start_datetime", "end_datetime", "offset", "limit",
    } or not isinstance(write_inputs, dict) or not isinstance(description, str)
            or "any time overlap" not in description.lower()
            or "only touch at boundaries" not in description.lower()):
        return None
    for name in ("start_datetime", "end_datetime"):
        if (not isinstance(inputs[name], dict) or inputs[name].get("type") != "string"
                or not isinstance(inputs[name].get("description"), str)
                or FORMAT_DECLARATION not in inputs[name].get("description", "")
                or not isinstance(write_inputs.get(name), dict)
                or not isinstance(write_inputs[name].get("description"), str)
                or FORMAT_DECLARATION not in write_inputs[name].get("description", "")
                or ZONE.search(inputs[name]["description"])
                or ZONE.search(write_inputs[name]["description"])):
            return None
    for name in ("offset", "limit"):
        schema = inputs[name]
        if (not isinstance(schema, dict) or schema.get("type") != "integer"
                or type(schema.get("default")) is not int):
            return None
    if inputs["offset"]["default"] != 0 or not 0 < inputs["limit"]["default"] <= 100:
        return None
    return {
        "offset": 0, "limit": inputs["limit"]["default"],
        "time_domain": "Unconverted native calendar civil datetime, shared by proposal and public schemas",
        "schema_sha256": fingerprint([description, inputs, write_inputs]),
        "write_fields": tuple(write_inputs),
    }


@dataclass(frozen=True)
class Requirement:
    lower: datetime
    proposed_start: datetime
    proposed_end: datetime
    read_start: datetime
    read_end: datetime
    clause: str
    time_domain: str


def bind_requirement(task, proposal, contract):
    if not isinstance(task, str):
        return None, "current_public_task_not_text"
    if (proposal is None or proposal.get("action") != WRITE
            or not isinstance(proposal.get("action_input"), dict)):
        return None, "not_supported_calendar_write"
    body, anchor = public_task(task)
    arguments = proposal["action_input"]
    if set(arguments) - set(contract["write_fields"]):
        return None, "proposal_arguments_outside_public_schema"
    title = arguments.get("title")
    if not isinstance(title, str) or not title:
        return None, "proposal_title_not_explicit"
    if ZONE.search(body):
        return None, "explicit_or_mixed_timezone_not_uniquely_bound"
    if CONFLICT_PERMISSION.search(body):
        return None, "public_conflict_permission"
    if re.search(r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM)\b|\bat\s+\d{1,2}:\d{2}\b", body, re.I):
        return None, "fixed_time_clause_not_uniquely_reconciled"
    clauses = re.split(r"(?<=[.!?])\s+|\n+", body)
    candidates = []
    for clause in clauses:
        if not re.search(r"\b(?:schedule|create|book)\b", clause, re.I):
            continue
        if not re.search(r"\b(?:calendar|event|events|meeting|appointment)\b", clause, re.I):
            continue
        titles = re.findall(
            r"\b(?:title\s+(?:them|it)|titled|called|named)\s+['\"]([^'\"]+)['\"]", clause, re.I,
        )
        if titles != [title]:
            continue
        if re.search(r"\b(?:at|exactly at)\s+\d", clause, re.I):
            continue
        availability = re.findall(
            r"\b(?:first|earliest)\s+(?:time\s+slot\s+(?:free|available)|"
            r"(?:free|available)\s+(?:time\s+)?slot)\s+from\s+"
            r"(\d{1,2}(?::\d{2})?)\s*(AM|PM)?\b", clause, re.I,
        )
        durations = re.findall(r"\b(\d+)[ -](hour|minute)s?\b", clause, re.I)
        if len(availability) == len(durations) == 1:
            candidates.append((clause, availability[0], durations[0]))
    if len(candidates) != 1:
        return None, "no_unique_task_local_availability_requirement"
    clause, clock, duration = candidates[0]
    try:
        start = native_datetime(arguments.get("start_datetime"))
        end = native_datetime(arguments.get("end_datetime"))
        minutes = int(duration[0]) * (60 if duration[1].lower() == "hour" else 1)
        if not 0 < minutes < 24 * 60 or end - start != timedelta(minutes=minutes):
            return None, "proposal_duration_not_grounded"
        if start.date() != end.date():
            return None, "cross_day_proposal_not_bound"
        hours, separator, minute = clock[0].partition(":")
        hour, minute = int(hours), int(minute) if separator else 0
        if clock[1]:
            if not 1 <= hour <= 12:
                return None, "ambiguous_clock_unit"
            hour = hour % 12 + (12 if clock[1].upper() == "PM" else 0)
        elif not separator:
            return None, "ambiguous_clock_unit"
        lower = start.replace(hour=hour, minute=minute, second=0)
        if start < lower or end <= start:
            return None, "proposal_outside_requested_search_window"
        explicit = re.findall(r"\bon\s+(\d{4}-\d\d-\d\d)\b", clause, re.I)
        relative = re.search(r"\bon the 1st day of the next (two|2) months\b", clause, re.I)
        if len(explicit) == 1 and relative is None:
            if start.date().isoformat() != explicit[0]:
                return None, "proposal_date_not_grounded"
        elif not explicit and relative is not None and anchor is not None:
            allowed = set()
            for shift in (1, 2):
                month = anchor.year * 12 + anchor.month - 1 + shift
                allowed.add((month // 12, month % 12 + 1, 1))
            if (start.year, start.month, start.day) not in allowed:
                return None, "proposal_date_not_grounded"
        else:
            return None, "date_scope_not_uniquely_bound"
    except ValueError:
        return None, "invalid_native_datetime_or_duration"
    return Requirement(
        lower, start, end, start.replace(hour=0, minute=0, second=0),
        start.replace(hour=23, minute=59, second=59), clause, contract["time_domain"],
    ), "task_local_requirement_bound"


def calendar_page(observation, action, default_limit):
    if action is None or action.get("action") != READ or not isinstance(observation, str):
        return None
    arguments = action.get("action_input")
    if not isinstance(arguments, dict) or set(arguments) - {
        "start_datetime", "end_datetime", "offset", "limit",
    }:
        return None
    try:
        start = native_datetime(arguments.get("start_datetime"))
        end = native_datetime(arguments.get("end_datetime"))
        interval, total = top_literal(observation, "range"), top_literal(observation, "total")
        offset, limit = arguments.get("offset", 0), arguments.get("limit", default_limit)
        if (start >= end or not isinstance(interval, (list, tuple)) or len(interval) != 2
                or any(type(value) is not int for value in [*interval, total, offset, limit])
                or limit <= 0 or interval[0] != offset
                or not 0 <= interval[0] <= interval[1] <= total
                or interval[1] != min(offset + limit, total)):
            return None
        tree = ast.parse(observation, mode="eval").body
        if not isinstance(tree, ast.Dict):
            return None
        event_lists = [value for key, value in zip(tree.keys, tree.values)
                       if isinstance(key, ast.Constant) and key.value == "events"]
        if len(event_lists) != 1 or not isinstance(event_lists[0], ast.List):
            return None
        if len(event_lists[0].elts) != interval[1] - interval[0]:
            return None
        for event in event_lists[0].elts:
            if (isinstance(event, ast.Call) and isinstance(event.func, ast.Name)
                    and event.func.id == "CalendarEvent" and not event.args):
                fields = {value.arg: value.value for value in event.keywords}
                if None in fields or len(fields) != len(event.keywords):
                    return None
            elif isinstance(event, ast.Dict):
                fields = {ast.literal_eval(key): value for key, value in zip(event.keys, event.values)}
                if len(fields) != len(event.keys):
                    return None
            else:
                return None
            values = {name: ast.literal_eval(fields[name])
                      for name in ("event_id", "start_datetime", "end_datetime")}
            if (not isinstance(values["event_id"], str) or not values["event_id"]
                    or any(type(values[name]) not in (int, float) or not math.isfinite(values[name])
                           for name in ("start_datetime", "end_datetime"))
                    or values["start_datetime"] > values["end_datetime"]):
                return None
    except (KeyError, ValueError, SyntaxError, TypeError):
        return None
    return start, end, {
        "start": interval[0], "end": interval[1], "total": total,
        "public_receipt_sha256": fingerprint(observation),
    }


@dataclass
class CalendarReadState:
    task: str = ""
    pages: dict = field(default_factory=dict)
    pending_read: dict | None = None

    def clear(self):
        self.pages.clear()
        self.pending_read = None

    def invalidate_calendar_write(self, action):
        if action["action"] != WRITE:
            self.clear()
            return
        try:
            start = native_datetime(action["action_input"].get("start_datetime"))
            end = native_datetime(action["action_input"].get("end_datetime"))
            if start >= end:
                raise ValueError("Unbound mutation interval")
        except ValueError:
            self.clear()
            return
        self.pages = {span: pages for span, pages in self.pages.items()
                      if not (span[0] < end and start < span[1])}
        self.pending_read = None

    def observe(self, row, default_limit):
        kind = row.get_type()
        if kind == "task":
            self.clear()
            self.task = getattr(row, "content", "")
        elif kind in PUBLIC_INTERRUPTS:
            self.clear()
            if kind in ("agent_user_interface", "task_reminder", "hint"):
                self.task = getattr(row, "content", "")
        elif kind == "error":
            self.clear()
        elif kind == "llm_output":
            text = getattr(row, "content", None)
            actual = action_object(text) if isinstance(text, str) else None
            self.pending_read = actual if actual is not None and actual["action"] == READ else None
            if actual is not None and actual["action"].startswith("Calendar__") and actual["action"] != READ:
                if not actual["action"].split("__", 1)[1].startswith(("get_", "read_", "search_", "list_")):
                    self.invalidate_calendar_write(actual)
        elif kind == "observation":
            content = getattr(row, "content", None)
            if isinstance(content, str) and re.search(r'(?i)(?:error:|traceback \(|"error"\s*:)', content):
                self.clear()
                return
            had_pending_read = self.pending_read is not None
            page = calendar_page(content, self.pending_read, default_limit)
            self.pending_read = None
            if had_pending_read and page is None:
                self.clear()
                return
            if page is not None:
                start, end, receipt = page
                existing = self.pages.setdefault((start, end), [])
                if any(row["total"] != receipt["total"] or (
                    row["start"] == receipt["start"] and row["end"] == receipt["end"]
                    and row["public_receipt_sha256"] != receipt["public_receipt_sha256"]
                ) for row in existing):
                    existing.clear()
                existing.append(receipt)

    def decision(self, requirement, contract):
        for (start, end), pages in self.pages.items():
            proof = coverage(pages)
            if start <= requirement.lower and end >= requirement.proposed_end and proof["complete"]:
                return None, "valid_public_availability_read", {
                    "scope": [start.strftime(FORMAT), end.strftime(FORMAT)],
                    "coverage": proof, "receipt_sha256": [row["public_receipt_sha256"] for row in pages],
                }
        span = (requirement.read_start, requirement.read_end)
        prefix = coverage(self.pages.get(span, []))["covered_prefix"]
        return {
            "action": READ,
            "action_input": {
                "start_datetime": span[0].strftime(FORMAT), "end_datetime": span[1].strftime(FORMAT),
                "offset": prefix, "limit": contract["limit"],
            },
        }, "availability_read_required", {
            "task_clause": requirement.clause, "time_domain": requirement.time_domain,
            "required_scope": [requirement.lower.strftime(FORMAT), requirement.proposed_end.strftime(FORMAT)],
        }


class CalendarPrereadEngine:
    def __init__(self, agent, provider, *, record):
        self.agent, self.provider, self.record = agent, provider, record
        self.state = CalendarReadState()
        self.cursor = 0
        self.last_log = None
        self.initialized = False
        self.schema_sha = None
        self.last_decision = None

    def sync(self, contract):
        logs = self.agent.logs
        if self.cursor > len(logs) or (self.cursor and logs[self.cursor - 1] is not self.last_log):
            self.state = CalendarReadState()
            self.cursor, self.initialized = 0, False
        fresh = logs[self.cursor:]
        interrupted = self.initialized and any(
            row.get_type() in PUBLIC_INTERRUPTS | {"task", "error"} for row in fresh
        )
        if contract is None:
            self.state.clear()
            limit = 10
        else:
            if self.schema_sha != contract["schema_sha256"]:
                self.state.clear()
            limit = contract["limit"]
            self.schema_sha = contract["schema_sha256"]
        for row in fresh:
            self.state.observe(row, limit)
        self.cursor = len(logs)
        self.last_log = logs[-1] if logs else None
        self.initialized = True
        return interrupted

    def __call__(self, prompt, **kwargs):
        self.last_decision = None
        contract = read_contract(self.agent.action_executor.tools)
        interrupted = self.sync(contract)
        response = self.provider(prompt, **kwargs)
        contract_after = read_contract(self.agent.action_executor.tools)
        interrupted = self.sync(contract_after) or interrupted
        text, metadata = response if isinstance(response, tuple) and len(response) == 2 else (response, None)
        proposed = action_object(text) if isinstance(text, str) else None
        selected, proof = None, {}
        reason = "no_supported_public_calendar_contract"
        if contract_after is not None and contract_after == contract:
            requirement, reason = bind_requirement(self.state.task, proposed, contract_after)
            if requirement is not None:
                paid = (isinstance(metadata, dict)
                        and all(type(metadata.get(key)) is int and metadata[key] >= 0
                                for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                        and metadata["prompt_tokens"] + metadata["completion_tokens"] == metadata["total_tokens"]
                        and metadata["total_tokens"] > 0
                        and type(metadata.get("completion_duration")) in (int, float)
                        and math.isfinite(metadata["completion_duration"])
                        and metadata["completion_duration"] > 0)
                if interrupted:
                    reason = "public_interruption_requires_native_control"
                elif not paid:
                    reason = "paid_usage_or_measured_duration_not_verified"
                else:
                    selected, reason, proof = self.state.decision(requirement, contract_after)
        executed = text
        if selected is not None:
            executed = ("Thought: Read the public calendar before deciding this availability-dependent write.\n"
                        "Action:\n" + json.dumps(selected, ensure_ascii=False))
        self.last_decision = {
            "source": "calendar_availability_preread_v1", "agent_id": self.agent.agent_id,
            "decision": reason, "intervened": selected is not None, "proof": proof,
            "model_proposal": text, "returned_native_text": executed,
            "proposal_is_audit_only": selected is not None,
            "deferred_write_will_not_be_auto_executed": True,
            "model_calls_this_step": 1, "additional_model_calls": 0,
            "provider_metadata": metadata, "metadata_unchanged": True,
            "native_messages_kwargs_unchanged": True,
        }
        self.record(self.last_decision)
        return (executed, metadata) if selected is not None else response


class CalendarPrereadReceiptEngine(NativeFallbackEngine):
    def __init__(self, agent, provider, *, skill, record, execute_followups):
        guard = CalendarPrereadEngine(agent, provider, record=record)

        def record_actual_dispatch(value):
            decision = guard.last_decision
            if (value.get("source") == "receipt_graph_model" and decision is not None
                    and decision["intervened"]):
                value = {
                    **value, "source": "receipt_graph_model_calendar_preread",
                    "original_response": decision["model_proposal"],
                    "native_fallback_unchanged": False, "calendar_preread_intervention": True,
                    "model_metadata_unchanged": True,
                }
            record(value)

        super().__init__(
            agent, guard, skill=skill, record=record_actual_dispatch, execute_followups=execute_followups,
        )


def with_public_receipt_compiler(agent, provider, *, skill, record, execute_followups):
    return CalendarPrereadReceiptEngine(
        agent, provider, skill=skill, record=record, execute_followups=execute_followups,
    )


def install_adapter(config, directory, role, mode, skill):
    from phiagent.harness import gaia2_202060_progress as progress

    progress.ReceiptEngine = CalendarPrereadReceiptEngine
    progress.install_adapter(config, directory, role, mode, skill)
