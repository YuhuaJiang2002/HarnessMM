"""Compile the next read from pinned public pagination receipts, never private state."""
from __future__ import annotations
import ast
import io
import json
import re
import tokenize


def top_literal(text, name):
    """Extract one top-level dictionary field without evaluating object reprs."""
    if not isinstance(text, str) or not text.strip().startswith("{"):
        return None
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        depth, matches = 0, []
        for index, token in enumerate(tokens):
            if token.type == tokenize.STRING and depth == 1 and ast.literal_eval(token.string) == name:
                if tokens[index + 1].string != ":":
                    return None
                nested, parts = 0, []
                for value in tokens[index + 2:]:
                    if nested == 0 and value.type == tokenize.OP and value.string in (",", "}"):
                        break
                    parts.append(value.string)
                    if value.type == tokenize.OP:
                        nested += value.string in ("{", "[", "(")
                        nested -= value.string in ("}", "]", ")")
                matches.append(ast.literal_eval(" ".join(parts)))
            if token.type == tokenize.OP:
                depth += token.string in ("{", "[", "(")
                depth -= token.string in ("}", "]", ")")
        return matches[0] if depth == 0 and len(matches) == 1 else None
    except (ValueError, SyntaxError, TypeError, IndexError, tokenize.TokenError, IndentationError):
        return None


def required_set(task, tool):
    if tool == "Emails__list_emails":
        from phiagent.harness.gaia2_state_progress import enumeration_goal
        return enumeration_goal(task, "email")
    if tool == "Contacts__get_contacts":
        return bool(re.search(r"\bcontacts?\b", task, re.I) and re.search(
            r"\b(all|every|any|oldest|youngest|count|number|more than|most|least)\b", task, re.I))
    if tool == "Cabs__get_ride_history":
        return bool(re.search(r"\b(cabs?|rides?)\b", task, re.I) and re.search(
            r"\b(all|every|among|expensive|cheapest|longest|shortest|top|total)\b", task, re.I))
    return False


def range_receipt(action, observation, task):
    if not isinstance(action, dict):
        return None
    tool, args = action.get("action"), action.get("action_input")
    if not isinstance(args, dict) or not required_set(task, tool):
        return None
    if tool == "Cabs__get_ride_history":
        if set(args) - {"offset", "limit"}:
            return None
        interval, total = top_literal(observation, "range"), top_literal(observation, "total")
    elif tool == "Contacts__get_contacts":
        if set(args) - {"offset"}:
            return None
        metadata = top_literal(observation, "metadata")
        if not isinstance(metadata, dict) or set(metadata) != {"range", "total"}:
            return None
        interval, total = metadata["range"], metadata["total"]
    elif tool == "Emails__list_emails":
        if set(args) - {"offset", "limit", "folder_name"}:
            return None
        from phiagent.harness.gaia2_public_progress import email_page
        page = email_page(observation)
        if page is None:
            return None
        interval, total = page[:2], page[2]
    else:
        return None
    offset = args.get("offset", 0)
    if (not isinstance(interval, (tuple, list)) or len(interval) != 2
            or any(type(value) is not int for value in (*interval, total, offset))
            or interval[0] != offset or not 0 <= interval[0] < interval[1] <= total):
        return None
    if tool != "Contacts__get_contacts":
        limit = args.get("limit", 10)
        if type(limit) is not int or limit <= 0 or interval[1] != min(offset + limit, total):
            return None
    scope = json.dumps([tool, {key: value for key, value in args.items() if key != "offset"}],
                       sort_keys=True)
    return {"scope": scope, "tool": tool, "start": interval[0], "end": interval[1], "total": total,
            "next_action": {"action": tool, "action_input": {**args, "offset": interval[1]}}
            if interval[1] < total else None}


def coverage(pages):
    """Certify contiguous coverage only; terminal pages alone are insufficient."""
    if not pages:
        return {"complete": False, "covered_prefix": 0, "total": None}
    totals = {page["total"] for page in pages}
    if len(totals) != 1:
        return {"complete": False, "covered_prefix": 0, "total": None}
    cursor = 0
    for page in sorted(pages, key=lambda row: (row["start"], row["end"])):
        if page["start"] > cursor:
            break
        cursor = max(cursor, page["end"])
    total = next(iter(totals))
    return {"complete": cursor == total, "covered_prefix": cursor, "total": total}
