"""Pure public-record decoding and exact reductions; no agent, model, or scorer access."""
from __future__ import annotations

import ast
from collections import Counter
from decimal import Decimal
import io
import re
import tokenize


RECORD_TYPES = frozenset({"Ride", "Contact", "Apartment", "Email"})


def parse_public_records(text):
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    normalized = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == tokenize.OP and token.string == "<":
            end = index + 1
            while end < len(tokens) and tokens[end].string != ">":
                end += 1
            inner = tokens[index + 1:end]
            if (end == len(tokens) or len(inner) != 5
                    or [value.type for value in inner] != [
                        tokenize.NAME, tokenize.OP, tokenize.NAME, tokenize.OP, tokenize.STRING,
                    ] or inner[1].string != "." or inner[3].string != ":"):
                raise ValueError("Unsupported public enum representation")
            normalized.append((tokenize.STRING, inner[-1].string))
            index = end + 1
        else:
            normalized.append((token.type, token.string))
            index += 1

    def decode(node):
        if isinstance(node, ast.Constant):
            if node.value is None or isinstance(node.value, (str, int, float, bool)):
                return node.value
        elif isinstance(node, (ast.List, ast.Tuple)):
            return [decode(value) for value in node.elts]
        elif isinstance(node, ast.Dict):
            keys = [decode(key) for key in node.keys]
            if any(not isinstance(key, (str, int)) for key in keys) or len(set(keys)) != len(keys):
                raise ValueError("Unsupported or duplicate public-record key")
            return {key: decode(value) for key, value in zip(keys, node.values)}
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = decode(node.operand)
            if type(value) in (int, float):
                return -value
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in RECORD_TYPES and not node.args):
            names = [value.arg for value in node.keywords]
            if any(name is None for name in names) or len(set(names)) != len(names):
                raise ValueError("Ambiguous public record fields")
            return {"record_type": node.func.id, **{
                value.arg: decode(value.value) for value in node.keywords
            }}
        raise ValueError("Public record contains an unsupported expression")

    return decode(ast.parse(tokenize.untokenize(normalized), mode="eval").body)


def complete_public_pages(pages, *, rows_key, id_key, metadata_key=None):
    by_id, observed_intervals = {}, []
    for page in pages:
        metadata = page[metadata_key] if metadata_key else page
        interval, total = metadata["range"], metadata["total"]
        if (len(interval) != 2 or any(type(value) is not int for value in [*interval, total])
                or not 0 <= interval[0] <= interval[1] <= total):
            raise ValueError("Invalid public pagination interval")
        rows = page[rows_key]
        rows = list(rows.values()) if isinstance(rows, dict) else rows
        if len(rows) != interval[1] - interval[0]:
            raise ValueError("Public page count disagrees with its range")
        for row in rows:
            ident = row[id_key]
            if ident in by_id and by_id[ident] != row:
                raise ValueError("Public records changed across reads")
            by_id[ident] = row
        observed_intervals.append((interval[0], interval[1], total))
    if not observed_intervals or len({value[2] for value in observed_intervals}) != 1:
        raise ValueError("No stable public population size")
    cursor = 0
    for start, end, _ in sorted(observed_intervals):
        if start > cursor:
            raise ValueError("Public coverage has an unread gap")
        cursor = max(cursor, end)
    total = observed_intervals[0][2]
    if cursor != total or len(by_id) != total:
        raise ValueError("Public population is incomplete")
    return list(by_id.values()), {"complete": True, "unique_rows": total}


def unique_numeric_postcode(address):
    candidates = re.findall(r"(?<!\d)\d{4,6}(?!\d)", address)
    if len(candidates) != 1:
        raise ValueError("Numeric postcode is absent or ambiguous in public address")
    return candidates[0]


def exact_mode(values, *, numeric_tie_break=False):
    counts = Counter(values)
    if not counts:
        raise ValueError("Cannot reduce an empty public population")
    maximum = max(counts.values())
    tied = [value for value, count in counts.items() if count == maximum]
    if len(tied) > 1 and not numeric_tie_break:
        raise ValueError("Public request has not resolved the mode tie")
    if numeric_tie_break:
        if any(not isinstance(value, str) or not value.isdecimal() for value in tied):
            raise ValueError("The requested numeric tie break has nonnumeric keys")
        selected = min(tied, key=lambda value: (int(value), value))
    else:
        selected = tied[0]
    return selected, dict(counts)


def exact_extreme(rows, field, *, minimum=False, preferred_tie=None):
    if not rows:
        raise ValueError("Cannot select from an empty public population")
    values = []
    for row in rows:
        value = row[field]
        if type(value) not in (int, float, Decimal):
            raise ValueError("Ranking field is not an explicit public number")
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError("Ranking field is not finite")
        values.append(number)
    target = min(values) if minimum else max(values)
    tied = [row for row, value in zip(rows, values) if value == target]
    if len(tied) > 1 and preferred_tie is not None:
        key, value = preferred_tie
        preferred = [row for row in tied if row.get(key) == value]
        if preferred:
            tied = preferred
    if len(tied) != 1:
        raise ValueError("Public evidence/request does not uniquely resolve the ranking")
    return tied[0]
