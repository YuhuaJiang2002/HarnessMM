"""Strict literal repair for a final Qwen terminal call stranded in reasoning.

A prospective shared-parent adapter; no commands are inferred or edited.
It applies only when the native response has no structured tool call and no
visible answer. Original wire text and provider usage must remain archived.
"""
import math
import re


def final_literal_terminal_call(reasoning, *, structured_tool_calls=0, visible_answer=''):
    if structured_tool_calls or visible_answer.strip() or not isinstance(reasoning, str):
        return None
    text = reasoning.rstrip()
    if text.endswith('<|im_end|>'):
        text = text[:-len('<|im_end|>')].rstrip()
    # qwen3 streaming may consume the opening tool_call token. A complete
    # function and terminal envelope closing token are still mandatory.
    if text.count('<function=terminal>') != 1:
        return None
    match = re.search(r'(?:<tool_call>\s*)?<function=terminal>(.*?)</function>\s*</tool_call>$', text, re.S)
    if not match:
        return None
    body = match.group(1)
    parameters = list(re.finditer(r'<parameter=([a-z_]+)>(.*?)</parameter>', body, re.S))
    if not parameters:
        return None
    remaining = re.sub(r'<parameter=([a-z_]+)>.*?</parameter>', '', body, flags=re.S)
    if remaining.strip():
        return None
    arguments = {}
    for parameter in parameters:
        key, literal = parameter.groups()
        if key not in {'command', 'cwd', 'timeout_sec'} or key in arguments:
            return None
        if '<parameter=' in literal or '<function=' in literal or '\0' in literal:
            return None
        if key == 'timeout_sec':
            if not re.fullmatch(r'\s*\d+(?:\.\d+)?\s*', literal):
                return None
            value = float(literal)
            if not math.isfinite(value) or value <= 0:
                return None
            arguments[key] = int(value) if value.is_integer() else value
        else:
            # Preserve the authored literal including every newline/space.
            arguments[key] = literal
    if not arguments.get('command', '').strip():
        return None
    if 'cwd' in arguments and (not arguments['cwd'].startswith('/') or any(c in arguments['cwd'] for c in '\r\n')):
        return None
    return {'name': 'terminal', 'arguments': arguments,
            'source_span': [match.start(), match.end()],
            'repair_identity': 'final-literal-Qwen-terminal-markup-v1',
            'invented_arguments': False}
