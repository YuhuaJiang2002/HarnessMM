"""Single-call CoT action/bank authorship with exact public input compression."""
from copy import deepcopy
import hashlib
import json
import re

CONTRACT = 'native_parent_fused_cot_bank_v16'
INSTRUCTION = '''In the same native Thought/Action response, choose the current action
first and add jev_bank to that Action JSON. It is a continuation annotation,
not a second action. jev_bank has note and candidates; each candidate has when,
action and action_input, with optional foreach. At most eight candidates.
Keep the current action, full reasoning, tool arguments, task constraints and
native end marker. Empty candidates are valid when more evidence is needed.
Declare only useful later actions. Every later action is independently selected
or declined by the one-token judge after an actual public observation.
Use literal known values or binding objects: {"$obs":["field",0,"id"]} for
the current action's future actual result; {"$prev":["offset"]} for its
arguments; {"$receipt":{"tool":"exact tool","arguments":{},"path":[]}}
for an earlier actual public receipt. foreach uses an actual observed list,
an exact path and equality-only where; $item is valid only inside that iterator.
Do not invent fields, identifiers, expressions, timing or completed effects.
Do not propose completion, waits or user messages as fast candidates.
Do not repeat completed subgoals. Keep useful bank candidates across unrelated
reads; when no new continuation is needed, use an empty bank.
Tool records may use JSON arrays [name, description, output_type, inputs].
All descriptions, argument schemas, defaults and tools remain present.
An exact duplicate observation reference denotes the full retained observation
at the named step; it has identical bytes and is historical evidence.'''

OBSERVATION = re.compile(
    r'^(\[OUTPUT OF STEP (?P<step>[0-9]+)\] Observation:\n\*\*\*\n)'
    r'(?P<body>.*?)(\n\*\*\*\n?)$', re.DOTALL)


def project(messages, descriptions, records):
    original = deepcopy(messages)
    if not original or original[0].get('role') != 'system':
        raise ValueError('Require the full native system prompt')
    result = deepcopy(original)
    catalogue = '\n'.join(descriptions.values())
    packed = '\n'.join(json.dumps([r['name'], r['description'], r['output_type'], r['inputs']],
                                  ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                       for r in records)
    restored = [dict(zip(('name', 'description', 'output_type', 'inputs'), json.loads(line)))
                for line in packed.splitlines()]
    if restored != records:
        raise ValueError('Compact tool contracts did not round trip exactly')
    system = result[0]['content']
    schema_applied = bool(catalogue and system.count(catalogue) == 1 and len(packed) < len(catalogue))
    if schema_applied:
        result[0]['content'] = system.replace(catalogue, packed, 1)
    retained = {}
    references = []
    for i in range(len(result) - 1, 0, -1):
        m = original[i]
        match = OBSERVATION.fullmatch(m.get('content', '')) if m.get('role') == 'user' and isinstance(m.get('content'), str) else None
        if match is None or len(match['body']) < 512:
            continue
        body = match['body']
        if body in retained:
            target, step = retained[body]
            marker = f'[Exact duplicate of public observation at step {step}; sha256={hashlib.sha256(body.encode()).hexdigest()}]'
            result[i]['content'] = m['content'][:match.start('body')] + marker + m['content'][match.end('body'):]
            references.append({'message': i, 'retained_message': target, 'retained_step': step})
        else:
            retained[body] = i, match['step']
    # The task and latest observation are preserved byte-for-byte. Only identical
    # earlier observation bodies refer to a later full copy in the same request.
    assert result[-1] == original[-1]
    result[0]['content'] += '\n\n' + INSTRUCTION
    if messages != original:
        raise ValueError('Projection mutated native logs')
    return result, {'source': 'fused_cot_public_projection', 'mechanism_revision': CONTRACT,
                    'schema_roundtrip': True, 'schema_compacted': schema_applied,
                    'exact_duplicate_references': references, 'unique_observations_removed': 0,
                    'standalone_author_calls': 0, 'new_model_calls_for_projection': 0,
                    'native_logs_mutated': False, 'tools_removed': 0,
                    'original_characters': sum(len(m.get('content', '')) for m in original),
                    'projected_characters': sum(len(m.get('content', '')) for m in result)}


def native_prompt(agent, messages, record):
    from are.simulation.tool_box import get_tool_description_with_args
    tools = agent.action_executor.tools
    if not all(all(hasattr(t, k) for k in ('description', 'output_type', 'inputs')) for t in tools.values()):
        projected, audit = project(messages, {}, [])
        record({'agent_id': agent.agent_id, 'metadata_available': False, **audit})
        return projected
    descriptions = {n: get_tool_description_with_args(t) for n, t in tools.items()}
    records = [{'name': n, 'description': t.description, 'output_type': t.output_type,
                'inputs': deepcopy(t.inputs)} for n, t in tools.items()]
    projected, audit = project(messages, descriptions, records)
    record({'agent_id': agent.agent_id, **audit})
    return projected
