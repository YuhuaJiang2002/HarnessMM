"""CoT authorship grounded in the runtime's actual public binding grammar."""
from copy import deepcopy
import hashlib
import json
from phiagent.harness import gaia2_receipt_complete_author_v12 as _previous

for _name in dir(_previous):
    if not _name.startswith('_'):
        globals()[_name] = getattr(_previous, _name)

CONTRACT = 'native_parent_executable_source_author_v13'
AUTHOR = """Author a reusable continuation bank from the task, original public tool
receipts and native reasoning. Track completed and remaining subgoals across ALL
receipts. Each candidate is separately selected by a one-token judge or deferred
to native CoT. Preserve ordering, exceptions, authorization and timing. Use exact
native tools and argument types. Return only the supplied bank schema within the
unchanged output capacity. Keep the note concise; an empty bank is valid.

Compile each candidate mentally against public_binding_sources before emitting
it. Their shapes come from the actual runtime parser, without inferred fields.
An opaque source has NO bindable fields; use known public literals or leave its
interpretation to native CoT. A JSON list has integer indices, never string "0".
An object has exactly its observed keys. Do not wrap a list in invented keys such
as contacts, events, rides or conversation_id. A scalar identifier binds at [].

Use binding OBJECTS only: {"$obs":["field",0,"id"]} means the supplied completed
action's result; {"$prev":["offset"]} means that completed action's arguments.
Earlier results use {"$receipt":{"tool":"exact native tool","arguments":{},
"path":["actual field"]}} with that actual query's arguments. These references
retain their original source after intervening actions. Never emit $obs:field,
Python expressions, dotted binding strings, future observations or placeholders.

foreach must select an actual observed LIST in an explicit source. Its path
addresses that list; foreach.receipt identifies an earlier actual tool/query.
$item is available ONLY inside that iterator: {"$item":[]} for scalar members,
{"$item":["id"]} for object members. where is exact equality on actual object
fields. It is NOT a query language: no contains, >, <, not_in, or expressions.
For scalar lists use where:{}. If a task needs a semantic/numeric filter, CoT
must interpret the actual public values and declare grounded literal actions or
an already observed exact filter; do not put the condition into an unsupported
where expression. Each member is retained and receives a separate fast decision.

For every write, supply the actual required recipient, object, content and task
conditions even when a native default makes an empty argument object legal.
Do not substitute a time lookup for waiting, repeat completed reads without new
evidence, or declare completion, waits or invented values. Reads may request
unknown results when their INPUTS are grounded. Missing input prerequisites or
uninterpreted task conditions require native CoT; retain valid sibling candidates.
Only actual public receipts are evidence; bank intentions are not observations."""


def public_shape(value, depth=0):
    """All member shapes, deduplicated structurally; no values or member truncation."""
    kind = ('null' if value is None else 'boolean' if type(value) is bool else
            'integer' if type(value) is int else 'number' if type(value) is float else
            'string' if isinstance(value, str) else 'object' if isinstance(value, dict)
            else 'list' if isinstance(value, list) else None)
    if kind is None:
        raise ValueError('Unsupported public value')
    result = {'kind': kind}
    if depth >= 8:
        return result
    if kind == 'object':
        result['fields'] = {key: public_shape(item, depth + 1)
                            for key, item in value.items()}
    elif kind == 'list':
        shapes = {}
        for item in value:
            shape = public_shape(item, depth + 1)
            shapes[json.dumps(shape, sort_keys=True, ensure_ascii=False)] = shape
        result.update(count=len(value), member_shapes=list(shapes.values()),
                      member_scope_truncated=False)
    return result


def binding_sources(receipts):
    result = []
    for receipt in receipts:
        text = receipt.get('observation')
        call = receipt.get('native_tool_call') or {}
        if (not isinstance(text, str) or type(receipt.get('log_index')) is not int
                or receipt.get('historical_receipt_not_current_state') is not True
                or hashlib.sha256(text.encode()).hexdigest()
                != receipt.get('native_observation_sha256')):
            raise ValueError('Catalogue requires original public receipt provenance')
        row = {'log_index': receipt['log_index'], 'native_tool_call': deepcopy(call),
               'native_observation_sha256': receipt['native_observation_sha256']}
        try:
            row['shape'] = public_shape(observation_data(text))
            row['status'] = 'BINDABLE_PUBLIC_SOURCE'
        except (ValueError, TypeError, KeyError, SyntaxError, IndexError):
            row.update(status='OPAQUE_PUBLIC_SOURCE', bindable_fields=False)
        result.append(row)
    return result


def author_request(*args, **kwargs):
    body = _previous.author_request(*args, **kwargs)
    body['messages'][0]['content'] = AUTHOR
    state = json.loads(body['messages'][1]['content'])
    state['public_binding_sources'] = binding_sources(state['original_public_tool_receipts'])
    body['messages'][1]['content'] = json.dumps(state, ensure_ascii=False)
    return body
