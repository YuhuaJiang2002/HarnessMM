"""Paid CoT authors individually executable native operations from public evidence."""
from copy import deepcopy
from hashlib import sha256
import json
from phiagent.harness.gaia2_grounded_bank_author_v7 import author_request as original_request
from phiagent.harness.gaia2_cot_jev import parse_bank

CONTRACT = 'native_parent_fresh_public_direct_operations_v8'
AUTHOR = """The supplied native action has executed. Its observation is the actual returned
public result. Reason from the full task, this result, the prior native reasoning,
the completed action and the supplied original public tool receipt. Author a bank
of useful next native operations. Every operation is separate: a one-token judge
will select one or abstain. Keep all task filters, exceptions, direction, timing,
ordering and authorization. The note explains the plan; the action_input must
contain the actual argument values needed to carry out each declared operation.

Use exactly the supplied native tool names, parameter names and argument types.
Omit irrelevant optional parameters; never replace needed task arguments with an
empty object. State what observed evidence warrants each operation in when.
Use literal arguments whose values are already public. A public-field binding is
an object, never a dotted string: {\"$obs\":[\"field\",0,\"id\"]} reads the current
observation; {\"$prev\":[\"offset\"]} reads the completed action's arguments.
{\"$receipt\":{\"tool\":\"exact native tool\",\"path\":[\"field\",0,\"id\"],
\"arguments\":{}}} reads that tool's latest original public receipt matching its
specified arguments. Omit arguments to match the latest call of that tool.
{\"$add\":[term,term]} adds two known numbers. Use only actual observed field paths.
Do not put an operator name into a field path or write strings such as $obs.id.

Declare ordinary single operations, including reads needed for the next decision.
Do not use foreach or $item. A time result, count, scalar identifier or empty
lookup result is not a record list. Declare each justified record operation with
its own complete native argument object. If needed data are unavailable, omit
that operation and explain the unresolved prerequisite in the note. Never invent
a future observation, identifier, field name, completion, wait or user message.
An empty bank is correct when further interpretation or native CoT is required.
Return the supplied bank schema. No declared operation executes automatically."""


def author_state_key(task, observation, completed_action, receipts):
    # Latest evidence for each exact native query suffices to identify a repeated
    # public state. Log indices and repeated copies are not new public facts.
    latest = {}
    for receipt in receipts:
        call = receipt.get('native_tool_call', {})
        query = json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        latest[query] = receipt.get('native_observation_sha256')
    value = {'task':task,'observation':observation,'completed_action':completed_action,
             'latest_public_receipt_hashes':latest}
    return sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',', ':')).encode()).hexdigest()


def author_request(model,tools,task,observation,completed_action,native_reasoning,receipts):
    body = deepcopy(original_request(model,tools,task,observation,completed_action,native_reasoning,receipts))
    body['messages'][0]['content'] = AUTHOR
    schema = body['response_format']['json_schema']['schema']
    for candidate in schema['properties']['candidates']['items'].get('anyOf', []):
        candidate['properties'].pop('foreach', None)
    references = schema.get('$defs', {}).get('cot_jev_public_argument_reference', {}).get('anyOf', [])
    references[:] = [r for r in references if r.get('required') != ['$item']]
    return body


def parse_author(raw,tools):
    choice = raw['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise ValueError('A direct-operation bank author must finish naturally')
    bank = parse_bank({'jev_bank':json.loads(choice['message']['content'])},tools,
                      allow_routine_writes=True,continuation_bank=True)
    def inspect(value):
        if isinstance(value, dict):
            if '$item' in value:raise ValueError('Direct operations have no implicit record iterator')
            for child in value.values():inspect(child)
        elif isinstance(value,list):
            for child in value:inspect(child)
        elif isinstance(value,str) and value.startswith(('$obs.','$prev.','$receipt.','$item.')):
            raise ValueError('Public bindings are typed objects, not dotted literal strings')
    for candidate in bank['candidates']:
        if 'foreach' in candidate:raise ValueError('Declare each direct native operation separately')
        inspect(candidate['action_input'])
    return bank


def declare_unavailable_branches(messages, failures):
    if not failures:return messages
    result=deepcopy(messages);state=json.loads(result[1]['content'])
    state['unavailable_candidate_templates']=deepcopy(failures)
    state['unavailable_scope']='These other CoT-declared branches lack complete public bindings and cannot be selected. Their prerequisites remain unresolved; choose only a listed grounded operation or abstain.'
    result[1]['content']=json.dumps(state,ensure_ascii=False,separators=(',', ':'))
    return result
