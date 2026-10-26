"""CoT-authored public iterator banks retained across native reasoning fallbacks."""
from copy import deepcopy
import hashlib
import json

from phiagent.harness.gaia2_grounded_bank_author_v7 import (
    author_request as original_request, parse_author,
)
from phiagent.harness.gaia2_direct_bank_author_v8 import declare_unavailable_branches
from phiagent.harness.gaia2_cot_jev import digest, observation_data, at_path
from phiagent.harness.gaia2_typed_continuation_bank import receipt_value, bind_arguments
from phiagent.harness.gaia2_dual_rsi import well_typed

CONTRACT = 'native_parent_persistent_public_iterator_bank_v9'
AUTHOR = """The supplied native action has executed and its observation is the actual
public result. Reason from the full task, returned data, completed action, prior
native reasoning and original public receipt. Author a continuation bank covering
the useful remaining operations supported by these facts. The bank persists
across intervening tool reads and native reasoning; one-token decisions select
each action separately or abstain. Keep all task filters, exceptions, ordering,
timing and authorization in the note and candidate conditions. Never declare
completion, waits, user messages, future observations or invented field names.
Use exactly the supplied native tool names, argument names and argument types.
Omit irrelevant optional parameters. Known public values may be literals.
Bindings are objects: {"$obs":["field",0,"id"]} refers to THIS supplied result;
{"$prev":["offset"]} refers to THIS completed action's arguments. They keep
their original public source across intervening reads. An exact earlier query
may be referenced by {"$receipt":{"tool":"native tool","arguments":{},
"path":["field"]}}. Omit arguments only if the latest call of that tool is intended.
For an actual observed list, declare foreach with path and where. Object-list
members use {"$item":["actual_field"]}; a list of scalar identifiers uses
{"$item":[]} with where:{}. A scalar, count, time or object is not a list.
Use only observed paths and types. An earlier list may use foreach.receipt
with its exact native query. Each declared iterator presents its next unissued
member for a separate one-token selection; it never executes a batch. Every
member is retained in the declared scope. Do not enumerate a long list as
eight arbitrary examples. Missing prerequisites require native CoT; an empty
bank is correct when no operation can be grounded. Return the supplied schema."""


def author_request(*args, **kwargs):
    body = original_request(*args, **kwargs)
    body['messages'][0]['content'] = AUTHOR
    return body


def author_state_key(task, observation, completed_action, receipts):
    # An unrelated new receipt does not turn the same completed query/result into
    # new evidence for another paid author. No hidden environment state is used.
    return digest({'task': task, 'observation': observation, 'action': completed_action})


def origin_for(observation, completed_action, receipts):
    arguments = completed_action.get('action_input')
    if not isinstance(arguments, dict):
        raise ValueError('The original completed native arguments are required')
    reference = {'tool': completed_action['action'], 'arguments': deepcopy(arguments), 'path': []}
    _, receipt = receipt_value(reference, receipts)
    expected = hashlib.sha256(observation.encode()).hexdigest()
    if receipt['native_observation_sha256'] != expected:
        raise ValueError('Bank origin must be the original completed public observation')
    return {'reference': reference, 'observation_sha256': expected,
            'log_index': receipt['log_index'], 'previous': deepcopy(arguments)}


def origin_current(origin, receipts):
    if origin is None:
        return False
    try:
        _, receipt = receipt_value(origin['reference'], receipts)
        return receipt['native_observation_sha256'] == origin['observation_sha256']
    except (KeyError, TypeError, ValueError, IndexError, SyntaxError):
        return False


def anchored_bank(bank, origin):
    """Bind CoT-declared source operators without changing their plan or scope."""
    def freeze(value):
        if isinstance(value, dict):
            if '$obs' in value:
                if len(value) != 1:
                    raise ValueError('An observation operator cannot hide fields')
                return {'$receipt': {**deepcopy(origin['reference']), 'path': deepcopy(value['$obs'])}}
            if '$prev' in value:
                if len(value) != 1:
                    raise ValueError('A previous-argument operator cannot hide fields')
                # Resolve each declared path inside the candidate-local binder,
                # using the retained original arguments supplied by the provider.
                return deepcopy(value)
            return {key: freeze(child) for key, child in value.items()}
        if isinstance(value, list):
            return [freeze(child) for child in value]
        return deepcopy(value)
    result = deepcopy(bank)
    for candidate in result['candidates']:
        candidate['action_input'] = freeze(candidate['action_input'])
        if 'foreach' in candidate and 'receipt' not in candidate['foreach']:
            candidate['foreach']['receipt'] = {
                key: deepcopy(value) for key, value in origin['reference'].items() if key != 'path'}
    return result


def instantiate_frontier(bank, observation, previous, tools, seen, receipts):
    """One next unissued member per declared template, with full scope evidence."""
    if len(bank['candidates']) > 8:
        raise ValueError('The CoT template capacity changed')
    rows, failures, used, identities = [], [], {}, set()
    for index, template in enumerate(bank['candidates']):
        try:
            local_used, members, scope_digest = {}, [(0, None)], None
            if 'foreach' in template:
                scope = template['foreach']
                source, receipt = receipt_value({**scope['receipt'], 'path': scope['path']}, receipts)
                local_used[receipt['log_index']] = receipt
                if not isinstance(source, list):
                    raise ValueError('foreach requires an actual public list, never a scalar')
                if scope['where'] and any(not isinstance(item, dict) for item in source):
                    raise ValueError('Object-field filters cannot be applied to scalar members')
                members = [(ordinal, item) for ordinal, item in enumerate(source)
                           if not scope['where'] or all(key in item and item[key] == expected
                               for key, expected in scope['where'].items())]
                scope_digest = digest(source)
            remaining, next_row = 0, None
            for ordinal, member in members:
                arguments = bind_arguments(template['action_input'], observation, previous,
                                           member, receipts, local_used)
                json.dumps(arguments, allow_nan=False)
                specs = tools[template['action']].inputs
                if set(arguments) - set(specs) or any(
                    key not in arguments and 'default' not in spec
                    or key in arguments and not well_typed(arguments[key], spec)
                    for key, spec in specs.items()):
                    raise ValueError('Candidate violates the declared native argument contract')
                action = {'action': template['action'], 'action_input': arguments}
                fingerprint = digest(action)
                if fingerprint in seen or fingerprint in identities:
                    continue
                remaining += 1
                if next_row is None:
                    next_row = {'action': action, 'when': template['when'], 'template': index,
                                'binding_receipt_indices': sorted(local_used),
                                'declared_scope_member_ordinal': ordinal}
            if next_row is not None:
                next_row.update(declared_scope_members=len(members),
                                remaining_unissued_members=remaining,
                                complete_scope_sha256=scope_digest,
                                scope_truncated=False, automatic_batch_execution=False)
                identities.add(digest(next_row['action']))
                rows.append(next_row)
                used.update(local_used)
        except (KeyError, IndexError, TypeError, ValueError, SyntaxError) as error:
            failures.append({'template': index, 'reason': str(error)})
    return rows, failures, [deepcopy(used[index]) for index in sorted(used)]


def public_state_candidates(bank, observation, previous, tools, seen, *, native_receipts=None):
    from phiagent.harness.gaia2_public_literal_table import UnsupportedLiteral
    receipt = {'native_observation_sha256': hashlib.sha256(observation.encode()).hexdigest(),
               'native_observation_preserved': True, 'iterator_contract': CONTRACT}
    try:
        public = observation_data(observation)
    except (SyntaxError, ValueError, TypeError, UnsupportedLiteral) as error:
        return [], [], {**receipt, 'status': 'UNKNOWN', 'route': 'full_cot', 'reason': str(error)}
    rows, failures, used = instantiate_frontier(bank, public, previous, tools, seen, native_receipts)
    return rows, failures, {**receipt, 'status': 'PARSED', 'public_value_type': type(public).__name__,
                           'bank_binding_receipts': used, 'complete_declared_scopes_preserved': True}
