"""Prospective slow-output contract; native qualification precedes deployment."""
from copy import deepcopy

CONTRACT = 'native_parent_declared_continuation_bank_v4'
BANK_OBLIGATION = '''Continuation-bank output contract:
Keep the native Thought/Action response, selected current action, actual native
tool names, argument types and end marker. In the current Action JSON, explicitly
include jev_bank with a note and candidates. Author useful later actions from
your current reasoning using only the public receipt bindings already declared
in the system contract. If a later action is not supported, emit candidates: [].
This annotation does not execute anything. Every later action requires a separate
one-token native selector or abstention. Do not invent observations or arguments.'''


def declared_bank_prompt(parent_prompt):
    messages = deepcopy(parent_prompt)
    if not messages or messages[0].get('role') != 'system':
        raise ValueError('Preserve the capable native parent system message')
    candidates = [m for m in messages if m.get('role') == 'user']
    if not candidates or not isinstance(candidates[-1].get('content'), str):
        raise ValueError('Require a string-valued native public user turn')
    if BANK_OBLIGATION in candidates[-1]['content']:
        raise ValueError('Do not repeat the declaration contract')
    candidates[-1]['content'] += '\n\n' + BANK_OBLIGATION
    return messages
