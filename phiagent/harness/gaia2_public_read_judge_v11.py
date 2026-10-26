"""Disambiguate a grounded public read from unsupported arguments or effects."""
from copy import deepcopy
from phiagent.harness import gaia2_public_source_bank_v10 as previous

for _name in dir(previous):
    if not _name.startswith('_'):
        globals()[_name]=getattr(previous,_name)
CONTRACT='native_parent_grounded_public_read_judge_v11'
CHOICE_SYSTEM='''Select one listed CoT-authored native operation that advances the full
user task using the supplied public evidence. Check its exact arguments, goal,
filters, exceptions, timing, ordering and prior issued operations. Bank notes and
conditions are hypotheses; current observations govern their truth.

A task-needed read, lookup or search may legitimately discover a result that is
not known yet. Unknown output is the purpose of that operation, not a missing
input argument. Select such an investigative operation when its declared query,
known inputs and task purpose are supported. Do not require the result of a read
before selecting that same read. Respect time windows and complete task scope.

For a state change, require public support for every argument and applicable
condition, including authorization, filters, recipients, ordering and timing.
A planned effect or issued action does not prove that the effect occurred. Do not
repeat an operation already achieved or select an operation with an unbound
argument. Choose ? if no listed operation is warranted, its required input or
condition is unavailable, an intervening event changes the task, or fresh CoT is
needed to interpret evidence. Always retain the option to abstain.

Return exactly one supplied finite label. You may select or reject declared
operations only; do not generate new arguments, actions, reasoning or answers.'''


def public_read_judge_messages(messages):
    if not messages or messages[0].get('role')!='system' or not isinstance(messages[0].get('content'),str):
        raise ValueError('Require the original complete native finite-choice messages')
    value=deepcopy(messages)
    value[0]['content']=CHOICE_SYSTEM
    return value
