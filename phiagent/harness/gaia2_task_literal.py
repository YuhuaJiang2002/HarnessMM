"""Preserve explicit task qualifiers in proposed native names, without inference."""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import re

_DEDICATED_FIELDS = {
    'tentative': {'tentative', 'is_tentative', 'status'},
    'provisional': {'provisional', 'status'},
    'draft': {'draft', 'is_draft', 'status'},
    'private': {'private', 'is_private', 'visibility', 'privacy'},
    'confidential': {'confidential', 'visibility', 'privacy', 'access'},
    'urgent': {'priority', 'urgency'},
    'temporary': {'temporary', 'is_temporary', 'status'},
}
_QUALIFIER = '|'.join(_DEDICATED_FIELDS)
_NEGATED = re.compile(r"\b(?:not|never|avoid|don['\u2019]t)\s+(?:\w+\s+){0,2}$", re.I)
_FOLLOWING = (
    r'(?=\s+(?:event|meeting|appointment|note|document|folder|message|record|'
    r'with|for|on|at|from|and|that|which)\b|\s*[.,;!?]|\s*$)'
)


def restoration_key(action, changes):
    fields={change['field'] for change in changes}
    identity={k:v for k,v in action['action_input'].items()
              if k not in fields and k not in ('description','content','body','note')}
    payload={'tool':action['action'],'identity':identity,
             'restorations':[(c['field'],c['before'],c['after'],c['task_quote']) for c in changes]}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def preserve_task_literal(action, task, tools):
    if not isinstance(action, dict) or not isinstance(task, str):
        return action, []
    tool=action.get('action');arguments=action.get('action_input')
    if tool not in tools or not isinstance(arguments, dict):
        return action, []
    operation=tool.split('__')[-1]
    if not re.match(r'^(?:create|add|schedule|rename|update)(?:_|$)', operation):
        return action, []
    inputs=tools[tool].get('inputs', {})
    if not isinstance(inputs, Mapping):
        return action, []
    result=deepcopy(action);changes=[]
    for field in ('title','name'):
        schema=inputs.get(field);old=arguments.get(field)
        if (not isinstance(schema, Mapping) or schema.get('type') not in ('string','str') or
                not isinstance(old, str) or not old.strip() or old!=old.strip()):
            continue
        expression=re.compile(
            r'\b(?:create|schedule|add|make|rename)\s+(?:a|an|the)\s+(?:new\s+)?'
            r'(?P<literal>(?P<qualifiers>(?:(?:'+_QUALIFIER+r')\s+)+)'
            +re.escape(old)+r')'+_FOLLOWING, re.I)
        targets={}
        for match in expression.finditer(task):
            if _NEGATED.search(task[max(0,match.start()-40):match.start()]):
                continue
            qualifiers=match['qualifiers'].lower().split()
            if any(_DEDICATED_FIELDS[q].intersection(inputs) for q in qualifiers):
                continue
            targets[match['literal']]=qualifiers
        if len(targets)!=1:
            continue
        target,qualifiers=next(iter(targets.items()))
        if target.casefold()==old.casefold():
            continue
        result['action_input'][field]=target
        changes.append({'field':field,'before':old,'after':target,'task_quote':target,
                        'qualifiers':qualifiers,'selection_model_calls':0})
    return result,changes
