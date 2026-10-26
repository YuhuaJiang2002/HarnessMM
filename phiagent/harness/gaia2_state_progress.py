"""Public enumeration subgoals with persistent range memory and native actions.

Only read-only email/contact pagination is compiled. Task completion and all
semantic transformations remain with the original actor. No evaluator state.
"""
import ast
from dataclasses import dataclass, field
import io
import json
import re
import tokenize

from phiagent.harness.gaia2_public_progress import NativePublicProgress, email_page
from phiagent.harness.gaia2_jev import fingerprint, action_object


def contact_page(observation):
    """Read only a top-level metadata literal in the native dictionary repr.

Contact reprs contain Enum reprs, so the whole result is not a Python literal.
Tokenization distinguishes quoted contact content from actual metadata keys.
"""
    if not isinstance(observation, str) or not observation.strip().startswith('{'):
        return None
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(observation).readline))
        depth, candidates = 0, []
        for i, token in enumerate(tokens):
            if token.type == tokenize.STRING and depth == 1 and ast.literal_eval(token.string) == 'metadata':
                if tokens[i+1].string != ':' or tokens[i+2].string != '{':
                    return None
                level, parts = 0, []
                for nested in tokens[i+2:]:
                    parts.append(nested.string)
                    if nested.type == tokenize.OP:
                        level += nested.string in ('{','[','(')
                        level -= nested.string in ('}',']',')')
                    if level == 0:
                        break
                candidates.append(ast.literal_eval(' '.join(parts)))
            if token.type == tokenize.OP:
                depth += token.string in ('{','[','(')
                depth -= token.string in ('}',']',')')
        if depth != 0 or len(candidates) != 1:
            return None
        metadata = candidates[0]
        if set(metadata) != {'range','total'}:
            return None
        start,end = metadata['range'];total=metadata['total']
        if any(type(x) is not int for x in (start,end,total)) or not 0 <= start < end <= total:
            return None
        return start,end,total
    except (SyntaxError, ValueError, TypeError, KeyError, IndexError, tokenize.TokenError, IndentationError):
        return None


def enumeration_goal(task, domain):
    noun = r'(?:e-?mails?|inbox)' if domain == 'email' else r'(?:contacts?|address\s+book)'
    # Require an explicit exhaustive subgoal near its domain, not unrelated 'all'.
    return bool(re.search(r'\b(?:all|every|entire|whole)\b(?:\s+(?!(?:and|then|but|or)\b)[\w-]+){0,5}\s+' + noun + r'\b', task, re.I)
                or re.search(r'\b(?:enumerate|list)\s+(?:my\s+|the\s+)?' + noun + r'\b', task, re.I))


@dataclass
class StatePublicProgress(NativePublicProgress):
    require_stall: bool = False
    persistent_memory: bool = True
    include_contacts: bool = True
    pages: dict = field(default_factory=dict)
    completed: dict = field(default_factory=dict)
    epoch: int = 0

    def remember(self, response, legal_tools):
        action = action_object(response)
        operation = (action or {}).get("action", "").split("__")[-1]
        if operation.startswith(("add_", "delete_", "edit_", "move_", "send_", "reply_", "forward_", "remove_", "create_")):
            self.pages.clear(); self.completed.clear(); self.attempted.clear(); self.epoch += 1
        return super().remember(response, legal_tools)

    def decide(self, *, observation, error, intervened, legal_tools):
        if intervened:
            self.pages.clear(); self.completed.clear(); self.attempted.clear()
            self.stalled = False
            self.epoch += 1
        if not self.persistent_memory:
            self.pages.clear(); self.completed.clear(); self.attempted.clear()
        prior = self.previous_action or {}
        name, args = prior.get('action',''), prior.get('action_input',{})
        current = fingerprint(prior, observation) if prior and isinstance(observation,str) else None
        if error or current is not None and current == self.previous_feedback:
            self.stalled = True
        self.previous_feedback = current
        domain, page, schema = None, None, False
        if name.endswith('__list_emails'):
            domain='email';page=email_page(observation)
            schema=(set(args) <= {'folder_name','offset','limit'} and type(args.get('limit',10)) is int
                    and args.get('limit',10)>0 and (page is None or page[1]==min(page[0]+args.get('limit',10),page[2])))
        elif self.include_contacts and name.endswith('__get_contacts'):
            domain='contacts';page=contact_page(observation);schema=set(args)<= {'offset'}
        offset=args.get('offset',0)
        schema=bool(schema and type(offset) is int and offset>=0)
        goal=domain is not None and enumeration_goal(self.task,domain)
        key=json.dumps([name,{k:v for k,v in args.items() if k!='offset'}],sort_keys=True)
        valid=bool(page and schema and offset==page[0] and not error and not intervened)
        old=self.pages.get(key,{})
        # A changed total invalidates previously accumulated facts for this list.
        if valid and old and any(v['total']!=page[2] for v in old.values()):
            self.pages.pop(key,None);self.completed.pop(key,None)
            self.attempted.clear();old={};self.epoch+=1
        new_range=valid and str(page[0]) not in old
        if valid:
            self.pages.setdefault(key,{})[str(page[0])]={'end':page[1],'total':page[2],'public_feedback_sha256':current}
            if page[1]==page[2]:self.completed[key]={'terminal_offset':page[1],'public_feedback_sha256':current}
        action={'action':name,'action_input':{**args,'offset':page[1]}} if valid and page[1]<page[2] else None
        signature=json.dumps(action,sort_keys=True) if action else None
        checks={'public_observation':isinstance(observation,str) and bool(observation.strip()),
                'no_public_error':not error,'no_intervening_event':not intervened,
                'explicit_enumeration_subgoal':bool(goal),'native_schema_and_range':valid,
                'unseen_range':bool(new_range),'remaining_public_page':action is not None,
                'native_tool_available':name in legal_tools,
                'unattempted_action':action is not None and signature not in self.attempted,
                'stall_gate':self.stalled or not self.require_stall}
        execute=all(checks.values())
        candidates=[] if action is None else [{'action':action,'priority':0,'stage':'acquire_remaining_public_page',
            'argument_provenance':{'offset':'preceding_native_range_end','other_arguments':'preceding_actor_action'},
            'checks':checks}]
        record={'event':'decision','execute':execute,'checks':checks,'selection_model_calls':0,'periodic_refresh':False,
                'compiler':'native_enumeration_state_v2','candidate_origin':'native_schema_and_public_range',
                'ordering':'compiler_defined_subgoal_priority_then_tool_name','candidates':candidates,
                'state':{'epoch':self.epoch,'domain':domain,'stage':'acquire_remaining_public_page' if action else 'return_to_actor',
                         'observed_ranges':self.pages.get(key,{}),'terminal_page':self.completed.get(key),
                         'persistent_memory':self.persistent_memory,'recovery_latched':self.stalled},
                'feedback_sha256':current,'public_range':page,'fast_steps':self.fast_steps}
        if not execute:return None,record
        self.fast_steps+=1;self.previous_action=action;self.attempted.add(signature)
        return 'Thought: Advance the explicitly requested enumeration using its public range.\nAction:\n'+json.dumps(action),record
