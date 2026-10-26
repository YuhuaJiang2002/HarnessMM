"""Single-call CoT action/bank serialization with native execution and abstention."""
from copy import deepcopy
import json,time

FORMAT_INSTRUCTION = '''For this CoT call, return exactly the JSON object required by the response schema: reasoning, action, action_input, jev_bank. The reasoning field contains your actual native CoT. Choose the actual current native tool and its exact arguments. In jev_bank, explicitly give the remaining plan and up to eight useful LATER operations, never a duplicate of the current action. Bind each later argument to a task literal or the actual public receipt operators described above. A required investigative read may have unknown output while all its inputs are supported. Give an empty candidates array only when no justified later operation can be authored; explain the missing evidence in note. Do not guess fields, identifiers, authorization, effects or timing. Every proposed next operation still needs the independent one-token judge and original public binding/native execution checks. Do not write Thought:/Action: outside this JSON; the native adapter preserves the exact generated reasoning and action values.'''

def response_format(tools):
    from phiagent.harness.gaia2_cot_jev import cot_response_format
    return cot_response_format(tools,allow_routine_writes=True,continuation_bank=True)

def native_text(raw,tools):
    from phiagent.harness.gaia2_cot_jev import parse_cot_envelope
    value=parse_cot_envelope(raw,tools)
    if not isinstance(value['jev_bank'],dict):raise ValueError('Explicit bank required')
    action={k:deepcopy(value[k])for k in ['jev_bank','action','action_input']}
    return 'Thought: '+value['reasoning']+'\nAction:\n'+json.dumps(action,ensure_ascii=False),value

def generate(provider,messages,tools,stop_sequences):
    from scripts.run_gaia2_jev import native_usage,native_response_text
    from phiagent.harness.gaia2_dual_rsi import combine_usage
    prompt=deepcopy(messages)
    from phiagent.harness.gaia2_typed_continuation_bank import INSTRUCTION
    prompt.append({'role':'user','content':FORMAT_INSTRUCTION+'\n\n'+INSTRUCTION})
    started=time.monotonic()
    fmt=response_format(tools)
    raw=provider.transport(provider.endpoint(),{'model':provider.config['model']['served_name'],
        'messages':prompt,'response_format':fmt},timeout=1830)
    original=raw['choices'][0]['message']['content']
    usage={**native_usage(raw,provider.config['model']['served_name'],131072),
           'completion_duration':time.monotonic()-started}
    provider.record({'source':'public_actor_messages','messages':prompt,'response':original,
        'usage':usage,'component':'structured_native_cot_bank_v9'})
    try:
        text,value=native_text(original,tools)
        provider.record({'source':'structured_bank_serialization_v9','bank_field_present':True,
            'authored_candidates':len(value['jev_bank'].get('candidates',[])),
            'real_cot_and_action_values_preserved':True,'standalone_author_calls':0})
        return text,original,usage
    except (ValueError,KeyError,TypeError)as error:
        # A settled, invalid envelope has no executed effect. Preserve its cost;
        # a distinct native fallback is explicit and cannot silently retry it.
        provider.record({'source':'structured_bank_parse_fallback_v9','reason':str(error)})
        second=time.monotonic()
        raw=provider.transport(provider.endpoint(),{'model':provider.config['model']['served_name'],
            'messages':deepcopy(messages),'stop':list(stop_sequences)},timeout=1830)
        original=raw['choices'][0]['message']['content']
        extra={**native_usage(raw,provider.config['model']['served_name'],131072),
               'completion_duration':time.monotonic()-second}
        provider.record({'source':'public_actor_messages','messages':messages,'response':original,
            'usage':extra,'component':'structured_bank_native_fallback_v9'})
        return native_response_text(original,stop_sequences),original,combine_usage(usage,extra)
