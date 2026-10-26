"""Pure native wire/receipt guards; synthetic public calls, no remote execution."""
from copy import deepcopy
import json
import pytest
from phiagent.harness.codex_terminal_bank import decision_packet,compile_response,encode_response,request_for
from scripts.serve_codex_deepseek import terminal_response


def native_request():
    candidates=[{'label':'A','after':'root','condition':'receipt contains READY',
                 'tool':'exec_command','arguments':{'cmd':'printf DONE'}}]
    return {'model':'deepseek-flash','stream':True,'tools':[
        {'type':'function','name':'exec_command','parameters':{'type':'object','properties':{'cmd':{'type':'string'}},'required':['cmd'],'additionalProperties':False}}],
        'input':[
            {'type':'message','role':'user','content':[{'type':'input_text','text':'installation marker'}]},
            {'type':'reasoning','content':'private reasoning marker'},
            {'type':'message','role':'assistant','content':[{'type':'output_text','text':'<CHISEL_PLAN>'+json.dumps({'candidates':candidates})+'</CHISEL_PLAN>'}]},
            {'type':'function_call','name':'exec_command','call_id':'root-call','arguments':'{"cmd":"printf READY"}'},
            {'type':'function_call_output','call_id':'root-call','output':'Chunk id: example\nProcess exited with code 0\nREADY'}]}


def paid_label(label='A'):
    return {'model':'deepseek-flash','choices':[{'message':{'role':'assistant','content':label},'finish_reason':'length',
            'logprobs':{'content':[{'token':label,'logprob':-0.1,'top_logprobs':[{'token':label,'logprob':-0.1}]}]}}],
            'usage':{'prompt_tokens':17,'completion_tokens':1,'total_tokens':18,
                     'prompt_cache_hit_tokens':3,'prompt_cache_miss_tokens':14}}


def test_native_task_receipts_and_schemas_are_exact_and_private_parts_excluded():
    body=native_request();before=deepcopy(body);context=decision_packet(body,'Read READY, then print DONE.')
    packet=context['state'];assert body==before
    text=json.dumps(packet)
    assert 'READY'in text and 'printf DONE'in text
    assert 'private reasoning marker'not in text and 'installation marker'not in text
    assert packet['native_tool_contract'][0]['parameters']==body['tools'][0]['parameters']
    assert context['implementation']=='codex-native-responses-literal-terminal-bank/v1'
    fast=request_for(context);assert fast['max_tokens']==1 and fast['logprobs']is True


@pytest.mark.parametrize('change',['wrong_id','duplicate_receipt','new_user','repeat_root','schema_change'])
def test_stale_ambiguous_or_changed_native_surface_never_supplies_a_fast_packet(change):
    body=native_request()
    if change=='wrong_id':body['input'][-1]['call_id']='unrelated'
    elif change=='duplicate_receipt':body['input'].append(deepcopy(body['input'][-1]))
    elif change=='new_user':body['input'].append({'type':'message','role':'user','content':'Cancel the next command'})
    elif change=='repeat_root':body['input'][2]['content'][0]['text']=body['input'][2]['content'][0]['text'].replace('printf DONE','printf READY')
    else:body['tools'][0]['parameters']['properties']['cmd']={'type':'number'}
    assert decision_packet(body,'Public original task')is None


def test_native_function_frame_and_sse_terminal_usage_reconcile_with_one_paid_label():
    body=native_request();context=decision_packet(body,'Public original task');response,decision=compile_response(body,context,paid_label())
    call=response['output'][-1]
    assert response['output'][0]['content']==[{'type':'reasoning_text','text':'A'}]
    assert call['name']=='exec_command'and json.loads(call['arguments'])=={'cmd':'printf DONE'}
    assert call['call_id']=='chisel_'+context['bank']['sha256'][:24]+'_A'
    raw,kind=encode_response(response,True);parsed=terminal_response(raw,kind)
    assert parsed==response and parsed['usage']['total_tokens']==18 and parsed['usage']['output_tokens']==1
    assert parsed['usage']['input_tokens']==17 and parsed['usage']['input_tokens_details']['cached_tokens']==3
    changed=deepcopy(body);changed['input'][-1]['output']='different public receipt'
    with pytest.raises(ValueError,match='another native request'):compile_response(changed,context,paid_label())


def test_abstention_retains_its_paid_control_and_invalid_labels_fail_closed():
    body=native_request();context=decision_packet(body,'Public original task')
    response,decision=compile_response(body,context,paid_label('?'))
    assert response is None and decision['native_usage']['total_tokens']==18
    with pytest.raises(ValueError):compile_response(body,context,paid_label('Z'))


def test_paid_selector_is_once_per_exact_request_and_native_receipt_is_observed(tmp_path,monkeypatch):
    import hashlib
    import phiagent.harness.codex_terminal_bank as module
    calls=[]
    class Received:
        headers={'X-PhiAgent-Call-ID':'unique-physical-fast'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps(paid_label()).encode()
    def receive(request,timeout):
        calls.append(json.loads(request.data));return Received()
    provider=module.TerminalBankProvider(tmp_path/'bank',provider_url='http://127.0.0.1:1',relay_secret='synthetic secret')
    monkeypatch.setattr(provider.opener,'open',receive)
    task='Public original task';provider.register('trial',public_task=task,enabled=True,task_sha256=hashlib.sha256(task.encode()).hexdigest())
    body=native_request();raw=json.dumps(body).encode()
    first=provider('trial',body,raw);assert first==provider('trial',body,raw)and len(calls)==1
    response=terminal_response(*first);call=response['output'][-1]
    next_body=deepcopy(body);next_body['input'].extend(response['output']+[{'type':'function_call_output','call_id':call['call_id'],'output':'DONE'}])
    assert provider('trial',next_body,json.dumps(next_body).encode())is None
    events=[json.loads(x)for x in(provider.root/'events.jsonl').read_text().splitlines()]
    observed=[x for x in events if x['kind']=='codex-terminal-bank/native-execution-observed']
    paid=[x for x in events if x['kind']=='codex-terminal-bank/paid-decision']
    assert len(observed)==len(paid)==1 and observed[0]['native_output']=='DONE'
    assert paid[0]['physical_call_id']=='unique-physical-fast'and paid[0]['decision']['native_usage']['total_tokens']==18
    altered=deepcopy(next_body);altered['input'][-2]['arguments']='{"cmd":"different"}'
    provider.observed.clear()
    with pytest.raises(ValueError,match='changed its compiled'):provider('trial',altered,json.dumps(altered).encode())


def test_provider_state_dependency_requires_full_history_and_stateless_native_endpoint():
    body=native_request();body['previous_response_id']='provider-owned-parent'
    assert decision_packet(body,'Public original task')is None
    body.pop('previous_response_id');body['store']=True
    assert decision_packet(body,'Public original task')is None
