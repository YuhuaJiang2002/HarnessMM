"""Native-boundary contract tests; no models, benchmarks, or network calls."""
from contextlib import nullcontext
import json
import pytest
from phiagent.harness.codex_native_app_bank import NativeBankBoundary, POLICY, instruction, fast_request, public_user_text

TOOLS=[{'name':'Files__cat','inputSchema':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}},
       {'name':'Files__exists','inputSchema':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}}]


def response(candidates):
    return {'output':[{'type':'message','role':'assistant','content':[
        {'type':'output_text','text':'<CHISEL_PLAN>'+json.dumps({'candidates':candidates})+'</CHISEL_PLAN>'},
        {'type':'reasoning','text':'private reasoning must never enter fast input'}]},
        {'type':'function_call','call_id':'root-1','name':'mcp__are__Files__cat','arguments':'{"path":"receipt.txt"}'}]}


def candidate(label='A',after='root',tool='Files__exists',path='done.txt'):
    return {'label':label,'after':after,'condition':'public receipt is READY','tool':tool,'arguments':{'path':path}}


def fast(label):
    return {'model':'deepseek-flash','usage':{'prompt_tokens':17,'completion_tokens':1,'total_tokens':18,
        'prompt_cache_hit_tokens':8,'prompt_cache_miss_tokens':9},
        'choices':[{'message':{'content':label},'logprobs':{'content':[{'token':label,'logprob':-0.1}]}}]}


def setup(enabled, labels):
    events=[];calls=[];packets=[]
    def execute(name,args):
        calls.append((name,args));return {'content':[{'type':'text','text':'READY'}],'isError':False}
    def judge(packet):
        packets.append(packet);return fast(next(labels))
    boundary=NativeBankBoundary(TOOLS,enabled=enabled,judge=judge,execute=execute,generation=nullcontext,
                                finished=lambda:False,audit=events.append)
    return boundary,events,calls,packets


def test_fast_disabled_control_preserves_the_identical_slow_bank_and_root_only():
    b,events,calls,packets=setup(False,iter([]));b.register(response([candidate()]))
    result=b.call('Files__cat',{'path':'receipt.txt'})
    assert len(calls)==1 and not packets and result['content']==[{'type':'text','text':'READY'}]
    assert events[0]['kind']=='codex-bank/registered'


def test_native_fast_then_abstain_returns_complete_receipts_and_resumes_slow():
    b,events,calls,packets=setup(True,iter(['A','?']))
    b.register(response([candidate(),candidate('B','A','Files__cat','done.txt')]))
    result=b.call('Files__cat',{'path':'receipt.txt'})
    assert len(calls)==2 and len(packets)==2
    assert calls[-1]==('Files__exists',{'path':'done.txt'})
    assert 'CHISEL_NATIVE_ACTION_RECEIPT'in result['content'][-1]['text']
    assert 'private reasoning' not in json.dumps(packets)
    assert [x['kind']for x in events].count('codex-bank/native-action')==1
    b.call('Files__cat',{'path':'next-slow.txt'})
    assert len(calls)==3 and len(packets)==2


def test_duplicate_root_or_schema_mismatch_never_dispatches_fast_actions():
    for bad in [candidate(tool='Files__cat',path='receipt.txt'),{**candidate(),'arguments':{}},candidate(after='future')]:
        b,events,calls,packets=setup(True,iter([]));b.register(response([bad]));b.call('Files__cat',{'path':'receipt.txt'})
        assert not packets and len(calls)==1 and events[0]['kind']=='codex-bank/invalid'


def test_cot_control_has_identical_bank_contract_and_fast_uses_the_actual_public_text():
    assert instruction({**POLICY,'enabled':True})==instruction({**POLICY,'enabled':False})
    b,events,calls,packets=setup(True,iter(['?']))
    b.register(response([candidate()]));b.call('Files__cat',{'path':'receipt.txt'})
    body=fast_request(packets[0])
    assert json.loads(body['messages'][1]['content'])['public_results'][0]['value']['native_text']=='READY'
    assert body['max_tokens']==1 and body['thinking']=={'type':'disabled'}


def test_observed_error_and_public_overflow_return_to_slow_without_inventing_actions():
    b,events,calls,packets=setup(True,iter(['A']))
    b.execute=lambda name,args:{'content':[{'type':'text','text':'x'*32769}]}
    b.register(response([candidate()]));result=b.call('Files__cat',{'path':'receipt.txt'})
    assert not packets and events[-1]['kind']=='codex-bank/public-overflow'
    assert len(result['content'][0]['text'])==32769


def test_actual_codex_namespace_and_public_user_constraints_reach_the_fast_packet():
    b,events,calls,packets=setup(True,iter(['A']))
    slow=response([candidate()]);slow['output'][-1]['name']='mcp__are.Files__cat'
    task=public_user_text({'input':[{'role':'user','content':[{'type':'input_text','text':'Read only the declared paths.'}]},
                                  {'role':'assistant','content':[{'type':'reasoning','text':'private marker'}]}]})
    b.register(slow,public_task=task);b.call('Files__cat',{'path':'receipt.txt'})
    assert len(calls)==2 and packets[0]['public_user_request']==['Read only the declared paths.']
    assert 'private marker'not in json.dumps(packets)
    b.register({**slow,'output':[slow['output'][0],{**slow['output'][-1],'name':'mcp__other.Files__cat'}]})
    assert b.bank is None and events[-1]['kind']=='codex-bank/invalid'


def test_separate_native_response_namespace_matches_the_actual_provider_envelope():
    b,events,calls,packets=setup(True,iter(['A']))
    slow=response([candidate()]);slow['output'][-1].update(name='Files__cat',namespace='mcp__are')
    b.register(slow);b.call('Files__cat',{'path':'receipt.txt'})
    assert len(calls)==2 and len(packets)==1
    slow['output'][-1]['namespace']='mcp__other';b.register(slow)
    assert b.bank is None


def test_fast_task_is_the_already_delivered_native_notifications_without_actor_setup():
    b,events,calls,packets=setup(True,iter(['A']))
    notifications=[{'type':'USER_MESSAGE','message':'Read done.txt after READY.'}]
    b.native_notifications=lambda:notifications
    b.register(response([candidate()]),public_task=['INSTALLATION: emit CHISEL_PLAN before tools'])
    b.call('Files__cat',{'path':'receipt.txt'})
    packet=packets[0]
    assert packet['public_user_request']==notifications
    assert 'INSTALLATION'not in json.dumps(packet)
    assert packet['root']['name']=='Files__cat'
    notifications.append({'type':'USER_MESSAGE','message':'New requirement after the previous receipt.'})
    assert len(packet['public_user_request'])==1
