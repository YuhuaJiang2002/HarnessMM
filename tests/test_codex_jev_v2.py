from contextlib import nullcontext
import json
import pytest
from phiagent.harness.codex_jev_v2 import CodexJevV2, prepare_packet
from tests.test_codex_native_app_bank import TOOLS, response, candidate, fast

CONFIG={'phase':'train','harness':'codex','fast_families':['Files__exists'],'max_fast_bytes':4096}

def test_codex_numeric_action_retains_constraints_and_executes_literal_arguments():
    calls=[];packets=[];events=[]
    def execute(name,args):
        calls.append((name,args));return {'content':[{'type':'text','text':'READY'}],'isError':False}
    def judge(p):packets.append(p);return fast('0')
    b=CodexJevV2(TOOLS,v2=CONFIG,enabled=True,judge=judge,execute=execute,
                generation=nullcontext,finished=lambda:False,audit=events.append)
    c=candidate();c['condition']='The exact original native result is READY'
    b.register(response([c]),public_task=['Read only declared paths.'])
    b.call('Files__cat',{'path':'receipt.txt'})
    assert calls==[('Files__cat',{'path':'receipt.txt'}),('Files__exists',{'path':'done.txt'})]
    assert packets[0]['constraints']==['Read only declared paths.']
    assert packets[0]['candidates'][0]['id']==0
    assert 'native_tools' not in packets[0] and 'private reasoning' not in json.dumps(packets)
    assert sum(e['kind']=='codex-bank/native-action' for e in events)==1

def test_codex_no_train_evidence_cannot_be_relabelled_as_qualified():
    with pytest.raises(ValueError):prepare_packet({}, {**CONFIG,'phase':'test'})
    with pytest.raises(ValueError):prepare_packet({}, {**CONFIG,'harness':'deepseek'})
