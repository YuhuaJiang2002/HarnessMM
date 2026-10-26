"""Pure native Responses action-bank boundary for Codex Terminal-Bench.

The caller supplies the exact public task and an independently metered judge.
This module neither executes commands nor reads a benchmark or provider key.
The original Codex tool dispatcher must execute every emitted function call.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid
import threading
from urllib.request import Request,build_opener,ProxyHandler

from phiagent.harness.codex_native_app_bank import fast_request, contract

POLICY = {'version':4,'enabled':True,'maximum_candidates':16,
          'maximum_fast_steps':16,'maximum_public_bytes':32768}
NATIVE_TOOLS = frozenset({'exec_command','write_stdin'})
MODULE = Path(__file__).resolve().parents[2]/'integrations/deepseek_harness/cot_jev.mjs'


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def native_history(body, public_task):
    """Map only delivered public outputs; keep opaque reasoning out of fast state."""
    if not isinstance(public_task,str)or not public_task.strip():
        raise ValueError('The original delivered public task is required')
    if (body.get('model')!='deepseek-flash' or not isinstance(body.get('input'),list)
            or body.get('previous_response_id')is not None or body.get('store',False)is not False):
        return None
    history=[{'role':'user','content':[{'type':'text','text':public_task}]}]
    pending=[];seen_output=set();seen_call=set();started=False
    for item in body['input']:
        if not isinstance(item,dict):return None
        kind=item.get('type')
        if kind=='message'and item.get('role')=='user':
            # A new user turn after tool activity invalidates the previous bank.
            if started:
                if pending:history.append({'role':'assistant','content':pending});pending=[]
                history.append({'role':'user','content':[]})
        elif kind=='message'and item.get('role')=='assistant':
            for p in item.get('content',[]):
                if p.get('type')in ('output_text','text')and isinstance(p.get('text'),str):
                    pending.append({'type':'text','text':p['text']})
        elif kind=='function_call':
            if item.get('namespace')is not None or item.get('name')not in NATIVE_TOOLS:
                # An ordinary non-terminal action cannot inherit a terminal bank.
                if pending:history.append({'role':'assistant','content':pending});pending=[]
                history.append({'role':'assistant','content':[]});started=True
                continue
            call_id=item.get('call_id');args=item.get('arguments')
            if not isinstance(call_id,str)or call_id in seen_call or not isinstance(args,str):return None
            seen_call.add(call_id);started=True
            pending.append({'type':'tool-call','id':call_id,'name':item['name'],'arguments':args})
        elif kind=='function_call_output':
            call_id=item.get('call_id');output=item.get('output')
            if call_id in seen_output or not isinstance(output,str):return None
            seen_output.add(call_id)
            if pending:history.append({'role':'assistant','content':pending});pending=[]
            history.append({'role':'tool','toolCallId':call_id,'content':[{'type':'text','text':output}]})
        elif kind in ('custom_tool_call','custom_tool_call_output'):
            if pending:history.append({'role':'assistant','content':pending});pending=[]
            history.append({'role':'assistant','content':[]});started=True
    if pending:history.append({'role':'assistant','content':pending})
    return history


def decision_packet(body, public_task):
    """Compile the exact native history against its current unchanged tool schemas."""
    history=native_history(body,public_task)
    if history is None:return None
    tools=[];names=set()
    for tool in body.get('tools',[]):
        if not isinstance(tool,dict):return None
        if tool.get('type')=='function'and tool.get('name')in NATIVE_TOOLS:
            if tool['name']in names:raise ValueError('Ambiguous native terminal catalog')
            names.add(tool['name'])
            tools.append({'name':tool['name'],'parameters':deepcopy(tool['parameters'])})
    if not tools:return None
    options={'model':body['model'],'messages':history,'tools':tools}
    program=('import {decisionState} from '+json.dumps(MODULE.as_uri())+';'
             'let raw="";for await(const chunk of process.stdin)raw+=chunk;'
             'try{const x=JSON.parse(raw);process.stdout.write(JSON.stringify({valid:true,result:decisionState(x.options,x.policy)}));}'
             'catch(e){process.stdout.write(JSON.stringify({valid:false,error:e.message}));}')
    result=subprocess.run(['node','--input-type=module','-e',program],input=json.dumps({'options':options,'policy':POLICY}),
                          text=True,capture_output=True,timeout=10,check=True)
    value=json.loads(result.stdout)
    if not value['valid']:return None
    context=value['result']
    if context is None:return None
    # Both schema transformations and the original native request stay auditable.
    context['native_request_sha256']=digest(body)
    context['complete_native_catalog_sha256']=digest(body['tools'])
    context['public_task_sha256']=hashlib.sha256(public_task.encode()).hexdigest()
    context['implementation']='codex-native-responses-literal-terminal-bank/v1'
    return context


def compile_response(body, context, chat_response):
    """Turn one measured label into its previously slow-authored native function call."""
    if context['native_request_sha256']!=digest(body):
        raise ValueError('The bank decision belongs to another native request')
    candidates=context['state']['candidates']
    decision=contract('decision',response=chat_response,labels=[x['label']for x in candidates],model='deepseek-flash')
    if decision['label']=='?':return None,decision
    candidate=next(x for x in candidates if x['label']==decision['label'])
    call={'type':'function_call','id':'fc_'+uuid.uuid4().hex,'status':'completed',
          'call_id':'chisel_'+context['bank']['sha256'][:24]+'_'+candidate['label'],
          'name':candidate['tool'],'arguments':json.dumps(candidate['arguments'],ensure_ascii=False,separators=(',',':'))}
    usage=decision['native_usage']
    response={'id':'resp_'+uuid.uuid4().hex,'object':'response','created_at':int(time.time()),
              'status':'completed','error':None,'incomplete_details':None,'model':'deepseek-flash',
              # The native thinking transport requires a reasoning_text item
              # before every assistant tool turn. Carry only the actual paid
              # label; this is local transport compilation, not invented CoT.
              'output':[{'type':'reasoning','id':'rs_'+uuid.uuid4().hex,
                         'status':'completed','summary':[],
                         'content':[{'type':'reasoning_text','text':decision['label']}]},call],
              'parallel_tool_calls':False,
              'previous_response_id':body.get('previous_response_id'),
              'usage':{'input_tokens':usage['prompt_tokens'],'output_tokens':usage['completion_tokens'],
                       'total_tokens':usage['total_tokens'],
                       'input_tokens_details':{'cached_tokens':usage['prompt_cache_hit_tokens']},
                       'output_tokens_details':{'reasoning_tokens':0}}}
    return response,decision


def encode_response(response, stream):
    """Use native Responses event names; the callback records this local compilation."""
    if not stream:return json.dumps(response,separators=(',',':')).encode(),'application/json'
    reasoning,call=response['output'];empty={**call,'arguments':'','status':'in_progress'}
    events=[('response.created',{'response':{**response,'status':'in_progress','output':[],'usage':None}}),
            ('response.output_item.added',{'output_index':0,'item':{**reasoning,'status':'in_progress','content':[]}}),
            ('response.output_item.done',{'output_index':0,'item':reasoning}),
            ('response.output_item.added',{'output_index':1,'item':empty}),
            ('response.function_call_arguments.delta',{'output_index':1,'item_id':call['id'],'delta':call['arguments']}),
            ('response.function_call_arguments.done',{'output_index':1,'item_id':call['id'],'arguments':call['arguments']}),
            ('response.output_item.done',{'output_index':1,'item':call}),
            ('response.completed',{'response':response})]
    raw=''.join('event: '+kind+'\ndata: '+json.dumps({'type':kind,'sequence_number':i,**fields},separators=(',',':'))+'\n\n'
                for i,(kind,fields)in enumerate(events))
    return raw.encode(),'text/event-stream'


def request_for(context):
    return fast_request(context['state'])


class TerminalBankProvider:
    """Opt-in decision substitution, using the recorder's metered Chat endpoint.

    Slow Responses requests and responses pass through the original recorder.
    Fast replies are explicitly recorded as local compilations of paid labels.
    The subsequent original Codex request supplies execution receipts.
    """
    def __init__(self, root, *, provider_url, relay_secret):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.provider_url=provider_url.rstrip('/');self.relay_secret=relay_secret
        self.opener=build_opener(ProxyHandler({}))
        self.tasks={};self.locks={};self.cache={};self.emitted={};self.observed=set()
        self.registry_lock=threading.Lock();self.log_lock=threading.Lock()

    def record(self, event):
        with self.log_lock:
            with(self.root/'events.jsonl').open('a')as out:out.write(json.dumps(event,ensure_ascii=False)+'\n')

    def register(self, ident, *, public_task, enabled, task_sha256):
        from scripts.serve_codex_deepseek import trial_token
        trial_token(self.relay_secret,ident)
        if type(enabled)is not bool or hashlib.sha256(public_task.encode()).hexdigest()!=task_sha256:
            raise ValueError('Require the frozen original public task and explicit enabled flag')
        with self.registry_lock:
            if ident in self.tasks:raise ValueError('A native trial bank cannot be replaced')
            self.tasks[ident]=(public_task,enabled);self.locks[ident]=threading.Lock()
        self.record({'kind':'codex-terminal-bank/trial-registered','trial_id':ident,
                     'public_task_sha256':task_sha256,'enabled':enabled,'policy':POLICY})

    def observe(self, ident, body):
        if not isinstance(body.get('input'),list):return
        for item in body.get('input',[]):
            if not isinstance(item,dict):raise ValueError('Malformed native history item')
            key=(ident,item.get('call_id'))
            if item.get('type')!='function_call_output'or key not in self.emitted or key in self.observed:continue
            response=self.emitted[key];call=response['output'][-1]
            calls=[x for x in body['input']if x.get('type')=='function_call'and x.get('call_id')==key[1]]
            outputs=[x for x in body['input']if x.get('type')=='function_call_output'and x.get('call_id')==key[1]]
            if len(calls)!=1 or len(outputs)!=1 or any(calls[0].get(k)!=call[k]for k in ['name','arguments']):
                raise ValueError('The original Codex receipt changed its compiled native action')
            if not isinstance(item.get('output'),str):raise ValueError('Require original scalar native terminal output')
            self.observed.add(key)
            self.record({'kind':'codex-terminal-bank/native-execution-observed','trial_id':ident,
                         'native_call':call,'native_request_sha256':digest(body),
                         'native_output_sha256':hashlib.sha256(item['output'].encode()).hexdigest(),
                         'native_output':item['output'],'execution_source':'original Codex function_call_output'})

    def __call__(self, ident, body, raw):
        if ident not in self.tasks:return None
        public_task,enabled=self.tasks[ident]
        with self.locks[ident]:
            self.observe(ident,body)
            if not enabled:return None
            key=(ident,hashlib.sha256(raw).hexdigest())
            if key in self.cache:return self.cache[key]
            context=decision_packet(body,public_task)
            if context is None:return None
            folder=self.root/ident/uuid.uuid4().hex;folder.mkdir(parents=True)
            (folder/'native-request.json').write_bytes(raw)
            (folder/'CONTEXT.json').write_text(json.dumps(context,ensure_ascii=False,indent=2)+'\n')
            request=request_for(context);payload=json.dumps(request,ensure_ascii=False,separators=(',',':')).encode()
            from scripts.serve_codex_deepseek import trial_token,native_chat_usage
            req=Request(self.provider_url+'/'+ident+'/chat/completions',data=payload,
                        headers={'Authorization':'Bearer '+trial_token(self.relay_secret,ident),
                                 'Content-Type':'application/json'},method='POST')
            with self.opener.open(req,timeout=1800)as received:
                paid_raw=received.read();physical_call=received.headers.get('X-PhiAgent-Call-ID')
            (folder/'fast-response.body').write_bytes(paid_raw)
            chat=json.loads(paid_raw);native_chat_usage(chat)
            try:response,decision=compile_response(body,context,chat)
            except ValueError as error:
                response=None;decision={'invalid_decision':str(error),'native_usage':chat['usage']}
            event={'kind':'codex-terminal-bank/paid-decision','trial_id':ident,
                   'native_request_sha256':digest(body),'native_request_bytes_sha256':key[1],
                   'bank_sha256':context['bank']['sha256'],'context':str(folder/'CONTEXT.json'),
                   'context_sha256':hashlib.sha256((folder/'CONTEXT.json').read_bytes()).hexdigest(),
                   'physical_call_id':physical_call,'decision':decision,
                   'fast_response_sha256':hashlib.sha256(paid_raw).hexdigest(),
                   'local_native_compilation':response,'native_execution_claimed':False}
            self.record(event)
            result=None
            if response is not None:
                result=encode_response(response,body.get('stream',False))
                (folder/'local-native-response.body').write_bytes(result[0])
                self.emitted[(ident,response['output'][-1]['call_id'])]=response
            self.cache[key]=result
            return result
