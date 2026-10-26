#!/usr/bin/env python3
"""Matched, source-frozen native Codex terminal banks from complete TRAIN18."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse,fcntl,hashlib,json,os,secrets,socket,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phiagent.harness.codex_terminal_bank import TerminalBankProvider,POLICY,decision_packet,compile_response
from phiagent.harness.codex_terminal_bank_contract import CORE_METHODS,PARTITION_SHA256
from phiagent.harness.industrial_common_prompt import append_common_prompt,PROMPT_SHA256,PROMPT_VERSION
from phiagent.harness.codex_native_app_bank import instruction
from scripts.serve_codex_deepseek import make_server,read_credential,usage_summary,native_receipt_fields
from scripts.run_codex_terminal import executed_native_verdict
from scripts import run_industrial_capability_baselines as native
ARMS=('native','cot_only','cot_jev')
HOSTS={'h200-1','h200-2','h20-1','h20-2','h20-3','h20-4','a800-4'}
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,d,immutable=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if immutable and p.exists():raise ValueError('Preserve frozen artifact: '+str(p))
 tmp=p.with_suffix(p.suffix+'.writing');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def reference(p):return {'path':str(p),'sha256':sha(p)}

def summary(rows,ids,author_tokens):
 expected={(i,a)for i in ids for a in ARMS};keys=[(r['task_id'],r['arm'])for r in rows]
 if len(keys)!=len(set(keys))or not set(keys)<=expected:raise ValueError('Duplicate or unregistered native actor')
 valid=lambda r:r['valid']is True and type(r['native_score'])in (int,float)and r['native_score']in (0,1)and type(r['actor_tokens'])is int and r['actor_tokens']>=0
 out={'expected':len(expected),'settled':len(rows),'complete':set(keys)==expected and all(valid(r)for r in rows),'arms':{}}
 for arm in ARMS:
  sub=[r for r in rows if r['arm']==arm];complete=len(sub)==len(ids)and all(valid(r)for r in sub)
  paid=sum(r['actor_tokens']for r in sub if type(r['actor_tokens'])is int);auth=author_tokens if arm!='native'else 0
  out['arms'][arm]={'settled':len(sub),'complete':complete,'native_successes':sum(r['native_score']==1 for r in sub),
   'native_success_percent':100*sum(r['native_score']==1 for r in sub)/len(ids)if complete else None,
   'known_actor_token_subtotal':paid,'actor_tokens':paid+auth if complete else None,'shared_bank_author_tokens':auth,
   'native_fast_actions':sum(r.get('native_fast_actions',0)for r in sub)}
 if out['complete']:
  own=out['arms']['cot_jev'];controls=[out['arms'][a]for a in ('native','cot_only')]
  for percent,key in [(30,'admission_30_percent_gate_passed'),(40,'registered_40_percent_goal_passed')]:
   own[key]=all(own['native_successes']>=c['native_successes']and 100*own['actor_tokens']<=(100-percent)*c['actor_tokens']for c in controls)
 return out

def audit_physical_usage(usage,seen=None):
 if usage['usage_status']!='MEASURED'or usage['provider_models']!=['deepseek-flash']:raise ValueError('Complete same-model provider fees required')
 seen=set()if seen is None else seen;total=dict.fromkeys(('input_tokens','output_tokens','total_tokens'),0);calls={}
 for name in usage['receipts']:
  p=Path(name);r=load(p)
  if r['call_id']in seen:raise ValueError('A physical provider call was counted twice')
  seen.add(r['call_id']);raw=(p.parent/'response.body').read_bytes()
  actual=native_receipt_fields(raw,r['content_type'],r['http_status'],endpoint=r['endpoint'])
  if actual['usage_status']!='MEASURED'or actual['usage']!=r['usage']or r['request_sha256']!=sha(p.parent/'request.json')or r['body_rewritten']is not False:
   raise ValueError('Original wire usage or byte-preserving provider ledger changed')
  for k in total:total[k]+=actual['usage'][k]
  calls[r['call_id']]={'receipt':reference(p),'request':reference(p.parent/'request.json'),'response':reference(p.parent/'response.body'),
                      'endpoint':r['endpoint'],'usage':actual['usage']}
 if total!=usage['known_usage']or len(calls)!=usage['calls']:raise ValueError('Full physical actor fees do not reconcile')
 return total,calls

def bank_execution_audit(root,ident,public_task,arm,calls):
 path=root/'native-bank/events.jsonl'
 events=[json.loads(x)for x in path.read_text().splitlines()]if path.exists()else[]
 events=[x for x in events if x.get('trial_id')==ident];paid=[x for x in events if x['kind']=='codex-terminal-bank/paid-decision']
 proofs=[x for x in events if x['kind']=='codex-terminal-bank/native-execution-observed'];qualified=[];compiled={}
 if arm!='cot_jev'and(paid or proofs):raise ValueError('Fast calls cannot occur in either native control')
 for e in paid:
  c=calls[e['physical_call_id']];context_path=Path(e['context']);context=load(context_path);raw=(context_path.parent/'native-request.json').read_bytes();body=json.loads(raw)
  if sha(context_path)!=e['context_sha256']or hashlib.sha256(raw).hexdigest()!=e['native_request_bytes_sha256']or decision_packet(body,public_task)!=context:
   raise ValueError('Fast state differs from its frozen public native request')
  if c['endpoint']!='/chat/completions'or c['usage']['output_tokens']!=1:raise ValueError('Fast control is not the accepted paid one-token selector')
  chat=load(Path(c['response']['path']));local=e['local_native_compilation']
  if sha(Path(c['response']['path']))!=e['fast_response_sha256']:raise ValueError('Actual paid fast response changed')
  try:response,decision=compile_response(body,context,chat)
  except ValueError:
   if local is not None or 'invalid_decision'not in e['decision']:raise
   continue
  if decision!=e['decision']or(response is None)!=(local is None):raise ValueError('Paid typed label or abstention was misreported')
  if local is None:continue
  # Random native transport IDs and timestamps are local, explicitly logged;
  # the declared action, label and actual measured usage must match exactly.
  actual=response['output'][-1];stored=local['output'][-1]
  if any(actual[k]!=stored[k]for k in ('name','arguments','call_id'))or response['usage']!=local['usage']or local['output'][0]['content']!=response['output'][0]['content']:
   raise ValueError('Local native frame altered the slow-authored action or paid label')
  if stored['call_id']in compiled:raise ValueError('Duplicate fast native call identity')
  compiled[stored['call_id']]=stored
 for e in proofs:
  call=e['native_call'];original=compiled[call['call_id']]
  if call!=original or hashlib.sha256(e['native_output'].encode()).hexdigest()!=e['native_output_sha256']:
   raise ValueError('Fast native execution receipt changed')
  candidates=[load(Path(x['request']['path']))for x in calls.values()if x['endpoint']=='/responses']
  if not any(any(z.get('type')=='function_call_output'and z.get('call_id')==call['call_id']and z.get('output')==e['native_output']for z in b.get('input',[]))for b in candidates):
   raise ValueError('Original Codex did not return the compiled native execution receipt')
  qualified.append(e)
 return {'paid_fast_decisions':len(paid),'native_fast_actions':len(qualified),'native_fast_execution_proofs':qualified,
         'physical_actor_calls':len(calls),'clock':'Unmodified native Harbor agent timeout; selector and slow fallbacks included',
         'implementation':'original Codex Responses boundary with local literal compilation of a paid control label'}

class Campaign:
 def __init__(self,root):
  native.require_remote();self.root=root;self.cfg=load(root/'CONFIG.json');self.hashes=load(root/'SOURCE_SHA256.json')
  if ROOT!=root/'source':raise ValueError('Run only the new immutable source capsule')
  for n,h in self.hashes.items():
   if sha(ROOT/n)!=h:raise ValueError('Frozen source changed: '+n)
  if self.cfg['cumulative_token_limit']is not None or self.cfg['token_usage_mode']!='unlimited':raise ValueError('No cumulative token ceiling')
  self.parent=Path(self.cfg['parent_root']);self.native_cfg=load(self.parent/'NATIVE_CONFIG.json')
  self.partition_path=Path(self.native_cfg['partition_dir'])/'SPLIT_MANIFEST.json';self.partition=load(self.partition_path)
  if sha(self.partition_path)!=PARTITION_SHA256 or self.partition['counts']!={'train':18,'validation':18,'test':53}:raise ValueError('Preserve native complete18/18/53 population')
  self.acceptance_path=Path(self.cfg['native_acceptance']);accepted=load(self.acceptance_path)
  if accepted['status']!='VERIFIED_CODEX_NATIVE_TERMINAL_BANK_INTERFACE'or set(accepted['cases'])!={'cot_jev','cot_only','cot_jev_abstention'}:
   raise ValueError('All actual native CLI interface cases required')
  accepted_hashes=load(self.acceptance_path.parent.parent/'SOURCE_SHA256.json')
  for n in ('phiagent/harness/codex_terminal_bank.py','scripts/serve_codex_deepseek.py','phiagent/harness/codex_runtime.py',
            'phiagent/harness/codex_native_app_bank.py','integrations/deepseek_harness/cot_jev.mjs','integrations/deepseek_harness/codex_bank_contract.mjs'):
   if self.hashes[n]!=accepted_hashes[n]:raise ValueError('Accepted terminal boundary changed: '+n)
  self.rows=[];self.active={};self.phase='preparation';self.author_tokens=0

 def capacity(self):
  pool=load(self.root/'POOL.json');fresh=0<=time.time()-datetime.fromisoformat(pool['at']).timestamp()<300
  if not fresh or {x['alias']for x in pool['snapshots']}!=HOSTS:return False
  if next(x for x in pool['snapshots']if x['alias']=='a800-4')['returncode']!=0:return False
  actual=native.storage_capacity(self.native_cfg,self.root);save(self.root/'CAPACITY.json',actual)
  return actual['ready']

 def state(self,**extra):
  save(self.root/'STATE.json',{'at':now(),'pid':os.getpid(),'phase':self.phase,'active':list(self.active.values()),
   'cumulative_token_limit':None,'test_opened':self.phase=='test',**extra})

 def adopt(self):
  rows=load(self.parent/'TRAIN_NATIVE_ROWS.json');seen=set();profiles=[];retained=[]
  if len(rows)!=18 or {x['task_id']for x in rows}!=set(self.partition['splits']['train']):raise ValueError('Full capable TRAIN parent required')
  for row in rows:
   if row['phase']!='train'or row['arm']!='native'or row['seed']!=20260928 or row['status']!='SCORED'or not row['usage_complete']or not row['executed_native_tests_verified']:
    raise ValueError('Retained original native TRAIN identity changed')
   total,calls=audit_physical_usage({'usage_status':'MEASURED','provider_models':['deepseek-flash'],'receipts':row['provider_receipts'],
    'known_usage':row['usage'],'calls':len(row['provider_receipts'])},seen)
   entry=row.get('actor_receipt')or row.get('retention_proof',{}).get('original_actor')
   if not entry or sha(entry['path'])!=entry['sha256']:raise ValueError('Retained native actor source changed')
   record=load(entry['path'])
   if record['method']!='codex'or record['model']!='deepseek-flash'or record['context_control']['active_context_trigger_tokens']!=996147 or record['shared_prompt']['sha256']!=PROMPT_SHA256:
    raise ValueError('Use the actual capable unchanged native Codex parent')
   task=Path(row['task_path']);public=[]
   for c in calls.values():
    request=load(c['request']['path'])
    for m in request.get('input',[]):
     if m.get('type')=='function_call_output':public.append({'call_id':m['call_id'],'output':m.get('output')})
   # Each stateless native request repeats earlier receipts; deduplicate by ID.
   public={x['call_id']:x for x in public};receipts=list(public.values())
   excerpt=json.dumps(receipts[:4]+receipts[-6:],ensure_ascii=False)
   profiles.append({'task_id':row['task_id'],'phase':'train','native_score':row['score'],'actor_tokens':total['total_tokens'],
    'original_public_task':(task/'instruction.md').read_text(),'native_public_tool_receipt_excerpt':excerpt[:8000],
    'deterministic_trace_selection':'first4/last6 original public function outputs; no private reasoning',
    'original_actor':entry,'physical_call_count':len(calls)})
   retained.append({'task_id':row['task_id'],'arm':'native','phase':'train','seed':20260928,'valid':True,
    'native_score':int(row['score']),'actor_tokens':total['total_tokens'],'native_fast_actions':0,
    'paid_parent_row':row,'native_actor_replays':0})
  if sum(x['native_score']for x in retained)!=14 or sum(x['actor_tokens']for x in retained)!=61682026:raise ValueError('Complete native parent fees or capability changed')
  save(self.root/'ADOPTED_PARENT.json',{'rows':retained,'train_parent':reference(self.parent/'TRAIN_NATIVE_ROWS.json'),
       'complete_physical_calls':len(seen),'native_actor_replays':0},True)
  return retained,profiles

 def author(self,profiles):
  from phiagent.harness.codex_runtime import run_codex_trial
  self.phase='train_author';self.state()
  prompt={'phase':'train','training_ids':self.partition['splits']['train'],'public_native_train_profiles':profiles,
   'objective':'Author exactly ONE concise reusable native Codex terminal action-bank skill from all18 complete TRAIN public tasks and traces. '
    'Preserve native success of14/18 and target40% fewer full actor input/output tokens against both the native parent and identical fast-disabled skill. '
    'Slow CoT declares literal future exec_command/write_stdin actions in visible CHISEL_PLAN, outside private reasoning. '
    'The original Codex executes the single root; a charged one-token selector can execute a declared continuation from its actual public receipt, or abstain for more CoT. '
    'Use the exact original tool schemas: exec_command requires cmd, write_stdin requires session_id. Do not invent fields, future IDs, results or files. '
    'Author useful multi-action inspect/edit/build/verify chains when arguments are already known. Use task-relevant executable scripts to reduce repeated large-history narration and tool calls. '
    'Conditions must be established by actual post-root receipts, not exit zero alone; preserve every public artifact requirement and native verification. '
    'Do not copy the root into candidates, re-execute a paid completed command, bank a result-dependent unknown edit, or stop at a plan. '
    'Do not use validation/test tasks, oracle tests, hidden solutions or scenario-specific answers; keep native resources, clocks, context and output limits. '
    'Do not impose artificial token/turn/skill-character limits. Keep only executable guidance. The same literal skill and declaration contract go to both bank arms. '
    'Do not use author tools or write workspace files. Return only final JSON with instruction (nonempty) and supporting_task_ids (nonempty unique supplied TRAIN IDs).',
   'accepted_declaration_contract':instruction(POLICY)}
  save(self.root/'TRAIN_ONLY_AUTHOR_REQUEST.json',prompt,True)
  result=run_codex_trial(output=self.root/'author/native',prompt=json.dumps(prompt,ensure_ascii=False),
   trial_id=self.root.name+'-slow-author',provider_url=self.local_url,relay_secret_file=self.secret_file,
   codex_binary=Path(self.native_cfg['codex']['binary']),timeout_seconds=3600)
  save(self.root/'author/RESULT.json',result,True)
  if result['status']!='COMPLETED':raise ValueError('Preserve paid failed author; no automatic author replay')
  total,calls=audit_physical_usage(result['provider_usage']);self.author_tokens=total['total_tokens']
  answer=result['final_answer'].strip()
  if answer.startswith('```json\n')and answer.endswith('\n```'):answer=answer[len('```json\n'):-len('\n```')]
  value=json.loads(answer);skill=value['instruction'];support=value['supporting_task_ids']
  if not isinstance(skill,str)or not skill.strip()or not support or len(support)!=len(set(support))or not set(support)<=set(self.partition['splits']['train']):
   raise ValueError('Preserve paid author; skill format or TRAIN support unresolved')
  self.skill=self.root/'author/skill.txt';self.skill.write_text(skill)
  self.residual=self.root/'author/residual.txt';self.residual.write_text(instruction(POLICY)+'\n\n'+skill)
  contract={'schema':'codex-terminal-native-bank/v1','policy':POLICY,'model':'deepseek-flash',
   'cumulative_token_limit':None,'token_usage_mode':'unlimited','heldout_feedback_to_training':False,
   'native_context_trigger_tokens':996147,'source_frozen_before_execution':True,
   'partition_sha256':PARTITION_SHA256,'shared_prompt_sha256':PROMPT_SHA256,'populations':self.partition['splits'],
   'residual_sha256':sha(self.residual),'native_train_parent':{'task_count':18,'successes':14,'actor_tokens':61682026},
   'interface_acceptance_status':load(self.acceptance_path)['status'],'native_acceptance':reference(self.acceptance_path),
   'author':{'total_tokens':self.author_tokens,'receipt_sha256':sha(self.root/'author/RESULT.json'),'training_only':True,
             'physical_calls':calls,'supporting_task_ids':support,'skill_sha256':sha(self.skill)},
   'train_author_request':reference(self.root/'TRAIN_ONLY_AUTHOR_REQUEST.json'),'source_manifest':reference(self.root/'SOURCE_SHA256.json')}
  self.contract=self.root/'TRAIN_FROZEN_CONTRACT.json';save(self.contract,contract,True)

 def execute(self,phase,task_id,arm):
  folder=self.root/'trials'/phase/arm/'20260928'/task_id
  if folder.exists():raise ValueError('Paid actor cannot be replayed: '+str(folder))
  folder.mkdir(parents=True);task=Path(self.native_cfg['task_root'])/task_id;ident='cb-'+hashlib.sha256(str(folder).encode()).hexdigest()[:28]
  public=append_common_prompt((task/'instruction.md').read_text());self.bank.register(ident,public_task=public,enabled=arm=='cot_jev',task_sha256=hashlib.sha256(public.encode()).hexdigest())
  argv=[self.native_cfg['codex']['python'],'-B','-u',str(ROOT/'scripts/run_codex_terminal.py'),
   '--task',str(task),'--partition',str(self.partition_path),'--phase',phase,'--output',str(folder/'native-run'),
   '--trial-id',ident,'--provider-url',self.actor_url,'--provider-ledger',str(self.root/'provider'),
   '--relay-secret',str(self.secret_file),'--codex-binary',self.native_cfg['codex']['binary'],
   '--method','codex'if arm=='native'else CORE_METHODS[arm],'--context-compaction-threshold','996147',
   '--common-prompt-version',PROMPT_VERSION,'--common-prompt-sha256',PROMPT_SHA256,
   '--verifier-environment',str(self.root/'VERIFIER_ENV.json'),'--repetition-seed','20260928']
  if arm!='native':argv+=['--residual-file',str(self.residual),'--terminal-bank-contract',str(self.contract),
                        '--terminal-bank-contract-sha256',sha(self.contract),'--terminal-bank-arm',arm]
  if phase=='test':argv+=['--freeze',str(self.root/'FROZEN_FOR_TEST.json')]
  if self.native_cfg.get('sam_verifier_recovery')and task_id=='sam-cell-seg':argv+=['--sam-verifier-recovery',self.native_cfg['sam_verifier_recovery']]
  cache=self.native_cfg.get('native_verifier_caches',{}).get(task_id)
  if cache:argv+=['--native-verifier-cache',cache]
  env={**os.environ,'PYTHONPATH':str(ROOT),'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
  with(folder/'PROCESS.log').open('xb')as log:
   proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
   save(folder/'PROCESS.json',{'pid':proc.pid,'start_ticks':native.process_identity(proc.pid),'command':argv,'at':now()},True);code=proc.wait()
  record_path=folder/'native-run/RUN.json';record=load(record_path)if record_path.exists()else{}
  row={'phase':phase,'task_id':task_id,'arm':arm,'seed':20260928,'valid':False,'native_score':record.get('score'),
       'actor_tokens':None,'native_fast_actions':0,'native_run':str(record_path),'actor_replays':0}
  try:
   if code or record.get('status')!='SCORED'or not record.get('executed_native_tests_verified'):raise ValueError(record.get('error','Native actor/scorer did not settle'))
   total,calls=audit_physical_usage(record['provider_usage'])
   with self.bank.log_lock:audit=bank_execution_audit(self.root,ident,public,arm,calls)
   actual=load(record['native_result']);verdict=executed_native_verdict(actual,folder/'native-run/native')
   if verdict['score']!=row['native_score']:raise ValueError('Actual native verifier score changed')
   save(folder/'NATIVE_BANK_AUDIT.json',{'usage':total,'physical_calls':calls,**audit},True)
   row.update(valid=True,actor_tokens=total['total_tokens'],native_fast_actions=audit['native_fast_actions'],native_run_sha256=sha(record_path),audit=reference(folder/'NATIVE_BANK_AUDIT.json'))
  except (ValueError,KeyError,OSError)as error:
   row['error']=str(error);usage=record.get('provider_usage',{})
   if usage.get('usage_status')=='MEASURED':row['actor_tokens']=usage['known_usage']['total_tokens']
   save(folder/'UNRESOLVED.json',{'error':str(error),'paid_actor_preserved':True,'actor_replays':0},True)
  save(folder/'OUTCOME.json',row,True);return row

 def stage(self,phase,retained=()):
  self.phase=phase;ids=self.partition['splits'][phase];rows=list(retained)
  done={(x['task_id'],x['arm'])for x in rows};pending=[{'task_id':i,'arm':a,'phase':phase}for i in ids for a in ARMS if(i,a)not in done];blocked=False
  with ThreadPoolExecutor(max_workers=self.cfg['executor_workers'])as executor:
   while pending or self.active:
    while pending and len(self.active)<self.cfg['executor_workers']and not blocked and self.capacity():
     req=pending.pop(0);future=executor.submit(self.execute,phase,req['task_id'],req['arm']);self.active[future]=req
    self.state(stage=summary(rows,ids,self.author_tokens))
    if not self.active:
     if blocked:raise RuntimeError('Paid native actor/scorer unresolved; repair without replay')
     time.sleep(15);continue
    completed,_=wait(self.active,timeout=15,return_when=FIRST_COMPLETED)
    for future in completed:
     row=future.result();self.active.pop(future);rows.append(row);blocked|=not row['valid'];save(self.root/(phase.upper()+'_OUTCOMES.json'),rows)
  result=summary(rows,ids,self.author_tokens);save(self.root/(phase.upper()+'_COMPARISON.json'),result,True)
  if not result['complete']:raise ValueError('Complete metered native population required')
  return result

 def run(self):
  lock=(self.root/'OWNER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  save(self.root/'OWNER.json',{'pid':os.getpid(),'start_ticks':native.process_identity(os.getpid()),'at':now()},True)
  retained,profiles=self.adopt();self.secret_file=self.root/'relay.secret';self.secret_file.write_text(secrets.token_hex(32)+'\n');self.secret_file.chmod(0o600)
  holder={};secret=read_credential(self.secret_file)
  relay=make_server(self.root/'provider',read_credential(Path(self.cfg['credential_file'])),secret,
    bind='172.17.0.1',port=self.cfg['provider_port'],allow_native_fast=True,terminal_bank=lambda i,b,r:holder['bank'](i,b,r))
  self.local_url='http://172.17.0.1:'+str(relay.server_port);self.actor_url='http://172.17.0.1:'+str(relay.server_port)
  self.bank=holder['bank']=TerminalBankProvider(self.root/'native-bank',provider_url=self.local_url,relay_secret=secret)
  threading.Thread(target=relay.serve_forever,daemon=True).start()
  try:
   while not self.capacity():self.state(status='WAITING_FRESH_NATIVE_POOL');time.sleep(15)
   save(self.root/'VERIFIER_ENV.json',self.cfg['verifier_environment'],True)
   self.author(profiles);train=self.stage('train',retained)
   if train['arms']['cot_jev']['native_fast_actions']==0:raise ValueError('No real native fast execution in full TRAIN; revise before held-out')
   validation=self.stage('validation')
   save(self.root/'VALIDATION_SELECTION.json',{'phase':'validation','selected_candidate':reference(self.contract),
     'rule':'single eligible TRAIN-frozen native bank; retain all_outcomes after validation; benefit gates separate',
     'validation':reference(self.root/'VALIDATION_COMPARISON.json'),'heldout_feedback_to_training':False},True)
   files=[reference(self.contract),reference(self.residual),reference(self.root/'SOURCE_SHA256.json'),
          reference(self.root/'VALIDATION_SELECTION.json')]+[reference(ROOT/n)for n in self.hashes]
   save(self.root/'FROZEN_FOR_TEST.json',{'status':'FROZEN_FOR_TEST','at':now(),'partition_sha256':PARTITION_SHA256,
     'files':files,'test_execution_seeds':[20260928],'test_repetition_policy':'single_pass; first registered seed',
     'heldout_outcome_policy':'all_outcomes','cumulative_token_limit':None},True)
   test=self.stage('test');save(self.root/'COMPLETE_TEST.json',test,True);self.state(status='COMPLETE_NATIVE_TEST',stage=test)
  finally:relay.shutdown();relay.server_close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']='';os.umask(0o077)
 try:Campaign(a.root.resolve()).run()
 except BaseException as error:save(a.root/'BLOCKED.json',{'error':str(error),'traceback':traceback.format_exc(),'actor_replays':0});raise
