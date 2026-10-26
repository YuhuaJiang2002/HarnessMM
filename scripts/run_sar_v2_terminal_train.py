"""Crash-safe fixed-order SAR-v2 TRAIN6 -> TRAIN18 gate. No held-out execution."""
from pathlib import Path
import argparse,json,os,sys,fcntl,subprocess,hashlib,time,csv
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_industrial_capability_baselines as native
from phiagent.data_engine.provenance import write_json_atomic as write,utc_now
POLICY={'version':5,'enabled':True,'maximum_candidates':8,'maximum_fast_steps':8,'maximum_public_bytes':32768}
def ref(p):return {'path':str(p),'sha256':native.digest(p)}
def events_of(row):
 p=Path(row['cost']['sdk_worker_dir'])/'llm-streams.jsonl'
 return [json.loads(l)for l in p.read_text().splitlines()],p

def episode(row,arm,matched=None):
 events,path=events_of(row)
 fast=[e for e in events if e.get('kind')=='jev/request'];commits=[e for e in events if e.get('kind')=='jev/compiled-native-action'];det=[e for e in events if e.get('kind')=='sar/deterministic-commit'];cert=[e for e in events if e.get('kind')=='sar/substitution-certificate']
 slow=[e for e in events if e.get('kind')=='sar/slow-call'];starts=[e for e in events if e.get('kind')=='llm/start'];by_start={e['attempt_id']:e for e in starts}
 ends={e['call_id']:e for e in events if e.get('kind')=='bridge/tool/end'}
 banks={e['bank_sha256']:0 for e in events if e.get('kind')=='sar/bank-authored'}
 for e in cert:banks[e['bank_sha256']]=banks.get(e['bank_sha256'],0)+1
 missing=[e['call']['id']for e in commits if e['call']['id']not in ends]
 if len({e['call_id']for e in cert})!=len(cert):raise ValueError('Duplicate substitution certificate')
 usage=row['cost'].get('usage');total=usage['total_tokens']if usage else None
 accounting=native.load(Path(row['cost']['sdk_worker_dir'])/'usage.json');attempts=accounting.get('attempts',[])
 # Per-request full input includes all cache buckets; output includes native reasoning.
 costs={a['attempt_id']:a.get('usage')for a in attempts}
 def fullinput(u):return (u.get('inputTokens',0)+u.get('cacheReadTokens',0)+u.get('cacheWriteTokens',0))if u else 0
 slow_ids=[e['native_attempt_id']for e in slow]if arm!='native'else [e['attempt_id']for e in starts]
 inputs=[fullinput(costs.get(k))for k in slow_ids]
 fallback=[e for e in events if e.get('kind')in ('jev/slow-fallback','sar/fallback-slow-call')]
 fast_usage=[e['usage']for e in events if e.get('kind')=='jev/decision'and e.get('usage')]
 roots={e['root_id']for e in events if e.get('kind')=='sar/bank-authored'}
 author_ids={e['attempt_id']for e in events if e.get('kind')=='llm/chunk'
  and e.get('chunk',{}).get('type')=='block-end' and e.get('chunk',{}).get('block',{}).get('type')=='tool-call'
  and e['chunk']['block'].get('id')in roots}
 bank_tokens=sum((costs.get(k)or{}).get('totalTokens',0)for k in author_ids)
 summary={'task_id':row['task_id'],'native_or_chisel':arm,'success':row.get('native_score'),
  'total_input_tokens':usage['input_tokens']if usage else None,'total_output_tokens':usage['output_tokens']if usage else None,'total_actor_tokens':total,
  'slow_calls':len(slow_ids),'fast_requests':len(fast),'fast_commits':len(commits),'deterministic_fast_commits':len(det),
  'fallback_slow_calls':len(fallback),'banks_authored':len(banks),'actions_per_bank':list(banks.values()),
  'certified_eliminated_slow_callbacks':len(cert),'bank_author_tokens':bank_tokens,
  'bank_author_tokens_scope':'whole slow calls containing bank output; included in actor total, not added again',
  'fast_input_tokens':sum(u['totalTokens']-u['outputTokens']for u in fast_usage),'fast_output_tokens':sum(u['outputTokens']for u in fast_usage),
  'fallback_tokens':sum(e.get('combined_usage',{}).get('totalTokens',0)for e in events if e.get('kind')=='jev/fallback-usage')+sum((costs.get(e['native_attempt_id'])or{}).get('totalTokens',0)for e in fallback if e['kind']=='sar/fallback-slow-call'),
  'maximum_input_context':max(inputs,default=0),'mean_slow_call_input_tokens':sum(inputs)/len(inputs)if inputs else 0,
  'unknown_usage_calls':len(accounting.get('unknown_reasons',[])),'missing_receipts':len(missing),
  'native_result':ref(Path(row['native_result'])),'raw_stream_receipt':ref(path),'attempt_id':row['attempt_id'],
  'actor_done':(Path(row['cost']['sdk_worker_dir'])/'result.json').is_file(),'actor_usage_complete':usage is not None,
  'verifier_done':row.get('native_score')in(0,1),'verifier_result':row.get('native_score'),'receipt_hash':native.digest(path),
  'SCER':len(cert)/matched['slow_calls']if matched and matched['slow_calls']else None,
  'tokens_per_success':total if row.get('native_score')==1 else None,'tokens_per_episode':total,
  'fast_actions_per_bank':len(commits)/len(banks)if banks else 0,
  'fallback_rate':len(fallback)/(len(fallback)+len(commits))if fallback or commits else 0}
 if matched:summary['tokens_saved_per_certified_substitution']=(matched['total_actor_tokens']-total)/len(cert)if cert and total is not None else None
 return summary

def stage_summary(name,pairs):
 n=[p['native']for p in pairs];s=[p['sar']for p in pairs]
 complete=all(x['actor_usage_complete']and x['verifier_done']and not x['unknown_usage_calls']and not x['missing_receipts']for x in n+s)
 nt=sum(x['total_actor_tokens']or 0 for x in n);st=sum(x['total_actor_tokens']or 0 for x in s)
 ns=sum(x['slow_calls']for x in n);ss=sum(x['slow_calls']for x in s);cert=sum(x['certified_eliminated_slow_callbacks']for x in s)
 sr=sum(x['success']or 0 for x in s);nr=sum(x['success']or 0 for x in n)
 fast=sum(x['fast_commits']for x in s);fallback=sum(x['fallback_slow_calls']for x in s)
 r={'Stage':name,'Native SR':nr/len(n),'SAR SR':sr/len(s),'Native tokens':nt,'SAR tokens':st,'Saving':1-st/nt if nt else None,
 'Native slow':ns,'SAR slow':ss,'SCER':cert/ns if ns else None,'Fast commits':fast,'Actions/bank':[a for x in s for a in x['actions_per_bank']],
 'Fallback rate':fallback/(fallback+fast)if fallback+fast else 0,'Unknown usage':sum(x['unknown_usage_calls']for x in n+s),'complete':complete,'expected_tasks':len(pairs),
 'actor_complete':sum(x['actor_done']for x in s),'verdict_complete':sum(x['verifier_done']for x in s),'usage_complete':sum(x['actor_usage_complete']for x in s),'receipt_complete':sum(x['missing_receipts']==0 for x in s)}
 if name=='TRAIN6':r['gate_passed']=complete and sr>=nr and st*5<=nt*4 and ss<ns and sum(max(x['actions_per_bank'],default=0)>=3 for x in s)>=2
 else:r['gate_passed']=complete and sr>=10 and st<=19985526 and cert*10>=ns*3 and r['Fallback rate']<.15
 return r

def run(queue):
 native.require_remote();assert ROOT==queue/'source'
 lock=(queue/'OWNER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=native.load(queue/'CONFIG.json');parent=Path(cfg['native_parent']);preflight=native.load(Path(cfg['native_preflight']))
 assert preflight['status']=='PASS' and preflight['benchmark_calls']==0
 for name,h in native.load(queue/'SOURCE_SHA256.json').items():assert native.digest(ROOT/name)==h,name
 parent_report=native.load(parent/'REPORT.json');assert parent_report['coverage']=={'expected':18,'scored':18,'successes':10}and not parent_report['active']
 rows={r['task_id']:r for r in parent_report['attempts']};assert sum(r['cost']['usage']['total_tokens']for r in rows.values())==28550752
 arm=Path(cfg['train_run']);conf=native.load(Path(cfg['parent_config']));conf.update(maximum_workers=cfg['workers'],provider_durable_capture=True)
 # Original task, model, common prompt, limits, verifier, seeds and partition stay unchanged.
 if not arm.exists():native.prepare(conf,'deepseek',arm,extra_source_files=tuple(native.load(queue/'SOURCE_SHA256.json')))
 conf=native.load(arm/'CONFIG.json');manifest=native.load(arm/'SPLIT_MANIFEST.json');ids=manifest['splits']['train'];assert len(ids)==18
 write(queue/'FROZEN_TRAIN_ORDER.json',{'train_ids':ids,'quick_ids':ids[:6],'selection':'first six of frozen order; no outcome-based choice','partition':ref(arm/'SPLIT_MANIFEST.json')})
 pairs={}
 for i,ident in enumerate(ids):
  r=rows[ident];assert r['shared_prompt_sha256']==conf['shared_prompt_sha256']and r['partition_sha256']==conf['partition_sha256']and r['phase']=='train'
  assert native.digest(Path(r['native_result']))==r['native_result_sha256']
  pairs[ident]={'native':episode(r,'native')}
 def execute(ident):
  i=ids.index(ident);directory=arm/'tasks'/f'{i:02d}-{ident}'
  done=directory/'RESULT.json'
  if done.exists():row=native.load(done)
  elif (directory/'OWNER.json').exists():
   if not(directory/'native_result.json').exists():raise RuntimeError('Unsettled paid actor; no replay: '+ident)
   row=native.harvest(conf,'deepseek',arm,directory,ident)
  else:
   directory.mkdir(parents=True,exist_ok=False)
   cmd=native.trial_command(conf,'deepseek',arm,directory,ident);job=native.load(directory/'HARBOR_JOB.json')
   artifact=directory/'instruction.txt';artifact.write_text('Follow the frozen SAR-v2 persistent-bank protocol. Complete the original native task and all required verification.\n')
   job['agents'][0]['kwargs'].update(arm='chisel',cot_jev=POLICY,
    command_environment=conf['command_environment'],provider_durable_capture=True,
    chisel={'path':str(artifact),'sha256':native.digest(artifact),'artifact_kind':'training-derived',
      'provenance':{'identity':'human-specified SAR-v2 controller protocol; native slow CoT authors all literal action banks','train_diagnosis':cfg['train_diagnosis'],'validation_or_test_feedback_used':False}})
   write(directory/'HARBOR_JOB.json',job)
   # Write intent before actor launch; a crash in this gap blocks automatic replay.
   write(directory/'OWNER.json',{'status':'DISPATCH_INTENT','at':utc_now(),'command':cmd})
   with(directory/'actor.log').open('ab')as log:
    proc=subprocess.Popen(cmd,cwd=arm/'source',stdout=log,stderr=log,env={**os.environ,'PYTHONPATH':str(arm/'source'),'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL)
    write(directory/'OWNER.json',{'status':'RUNNING','pid':proc.pid,'at':utc_now(),'command':cmd});proc.wait()
   row=native.harvest(conf,'deepseek',arm,directory,ident)
  result=episode(row,'sar',pairs[ident]['native']);write(directory/'SUBSTITUTION_LEDGER.json',result)
  return ident,result
 for stage,selected in [('TRAIN6',ids[:6]),('TRAIN18',ids[6:])]:
  with ThreadPoolExecutor(max_workers=cfg['workers'])as pool:
   for f in as_completed([pool.submit(execute,i)for i in selected]):
    ident,result=f.result();pairs[ident]['sar']=result
    ready=[{'task_id':i,**pairs[i]}for i in ids if 'sar'in pairs[i]]
    write(queue/'PER_TASK_MATCHED.json',ready)
    write(queue/'STATE.json',{'at':utc_now(),'stage':stage,'settled':len(ready),'expected':6 if stage=='TRAIN6'else 18,'test_opened':False,'cumulative_token_limit':None})
    print(json.dumps({'stage':stage,'task_id':ident,'success':result['success'],'tokens':result['total_actor_tokens'],'slow_calls':result['slow_calls'],'certified':result['certified_eliminated_slow_callbacks']}),flush=True)
  ready=[{'task_id':i,**pairs[i]}for i in ids[:6]if 'sar'in pairs[i]]if stage=='TRAIN6'else [{'task_id':i,**pairs[i]}for i in ids]
  report=stage_summary(stage,ready);write(queue/(stage+'.json'),report);print(json.dumps(report),flush=True)
  if not report['gate_passed']:
   write(queue/'GATE_FAILED.json',{'stage':stage,'report':report,'validation_or_test_opened':False,'paid_work_preserved':True,'next':'TRAIN-only prospective mechanism revision after auditing these complete receipts'});return
 write(queue/'TRAIN_PASSED.json',{'at':utc_now(),'next':'Source-bound matched VALID18 requires the same frozen SAR configuration','test_opened':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);a=p.parse_args()
 try:run(a.queue.resolve())
 except Exception as e:write(a.queue/'BLOCKER.json',{'error':str(e),'at':utc_now(),'paid_work_preserved':True,'actor_replayed':False});raise
