"""Sequential first-four frozen TRAIN smoke. Never opens validation or TEST."""
from pathlib import Path
import argparse,json,os,sys,fcntl,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_industrial_capability_baselines as native
from scripts.audit_sar_subdag import episode,aggregate
from phiagent.data_engine.provenance import write_json_atomic as write,utc_now
POLICY={'version':6,'enabled':True,'maximum_candidates':16,'maximum_fast_steps':8,'maximum_public_bytes':65536}
ARMS=['native','cot_only','cot_jev','sar']
def run(queue):
 native.require_remote();assert ROOT==queue/'source'
 lock=(queue/'OWNER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=native.load(queue/'CONFIG.json');conf=native.load(Path(cfg['parent_config']));assert conf['model']=='deepseek-flash'
 for f,h in native.load(queue/'SOURCE_SHA256.json').items():assert native.digest(ROOT/f)==h,f
 assert native.load(queue/'preflight/RESULT.json')['status']=='PASS'
 manifest=native.load(Path(cfg['native_parent'])/'SPLIT_MANIFEST.json');ids=manifest['splits']['train'][:4]
 assert len(manifest['splits']['train'])==18 and len(manifest['splits']['validation'])==18 and len(manifest['splits']['test'])==53
 write(queue/'FROZEN_ORDER.json',{'tasks':ids,'criterion':'first four frozen TRAIN IDs; outcome independent','partition_sha256':native.digest(Path(cfg['native_parent'])/'SPLIT_MANIFEST.json'),'test_opened':False})
 parent=native.load(Path(cfg['native_parent'])/'REPORT.json');rows={r['task_id']:r for r in parent['attempts']};out={a:[]for a in ARMS}
 if(queue/'EPISODES.json').exists():out=native.load(queue/'EPISODES.json')
 for ident in ids:
  if not any(r['task_id']==ident for r in out['native']):
   r=rows[ident];assert r['phase']=='train'and r['partition_sha256']==conf['partition_sha256']and r['shared_prompt_sha256']==conf['shared_prompt_sha256'];out['native'].append(episode(r,False));write(queue/'EPISODES.json',out)
 for arm in ARMS[1:]:
  run=Path(cfg['train_root'])/arm
  if not run.exists():native.prepare({**conf,'maximum_workers':1,'provider_durable_capture':True},'deepseek',run,extra_source_files=tuple(native.load(queue/'SOURCE_SHA256.json')))
  frozen=native.load(run/'CONFIG.json')
  for i,ident in enumerate(ids):
   if(queue/'STOP').exists():write(queue/'STATE.json',{'status':'USER_PAUSED','at':utc_now(),'test_opened':False});return
   if any(r['task_id']==ident for r in out[arm]):continue
   directory=run/'tasks'/f'{i:02d}-{ident}'
   if(directory/'RESULT.json').exists():row=native.load(directory/'RESULT.json')
   elif(directory/'OWNER.json').exists():
    raise RuntimeError('Paid actor has dispatch intent; adopt after settlement; no automatic replay: '+str(directory))
   else:
    directory.mkdir(parents=True,exist_ok=False);cmd=native.trial_command(frozen,'deepseek',run,directory,ident)
    job=native.load(directory/'HARBOR_JOB.json');instruction=directory/'instruction.txt';instruction.write_text('Complete the original task and all required native verification.\n')
    policy={**POLICY,'version':7 if arm=='cot_jev'else 6,'enabled':arm!='cot_only'}
    job['agents'][0]['kwargs'].update(arm='chisel',cot_jev=policy,provider_durable_capture=True,
     usage_identity={'run_id':str(queue),'task_id':ident,'arm':arm},
     chisel={'path':str(instruction),'sha256':native.digest(instruction),'artifact_kind':'training-derived','provenance':{'identity':'prospective executable sub-DAG authoring; shared across three arms','validation_or_test_feedback_used':False}})
    write(directory/'HARBOR_JOB.json',job);write(directory/'OWNER.json',{'status':'DISPATCH_INTENT','at':utc_now(),'command':cmd})
    write(queue/'STATE.json',{'status':'SMOKE_RUNNING','arm':arm,'task_id':ident,'at':utc_now(),'test_opened':False,'cumulative_token_limit':None})
    with(directory/'actor.log').open('ab')as log:
     p=subprocess.Popen(cmd,cwd=run/'source',stdout=log,stderr=log,stdin=subprocess.DEVNULL,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(run/'source')})
     write(directory/'OWNER.json',{'status':'RUNNING','pid':p.pid,'command':cmd,'at':utc_now()});p.wait()
    row=native.harvest(frozen,'deepseek',run,directory,ident)
   result=episode(row);out[arm].append(result);write(queue/'EPISODES.json',out);write(queue/'AUDIT.json',{a:aggregate(out[a],4)for a in ARMS})
   if result['status']!='COMPLETE':raise RuntimeError('Native verifier or provider audit incomplete; preserve actor: '+ident)
 audit={a:aggregate(out[a],4)for a in ARMS};n,s=audit['native'],audit['sar'];complete=all(audit[a]['status']=='COMPLETE'for a in ARMS)
 cert=s['certified_substitutions'];ratio=s['substitutions_per_author_call'];saving=1-s['total_actor_tokens']/n['total_actor_tokens']if complete else None
 passed=complete and cert>=6 and ratio>=2 and s['slow_calls']<n['slow_calls'] and sum(a['malformed_native_actions']for a in audit.values())==0
 report={'stage':'mechanism_smoke_train4','status':'PASS'if passed else 'REJECT','arms':audit,'certified_substitutions':cert,'substitutions_per_author_call':ratio,'token_saving':saving,
 'substitution_rate':cert/n['slow_calls'],'slow_call_reduction':1-s['slow_calls']/n['slow_calls'],
 'malformed_native_actions':sum(a['malformed_native_actions']for a in audit.values()),'unknown_usage_count':sum(a['unknown_usage_count']for a in audit.values()),
 'gate_passed':passed,'test_opened':False,'validation_opened':False,'automatic_expansion':False,
 'next':'TRAIN8 only after manual audit of mechanism gate'if passed else 'TRAIN-only mechanism diagnosis; no 8/18/53 expansion'}
 write(queue/'SMOKE4.json',report);write(queue/'STATE.json',{'status':'SMOKE_PASSED'if passed else 'SMOKE_REJECTED','at':utc_now(),'test_opened':False});print(json.dumps(report),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);a=p.parse_args()
 try:run(a.queue.resolve())
 except Exception as e:write(a.queue/'BLOCKER.json',{'at':utc_now(),'error':str(e),'actor_replay':False,'test_opened':False});raise
