"""One-command complete native-verdict and physical actor-cost audit."""
from pathlib import Path
import argparse,json,hashlib,sys
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.serve_deepseek_durable_capture import audited_physical_accounting
from integrations.deepseek_harness.runner import summarize_streams

def load(p):return json.loads(Path(p).read_text())
def episode(row, require_append=True):
 sdk=Path(row['cost']['sdk_worker_dir']);events=[json.loads(l)for l in(sdk/'llm-streams.jsonl').read_text().rstrip('\n').split('\n')]
 wire=sdk.parent/'provider-wire';sdk_account=summarize_streams(events);historical=not require_append and not wire.exists()
 account=sdk_account if historical else audited_physical_accounting(wire,sdk_account)
 receipts=[load(p)for p in sorted((wire/'actor').glob('*/RECEIPT.json'))]
 ledger=wire/'ACTOR_USAGE.jsonl';entries=[json.loads(l)for l in ledger.read_text().rstrip('\n').split('\n')]if ledger.exists()else []
 physical={r['call_id']for r in receipts};complete_entries={r['physical_request_id']:r for r in entries if r['status']=='COMPLETE' and r['attempt']=='actor'}
 bindings_path=wire/'ACTOR_IDENTITY_BINDINGS.jsonl';bindings={e['physical_request_id']:e for e in (json.loads(l)for l in bindings_path.read_text().rstrip('\n').split('\n'))}if bindings_path.exists()else {}
 if historical:receipts=[{'call_id':r['attempt_id'],'usage':r.get('usage'),'status':'MEASURED'if r.get('usage')else 'UNKNOWN','route':'anthropic/v1/messages'}for r in sdk_account['attempts']]
 unknown=sum(not r.get('usage')or r.get('status')!='MEASURED'for r in receipts)
 errors=list(account.get('unknown_reasons',[]))
 if require_append and (physical!=set(complete_entries) or len(complete_entries)!=len(entries)):
  errors.append('append-only per-request ledger does not cover physical requests exactly once')
 if not receipts:errors.append('no physical receipts')
 if historical and not sdk_account.get('complete'):errors.append('historical native SDK per-attempt usage incomplete')
 for r in receipts:
  if require_append and r['call_id']in complete_entries:
   e=complete_entries[r['call_id']];u=r['usage']
   if e['input_tokens']+e['output_tokens']!=u['totalTokens']:errors.append('append-only usage differs from raw provider counters')
   identity=e if e.get('task_id')and e.get('arm')else bindings.get(r['call_id'],{})
   if identity.get('task_id')!=row['task_id']or not identity.get('arm')or identity.get('request_sha256')!=r['request_sha256']:errors.append('request identity lacks frozen task/arm binding')
 cert=[e for e in events if e['kind']=='sar/substitution-certificate'];ends={e['call_id']:e for e in events if e['kind']=='bridge/tool/end'}
 starts={e['attempt_id']:e for e in events if e['kind']=='llm/start'}
 for c in cert:
  begin=starts.get(c.get('bypassed_slow_callback_id'));end=ends.get(c['call_id'])
  if begin and end:
   lo=datetime.fromisoformat(begin['at'].replace('Z','+00:00'));hi=datetime.fromisoformat(end['at'].replace('Z','+00:00'))
   if any(r['route'].startswith('anthropic/')and lo<=datetime.fromisoformat(r['started_at_utc'])<=hi for r in receipts):errors.append('physical slow actor request occurred inside accepted boundary')
  else:errors.append('missing native callback boundary binding')
  if c.get('slow_requests_at_boundary')!=0 or c.get('slow_sequence_before')!=c.get('slow_sequence_after') or c['call_id']not in ends:errors.append('uncertified substitution')
  elif hashlib.sha256(json.dumps(ends[c['call_id']]['result'],separators=(',',':'),ensure_ascii=False).encode()).hexdigest()!=c['native_receipt_hash']:errors.append('native receipt hash differs')
 if len({c['call_id']for c in cert})!=len(cert):errors.append('duplicate certificates')
 u=account['usage'];verdict=Path(row['native_result']);verdict_ok=verdict.is_file()and row.get('native_score')in(0,1)and hashlib.sha256(verdict.read_bytes()).hexdigest()==row['native_result_sha256']
 banks=[e for e in events if e['kind']in('sar/bank-authored','sar/author-invalid-bank')];authors=len({e.get('author_callback_id',e.get('native_attempt_id',e.get('root_id')))for e in banks})
 slow=sum(r['route'].startswith('anthropic/')for r in receipts);fast=sum(r['route']=='chat/completions'for r in receipts)
 return {'task_id':row['task_id'],'success':row.get('native_score')if verdict_ok else None,'native_verdict_present':verdict_ok,
 'actor_requests':len(receipts),'requests_with_usage':len(receipts)-unknown,'unknown_usage_count':unknown,
 'input_tokens':u['totalTokens']-u['outputTokens']if u else None,'output_tokens':u['outputTokens']if u else None,'total_actor_tokens':u['totalTokens']if u else None,
 'slow_calls':slow,'fast_calls':fast,'slow_bank_author_calls':authors,'certified_substitutions':len(cert),
 'substitutions_per_author_call':len(cert)/authors if authors else 0,
 'abstentions':sum(e['kind']=='jev/decision'and e.get('label')=='?'for e in events),
 'fallbacks':sum(e['kind']=='sar/fallback-slow-call'for e in events),'certificates':cert,
 'malformed_native_actions':sum(e['kind']=='bridge/tool/error'and ('schema' in str(e.get('error')).lower()or 'argument' in str(e.get('error')).lower())for e in events),
 'status':'COMPLETE'if not errors and not unknown and verdict_ok else 'AUDIT_INCOMPLETE','audit_errors':errors,
 'actor_cost_scope':'all physical requests incl failures, fast, fallback, cached input; author cost included exactly once',
 'ledger_identity':'new append-only'if require_append else 'adopted historical complete native SDK per-attempt usage; no new wire capture',
 'native_verdict':str(verdict),'wire_directory':str(wire)}

def aggregate(rows,expected):
 complete=len(rows)==expected and all(r['status']=='COMPLETE'for r in rows)
 keys=['actor_requests','requests_with_usage','unknown_usage_count','slow_calls','fast_calls','slow_bank_author_calls','certified_substitutions','abstentions','fallbacks','malformed_native_actions']
 d={k:sum(r[k]for r in rows)for k in keys}
 d.update(tasks_expected=expected,tasks_executed=len(rows),native_verdicts_present=sum(r['native_verdict_present']for r in rows),status='COMPLETE'if complete else 'AUDIT_INCOMPLETE',figure3_readable=complete)
 for k in ['input_tokens','output_tokens','total_actor_tokens']:d[k]=sum(r[k]for r in rows)if complete else None
 d['successes']=sum(r['success']for r in rows)if complete else None
 d['substitutions_per_author_call']=d['certified_substitutions']/d['slow_bank_author_calls']if d['slow_bank_author_calls']else 0
 return d

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);a=p.parse_args();x=load(a.queue/'EPISODES.json');print(json.dumps({arm:aggregate(rows,4)for arm,rows in x.items()},indent=2))
