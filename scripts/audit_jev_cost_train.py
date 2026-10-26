"""Reconcile TRAIN physical actor requests; never open held-out artifacts.

Provider token totals are exact. Section bytes are exact; section tokens stay
unknown because this provider supplies no section counters/tokenizer revision.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

SECTIONS = ('system/base harness','tool schema / tool descriptions','skill library',
 'task specification','current observation','trajectory/history','candidate list',
 'previous tool outputs','recovery/error context','duplicated/static content','other')
SKILL_TEXT = ''
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encoded(x): return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def quantiles(xs):
    xs=sorted(xs)
    return {k:xs[min(len(xs)-1,max(0,math.ceil(q*len(xs))-1))] if xs else None
            for k,q in [('P50',.5),('P75',.75),('P90',.9),('P95',.95),('P99',.99),('max',1)]}
def usage(u):
    if not u: return None
    i=u.get('inputTokens');o=u.get('outputTokens');c=u.get('cacheReadTokens',0);w=u.get('cacheWriteTokens',0)
    if not all(type(v)is int and v>=0 for v in [i,o,c,w]): return None
    if u.get('totalTokens',i+o+c+w)!=i+o+c+w: raise ValueError('Contradictory provider counters')
    return dict(input_tokens=i+c+w,output_tokens=o,cached_input_tokens=c,total_tokens=i+c+w+o)
def context_sections(request):
    sections={k:0 for k in SECTIONS};seen=set();duplicate=0
    def add(k,x):
        nonlocal duplicate
        text=x if isinstance(x,str) else encoded(x)
        if k=='system/base harness' and SKILL_TEXT and SKILL_TEXT in text:
            sections['skill library']+=len(SKILL_TEXT.encode())*text.count(SKILL_TEXT)
            text=text.replace(SKILL_TEXT,'')
        n=len(text.encode());sections[k]+=n
        if text in seen:duplicate+=n
        seen.add(text)
    system=request.get('system') or request.get('systemPrompt')
    if system:add('system/base harness',system)
    if request.get('tools'):add('tool schema / tool descriptions',request['tools'])
    messages=request.get('messages',[])
    # Fast payload has explicit fields, rather than a native transcript.
    if len(messages)==2 and isinstance(messages[-1].get('content'),str):
        try:state=json.loads(messages[-1]['content'])
        except ValueError:state={}
        if 'candidates'in state:
            add('system/base harness',messages[0].get('content',''))
            mapping={'candidates':'candidate list','public_results':'current observation',
             'public_user_requests':'task specification','native_tool_contract':'tool schema / tool descriptions',
             'slow_root_action':'trajectory/history','native_receipts':'previous tool outputs',
             'observation':'current observation','state_id':'other','constraints':'task specification'}
            for k,v in state.items():add(mapping.get(k,'other'),v)
            return dict(utf8_bytes=sections,section_tokens={k:None for k in SECTIONS},
                        duplicate_bytes_diagnostic=duplicate,token_measurement='UNAVAILABLE')
    last_tool=max((i for i,m in enumerate(messages) if m.get('role')in ('tool','user') and
                  any(p.get('type')=='tool_result' for p in m.get('content',[])if isinstance(p,dict))),default=-1)
    last_sdk_tool=max((i for i,m in enumerate(messages)if m.get('role')=='tool'),default=-1)
    for index,m in enumerate(messages):
        role=m.get('role');parts=m.get('content',[])
        if isinstance(parts,str):parts=[{'type':'text','text':parts}]
        for part in parts:
            if not isinstance(part,dict):add('other',part);continue
            t=part.get('text','');kind=part.get('type')
            if role=='system':key='system/base harness'
            elif kind=='tool_result' or role=='tool':
                key='current observation'if index in (last_tool,last_sdk_tool)else 'previous tool outputs'
                if m.get('isError')or part.get('is_error'):key='recovery/error context'
            elif role=='user' and ('Current runtime context'in t or t.startswith('[ENVIRONMENT')):key='other'
            elif role=='user':key='task specification'
            elif '<CHISEL_PLAN>'in t:key='candidate list'
            else:key='trajectory/history'
            add(key,part)
    return dict(utf8_bytes=sections,section_tokens={k:None for k in SECTIONS},
                duplicate_bytes_diagnostic=duplicate,token_measurement='UNAVAILABLE')

def audit(root,out):
    global SKILL_TEXT
    candidate=load(root/'TRAIN_FROZEN_CANDIDATE.json')['descriptor']
    assert sha(candidate['path'])==candidate['sha256']
    SKILL_TEXT=Path(candidate['path']).read_text()
    report=load(root/'TRAIN_COMPARISON.json');rows=load(root/'TRAIN_OUTCOMES.json')
    assert report['complete'] and len(rows)==96 and all(r['phase']=='train'for r in rows)
    assert len({(r['task_id'],r['seed'],r['arm'])for r in rows})==96
    ledger=[];executions=[];family=defaultdict(Counter)
    for row in sorted(rows,key=lambda r:(r['arm'],r['task_id'],r['seed'])):
        summary_path=Path(row['summary']);assert sha(summary_path)==row['summary_sha256']
        s=load(summary_path);assert s['phase']=='train' and s['task_id']==row['task_id']
        if not s.get('actor_result'):
            recovery=s['native_model_failure_scoring'];p=Path(recovery['original_summary'])
            assert sha(p)==recovery['original_summary_sha256'];s=load(p);assert s['phase']=='train'
        sdk=Path(s['actor_result']).parent
        events=[json.loads(l)for l in (sdk/'llm-streams.jsonl').read_text().splitlines()]
        starts=[e for e in events if e['kind']=='llm/start'];by_attempt={e['attempt_id']:e for e in starts}
        sdk_usage=load(sdk/'usage.json');attempts={a['attempt_id']:a for a in sdk_usage['attempts']}
        requests=[e for e in events if e['kind']=='jev/request']
        decisions={e['decision_id']:e for e in events if e['kind']=='jev/decision'}
        fallbacks={e['decision_id']for e in events if e['kind']=='jev/slow-fallback'}
        fast_by_attempt={e['native_attempt_id']:e for e in requests}
        physical=(s['provider_usage']['actor'].get('durable_wire_evidence')or{}).get('receipts')
        execution_id=f"{row['arm']}:{row['seed']}:{row['task_id']}"
        def entry(index,aid,body,u,component,source,latency,success=True,fast=None):
            f=fast or fast_by_attempt.get(aid);d=decisions.get(f['decision_id'],{})if f else {}
            candidates=(json.loads(f['request']['messages'][-1]['content']).get('candidates',[])if f else [])
            return dict(execution_id=execution_id,task_id=row['task_id'],arm=row['arm'],phase='train',
             seed=row['seed'],turn_id=next((i for i,e in enumerate(starts)if e['attempt_id']==aid),None),
             request_index=index,physical_request_id=source['id'],component=component,
             **(u or dict(input_tokens=None,output_tokens=None,cached_input_tokens=None,total_tokens=None)),
             request_success=success,timeout=not success,retry=None,latency_seconds=latency,
             number_of_candidates=len(candidates)if f else None,fast_result=d.get('label'),
             abstained=(d.get('label')=='?')if f else None,
             fallback_triggered=(f['decision_id']in fallbacks)if f else False,
             candidate_valid=not d.get('invalid_decision',False)if f else None,
             context=context_sections(body),source=source,
             attempt_id=aid,fast_decision_id=f.get('decision_id')if f else None)
        current=[]
        if physical:
            for index,ref in enumerate(physical):
                p=Path(ref['path']);assert sha(p)==ref['sha256'];r=load(p);q=load(p.with_name('request.body'))
                assert sha(p.with_name('request.body'))==r['request_sha256']
                # Associate exact wire dispatch with its unique SDK time interval.
                at=datetime.fromisoformat(r['started_at_utc']);matches=[]
                for a in sdk_usage['attempts']:
                    start=datetime.fromisoformat(a['start']['at']);end=datetime.fromisoformat(a['end']['at'])
                    if start<=at<=end:matches.append(a['attempt_id'])
                assert len(matches)==1,(p,matches)
                aid=matches[0];f=fast_by_attempt.get(aid)
                component='jev_fast_selector'if r['route']=='chat/completions'else (
                    'native_slow'if row['arm']=='native'else
                    'slow_fallback'if f and f['decision_id']in fallbacks else 'candidate_generation')
                elapsed=(datetime.fromisoformat(r['finished_at_utc'])-at).total_seconds()
                e=entry(index,aid,q,usage(r['usage']),component,dict(id=r['call_id'],
                 receipt_path=str(p),receipt_sha256=sha(p),request_sha256=r['request_sha256'],
                 authority='physical_wire'),elapsed,r['upstream_status']==200,
                 fast=f if component=='jev_fast_selector'else None);current.append(e)
        else:
            # SDK counters are exact, but cannot prove invisible SDK transport retries.
            for index,a in enumerate(sdk_usage['attempts']):
                assert a['attempt_id']not in fast_by_attempt
                current.append(entry(index,a['attempt_id'],a['start']['request'],usage(a.get('usage')),
                 'native_slow',dict(id=a['attempt_id'],receipt_path=str(sdk/'usage.json'),
                 receipt_sha256=sha(sdk/'usage.json'),authority='sdk_attempt_not_physical_wire'),
                 a['end']['elapsed_seconds'],a['end']['completed']))
        totals={k:sum(e[k]for e in current)if all(e[k]is not None for e in current)else None
                for k in ['input_tokens','output_tokens','total_tokens']}
        assert totals['total_tokens']==row['actor_tokens'],(execution_id,totals,row['actor_tokens'])
        for f in requests:
            state=json.loads(f['request']['messages'][-1]['content']);d=decisions.get(f['decision_id'],{})
            for c in state.get('candidates',[]):
                family[c['tool']]['eligible']+=1
                family[c['tool']]['selected']+=d.get('label')==c['label']
                family[c['tool']]['abstain']+=d.get('label')=='?'
        for arm_event in events:
            if arm_event['kind']=='jev/compiled-native-action':
                family[arm_event['candidate']['tool']]['compiled']+=1
        ledger+=current
        for e in current:
            blocks=[v['chunk']['block']for v in events if v['kind']=='llm/chunk'
                    and v.get('attempt_id')==e['attempt_id'] and v['chunk'].get('type')=='block-end']
            visible=''.join(b.get('text','')for b in blocks if b.get('type')=='text')
            e['component_basis']='Physical route and correlated fallback; candidate_generation denotes the bank-enabled slow planning branch'
            e['semantic_component']=(e['component']if e['component']in ['native_slow','jev_fast_selector','slow_fallback'] else
                'candidate_generation'if '<CHISEL_PLAN>'in visible else
                'final_answer'if blocks and not any(b.get('type')=='tool-call'for b in blocks) else 'other')
            e['slow_authored_bank_in_response']='<CHISEL_PLAN>'in visible if e['component']!='jev_fast_selector'else None
        executions.append(dict(execution_id=execution_id,task_id=row['task_id'],arm=row['arm'],
         seed=row['seed'],native_score=row['native_score'],**totals,
         slow_requests=sum(e['component']!='jev_fast_selector'for e in current),
         fast_requests=len(requests),fallbacks=len(fallbacks),
         compiled_actions=sum(e['kind']=='jev/compiled-native-action'for e in events),
         missing_candidates=sum(e['kind']=='jev/full-cot'for e in events),
         invalid_banks=sum(e['kind']=='jev/invalid-bank'for e in events),
         inherited_reasoning_events=sum(e['kind']=='jev/inherited-native-reasoning'for e in events)))
    def aggregate(arm):
        es=[e for e in executions if e['arm']==arm];ls=[e for e in ledger if e['arm']==arm]
        by=defaultdict(lambda:dict(requests=0,input_tokens=0,output_tokens=0,total_tokens=0))
        section=Counter();growth=defaultdict(list);semantic=defaultdict(lambda:dict(requests=0,input_tokens=0,output_tokens=0,total_tokens=0))
        for e in ls:
            b=by[e['component']];b['requests']+=1
            for k in ['input_tokens','output_tokens','total_tokens']:b[k]+=e[k]
            b=semantic[e['semantic_component']];b['requests']+=1
            for k in ['input_tokens','output_tokens','total_tokens']:b[k]+=e[k]
            section.update(e['context']['utf8_bytes']);growth[e['turn_id']].append(e['input_tokens'])
        fast=[e for e in ls if e['component']=='jev_fast_selector'];fb=[e for e in ls if e['component']=='slow_fallback']
        af=[e for e in fast if e['fallback_triggered']]
        return dict(executions=len(es),successes=sum(e['native_score']for e in es),
         by_component=dict(by),by_semantic_component=dict(semantic),by_context_section_utf8_bytes=dict(section),
         by_context_section_tokens={k:None for k in SECTIONS},
         total_tokens=sum(e['total_tokens']for e in es),input_tokens=sum(e['input_tokens']for e in es),
         output_tokens=sum(e['output_tokens']for e in es),per_execution_tokens=quantiles([e['total_tokens']for e in es]),
         fast_input=quantiles([e['input_tokens']for e in fast]),slow_input=quantiles([e['input_tokens']for e in ls if e not in fast]),
         requests=len(ls),physical_request_count_authoritative=all(e['source']['authority']=='physical_wire'for e in ls),
         fast_invocations=len(fast),abstentions=len(af),abstention_rate=len(af)/len(fast)if fast else None,
         invalid_no_candidate_count=sum(e['missing_candidates']+e['invalid_banks']for e in es),
         fallback_slow_tokens=sum(e['total_tokens']for e in fb),fast_selector_tokens=sum(e['total_tokens']for e in fast),
         duplicated_fast_fallback_tokens=sum(e['total_tokens']for e in fb+af),
         slow_requests_per_execution=sum(e['slow_requests']for e in es)/len(es),
         fast_requests_per_execution=len(fast)/len(es),
         fast_coverage=sum(e['compiled_actions']>0 for e in es)/len(es),
         cost_conditional_on_success={str(s):quantiles([e['total_tokens']for e in es if e['native_score']==s])for s in [0,1]},
         context_growth=[dict(turn_id=k,requests=len(v),mean_input_tokens=sum(v)/len(v))for k,v in sorted(growth.items())],
         retry_failed_cost=None,retry_status='Wire retries can be counted prospectively with dispatch tags; unknown is not zero')
    out.mkdir(parents=True,exist_ok=True)
    # Shared authorship is a single physical request, retained once globally.
    # Its comparison charge appears once in each descendant's cost.
    author_receipts=list((root/'author/wire/actor').glob('*/RECEIPT.json'))
    shared=[]
    for p in author_receipts:
        r=load(p);body=load(p.with_name('request.body'));u=usage(r.get('usage'))
        assert u is not None and sha(p.with_name('request.body'))==r['request_sha256']
        shared.append(dict(execution_id='shared_train_skill_author',task_id=None,arm='shared_author',phase='train',
         seed=None,turn_id=0,request_index=len(shared),physical_request_id=r['call_id'],component='skill_generation',
         **u,request_success=r['upstream_status']==200,timeout=False,retry=None,
         latency_seconds=(datetime.fromisoformat(r['finished_at_utc'])-datetime.fromisoformat(r['started_at_utc'])).total_seconds(),
         number_of_candidates=None,fast_result=None,abstained=None,fallback_triggered=False,candidate_valid=None,
         context=context_sections(body),source=dict(receipt_path=str(p),receipt_sha256=sha(p),
          request_sha256=r['request_sha256'],authority='physical_wire'),charged_to_arms=['cot_only','cot_jev']))
    author_total=sum(e['total_tokens']for e in shared)
    assert author_total==report['arms']['cot_jev']['actor_tokens']-sum(e['total_tokens']for e in executions if e['arm']=='cot_jev')
    ledger_path=out/'jev_request_ledger_train.jsonl'
    ledger_path.write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n'for e in ledger+shared))
    result=dict(schema='jev-cost-attribution-train/v1',phase='train',source_run=str(root),
     comparison_sha256=sha(root/'TRAIN_COMPARISON.json'),outcomes_sha256=sha(root/'TRAIN_OUTCOMES.json'),
     request_ledger=dict(path=str(ledger_path),sha256=sha(ledger_path),requests=len(ledger)+len(shared)),
     shared_author_request_count=len(shared),shared_author_total_tokens=author_total,
     arms={a:aggregate(a)for a in ['native','cot_only','cot_jev']},executions=executions,
     families={k:dict(v)for k,v in family.items()},shared_author_tokens={a:report['arms'][a]['actor_tokens']-
      sum(e['total_tokens']for e in executions if e['arm']==a)for a in ['native','cot_only','cot_jev']},
     validation_or_test_trajectory_read=False,section_tokens_exact=False,
     section_measurement_blocker='No pinned provider-equivalent tokenizer/serialization or section counters in saved responses',
     test_300M_attribution='Not numerically attributable from TRAIN; TEST traces excluded. TRAIN drivers reported separately.')
    (out/'jev_cost_attribution_train.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# TRAIN request cost attribution','',f'Source: `{root}`. All 96 matched TRAIN executions; no held-out trajectories.',
     '', '| Arm | Input | Output | Actor + author | Slow / execution | Fast | Fallback | SR |',
     '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a,v in result['arms'].items():
        lines.append(f"| {a} | {v['input_tokens']} | {v['output_tokens']} | {v['total_tokens']+result['shared_author_tokens'][a]} | {v['slow_requests_per_execution']:.3f} | {v['fast_invocations']} | {v['abstentions']} | {100*v['successes']/v['executions']:.3f}% |")
    lines+=['','Provider totals are exact and recomputable from the JSONL ledger. Cached inputs are included.',
     'Context sections report exact UTF-8 bytes, with token fields null. Byte proportions are not token measurements.',
     'Native historical SDK attempts lack durable wire capture; invisible transport retries remain unverified.',
     'Shared slow authorship is reported separately and charged to each bank arm once. Physical authorship itself occurs once.']
    for a,v in result['arms'].items():
        lines+=['',f'## {a}', '', 'Component counters: `'+json.dumps(v['by_component'])+'`',
          '', 'Section byte counters: `'+json.dumps(v['by_context_section_utf8_bytes'])+'`',
          '', 'Execution quantiles: `'+json.dumps(v['per_execution_tokens'])+'`']
    (out/'jev_cost_attribution_train.md').write_text('\n'.join(lines)+'\n')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=audit(a.root,a.output);print(json.dumps({k:{f:v[f]for f in ['total_tokens','requests','fast_invocations','abstentions','slow_requests_per_execution']}for k,v in r['arms'].items()}))
