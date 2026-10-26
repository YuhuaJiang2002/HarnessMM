#!/usr/bin/env python3
"""Remote GAIA2 comparison using pinned official optimizers and native scoring.

Each method owns a service and immutable run. Scored train/development scenarios
are disjoint from all 160 final scenarios. No email-only or other ablation runs.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/SkillOpt'),str(ROOT/'external/AutoSaddler/src'),str(ROOT/'integrations/autosaddler_phiagent/src')]
from scripts import run_aime_four_method as lifecycle
from scripts import run_gaia2_jev as base
from phiagent.harness.gaia2_common_prompt import shared_actor_request
METHODS=('native','skillopt','autosaddler','chisel')
OBLIGATION_POLICIES=('task_literal_feedback_v12','task_literal_feedback_control_v12','task_literal_v11','task_literal_control_v11','qualified_binding_v10','qualified_advice_v10','literal_binding_isolated_v9','literal_advice_isolated_v9','literal_binding_v8','literal_advice_v8','dependency_recovery_v7','dependency_plan_only_v7','read_recovery_only_v7',
                    'obligation_review_v3','obligation_memory_v3','obligation_residual_v4','obligation_structured_v5',
                     'effect_contract_v6','effect_contract_only_v6','receipt_guard_only_v6')
SEED_SKILL='''# Complete the requested task
Keep the original task and all native tool constraints. Use exact identifiers from public observations, preserve user constraints and the required order of state changes, and inspect results before deciding the next action. After a repeated action with unchanged feedback or a tool error, reconsider the failed step instead of repeating it unchanged. Ask the user when required information is missing. Do not treat a successful tool call as proof of task completion.'''


def require_remote(config):
    if sys.platform!='linux' or socket.gethostname()!=config['hostname'] or not Path(config['run_directory']).is_relative_to('/opt/phiagent/work/gaia2-four-method'):
        raise RuntimeError('Only the authorized remote host may execute this experiment')
    if config['method'] not in METHODS or config['budget']['maximum_tokens'] is not None or config['budget']['token_usage_mode']!='unlimited':
        raise RuntimeError('Four named methods and unlimited cumulative-token accounting are required')


def prepare(config,root):
    require_remote(config)
    import pyarrow.parquet as pq
    if (root/'PREPARED.json').exists():raise FileExistsError('Immutable preparation exists')
    sources=config['data_sources'];data={};hashes={}
    for name,entry in sources.items():
        p=Path(entry['path']);h=base.digest(p)
        if h!=entry['sha256']:raise RuntimeError('Pinned data file changed')
        hashes[name]=h;rows=sorted(pq.read_table(p).to_pylist(),key=lambda x:x['scenario_id'])
        items=[]
        for row in rows:
            ident=row['scenario_id']
            if Path(ident).name!=ident:raise RuntimeError('Invalid identifier')
            dest=root/'data'/name/(ident+'.json');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(row['data'])
            items.append({'id':ident,'question':'Official GAIA2 scenario '+ident,'path':str(dest),'sha256':base.digest(dest)})
        data[name]=items
    if len(data['mini'])!=160:raise RuntimeError('Complete final mini population required')
    def public_task_key(item):
        scenario=base.read(Path(item['path']))
        events=[e for e in scenario.get('events',[]) if e.get('event_type')=='USER']
        return hashlib.sha256(json.dumps([e.get('action') for e in events],sort_keys=True).encode()).hexdigest()
    final_ids={x['id'] for x in data['mini']};final_tasks={public_task_key(x) for x in data['mini']}
    eligible=[]
    for item in data['optimization']:
        scenario=base.read(Path(item['path']))
        if not any(e.get('class_name')=='OracleEvent' for e in scenario.get('events',[])):
            raise RuntimeError('Unscored demo data cannot optimize a baseline')
        if item['id'] not in final_ids and public_task_key(item) not in final_tasks:eligible.append(item)
    eligible.sort(key=lambda x:hashlib.sha256((str(config['seed'])+x['id']).encode()).hexdigest())
    if len(eligible)<30:raise RuntimeError('Fewer than 30 distinct scored optimization tasks outside final population')
    payload={'train':eligible[:20],'development':eligible[20:30],'test':data['mini']}
    base.write(root/'OPTIMIZATION_SPLIT.json',{'eligible':len(eligible),'source_size':len(data['optimization']),
        'selection':'seeded hash of scenario ID; first20 train, next10 development; no score-based selection',
        'excluded_final_ids_and_public_task_hashes':True,'train_ids':[x['id'] for x in payload['train']],
        'development_ids':[x['id'] for x in payload['development']]})
    base.write(root/'DATA.json',payload)
    native=Path(__import__('importlib.util').util.find_spec('are.simulation').submodule_search_locations[0])
    base.write(root/'PREPARED.json',{'at_utc':base.now(),'dataset_hash':base.digest(root/'DATA.json'),'sources':hashes,
        'sizes':{k:len(v) for k,v in payload.items()},'native_source_sha256':{str(p.relative_to(native)):base.digest(p) for p in native.rglob('*.py')},
        'optimizer_scope':'Official engines with new GAIA2 text-skill adapters; official execution train20/development10, excluding final mini tasks',
        'model_calls':0,'benchmark_calls':0,'test_exposed_to_optimizers':False})


class Gateway(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,config,root,ledger):
        self.config,self.root,self.ledger=config,root,ledger
        super().__init__(('127.0.0.1',config['resource']['capture_port']),Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        config=self.server.config;model=config['model'];parts=self.path.strip('/').split('/')
        if parts[-3:]!=['v1','chat','completions'] or len(parts) not in (4,5):self.send_error(404);return
        role='/'.join(parts[:-3]);usage=None;rejected=False;sequence=None
        try:
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if body.get('model')!=model['served_name'] or body.get('stream',False):raise ValueError('Model identity mismatch')
            body,common_parent=shared_actor_request(body,role,config.get('shared_gaia2_prompt'))
            requested=body.pop('max_completion_tokens',body.get('max_tokens',32768))
            maximum=4096 if len(parts)==5 else min(int(requested),32768)
            body.update(max_tokens=maximum,temperature=0,seed=config['seed'],chat_template_kwargs={'enable_thinking':False})
            sequence=self.server.ledger.reserve(role)
            call=self.server.root/'provider'/f'call-{sequence:06d}';call.mkdir(parents=True)
            base.write(call/'request.json',{'role':role,'json':body})
            if common_parent is not None:base.write(call/'COMMON_PARENT_REQUEST.json',common_parent)
            # Native input-size accounting includes function schemas for optimizers.
            token_body={k:body[k] for k in ('model','messages','chat_template_kwargs','tools','tool_choice') if k in body}
            tokenized=base.get_json(f"http://127.0.0.1:{config['resource']['model_port']}/tokenize",token_body,timeout=60)
            base.write(call/'tokenize.json',tokenized);count=tokenized['count']
            if type(count)!=int or count<1:raise RuntimeError('Invalid native token count')
            if config.get('optimizer_context_repair') and role in ('skillopt_optimizer','autosaddler_optimizer'):
                from phiagent.harness.aime_transport_recovery import fit_optimizer_context
                def count_payload(payload):
                    token_request={k:payload[k] for k in ('model','messages','tools','chat_template_kwargs') if k in payload}
                    token_request['add_generation_prompt']=True
                    return base.get_json(f"http://127.0.0.1:{config['resource']['model_port']}/tokenize",token_request)['count']
                body,changes=fit_optimizer_context(body,count_payload,model['max_model_len'])
                if changes:
                    base.write(call/'CONTEXT_REPAIR.json',changes)
                    base.write(call/'request.json',{'role':role,'json':body})
                count=count_payload(body);maximum=body['max_tokens']
            if count+maximum>model['max_model_len']:
                rejected=True;base.write(call/'REJECTED_BEFORE_FORWARD.json',{'input_tokens':count,'max_tokens':maximum,'context':model['max_model_len'],'model_calls':0})
                payload=json.dumps({'error':'Native context capacity exceeded before inference'}).encode();status=413
            else:
                raw=base.get_json(f"http://127.0.0.1:{config['resource']['model_port']}/v1/chat/completions",body,timeout=1800)
                base.write(call/'response.json',raw);usage=base.native_usage(raw,model['served_name'],model['max_model_len'])
                payload=json.dumps(raw).encode();status=200
        except Exception as e:
            if sequence is not None:base.write(call/'FAILURE.json',{'error':str(e),'traceback':traceback.format_exc()})
            payload=json.dumps({'error':str(e)}).encode();status=502
        finally:
            if sequence is not None:self.server.ledger.settle(sequence,usage,preinference_rejection=rejected)
        self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers()
        try:self.wfile.write(payload)
        except (BrokenPipeError,ConnectionResetError):pass


def install_adapter(config,directory,role,mode,skill):
    from are.simulation.agents.default_agent.base_agent import BaseAgent
    from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMEngine
    from phiagent.harness.gaia2_jev import Gaia2Engine
    from phiagent.harness.gaia2_obligation_control import ObligationEngine
    old_step=BaseAgent.step;lock=threading.Lock()
    def record(value):
        with lock,(directory/'decisions.jsonl').open('a') as f:f.write(json.dumps(value,default=str)+'\n')
    def completion(engine,messages,stop_sequences=(),**kwargs):
        converted=[engine._convert_message_to_litellm_format(m) for m in messages];endpoint=engine.model_config.endpoint
        if not endpoint or not endpoint.startswith(f"http://127.0.0.1:{config['resource']['capture_port']}/{role}/"):raise RuntimeError('Unmetered model route')
        if mode=='jev' and '/actor/' in endpoint and config.get('chisel_policy') not in (*OBLIGATION_POLICIES,'selective_evidence_v1','selective_recovery_v1'):
            from phiagent.harness.gaia2_task_context import project
            converted,projection=project(converted,memory=True,
                preserve_reasoning=config.get('chisel_policy')=='stall_preserve_v2')
            record({'source':'public_context_projection','projection':projection})
        start=time.monotonic()
        body={'model':config['model']['served_name'],'messages':converted,'stop':list(stop_sequences)}
        if config.get('chisel_policy')=='obligation_structured_v5' and '/actor/' in endpoint:
            from phiagent.harness.gaia2_obligation_control import STRUCTURED_RESIDUAL_PREFIX,residual_response_format
            if converted and str(converted[0].get('content','')).startswith(STRUCTURED_RESIDUAL_PREFIX):
                body['response_format']=residual_response_format(json.loads(converted[-1]['content']))
        if config.get('chisel_policy') in ('effect_contract_v6','effect_contract_only_v6') and '/actor/' in endpoint:
            from phiagent.harness.gaia2_obligation_control import CONTRACT_PREFIX,contract_response_format
            if converted and str(converted[0].get('content','')).startswith(CONTRACT_PREFIX):
                body['response_format']=contract_response_format(json.loads(converted[-1]['content']))
        raw=base.get_json(endpoint.rstrip('/')+'/chat/completions',body,timeout=1830)
        usage=base.native_usage(raw,config['model']['served_name'],config['model']['max_model_len'])
        # Actor messages contain public observations only; never export judge prompts to optimizers.
        if '/actor/' in endpoint:record({'source':'public_actor_messages','messages':converted,'response':raw['choices'][0]['message']['content'],'usage':usage})
        return base.native_response_text(raw['choices'][0]['message']['content'],stop_sequences,judge='/judge/' in endpoint),{**usage,'completion_duration':time.monotonic()-start}
    def step(agent):
        if mode=='jev' and config.get('chisel_policy') in ('selective_evidence_v1','selective_recovery_v1'):
            from phiagent.harness.gaia2_selective_control import SelectiveEngine
            if not isinstance(agent.llm_engine,SelectiveEngine):
                agent.llm_engine=SelectiveEngine(agent,agent.llm_engine,skill=skill,record=record,
                    completion=config['chisel_policy']=='selective_evidence_v1')
            return old_step(agent)
        if mode=='jev' and config.get('chisel_policy') in OBLIGATION_POLICIES:
            if not isinstance(agent.llm_engine,ObligationEngine):
                variant=config['chisel_policy']
                literal_variants=('task_literal_v11','task_literal_control_v11','task_literal_feedback_v12','task_literal_feedback_control_v12')
                policy='qualified_binding_v10' if variant in literal_variants else variant
                agent.llm_engine=ObligationEngine(agent,agent.llm_engine,skill=skill,record=record,
                    review=policy in ('obligation_review_v3','obligation_residual_v4','obligation_structured_v5'),
                    blind_residual=policy in ('obligation_residual_v4','obligation_structured_v5'),
                    structured_residual=policy=='obligation_structured_v5',
                    contract=policy in ('effect_contract_v6','effect_contract_only_v6'),
                    receipt_guard=policy in ('effect_contract_v6','receipt_guard_only_v6'),
                    tool_plan=policy in ('qualified_binding_v10','qualified_advice_v10','dependency_recovery_v7','dependency_plan_only_v7','literal_binding_v8','literal_advice_v8','literal_binding_isolated_v9','literal_advice_isolated_v9'),
                    read_recovery=policy in ('qualified_binding_v10','qualified_advice_v10','dependency_recovery_v7','read_recovery_only_v7','literal_binding_v8','literal_advice_v8','literal_binding_isolated_v9','literal_advice_isolated_v9'),
                    literal_plan=policy in ('qualified_binding_v10','qualified_advice_v10','literal_binding_v8','literal_advice_v8','literal_binding_isolated_v9','literal_advice_isolated_v9'),
                    literal_guard=policy in ('qualified_binding_v10','literal_binding_v8','literal_binding_isolated_v9'),
                    independent_bindings=policy in ('qualified_binding_v10','qualified_advice_v10','literal_binding_isolated_v9','literal_advice_isolated_v9'),
                    qualified_bindings=policy in ('qualified_binding_v10','qualified_advice_v10'),
                    task_literal_guard=variant in ('task_literal_v11','task_literal_feedback_v12','task_literal_feedback_control_v12'),
                    task_literal_feedback=variant=='task_literal_feedback_v12',
                    variant=variant if variant in literal_variants else None)
            return old_step(agent)
        if not isinstance(agent.llm_engine,Gaia2Engine):
            engine_mode='stall' if mode=='jev' and config.get('chisel_policy')=='stall_preserve_v2' else mode
            agent.llm_engine=Gaia2Engine(agent,agent.llm_engine,mode=engine_mode,skill=skill,record=record,recovery_policy='native_enumeration_state_v2')
        return old_step(agent)
    LiteLLMEngine.chat_completion=completion;BaseAgent.step=step


def validate_population(rows,items,repetitions):
    # The official iterator assigns run_number only when num_runs > 1.
    numbers=(None,) if repetitions==1 else range(1,repetitions+1)
    expected={(x['id'],rep) for x in items for rep in numbers}
    actual={(x['base_scenario_id'],x['run_number']) for x in rows}
    if len(rows)!=len(expected) or actual!=expected:
        raise RuntimeError('Incomplete native population')


def worker(config,job):
    require_remote(config)
    from are.simulation.benchmark.scenario_executor import run_dataset
    directory=Path(job['directory']);data=directory/'scenarios';data.mkdir()
    for item in job['items']:
        p=Path(item['path'])
        if base.digest(p)!=item['sha256']:raise RuntimeError('Frozen scenario changed')
        (data/p.name).symlink_to(p)
    install_adapter(config,directory,job['role'],job['mode'],job['skill'])
    endpoint=f"http://127.0.0.1:{config['resource']['capture_port']}/{job['role']}"
    result=run_dataset(model=config['model']['served_name'],model_provider='local',dataset_path=str(data),agent='default',
        endpoint=endpoint+'/actor/v1',output_dir=str(directory/'traces'),
        max_concurrent_scenarios=config['resource'].get('native_workers_per_arm',2),
        executor_type='thread',scenario_timeout=7260,num_runs=job['repetitions'],simulated_generation_time_mode='measured',
        judge_model=config['model']['served_name'],judge_provider='local',judge_endpoint=endpoint+'/judge/v1')
    rows=result.to_polars().to_dicts()
    validate_population(rows,job['items'],job['repetitions'])
    base.write(directory/'NATIVE_RESULT.json',{'rows':rows,'run_config':result.run_config.model_dump(),'success_rate':result.success_rate()})


class Study:
    def __init__(self,config,root):
        self.config,self.root,self.data=config,root,base.read(root/'DATA.json')
        self.recovery_lock=threading.Lock();self.recovered={}
        if config.get('continuation'):
            self.recover_completed()
    def recover_completed(self,prior=None,expected_hash=None):
        initial=prior is None
        if initial:
            prior=Path(self.config['continuation']['prior_root'])
            expected_hash=self.config['continuation']['prior_provider_final_sha256']
            self.recovery_proofs=[];self.recovery_ledgers=[];self.recovery_seen=set()
        if str(prior) in self.recovery_seen:raise RuntimeError('Cyclic continuation chain')
        self.recovery_seen.add(str(prior))
        if base.digest(prior/'PROVIDER_FINAL.json')!=expected_hash:
            raise RuntimeError('Pinned predecessor ledger changed')
        ledger=base.read(prior/'PROVIDER_FINAL.json')
        if ledger['unknown_usage_calls'] or ledger['inflight_reserved_tokens']:
            raise RuntimeError('Cannot resume with unresolved predecessor costs')
        old=base.read(prior/'STARTED.json')['config']
        if old.get('continuation'):
            self.recover_completed(Path(old['continuation']['prior_root']),old['continuation']['prior_provider_final_sha256'])
        for key in ('method','seed','dataset_revision'):
            if old[key]!=self.config[key]:raise RuntimeError('Continuation identity changed: '+key)
        for key in ('repository','revision','served_name','temperature','seed','max_tokens','max_model_len','enable_thinking'):
            if old['model'][key]!=self.config['model'][key]:raise RuntimeError('Continuation model changed: '+key)
        for split in self.data:
            if [(x['id'],x['sha256']) for x in self.data[split]]!=[(x['id'],x['sha256']) for x in base.read(prior/'DATA.json')[split]]:
                raise RuntimeError('Continuation population changed')
        total=0
        for call in ledger['calls']:
            if call['status']!='SETTLED':raise RuntimeError('Predecessor call is not settled')
            directory=prior/'provider'/f"call-{call['sequence']:06d}"
            usage=base.native_usage(base.read(directory/'response.json'),old['model']['served_name'],old['model']['max_model_len'])
            if usage!=call['usage'] or base.read(directory/'request.json')['role']!=call['role']:
                raise RuntimeError('Predecessor native accounting mismatch')
            total+=usage['total_tokens']
        if total!=ledger['known_tokens']:raise RuntimeError('Predecessor cost sum mismatch')
        proofs=[]
        for job_path in sorted((prior/'rollouts').glob('*/JOB.json')):
            job=base.read(job_path);directory=job_path.parent;output=directory/'traces/output.jsonl'
            if not output.exists():continue
            if len(job['items'])!=1 or job['repetitions']!=1:raise RuntimeError('Only completed single training rollouts may be recovered')
            exports=[json.loads(x) for x in output.read_text().splitlines() if x.strip()]
            if len(exports)!=1:raise RuntimeError('Duplicate predecessor verdicts')
            export=exports[0];item=job['items'][0];meta=export['metadata']
            trace=Path(export['trace_id']);raw=base.read(trace)
            definition=raw['metadata']['definition'];decision=raw['metadata']['annotation']['validation_decision']
            if (export['task_id']!=item['id'] or meta['scenario_id']!=item['id'] or definition['scenario_id']!=item['id']
                or definition.get('run_number') is not None or meta['has_exception'] or decision not in ('Valid','Invalid')
                or export['score']!=int(decision=='Valid') or meta['status']!=('success' if decision=='Valid' else 'failed')):
                raise RuntimeError('Predecessor verdict mismatch')
            actor=[x for x in ledger['calls'] if x['role']==job['role']+'/actor']
            usage={k:sum(x['usage'][k] for x in actor) for k in base.USAGE_FIELDS}
            key=(item['id'],item['sha256'],job['skill'],job['mode'])
            if key in self.recovered:raise RuntimeError('Duplicate recovered task')
            self.recovered[key]=(directory,{'success_bool':decision=='Valid','has_exception':False},usage)
            proofs.append({'scenario_id':item['id'],'job':str(job_path),'job_sha256':base.digest(job_path),
                'output_sha256':base.digest(output),'trace_sha256':base.digest(trace),'actor_usage':usage})
        self.recovery_proofs.extend(proofs)
        self.recovery_ledgers.append({'root':str(prior),'sha256':base.digest(prior/'PROVIDER_FINAL.json'),'known_tokens':total})
        if initial:
            base.write(self.root/'CONTINUATION_AUDIT.json',{'prior_root':str(prior),'prior_ledger_sha256':base.digest(prior/'PROVIDER_FINAL.json'),
                'prior_ledgers':self.recovery_ledgers,'inherited_all_stage_tokens':sum(x['known_tokens'] for x in self.recovery_ledgers),
                'recovered_rollouts':self.recovery_proofs,'new_model_calls_for_recovery':0})
    def endpoint(self,role):return f"http://127.0.0.1:{self.config['resource']['capture_port']}/{role}/v1"
    def rollout(self,items,skill,directory,mode='cot',repetitions=1):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);role='r'+uuid.uuid4().hex
        job={'items':items,'skill':skill,'mode':mode,'role':role,'directory':str(directory),'repetitions':repetitions}
        base.write(directory/'JOB.json',job)
        command=[sys.executable,self.config.get('worker_script',str(Path(__file__).resolve())),'worker','--config',self.config['_config_path'],'--job',str(directory/'JOB.json')]
        with (directory/'worker.log').open('x') as f:
            p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
            base.write(directory/'WORKER.json',{'pid':p.pid,'command':command})
            try:
                if p.wait(timeout=7260*len(items)*repetitions+300):raise RuntimeError('Native rollout failed: '+str(directory))
            finally:base.stop_owned(p)
        native=base.read(directory/'NATIVE_RESULT.json')['rows']
        calls=[json.loads(x) for x in (self.root/'PROVIDER_CALLS.jsonl').read_text().splitlines() if x.strip()]
        actor=[x for x in calls if x['role']==role+'/actor'];judge=[x for x in calls if x['role']==role+'/judge']
        usage={k:sum(x.get('usage',{}).get(k,0) for x in actor if x.get('usage')) for k in base.USAGE_FIELDS}
        base.write(directory/'COST.json',{'role':role,'actor':usage,'judge_tokens':sum(x['usage']['total_tokens'] for x in judge if x.get('usage')),
            'actor_calls':len(actor),'judge_calls':len(judge),'unresolved_calls':sum(x['status']=='UNKNOWN' for x in actor+judge)})
        return native,usage
    def infer(self,item,skill,role,maximum=None):
        key=(item['id'],item['sha256'],skill,'cot')
        with self.recovery_lock:
            recovered=self.recovered.pop(key,None)
        if recovered:
            directory,row,usage=recovered
            dest=self.root/'recovered'/item['id'];dest.mkdir(parents=True,exist_ok=False)
            base.write(dest/'REUSED.json',{'directory':str(directory),'usage':usage,'new_model_calls':0})
        else:
            directory=self.root/'rollouts'/uuid.uuid4().hex
            native,usage=self.rollout([item],skill,directory)
            row=native[0]
        if row['success_bool'] is None or row['has_exception']:raise RuntimeError('Unresolved training rollout; do not turn infrastructure errors into optimizer feedback')
        public=[json.loads(x) for x in (directory/'decisions.jsonl').read_text().splitlines() if x.strip()]
        public=[x for x in public if x['source']=='public_actor_messages']
        response='\n\n'.join(x['response'] for x in public)
        question='\n'.join(str(x['content']) for x in public[0]['messages'] if x['role']=='user') if public else item['question']
        hard=int(row['success_bool'])
        return {'id':item['id'],'question':question,'response':response,'hard':hard,'soft':float(hard),'task_type':'gaia2',
            'agent_ok':True,'n_turns':len(public),'fail_reason':'' if hard else 'The official native task validator returned failure.',
            'usage':usage,'trajectory':str(directory),'public_messages':public[-1]['messages'] if public else []}
    def batch(self,items,skill,role,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        def one(item):
            row=self.infer(item,skill,role);dest=directory/'predictions'/item['id'];dest.mkdir(parents=True,exist_ok=True)
            base.write(dest/'result.json',row)
            base.write(dest/'conversation.json',[{'role':'user','content':row['question']},{'type':'message','turn':1,'content':row['response']}])
            (dest/'target_system_prompt.txt').write_text(skill);(dest/'target_user_prompt.txt').write_text(row['question'])
            (dest/'trajectory.txt').write_text(json.dumps(row['public_messages'],ensure_ascii=False)+'\n'+row['response'])
            (dest/'feedback.txt').write_text(row['fail_reason']);return row
        with ThreadPoolExecutor(max_workers=self.config['resource'].get('client_batch_workers',2)) as pool:rows=list(pool.map(one,items))
        base.write(directory/'RESULT.json',rows);return rows
    def skillopt(self):
        from skillopt.envs.base import EnvAdapter
        from skillopt.engine.trainer import ReflACTTrainer
        from skillopt.config import load_config,flatten_config
        study=self
        class Adapter(EnvAdapter):
            analyst_workers=2;failure_only=False;minibatch_size=8;edit_budget=4
            def build_train_env(self,batch_size,seed,**kwargs):
                rows=list(study.data['train']);random.Random(seed).shuffle(rows);return rows[:batch_size]
            def build_eval_env(self,env_num,split,seed,**kwargs):
                if 'test' in split:raise RuntimeError('Test data sealed until optimization ends')
                return study.data['development']
            def rollout(self,env_manager,skill_content,out_dir,**kwargs):
                return study.batch(env_manager,skill_content,'skillopt_train',out_dir)
            def get_task_types(self):return ['gaia2']
        cfg=flatten_config(load_config(str(ROOT/'external/SkillOpt/configs/_base_/default.yaml')))
        seedfile=self.root/'seed_skill.md';seedfile.write_text(SEED_SKILL)
        cfg.update(out_root=str(self.root/'skillopt'),skill_init=str(seedfile),env_name='gaia2',
            optimizer_model=self.config['model']['served_name'],target_model=self.config['model']['served_name'],
            optimizer_backend='qwen_chat',target_backend='qwen_chat',qwen_chat_base_url=self.endpoint('skillopt_optimizer'),
            qwen_chat_api_key='EMPTY',qwen_chat_thinking_mode='disabled',qwen_chat_temperature=0,
            qwen_chat_max_tokens=32768,qwen_chat_timeout_seconds=1800,
            train_size=len(self.data['train']),batch_size=len(self.data['train']),
            num_epochs=4,sel_env_num=len(self.data['development']),test_env_num=0,
            eval_test=False,seed=self.config['seed'],analyst_workers=2)
        base.write(self.root/'SKILLOPT_CONFIG.json',cfg)
        ReflACTTrainer(cfg,Adapter()).train()
        if self.config.get('strict_optimizer_completion'):
            from phiagent.harness.aime_transport_recovery import skillopt_completion
            proof=skillopt_completion(self.root/'skillopt',4)
            base.write(self.root/'SKILLOPT_OPTIMIZER_AUDIT.json',proof)
            if proof['status']!='PASS':raise RuntimeError('SkillOpt did not complete all optimizer updates')
        return (self.root/'skillopt/best_skill.md').read_text()
    def autosaddler(self):
        from autosaddler.v2.config.registry import ScenarioRegistration,build_runtime,default_registry
        from autosaddler.v2.plugins.api import ScenarioPlugin,SCENARIO_PLUGIN_API_VERSION
        from autosaddler.v2.core.domain import ArtifactRef,Case,Cost,Evaluation,Observation,sha256_digest
        from autosaddler.v2.core.ports import ScenarioComponents
        from autosaddler.v2.prompting.models import Usage
        from phiagent_autosaddler.text_skill import TextSkillHarnessSpace,TextSkillEvidenceBuilder,TextSkillPromptPack
        from phiagent_autosaddler.openai_function_provider import build_openai_function_provider
        if self.config.get('optimizer_workspace_repair'):
            original_pack=TextSkillPromptPack
            class TextSkillPromptPack(original_pack):
                def _evidence(self,value):
                    return json.dumps(json.loads(super()._evidence(value)),indent=2,ensure_ascii=False)+'\n'
                def session(self,kind,context):
                    spec=super().session(kind,context)
                    if kind!='diagnose_patch':return spec
                    from phiagent.harness.aime_transport_recovery import readable_training_evidence
                    files=dict(spec.workspace_files)
                    files.update(readable_training_evidence(json.loads(files['training_evidence.json'])))
                    return replace(spec,workspace_files=files,task_prompt=spec.task_prompt+
                        '\nThe complete case index is training_evidence_index.json. Original responses '
                        'are unescaped text in training_cases/. Use narrow grep/view ranges.')
        import yaml
        study=self
        def factory(*,settings,base_dir,run_dir,store,ledger):
            harness=TextSkillHarnessSpace(baseline_skill=SEED_SKILL,store_root=run_dir/'candidates',materialization_root=run_dir/'materialized')
            class Evaluator:
                fingerprint='pinned-gaia2-native-'+study.config['dataset_revision']
                async def evaluate(self,candidate,cases,context):
                    materialized=harness.materialize(candidate,'evaluate')
                    try:skill=base.read(materialized.root/'candidate.json')['skill']
                    finally:materialized.release()
                    private={r['id']:r for r in study.data[context.split]}
                    observations=[]
                    for case in cases:
                        cached=context.attempt_sink.completed(candidate_id=candidate.candidate_id,case_id=case.case_id,repetition=0)
                        if cached is not None:observations.append(cached);continue
                        attempt,number=context.attempt_sink.start(candidate_id=candidate.candidate_id,case_id=case.case_id,repetition=0)
                        started=time.monotonic()
                        try:row=study.infer(private[case.case_id],skill,'autosaddler_train')
                        except Exception:
                            context.attempt_sink.fail(attempt,'provider_failure',Cost(rollouts=1));raise
                        usage=row['usage'];cost=Cost(rollouts=1,input_tokens=usage['prompt_tokens'],output_tokens=usage['completion_tokens'],wall_seconds=time.monotonic()-started)
                        context.attempt_sink.observe_usage(attempt,Usage(input_tokens=usage['prompt_tokens'],output_tokens=usage['completion_tokens'],total_tokens=usage['total_tokens'],model=study.config['model']['served_name'],role='actor'))
                        dest=context.artifact_dir/(hashlib.sha256(case.case_id.encode()).hexdigest()+'.json');base.write(dest,row)
                        ref=ArtifactRef(uri=str(dest.relative_to(run_dir)),kind='task-agent-raw-response')
                        obs=Observation.create(candidate_id=candidate.candidate_id,case_id=case.case_id,split=context.split,repetition=0,
                             disposition='success' if row['hard'] else 'task_failure',score=float(row['hard']),evaluator_fingerprint=self.fingerprint,
                             output=ref,attempts=number,cost=cost,metadata={'response':row['response'],'usage':usage})
                        context.attempt_sink.complete(attempt,obs,cost);observations.append(obs)
                    return Evaluation(evaluation_id=sha256_digest(context.operation_id),candidate_id=candidate.candidate_id,split=context.split,
                        purpose=context.purpose,iteration=context.iteration,requested_case_ids=tuple(c.case_id for c in cases),observations=tuple(observations),
                        artifact_dir=ArtifactRef(uri=str(context.artifact_dir.relative_to(run_dir)),kind='evaluation-directory'))
            cases=lambda split:tuple(Case(case_id=r['id'],split=split,payload={'question':r['question']}) for r in study.data[split])
            return ScenarioComponents(name='phiagent_gaia2',version='1',harness_space=harness,evaluator=Evaluator(),
                evidence_builder=TextSkillEvidenceBuilder(store,'gaia2'),prompt_pack=TextSkillPromptPack(store,benchmark='GAIA2 official execution development',skill_document=SEED_SKILL),
                train_cases=cases('train'),development_cases=cases('development'),required_capabilities=frozenset({'read_workspace','edit_workspace','load_skills'}),resolved_entities={})
        registry=default_registry();plugin=ScenarioPlugin(name='phiagent_gaia2',api_version=SCENARIO_PLUGIN_API_VERSION,factory=factory)
        registry.register_scenario(ScenarioRegistration(plugin=plugin,source='builtin'));registry.providers['phiagent_openai_function']=build_openai_function_provider
        model=self.config['model']['served_name']
        cfg={'schema_version':'autosaddler/v2','scenario':{'type':'phiagent_gaia2','settings':{}},
             'optimization':{'task_selection':{'type':'fixed','batch_size':len(self.data['train']),'seed':self.config['seed']},'acceptance':{'type':'matched_valid_strict_improvement'},
                 'development':{'type':'full_on_accept'},'ranking':{'type':'mean_development_score'},'budget':{'max_rollouts':100000,'max_iterations':4},
                 'diagnosis_patch_timeout_seconds':1800,'selection_timeout_seconds':1800,'reflection_timeout_seconds':1800,
                 'session_retries':self.config.get('recovery_session_retries',0),'session_retry_backoff_seconds':0},
             'provider':{'type':'phiagent_openai_function','capabilities':['read_workspace','edit_workspace','load_skills'],
                 'settings':{'model':model,'reasoning_effort':None,'max_tool_steps':48,'temperature':0,'max_completion_tokens':16384,'enable_thinking':False,
                     'finish_on_stalled_reads':self.config.get('optimizer_structured_finalization',False),
                     'provider':{'type':'openai','base_url':self.endpoint('autosaddler_optimizer'),'wire_api':'chat_completions','model_id':model,'wire_model':model}}},
             'storage':{'type':'local','run_root':str(self.root/'autosaddler')}}
        path=self.root/'AUTOSADDLER_CONFIG.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        runtime=build_runtime(path,run_id='gaia2-execution-development',registry=registry)
        result=runtime.engine.run();base.write(self.root/'AUTOSADDLER_SELECTED.json',{'id':result.selected_candidate_id,'development_score':result.development_score,'iterations':result.iterations})
        if self.config.get('strict_optimizer_completion'):
            from phiagent.harness.aime_transport_recovery import autosaddler_completion
            events=[json.loads(line) for line in runtime.store.events_path.read_text().splitlines() if line.strip()]
            proof=autosaddler_completion(events,4);proof['events_sha256']=base.digest(runtime.store.events_path)
            base.write(self.root/'AUTOSADDLER_OPTIMIZER_AUDIT.json',proof)
            if proof['status']!='PASS':raise RuntimeError('AutoSaddler did not complete every diagnosis/patch round')
        state=runtime.engine._state();materialized=runtime.scenario.harness_space.materialize(state.candidates[result.selected_candidate_id],'evaluate')
        try:return base.read(materialized.root/'candidate.json')['skill']
        finally:materialized.release()
    def execute(self):
        method=self.config['method']
        phase='OPTIMIZING_'+method.upper() if method in ('skillopt','autosaddler') else 'FROZEN_DEPLOYMENT'
        base.write(self.root/'PROGRESS.json',{'at_utc':base.now(),'phase':phase})
        skill=getattr(self,method)() if method in ('skillopt','autosaddler') else (SEED_SKILL if method=='chisel' else '')
        (self.root/'SELECTED_SKILL.md').write_text(skill)
        base.write(self.root/'FROZEN_SELECTION.json',{'at_utc':base.now(),'method':method,'skill_sha256':base.digest(self.root/'SELECTED_SKILL.md'),
            'test_outcomes_used':False,'optimizer_data':'official execution train20/development10, excluding final mini tasks'})
        base.write(self.root/'PROGRESS.json',{'at_utc':base.now(),'phase':'FINAL_FULL160_X3','method':method})
        rows,usage=self.rollout(self.data['test'],skill,self.root/'final',mode='native' if method=='native' else 'jev' if method=='chisel' else 'cot',repetitions=3)
        base.write(self.root/'RESULT.json',{'at_utc':base.now(),'method':method,'executions':len(rows),'successes':sum(x['success_bool'] is True for x in rows),
            'unresolved':sum(x['success_bool'] is None or x['has_exception'] for x in rows),'actor_usage':usage,'cost':base.read(self.root/'final/COST.json'),
            'model':self.config['model'],'dataset_revision':self.config['dataset_revision'],'native_result_sha256':base.digest(self.root/'final/NATIVE_RESULT.json'),
            'implementation':'pinned official optimization engines with GAIA2 text-skill adapters; native evaluator',
            'full_leaderboard_protocol':False,'paper_admitted':False})


def audit(config,root):
    require_remote(config);ledger=base.read(root/'PROVIDER_FINAL.json');result=base.read(root/'RESULT.json')
    if ledger['unknown_usage_calls'] or ledger['inflight_reserved_tokens']:raise RuntimeError('Unresolved provider cost')
    total=0;actor=0;role=base.read(root/'final/JOB.json')['role']
    for call in ledger['calls']:
        p=root/'provider'/f"call-{call['sequence']:06d}"
        if call['status']=='REJECTED_BEFORE_FORWARD':
            proof=base.read(p/'REJECTED_BEFORE_FORWARD.json')
            if proof['model_calls']!=0 or proof['input_tokens']+proof['max_tokens']<=proof['context']:raise RuntimeError('Invalid rejection evidence')
            continue
        raw=base.read(p/'response.json');usage=base.native_usage(raw,config['model']['served_name'],config['model']['max_model_len'])
        if usage!=call['usage'] or base.read(p/'request.json')['role']!=call['role']:raise RuntimeError('Raw native accounting mismatch')
        total+=usage['total_tokens']
        if call['role']==role+'/actor':actor+=usage['total_tokens']
    if total!=ledger['known_tokens'] or actor!=result['actor_usage']['total_tokens']:raise RuntimeError('Cost sum mismatch')
    rows=base.read(root/'final/NATIVE_RESULT.json')['rows'];data=base.read(root/'DATA.json')
    expected={(x['id'],rep) for x in data['test'] for rep in (1,2,3)}
    if len(rows)!=480 or {(x['base_scenario_id'],x['run_number']) for x in rows}!=expected:raise RuntimeError('Incomplete official population')
    if sum(x['success_bool'] is True for x in rows)!=result['successes']:raise RuntimeError('Score mismatch')
    for name,h in base.read(root/'SOURCE_FREEZE.json')['files_sha256'].items():
        if base.digest(ROOT/name)!=h:raise RuntimeError('Source changed: '+name)
    status='PASS' if result['unresolved']==0 else 'COMPLETE_WITH_UNRESOLVED_OUTCOMES'
    inherited=0
    if config.get('continuation'):
        continuation=base.read(root/'CONTINUATION_AUDIT.json');prior=Path(continuation['prior_root'])
        if base.digest(prior/'PROVIDER_FINAL.json')!=continuation['prior_ledger_sha256']:raise RuntimeError('Predecessor ledger changed')
        inherited=continuation['inherited_all_stage_tokens']
        if continuation.get('prior_ledgers'):
            for entry in continuation['prior_ledgers']:
                if base.digest(Path(entry['root'])/'PROVIDER_FINAL.json')!=entry['sha256']:raise RuntimeError('Ancestor ledger changed')
            if sum(x['known_tokens'] for x in continuation['prior_ledgers'])!=inherited:raise RuntimeError('Ancestor cost sum mismatch')
        for proof in continuation['recovered_rollouts']:
            job_path=Path(proof['job']);output=job_path.parent/'traces/output.jsonl'
            trace=Path(json.loads(output.read_text().splitlines()[0])['trace_id'])
            if base.digest(job_path)!=proof['job_sha256'] or base.digest(output)!=proof['output_sha256'] or base.digest(trace)!=proof['trace_sha256']:
                raise RuntimeError('Recovered evidence changed')
            if not (root/'recovered'/proof['scenario_id']/'REUSED.json').exists():raise RuntimeError('Recovered rollout was not reused')
    base.write(root/'VERIFIED.json',{'at_utc':base.now(),'status':status,'result_sha256':base.digest(root/'RESULT.json'),'all_stage_tokens':total+inherited,
        'current_run_tokens':total,'inherited_all_stage_tokens':inherited,'unknown_usage_calls':0,
        'official_native_scores_recounted':480,'data_sha256':base.digest(root/'DATA.json')})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','worker','audit'));p.add_argument('--config',type=Path,required=True);p.add_argument('--job',type=Path)
    args=p.parse_args();config=base.read(args.config);require_remote(config);root=Path(config['run_directory'])
    os.environ.update(DEMO_FS_PATH=config['filesystem'],FS_PATH=str(root/'filesystem-cache'),OPENAI_API_KEY='EMPTY',HF_HUB_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
    if args.action=='prepare':prepare(config,root)
    elif args.action=='worker':worker(config,base.read(args.job))
    elif args.action=='audit':audit(config,root)
    else:
        lifecycle.Study=Study;lifecycle.Gateway=Gateway;lifecycle.audit=audit
        lifecycle.run(config,root)

if __name__=='__main__':main()
