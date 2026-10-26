#!/usr/bin/env python3
"""Remote-only AIME study: official optimizers, sealed 2025 test, native costs.

The 2024 train/development split is the sole optimization source. CHISEL transfers
the existing answer-first, confidence-gated fallback mechanism from LiveMath.
"""
from __future__ import annotations
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'external/SkillOpt'), str(ROOT/'external/AutoSaddler/src'),
               str(ROOT/'integrations/autosaddler_phiagent/src')]
from scripts import run_gaia2_jev as base

SEED_SKILL = '''# Integer mathematics
## Model the problem
Translate the stated constraints exactly. Identify the desired integer quantity.
## Solve
Use exact arithmetic and a short derivation; check endpoints and extraneous roots.
## Verify
Substitute the result into the original constraints. End with ANSWER: followed by the integer.
'''
FAST_SKILL = '''Solve this integer mathematics problem using a concise exact derivation.
Check arithmetic and the original constraints. Before the answer, report
CONFIDENCE: an integer from 0 to 100 estimating the probability your answer is correct.
The final line must be ANSWER: followed by the integer. Do not claim confidence if unresolved.'''


def remote(config):
    root = Path(config['run_directory'])
    if sys.platform != 'linux' or socket.gethostname() != config['hostname'] or not root.is_relative_to('/opt/phiagent/work/aime-comparison'):
        raise RuntimeError('Benchmark execution is restricted to the frozen remote host')
    if config['budget']['token_usage_mode'] != 'unlimited' or config['budget']['maximum_tokens'] is not None:
        raise RuntimeError('Cumulative token accounting must be explicitly unlimited')
    return root


def official(config):
    path = ROOT/'official/aime_common.py'
    if base.digest(path) != config['scorer_sha256']:
        raise RuntimeError('Pinned official scorer changed')
    spec = importlib.util.spec_from_file_location('pinned_aime_common', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def score_response(module, response, target):
    from inspect_ai.scorer import Target
    from inspect_ai.model import ModelOutput
    state = SimpleNamespace(output=ModelOutput.from_content(model='frozen', content=response))
    score = asyncio.run(module.aime_scorer()(state, Target(str(target))))
    return int(score.value == 'C'), score.answer


def probability_readout_args(config):
    mode = config.get('resource', {}).get('logprobs_mode')
    if mode is None:
        return []
    if mode != 'processed_logprobs':
        raise ValueError('Typed native choices require probabilities after the declared-token mask')
    return ['--logprobs-mode', mode, '--max-logprobs', '20']


def prepare(config, root):
    import requests
    import pyarrow.parquet as pq
    if (root/'DATA.json').exists():
        raise FileExistsError('Immutable preparation already exists')
    proxy = 'socks5h://127.0.0.1:18190'
    listener = subprocess.check_output(['ss','-ltn'], text=True)
    if '127.0.0.1:18190' not in listener:
        raise RuntimeError('Workstation-owned loopback SOCKS listener missing')
    session = requests.Session(); session.trust_env=False
    session.proxies={'https':proxy,'http':proxy}
    check=session.get('https://huggingface.co',timeout=30);check.raise_for_status()
    snapshots={}; datasets={}
    for year, identity in config['datasets'].items():
        repo, rev = identity['repository'], identity['revision']
        response=session.get(f'https://huggingface.co/api/datasets/{repo}/tree/{rev}',params={'recursive':'true'},timeout=60);response.raise_for_status()
        inventory=response.json(); snapshots[year]={'identity':identity,'files':{}}
        records=[]
        files=[r for r in inventory if r['type']=='file' and r['path'].endswith(('.parquet','.jsonl'))]
        if not files: raise RuntimeError('Pinned dataset has no supported Parquet or JSONL source')
        for row in files:
            name=row['path']; dest=root/'data'/year/name;dest.parent.mkdir(parents=True,exist_ok=True)
            response=session.get(f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}',timeout=120);response.raise_for_status();dest.write_bytes(response.content)
            snapshots[year]['files'][name]=base.digest(dest)
            if name.endswith('.parquet'):
                records.extend(pq.read_table(dest).to_pylist())
            else:
                records.extend(json.loads(line) for line in dest.read_text().splitlines() if line.strip())
        items=[]
        for row in records:
            question=row.get('problem',row.get('Problem'));answer=row.get('answer',row.get('Answer'))
            identifier=str(row.get('id',row.get('ID')))
            if question is None or answer is None:raise RuntimeError('Unknown dataset schema')
            items.append({'id':year+'-'+identifier,'question':str(question),'answer':str(answer)})
        if len(items)!=30 or len({x['id'] for x in items})!=30:raise RuntimeError('Expected full 30-question year')
        datasets[year]=sorted(items,key=lambda x:x['id'])
    train_year=list(datasets['2024']);random.Random(config['seed']).shuffle(train_year)
    data={'train':train_year[:20],'development':train_year[20:],'test':datasets['2025']}
    if set(x['question'] for x in datasets['2024']) & set(x['question'] for x in datasets['2025']):
        raise RuntimeError('Cross-year question overlap')
    base.write(root/'DATA.json',data);base.write(root/'DATA_PROVENANCE.json',snapshots)
    module=official(config)
    for response,target in [('ANSWER: 42','42'),('$$\n\\boxed{70}\n$$','70')]:
        if score_response(module,response,target)[0]!=1:raise RuntimeError('Official scorer import check failed')
    base.write(root/'PREPARED.json',{'at_utc':base.now(),'dataset_hash':base.digest(root/'DATA.json'),
               'split_sizes':{k:len(v) for k,v in data.items()},'scorer_sha256':config['scorer_sha256'],
               'test_exposure_to_optimizers':False,'model_calls':0})


class Gateway(ThreadingHTTPServer):
    def __init__(self,config,root,ledger):
        from scripts.aime_client_admission import policy
        self.config,self.root,self.ledger=config,root,ledger
        self.admission_policy=policy(config)
        limit=self.admission_policy['limit']
        self.admission=threading.BoundedSemaphore(limit) if limit else None
        super().__init__(('127.0.0.1',config['resource']['capture_port']),Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        config=self.server.config;model=config['model'];usage=None;admitted=False
        parts=self.path.strip('/').split('/')
        if len(parts)!=4 or parts[1:]!=['v1','chat','completions']:
            self.send_error(404);return
        try:
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if body.get('model')!=model['served_name'] or body.get('stream',False):raise ValueError('Model/stream mismatch')
            maximum=body.pop('max_completion_tokens',body.get('max_tokens',model['max_tokens']))
            body.update(max_tokens=min(int(maximum),model['max_tokens']),temperature=0,seed=config['seed'],
                        chat_template_kwargs={'enable_thinking':False})
            context_changes=[]
            if config.get('optimizer_context_repair') and parts[0] in ('autosaddler_optimizer','skillopt_optimizer'):
                from phiagent.harness.aime_transport_recovery import fit_optimizer_context
                def count(payload):
                    token_request={k:payload[k] for k in ('model','messages','tools','chat_template_kwargs') if k in payload}
                    token_request['add_generation_prompt']=True
                    return base.get_json(f"http://127.0.0.1:{config['resource']['model_port']}/tokenize",token_request)['count']
                body,context_changes=fit_optimizer_context(body,count,model['max_model_len'])
            if self.server.admission is not None:
                if not self.server.admission.acquire(blocking=False):
                    raise RuntimeError('Client admission busy; no model request or cost reservation made')
                admitted=True
            sequence=self.server.ledger.reserve(parts[0])
        except Exception as e:
            if admitted:self.server.admission.release()
            self.send_error(429,str(e));return
        directory=self.server.root/'provider'/f'call-{sequence:06d}';directory.mkdir(parents=True)
        base.write(directory/'request.json',{'role':parts[0],'json':body})
        if context_changes:base.write(directory/'CONTEXT_REPAIR.json',context_changes)
        try:
            base.write(directory/'ADMISSION.json',{'at_utc':base.now(),
                'policy':self.server.admission_policy,
                'queue_avoided_by_client_admission':self.server.admission is not None})
            raw=base.get_json(f"http://127.0.0.1:{config['resource']['model_port']}/v1/chat/completions",body,
                             timeout=self.server.admission_policy['provider_timeout'])
            base.write(directory/'response.json',raw)
            usage=base.native_usage(raw,model['served_name'],model['max_model_len'])
            payload=json.dumps(raw).encode();status=200
        except Exception as e:
            base.write(directory/'FAILURE.json',{'error':str(e),'traceback':traceback.format_exc()})
            payload=json.dumps({'error':str(e)}).encode();status=502
        finally:
            try:self.server.ledger.settle(sequence,usage)
            finally:
                if admitted:self.server.admission.release()
        self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)


class Study:
    def __init__(self,config,root):
        from scripts.aime_client_admission import policy,slots
        self.config,self.root,self.module=config,root,official(config)
        self.client_policy=policy(config);self.client_slots=slots(config)
        self.data=base.read(root/'DATA.json')
        self.prior_responses={}
        if config.get('inherit_provider_ledger'):
            prior=Path(config['inherit_provider_ledger'])
            if base.digest(prior/'PROVIDER_FINAL.json')!=config['prior_provider_sha256'] or base.digest(prior/'DATA.json')!=base.digest(root/'DATA.json'):
                raise RuntimeError('AIME continuation identity changed')
            for call in base.read(prior/'PROVIDER_FINAL.json')['calls']:
                if call['role']!='autosaddler_train' or config.get('autosaddler_resume_from'):continue
                directory=prior/'provider'/f"call-{call['sequence']:06d}"
                request=base.read(directory/'request.json')['json'];raw=base.read(directory/'response.json')
                if base.native_usage(raw,config['model']['served_name'],config['model']['max_model_len'])!=call['usage']:
                    raise RuntimeError('AIME cached usage mismatch')
                key=json.dumps([request['messages'],request['max_tokens']],sort_keys=True)
                if key in self.prior_responses:raise RuntimeError('Ambiguous AIME cached response')
                self.prior_responses[key]=(raw,call['sequence'])
    def endpoint(self,role):return f"http://127.0.0.1:{self.config['resource']['capture_port']}/{role}/v1"
    def infer(self,item,skill,role,maximum=32768):
        with self.client_slots:
            return self._infer_admitted(item,skill,role,maximum)
    def _infer_admitted(self,item,skill,role,maximum=32768):
        messages=[{'role':'user','content':self.module.USER_PROMPT_TEMPLATE.format(prompt=item['question'])}]
        if skill:messages.insert(0,{'role':'system','content':skill})
        key=json.dumps([messages,maximum],sort_keys=True)
        cached=self.prior_responses.pop(key,None) if role=='autosaddler_train' else None
        if cached:
            raw,sequence=cached;directory=self.root/'reused';directory.mkdir(exist_ok=True)
            base.write(directory/(str(sequence)+'.json'),{'prior_sequence':sequence,'item_id':item['id'],'new_model_calls':0})
        else:
            raw=base.get_json(self.endpoint(role)+'/chat/completions',{'model':self.config['model']['served_name'],
                             'messages':messages,'max_tokens':maximum},
                              timeout=self.client_policy['client_timeout'])
        response=raw['choices'][0]['message']['content']
        hard,answer=score_response(self.module,response,item['answer'])
        return {'id':item['id'],'question':item['question'],'response':response,'hard':hard,'soft':float(hard),
                'task_type':'integer_math','predicted_answer':answer,'agent_ok':True,'n_turns':1,
                'fail_reason':'' if hard else 'Final integer differs from the reference answer.',
                'usage':raw['usage'],'raw_response':raw}
    def batch(self,items,skill,role,directory,maximum=32768):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        def one(item):
            from scripts.aime_client_admission import reuse_prediction
            dest=directory/'predictions'/item['id']
            reused=reuse_prediction(self.root,self.config,dest,item,skill,
                self.module.USER_PROMPT_TEMPLATE.format(prompt=item['question']),role,maximum)
            if reused is not None:return reused
            row=self.infer(item,skill,role,maximum)
            dest=directory/'predictions'/item['id'];dest.mkdir(parents=True,exist_ok=True)
            base.write(dest/'result.json',row)
            base.write(dest/'conversation.json',[
                {'role':'user','content':item['question']},
                {'type':'message','turn':1,'content':row['response']},
                {'role':'system','content':f"Official integer accuracy: {row['hard']}; reference answer: {item['answer']}"}])
            (dest/'target_system_prompt.txt').write_text(skill)
            (dest/'target_user_prompt.txt').write_text(self.module.USER_PROMPT_TEMPLATE.format(prompt=item['question']))
            (dest/'trajectory.txt').write_text('Question:\n'+item['question']+'\nResponse:\n'+row['response'])
            (dest/'feedback.txt').write_text(row['fail_reason'])
            return row
        with ThreadPoolExecutor(max_workers=self.client_policy['workers']) as pool:rows=list(pool.map(one,items))
        base.write(directory/'RESULT.json',rows)
        return rows
    def skillopt(self):
        from skillopt.envs.base import EnvAdapter
        from skillopt.engine.trainer import ReflACTTrainer
        from skillopt.config import load_config,flatten_config
        study=self
        class Adapter(EnvAdapter):
            analyst_workers=study.client_policy['analysts'];failure_only=False;minibatch_size=8;edit_budget=4
            def build_train_env(self,batch_size,seed,**kwargs):
                rows=list(study.data['train']);random.Random(seed).shuffle(rows);return rows[:batch_size]
            def build_eval_env(self,env_num,split,seed,**kwargs):
                if 'test' in split:raise RuntimeError('Test data sealed until optimization ends')
                return study.data['development']
            def rollout(self,env_manager,skill_content,out_dir,**kwargs):
                return study.batch(env_manager,skill_content,'skillopt_train',out_dir)
            def get_task_types(self):return ['integer_math']
        cfg=flatten_config(load_config(str(ROOT/'external/SkillOpt/configs/_base_/default.yaml')))
        seedfile=self.root/'seed_skill.md';seedfile.write_text(SEED_SKILL)
        cfg.update(out_root=str(self.root/'skillopt'),skill_init=str(seedfile),env_name='aime',
            optimizer_model=self.config['model']['served_name'],target_model=self.config['model']['served_name'],
            optimizer_backend='qwen_chat',target_backend='qwen_chat',qwen_chat_base_url=self.endpoint('skillopt_optimizer'),
            qwen_chat_api_key='EMPTY',qwen_chat_thinking_mode='disabled',qwen_chat_temperature=0,
            qwen_chat_max_tokens=32768,qwen_chat_timeout_seconds=self.client_policy['client_timeout'],
            train_size=len(self.data['train']),batch_size=len(self.data['train']),
            num_epochs=4,sel_env_num=len(self.data['development']),test_env_num=0,eval_test=False,
            seed=self.config['seed'],analyst_workers=self.client_policy['analysts'])
        base.write(self.root/'SKILLOPT_CONFIG.json',cfg)
        ReflACTTrainer(cfg,Adapter()).train()
        if self.config.get('strict_optimizer_completion'):
            from phiagent.harness.aime_transport_recovery import skillopt_completion
            proof=skillopt_completion(self.root/'skillopt',4)
            base.write(self.root/'SKILLOPT_OPTIMIZER_AUDIT.json',proof)
            if proof['status']!='PASS':
                raise RuntimeError('SkillOpt has incomplete epoch-end updates; final evaluation blocked')
        return (self.root/'skillopt/best_skill.md').read_text()
    def autosaddler(self):
        from autosaddler.v2.config.registry import ScenarioRegistration,build_runtime,default_registry
        from autosaddler.v2.plugins.api import ScenarioPlugin,SCENARIO_PLUGIN_API_VERSION
        from autosaddler.v2.core.domain import ArtifactRef,Case,Cost,Evaluation,Observation,sha256_digest
        from autosaddler.v2.core.ports import ScenarioComponents
        from autosaddler.v2.prompting.models import Usage
        from phiagent_autosaddler.text_skill import TextSkillHarnessSpace,TextSkillEvidenceBuilder,TextSkillPromptPack
        from phiagent_autosaddler.openai_function_provider import build_openai_function_provider
        if self.config.get('optimizer_context_repair') or self.config.get('optimizer_workspace_repair'):
            original_pack=TextSkillPromptPack
            workspace_repair=self.config.get('optimizer_workspace_repair',False)
            class TextSkillPromptPack(original_pack):
                def _evidence(self,value):
                    # Preserve every value, while making grep and line ranges usable.
                    return json.dumps(json.loads(super()._evidence(value)),indent=2,separators=(',',':'),ensure_ascii=False)+'\n'
                def session(self,kind,context):
                    spec=super().session(kind,context)
                    if kind!='diagnose_patch' or not workspace_repair:return spec
                    from phiagent.harness.aime_transport_recovery import readable_training_evidence
                    files=dict(spec.workspace_files)
                    files.update(readable_training_evidence(json.loads(files['training_evidence.json'])))
                    return replace(spec,workspace_files=files,task_prompt=spec.task_prompt+
                        '\nThe complete case index is training_evidence_index.json; every original response '
                        'is also available as unescaped text in training_cases/. Use the index and narrow '
                        'grep/view ranges instead of repeatedly reading truncated JSON lines. The original '
                        'training_evidence.json remains unchanged in meaning.')
        import yaml
        study=self
        def factory(*,settings,base_dir,run_dir,store,ledger):
            harness=TextSkillHarnessSpace(baseline_skill=SEED_SKILL,store_root=run_dir/'candidates',materialization_root=run_dir/'materialized')
            class Evaluator:
                fingerprint='pinned-inspect-aime2024-'+study.config['scorer_sha256']
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
                        # The official scorer owns its event loop. Run the synchronous
                        # inference/scoring adapter outside AutoSaddler's active loop.
                        try:row=await asyncio.to_thread(study.infer,private[case.case_id],skill,'autosaddler_train')
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
            return ScenarioComponents(name='phiagent_aime',version='1',harness_space=harness,evaluator=Evaluator(),
                evidence_builder=TextSkillEvidenceBuilder(store,'aime'),prompt_pack=TextSkillPromptPack(store,benchmark=study.config.get('benchmark_label','AIME 2024'),skill_document=SEED_SKILL),
                train_cases=cases('train'),development_cases=cases('development'),required_capabilities=frozenset({'read_workspace','edit_workspace','load_skills'}),resolved_entities={})
        registry=default_registry();plugin=ScenarioPlugin(name='phiagent_aime',api_version=SCENARIO_PLUGIN_API_VERSION,factory=factory)
        registry.register_scenario(ScenarioRegistration(plugin=plugin,source='builtin'));registry.providers['phiagent_openai_function']=build_openai_function_provider
        model=self.config['model']['served_name']
        cfg={'schema_version':'autosaddler/v2','scenario':{'type':'phiagent_aime','settings':{}},
             'optimization':{'task_selection':{'type':'fixed','batch_size':len(self.data['train']),'seed':self.config['seed']},'acceptance':{'type':'matched_valid_strict_improvement'},
                 'development':{'type':'full_on_accept'},'ranking':{'type':'mean_development_score'},'budget':{'max_rollouts':100000,'max_iterations':4},
                 'diagnosis_patch_timeout_seconds':1800,'selection_timeout_seconds':1800,'reflection_timeout_seconds':1800,'session_retries':self.config.get('recovery_session_retries',0),'session_retry_backoff_seconds':0},
             'provider':{'type':'phiagent_openai_function','capabilities':['read_workspace','edit_workspace','load_skills'],
                 'settings':{'model':model,'reasoning_effort':None,'max_tool_steps':48,'temperature':0,'max_completion_tokens':16384,'enable_thinking':False,
                     'finish_on_stalled_reads':self.config.get('optimizer_structured_finalization',False),
                     'provider':{'type':'openai','base_url':self.endpoint('autosaddler_optimizer'),'wire_api':'chat_completions','model_id':model,'wire_model':model}}},
             'storage':{'type':'local','run_root':str(self.root/'autosaddler')}}
        path=self.root/'AUTOSADDLER_CONFIG.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        runtime=build_runtime(path,run_id='aime2024-development',registry=registry)
        if self.config.get('autosaddler_resume_from'):
            from autosaddler.v2.storage.local import LocalRunStore
            prior=Path(self.config['autosaddler_resume_from'])
            source=LocalRunStore(run_dir=prior,run_id='aime2024-development')
            source.validate_integrity()
            if runtime.store.events():raise RuntimeError('Resume target already contains events')
            # New runtime manifest records repaired transport settings. Copy the exact
            # official event ledger and immutable artifacts; leave the source untouched.
            for item in prior.iterdir():
                if item.name in {'resolved','resolved_config.yaml','manifest.json','snapshot.json','metrics-summary.json','evolution_dag.json'}:continue
                target=runtime.store.run_dir/item.name
                if item.is_dir():shutil.copytree(item,target,dirs_exist_ok=True)
                else:shutil.copy2(item,target)
            runtime.store.refresh_projections();runtime.store.validate_integrity()
            base.write(self.root/'AUTOSADDLER_RESUME.json',{'source':str(prior),
                'source_events_sha256':base.digest(prior/'events.jsonl'),
                'copied_events_sha256':base.digest(runtime.store.events_path),
                'source_manifest_sha256':base.digest(prior/'manifest.json'),
                'official_engine_modified':False,'recovery_session_retries':self.config['recovery_session_retries']})
        result=runtime.engine.run();base.write(self.root/'AUTOSADDLER_SELECTED.json',{'id':result.selected_candidate_id,'development_score':result.development_score,'iterations':result.iterations})
        if self.config.get('strict_optimizer_completion'):
            from phiagent.harness.aime_transport_recovery import autosaddler_completion
            events=[json.loads(line) for line in runtime.store.events_path.read_text().splitlines() if line.strip()]
            proof=autosaddler_completion(events,4)
            proof['events_sha256']=base.digest(runtime.store.events_path)
            base.write(self.root/'AUTOSADDLER_OPTIMIZER_AUDIT.json',proof)
            if proof['status']!='PASS':
                raise RuntimeError('AutoSaddler did not complete every diagnosis/patch round; final evaluation blocked')
        state=runtime.engine._state();materialized=runtime.scenario.harness_space.materialize(state.candidates[result.selected_candidate_id],'evaluate')
        try:return base.read(materialized.root/'candidate.json')['skill']
        finally:materialized.release()
    def chisel(self,items,fallback,threshold,directory):
        if self.config.get('chisel_policy')=='proof_review_v1':
            from scripts.run_aime_proof_development import PROOF_SKILL,REVIEW_SKILL
            directory=Path(directory)
            drafts=self.batch(items,PROOF_SKILL,'chisel_proof_draft',directory/'draft',
                              maximum=self.config['model']['max_tokens'])
            def review(pair):
                item,draft=pair
                revised={**item,'question':'ORIGINAL PROBLEM:\n'+item['question']+
                         '\n\nPROPOSED DERIVATION (may contain errors):\n'+draft['response']}
                row=self.infer(revised,REVIEW_SKILL,'chisel_proof_review',
                               maximum=self.config['model']['max_tokens'])
                row['draft_usage']=dict(draft['usage']);row['review_usage']=dict(row['usage'])
                row['usage']={k:row['usage'][k]+draft['usage'][k] for k in base.USAGE_FIELDS}
                row['draft_raw_response']=draft['raw_response']
                folder=directory/'predictions'/item['id'];folder.mkdir(parents=True,exist_ok=False)
                base.write(folder/'result.json',row)
                return row
            with ThreadPoolExecutor(max_workers=self.client_policy['workers']) as pool:rows=list(pool.map(review,zip(items,drafts)))
            base.write(directory/'RESULT.json',rows)
            return rows
        rows=self.batch(items,FAST_SKILL,'chisel_draft',Path(directory)/'draft',2048)
        for item,row in zip(items,rows):
            matches=re.findall(r'CONFIDENCE:\s*(\d+)',row['response'],re.I)
            confidence=int(matches[-1]) if matches else -1
            row['draft_usage']=dict(row['usage']);row['confidence']=confidence;row['fallback']=False
            valid=True
            if self.config.get('chisel_draft_guard')=='complete_integer_v1':
                from phiagent.harness.aime_draft_guard import inspect_draft
                row['draft_guard']=inspect_draft(row['response'],row['raw_response']['choices'][0].get('finish_reason'))
                valid=row['draft_guard']['accepted']
                confidence=row['draft_guard']['confidence'];row['confidence']=confidence
            if not valid or confidence<threshold or not row['predicted_answer']:
                second=self.infer(item,fallback,'chisel_fallback');usage={k:row['usage'][k]+second['usage'][k] for k in base.USAGE_FIELDS}
                row.update({k:second[k] for k in ('response','predicted_answer','hard','soft')});row['usage']=usage;row['fallback']=True;row['fallback_raw_response']=second['raw_response']
        base.write(Path(directory)/'RESULT.json',rows);return rows
    def execute(self):
        # Optimizers receive only 2024 items; final-test prompts are frozen first.
        base.write(self.root/'PROGRESS.json',{'phase':'SKILLOPT_2024_TRAINING','at_utc':base.now()})
        optimizer_roots=self.config.get('frozen_optimizer_roots')
        if optimizer_roots:
            if set(optimizer_roots)!={'skillopt','autosaddler'}:raise RuntimeError('Both corrected optimizers are required')
            for name,location in optimizer_roots.items():
                prior=Path(location);proof=base.read(prior/'VERIFIED.json');result=base.read(prior/'RESULT.json')
                if proof['status']!='PASS' or proof['result_sha256']!=base.digest(prior/'RESULT.json'):
                    raise RuntimeError('Corrected optimizer is not verified')
                if result['method']!=name or result['selected_skill_sha256']!=base.digest(prior/f'selected_{name}.md'):
                    raise RuntimeError('Corrected selected skill identity differs')
                if result['model']!=self.config['model']:raise RuntimeError('Corrected optimizer backbone differs')
                shutil.copy2(prior/f'{name.upper()}_OPTIMIZER_AUDIT.json',self.root/f'{name.upper()}_OPTIMIZER_AUDIT.json')
            skill=(Path(optimizer_roots['skillopt'])/'selected_skillopt.md').read_text()
        elif self.config.get('inherit_provider_ledger'):
            selected=Path(self.config['inherit_provider_ledger'])/'selected_skillopt.md'
            if base.digest(selected)!=self.config['prior_skillopt_sha256']:raise RuntimeError('Selected SkillOpt skill changed')
            skill=selected.read_text()
        else:skill=self.skillopt()
        (self.root/'selected_skillopt.md').write_text(skill)
        base.write(self.root/'PROGRESS.json',{'phase':'AUTOSADDLER_2024_TRAINING','at_utc':base.now()})
        auto=(Path(optimizer_roots['autosaddler'])/'selected_autosaddler.md').read_text() if optimizer_roots else self.autosaddler()
        (self.root/'selected_autosaddler.md').write_text(auto)
        # Thresholds and tie-break are frozen before any 2025 model call.
        candidates=[]
        if self.config.get('chisel_policy')=='proof_review_v1':
            frozen=base.read(self.root/'PROOF_POLICY_PROVENANCE.json')
            if not frozen['development_gate_passed'] or frozen['test_details_used_for_selection']:
                raise RuntimeError('Proof policy requires its frozen development gate')
            for name,digest in frozen['policy_source_sha256'].items():
                if base.digest(ROOT/name)!=digest:raise RuntimeError('Frozen proof policy changed')
            selected={'threshold':None,'policy':'proof_review_v1','development_provenance_sha256':base.digest(self.root/'PROOF_POLICY_PROVENANCE.json')}
        else:
            for threshold in (0,85,95,101):
                rows=self.chisel(self.data['development'],skill,threshold,self.root/f'chisel-selection/{threshold}')
                candidates.append({'threshold':threshold,'correct':sum(r['hard'] for r in rows),'tokens':sum(r['usage']['total_tokens'] for r in rows)})
            selected=sorted(candidates,key=lambda r:(-r['correct'],r['tokens'],r['threshold']))[0]
        base.write(self.root/'FROZEN_TEST_POLICY.json',{'candidates':candidates,'selected':selected,
             'tie_break':'Frozen proof-review policy; no new test-based selection' if self.config.get('chisel_policy')=='proof_review_v1' else 'maximum development exact matches, then minimum tokens, then threshold',
             'skillopt_sha256':base.digest(self.root/'selected_skillopt.md'),'autosaddler_sha256':base.digest(self.root/'selected_autosaddler.md'),
             'test_model_calls_before_freeze':0})
        base.write(self.root/'PROGRESS.json',{'phase':'FULL_AIME2025_TEST','at_utc':base.now()})
        results={}
        for name,prompt in [('no_skill',''),('skillopt',skill),('autosaddler',auto)]:
            results[name]=self.batch(self.data['test'],prompt,name+'_test',self.root/'test'/name)
        results['chisel']=self.chisel(self.data['test'],skill,selected['threshold'],self.root/'test/chisel')
        if self.config.get('chisel_policy')=='proof_review_v1':
            results['chisel_draft']=base.read(self.root/'test/chisel/draft/RESULT.json')
            base.write(self.root/'test/chisel_draft/RESULT.json',results['chisel_draft'])
        methods={name:{'tasks':len(rows),'correct':sum(r['hard'] for r in rows),'accuracy_percent':100*sum(r['hard'] for r in rows)/len(rows),
                       'total_tokens':sum(r['usage']['total_tokens'] for r in rows),'rows_sha256':base.digest(self.root/'test'/name/'RESULT.json')}
                 for name,rows in results.items()}
        base.write(self.root/'RESULT.json',{'status':'COMPLETE_PENDING_INDEPENDENT_RESCORING','methods':methods,'model':self.config['model'],
                    'dataset':self.config['datasets']['2025'],'scorer_sha256':self.config['scorer_sha256'],'selection':selected,
                    'baseline_identity':'Pinned official SkillOpt and AutoSaddler engines with new integer-math adapters',
                    'chisel_identity':'Full derivation followed by proof review; frozen on 2024 development' if self.config.get('chisel_policy')=='proof_review_v1' else 'LiveMath answer-first confidence gate transferred to integer mathematics; selected on 2024 only',
                    'chisel_draft_guard':self.config.get('chisel_draft_guard')})


def run(config,root):
    readout_args=probability_readout_args(config)
    if (root/'STARTED.json').exists():raise FileExistsError('New experiment directory required')
    for name,sha in base.read(root/'SOURCE_FREEZE.json')['files_sha256'].items():
        if base.digest(ROOT/name)!=sha:raise RuntimeError('Frozen source changed: '+name)
    if base.digest(root/'DATA.json')!=base.read(root/'PREPARED.json')['dataset_hash']:raise RuntimeError('Data changed')
    packages={};resource_forks=[]
    for distribution in importlib.metadata.distributions():
        path=Path(str(getattr(distribution,'_path','')))
        if path.name.startswith('._'):
            resource_forks.append(str(path));continue
        metadata=distribution.metadata
        if metadata['Name']:packages[metadata['Name']]=distribution.version
    base.write(root/'STARTED.json',{'at_utc':base.now(),'command':sys.argv,'hostname':socket.gethostname(),'config':config,
           'packages':packages,'ignored_apple_double_resource_forks':resource_forks})
    stop=threading.Event();process=None;gateway=None;ledger=None
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with ExitStack() as stack:
        try:
            assets=Path(config['model']['assets']);manifest=assets/config['model']['manifest']
            if base.digest(manifest)!=config['model']['manifest_sha256']:raise RuntimeError('Model manifest changed')
            for row in base.read(manifest)['files']:
                if base.digest(assets/'model'/row['path'])!=row['sha256']:raise RuntimeError('Model file changed')
            gpu=base.select_gpu(config,root,stack,stop);service=root/'service';(service/'w').mkdir(parents=True)
            python=assets/'runtime/bin/python'
            command=[str(python),'-m','vllm.entrypoints.openai.api_server','--model',str(assets/'model'),'--served-model-name',config['model']['served_name'],
                     '--host','127.0.0.1','--port',str(config['resource']['model_port']),'--dtype','bfloat16','--tensor-parallel-size','1',
                     '--max-model-len','131072','--max-num-seqs',str(config['resource'].get('max_num_seqs',8)),'--gpu-memory-utilization','0.8','--generation-config','vllm',
                     '--max-num-batched-tokens','4096','--gdn-prefill-backend','triton','--default-chat-template-kwargs','{"enable_thinking":false}',
                     '--attention-config','{"backend":"FLASH_ATTN","flash_attn_version":3}','--enable-auto-tool-choice','--tool-call-parser','qwen3_xml',
                     '--language-model-only','--enforce-eager','--seed',str(config['seed'])]
            command.extend(readout_args)
            environment=base.runtime_environment(service,python,gpu_uuid=gpu['uuid'])
            if config.get('require_reproducible_inference'):
                from scripts.aime_reproducible_runtime import apply_reproducible_runtime,validate_reproducible_runtime
                command,environment=apply_reproducible_runtime(command,environment)
                environment['VLLM_SERVER_DEV_MODE']='1'
                base.write(root/'REPRODUCIBLE_RUNTIME.json',validate_reproducible_runtime(command,environment))
            base.write(service/'COMMAND.json',command)
            log=stack.enter_context((service/'service.log').open('x'));process=subprocess.Popen(command,cwd=service,
                env=environment,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            base.write(service/'OWNER.json',{'pid':process.pid,'command':command})
            deadline=time.time()+config['budget']['maximum_wall_seconds'];startup=time.time()+1800
            while True:
                if process.poll() is not None or time.time()>startup:raise RuntimeError('Owned model startup failed')
                try:
                    if base.get_json(f"http://127.0.0.1:{config['resource']['model_port']}/v1/models")['data']:break
                except Exception:time.sleep(2)
            if config.get('require_reproducible_inference'):
                from scripts.aime_reproducible_runtime import validate_server_info
                observed=base.get_json(f"http://127.0.0.1:{config['resource']['model_port']}/server_info?config_format=json")
                base.write(root/'RUNTIME_OBSERVED.json',validate_server_info(observed,model_path=str(assets/'model')))
            ledger=base.Ledger(root,config['budget'],131072,deadline)
            if config.get('inherit_provider_ledger'):
                prior=Path(config['inherit_provider_ledger'])
                if base.digest(prior/'PROVIDER_FINAL.json')!=config['prior_provider_sha256']:raise RuntimeError('Prior ledger changed')
                ledger.inherit(base.read(prior/'PROVIDER_FINAL.json'),prior)
            gateway=Gateway(config,root,ledger)
            threading.Thread(target=gateway.serve_forever,daemon=True).start()
            Study(config,root).execute()
        except BaseException as e:
            base.write(root/'FAILURE.json',{'at_utc':base.now(),'error':str(e),'traceback':traceback.format_exc()});raise
        finally:
            if gateway:gateway.shutdown();gateway.server_close()
            base.stop_owned(process)
            if ledger:base.write(root/'PROVIDER_FINAL.json',ledger.snapshot())
            base.write(root/'RETIREMENT.json',{'at_utc':base.now(),'gpu_state':base.gpu_snapshot()})
    audit(config,root)


def audit(config,root):
    result=base.read(root/'RESULT.json');ledger=base.read(root/'PROVIDER_FINAL.json')
    if ledger['unknown_usage_calls'] or ledger['inflight_reserved_tokens']:
        raise RuntimeError('Unresolved usage cannot support final results')
    total=0
    for row in ledger['calls']:
        raw=base.read(root/'provider'/f"call-{row['sequence']:06d}"/'response.json')
        usage=base.native_usage(raw,config['model']['served_name'],config['model']['max_model_len'])
        if usage!=row['usage']:raise RuntimeError('Native usage replay mismatch')
        total+=usage['total_tokens']
    if total!=ledger['known_tokens']:raise RuntimeError('Cumulative cost mismatch')
    if config.get('strict_optimizer_completion'):
        from phiagent.harness.aime_transport_recovery import autosaddler_completion,skillopt_completion
        locations=config.get('frozen_optimizer_roots',{})
        auto_root=Path(locations.get('autosaddler',root));skill_root=Path(locations.get('skillopt',root))
        path=auto_root/'autosaddler/aime2024-development/events.jsonl'
        proof=autosaddler_completion([json.loads(x) for x in path.read_text().splitlines() if x.strip()],4)
        if proof['status']!='PASS':raise RuntimeError('Incomplete official optimizer')
        if base.read(root/'AUTOSADDLER_OPTIMIZER_AUDIT.json')['events_sha256']!=base.digest(path):
            raise RuntimeError('Optimizer event ledger changed after selection')
        if skillopt_completion(skill_root/'skillopt',4)['status']!='PASS':raise RuntimeError('Incomplete SkillOpt optimizer')
    module=official(config);data=base.read(root/'DATA.json');items={r['id']:r for r in data['test']}
    for name,summary in result['methods'].items():
        path=root/'test'/name/'RESULT.json';rows=base.read(path)
        if base.digest(path)!=summary['rows_sha256'] or len(rows)!=30 or {r['id'] for r in rows}!=set(items):
            raise RuntimeError('Final result coverage or hash mismatch')
        correct=0
        for row in rows:
            hard,_=score_response(module,row['response'],items[row['id']]['answer'])
            if hard!=row['hard']:raise RuntimeError('Official scorer replay differs')
            correct+=hard
        if correct!=summary['correct']:raise RuntimeError('Final accuracy aggregation differs')
    if config.get('chisel_policy')=='proof_review_v1':
        drafts={r['id']:r for r in base.read(root/'test/chisel/draft/RESULT.json')}
        for row in base.read(root/'test/chisel/RESULT.json'):
            draft=drafts[row['id']]
            if row['draft_raw_response']!=draft['raw_response'] or row['draft_usage']!=draft['usage']:
                raise RuntimeError('Proof review is not bound to its recorded draft')
            if any(row['usage'][k]!=row['draft_usage'][k]+row['review_usage'][k] for k in base.USAGE_FIELDS):
                raise RuntimeError('Proof review cost omitted a stage')
            if (row['draft_usage']!=base.native_usage(draft['raw_response'],config['model']['served_name'],config['model']['max_model_len']) or
                row['review_usage']!=base.native_usage(row['raw_response'],config['model']['served_name'],config['model']['max_model_len'])):
                raise RuntimeError('Proof stage costs differ from raw provider responses')
    selected={name:base.digest(root/f'selected_{name}.md') for name in ('skillopt','autosaddler')}
    identical=selected['skillopt']==selected['autosaddler']
    result['selected_skill_sha256']=selected
    result['selected_skills_identical']=identical
    if config.get('strict_optimizer_completion'):
        result['optimizer_completion']={name:base.read(root/f'{name.upper()}_OPTIMIZER_AUDIT.json')
                                        for name in ('skillopt','autosaddler')}
        result['paper_admitted']=False
    if identical and config.get('require_reproducible_inference'):
        left=base.read(root/'test/skillopt/RESULT.json');right=base.read(root/'test/autosaddler/RESULT.json')
        if [(r['id'],r['response']) for r in left]!=[(r['id'],r['response']) for r in right]:
            raise RuntimeError('Identical selected skills produced different responses; final attribution blocked')
    if config.get('historical_cost_bindings'):
        historical=0
        for binding in config['historical_cost_bindings']:
            path=Path(binding['ledger_path'])
            if base.digest(path)!=binding['sha256']:raise RuntimeError('Historical cost ledger changed')
            prior=base.read(path)
            if prior['unknown_usage_calls'] or prior['inflight_reserved_tokens']:raise RuntimeError('Historical cost unresolved')
            historical+=prior['known_tokens']
        if config.get('reproducibility_cost_binding'):
            binding=config['reproducibility_cost_binding'];path=Path(binding['path'])
            if base.digest(path)!=binding['sha256']:raise RuntimeError('Reproducibility cost ledger changed')
            prior=base.read(path)
            if prior['unknown_usage_calls']:raise RuntimeError('Reproducibility cost unresolved')
            result['reproducibility_verification_tokens']=prior['usage']['total_tokens']
            historical+=prior['usage']['total_tokens']
        result['historical_campaign_tokens']=historical
        result['cumulative_current_and_historical_tokens']=historical+total
    result.update(status='COMPLETE_VERIFIED_FULL30_FOUR_METHOD',all_stage_tokens=total,all_stage_calls=len(ledger['calls']),unknown_usage_calls=0)
    base.write(root/'RESULT.json',result)
    base.write(root/'VERIFIED.json',{'status':'PASS','result_sha256':base.digest(root/'RESULT.json'),
               'raw_provider_responses':len(ledger['calls']),'official_rescored_outcomes':sum(r['tasks'] for r in result['methods'].values()),'all_stage_tokens':total,
               'data_sha256':base.digest(root/'DATA.json'),'scorer_sha256':config['scorer_sha256']})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','audit']);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=base.read(args.config);root=remote(config)
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENAI_API_KEY']='EMPTY'
    if args.action=='prepare':prepare(config,root)
    elif args.action=='run':run(config,root)
    else:audit(config,root)
if __name__=='__main__':main()
