#!/usr/bin/env python3
"""Remote matched native Codex / slow-authored bank / charged Jev campaign."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import argparse, fcntl, hashlib, json, os, secrets, socket, subprocess, sys, threading, time, traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phiagent.harness.codex_native_app_bank import POLICY, instruction, contract
from phiagent.harness.codex_gaia2 import mcp_runtime_settings
from scripts.serve_codex_deepseek import make_server, read_credential

ARMS = ('native', 'cot_only', 'cot_jev')
HOSTS = {'h200-1', 'h200-2', 'h20-1', 'h20-2', 'h20-3', 'h20-4', 'a800-4'}


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())


def author_settings(url):
    # Reuse the native feature policy without starting or advertising MCP tools.
    return mcp_runtime_settings(SimpleNamespace(base_url=url), 1800)['config_overrides']


def save(path, value, immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists(): raise ValueError('Preserve frozen artifact: ' + str(path))
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)


def aggregate(rows, ids, seeds, author_tokens):
    expected = {(i, s, a) for i in ids for s in seeds for a in ARMS}
    keys = [(r['task_id'], r['seed'], r['arm']) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError('Duplicate or unregistered actor')
    valid = lambda r: (r['valid'] is True and type(r['native_score']) is int
                       and r['native_score'] in (0, 1) and type(r['actor_tokens']) is int
                       and r['actor_tokens'] >= 0)
    out = {'expected': len(expected), 'settled': len(rows),
           'complete': set(keys) == expected and all(valid(r) for r in rows), 'arms': {}}
    for a in ARMS:
        sub = [r for r in rows if r['arm'] == a]
        complete = len(sub) == len(ids) * len(seeds) and all(valid(r) for r in sub)
        successes = sum(r['native_score'] == 1 for r in sub)
        cost = sum(r['actor_tokens'] for r in sub if type(r['actor_tokens']) is int)
        author = author_tokens if a != 'native' else 0
        out['arms'][a] = {'complete': complete, 'settled': len(sub), 'native_successes': successes,
            'native_success_percent': 100 * successes / len(sub) if complete else None,
            'actor_tokens': cost + author if complete else None,
            'known_actor_token_subtotal': cost, 'shared_bank_author_tokens': author,
            'native_fast_actions': sum(r.get('native_fast_actions', 0) for r in sub)}
    if out['complete']:
        own = out['arms']['cot_jev']
        controls = [out['arms'][a] for a in ('native', 'cot_only')]
        for percent, key in [(30, 'admission_30_percent_gate_passed'), (40, 'registered_40_percent_goal_passed')]:
            own[key] = all(own['native_successes'] >= c['native_successes']
                           and 100 * own['actor_tokens'] <= (100 - percent) * c['actor_tokens'] for c in controls)
    return out


def audit_bank_episode(result_path, arm):
    """Reconcile actual mixed actor receipts and literal original ARE actions."""
    result_path = Path(result_path); result = load(result_path)
    actor = result['provider_usage']['actor']
    if result['status'] != 'COMPLETED' or actor['complete'] is not True:
        raise ValueError('Native verdict or complete actor accounting is unresolved')
    provider = Path(result.get('outputs',{}).get('provider_receipts',result_path.parent/'private/provider'))
    receipts = [load(p) for p in sorted((provider/'actor').glob('call-*/receipt.json'))]
    if len(receipts) != actor['requests'] or any(not r.get('usage') for r in receipts):
        raise ValueError('Every physical actor call must have its receipt')
    for key in ('input_tokens', 'output_tokens', 'total_tokens'):
        if sum(r['usage'][key] for r in receipts) != actor['usage'][key]:
            raise ValueError('Fast/slow accounting differs from complete actor ledger')
    fast = [r for r in receipts if r.get('native_fast_decision')]
    if arm != 'cot_jev' and fast: raise ValueError('A native or CoT-only control executed a fast call')
    events = [json.loads(s) for s in Path(result['outputs']['public_observations']).read_text().splitlines()]
    banks = {}; decisions = []; proofs = []; last_native = None
    for e in events:
        if e.get('kind') == 'codex-bank/registered': banks[e['bank_sha256']] = e
        elif e.get('kind') == 'codex-bank/decision':
            packet = e['packet']; response = e['response']
            encoded = json.dumps(packet, ensure_ascii=False, separators=(',', ':')).encode()
            if hashlib.sha256(encoded).hexdigest() != e['packet_sha256']:
                raise ValueError('Public fast packet changed')
            actual = contract('decision', response=response,
                              labels=[c['label'] for c in packet['candidates']], model='deepseek-flash')
            if actual != e['decision']: raise ValueError('Typed decision differs from native response')
            decisions.append(e)
        elif 'tool' in e and 'arguments' in e and 'result' in e: last_native = e
        elif e.get('kind') == 'codex-bank/native-action':
            bank = banks[e['bank_sha256']]
            action = next(c for c in bank['candidates'] if c['label'] == e['label'])
            decision = decisions[-1]
            if (e['tool'] != action['tool'] or e['arguments'] != action['arguments']
                or decision['packet']['bank_sha256'] != e['bank_sha256']
                or decision['decision'] != e['decision'] or e['decision']['label'] != e['label']
                or last_native is None or last_native['tool'] != e['tool']
                or last_native['arguments'] != e['arguments'] or last_native['result'] != e['native_result']):
                raise ValueError('Fast action does not bind a declared candidate to its actual native execution')
            proofs.append(e)
    invalid = sum(e.get('kind') == 'codex-bank/invalid-fast' for e in events)
    if len(decisions) + invalid != len(fast): raise ValueError('Paid fast calls lack public decision records')
    if any(r['usage']['output_tokens'] != 1 or r['wire_api'] != 'chat/completions' for r in fast):
        raise ValueError('The fast mechanism differs from the accepted one-token native interface')
    return {'actor_tokens': actor['usage']['total_tokens'], 'native_fast_calls': len(fast),
            'native_fast_actions': len(proofs), 'native_fast_execution_proofs': proofs,
            'public_observations_sha256': sha(result['outputs']['public_observations']),
            'result_sha256': sha(result_path), 'benchmark_evidence': True,
            'native_clock': 'original NativeSurface.generation, included fast and slow', 'actor_replays': 0}


class Campaign:
    def __init__(self, root):
        if sys.platform != 'linux' or socket.gethostname() != 'harnessmm-api-worker' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('Use the authorized A800 CPU/API runtime')
        self.root = root; self.source = root/'source'; self.cfg = load(root/'CONFIG.json')
        self.hashes = load(root/'SOURCE_SHA256.json')
        for n, h in self.hashes.items():
            if sha(self.source/n) != h: raise ValueError('Frozen source changed: ' + n)
        self.parent = Path(self.cfg['parent_root']); self.parent_cfg = load(self.parent/'CONFIG.json')
        self.partition_path = Path(self.parent_cfg['partition_manifest']); self.partition = load(self.partition_path)
        if sha(self.partition_path) != '5549b4a72340ae1d4cc883e419dc9d947af17b7a3f69fd1d60950c17108bebec':
            raise ValueError('Keep the original 32/32/96 partition')
        self.base_freeze = load(self.parent/'FREEZE.json')
        if self.cfg['cumulative_token_limit'] is not None or self.cfg['token_usage_mode'] != 'unlimited':
            raise ValueError('No cumulative token termination')
        if self.cfg['shared_gaia2_prompt'] != self.parent_cfg['shared_gaia2_prompt']:
            raise ValueError('The capable native parent and both bank arms need the identical common prompt')
        accepted = load(Path(self.cfg['native_acceptance']))
        if accepted['status'] != 'VERIFIED_CODEX_NATIVE_APP_BANK_INTERFACE' or accepted['benchmark_calls'] != 0:
            raise ValueError('Native Codex bank interface acceptance required')
        accepted_root = Path(self.cfg['native_acceptance']).parent.parent
        accepted_hashes = load(accepted_root/'SOURCE_SHA256.json')
        for n in ('phiagent/harness/codex_gaia2.py', 'phiagent/harness/codex_native_app_bank.py',
                  'scripts/serve_codex_deepseek.py', 'integrations/deepseek_harness/codex_bank_contract.mjs',
                  'integrations/deepseek_harness/cot_jev.mjs', 'phiagent/harness/codex_runtime.py'):
            if self.hashes[n] != accepted_hashes[n]: raise ValueError('Accepted native execution changed: ' + n)
        self.rows = []; self.active = {}; self.phase = 'preparation'; self.author_tokens = 0

    def capacity(self):
        d = load(self.root/'POOL.json')
        return ({r['alias'] for r in d['snapshots']} == HOSTS
                and 0 <= time.time()-datetime.fromisoformat(d['at']).timestamp() <= 300
                and next(r for r in d['snapshots'] if r['alias']=='a800-4')['returncode'] == 0)

    def state(self, **extra):
        save(self.root/'STATE.json', {'at': now(), 'pid': os.getpid(), 'phase': self.phase,
            'active': list(self.active.values()), 'settled': len(self.rows),
            'test_opened': self.phase == 'test', 'cumulative_token_limit': None, **extra})

    def adopt_parent(self):
        report = load(self.parent/'COMPLETE.json'); originals = load(self.parent/'TRAIN_OUTCOMES.json')['rows']
        if not report['capability_gate_passed'] or report['settled'] != 32 or report['native_successes'] < 8:
            raise ValueError('Complete capable native TRAIN32 required')
        rows = []; profiles = []
        for old in originals:
            p = Path(old['result']); result = load(p)
            if (sha(p) != old['result_sha256'] or result['status'] != 'COMPLETED'
                or result['split'] != 'train' or result['scenario_id'] != old['scenario_id']
                or result['provider_usage']['actor'] != old['actor_usage']):
                raise ValueError('Retained original actor identity or verdict changed')
            actor = old['actor_usage']; score = old['native_score']
            if not actor['complete'] or type(score) is not int: raise ValueError('Parent evidence unresolved')
            row = {'phase': 'train', 'task_id': old['scenario_id'], 'seed': 20260928, 'arm': 'native',
                'valid': True, 'native_score': score, 'actor_tokens': actor['usage']['total_tokens'],
                'result': str(p), 'result_sha256': sha(p), 'retained_paid': True, 'native_fast_actions': 0,
                'original_execution_freeze_sha256': sha(p.parent/'private/FREEZE.json'), 'new_actor_calls': 0}
            rows.append(row)
            public_path = Path(result['outputs']['public_observations'])
            public = [json.loads(x) for x in public_path.read_text().splitlines()]
            public = [{k:e[k] for k in ('tool','arguments','result') if k in e}
                      for e in public if 'tool' in e and 'result' in e]
            excerpt = json.dumps(public[-8:], ensure_ascii=False)
            profiles.append({'task_id': row['task_id'], 'phase': 'train', 'native_score': score,
                'actor_tokens': row['actor_tokens'], 'public_native_trace': excerpt[-2400:],
                'trace_truncated': len(excerpt)>2400, 'public_native_trace_sha256': sha(public_path)})
        if (len(rows)!=32 or {r['task_id'] for r in rows} != set(self.partition['splits']['train'])
            or sum(r['native_score'] for r in rows)!=12 or sum(r['actor_tokens'] for r in rows)!=29026003):
            raise ValueError('Complete paid Codex parent does not reconcile')
        save(self.root/'ADOPTED_PARENT.json', {'rows': rows, 'original_source_sha256': sha(self.parent/'SOURCE_SHA256.json'),
            'original_freeze_sha256': sha(self.parent/'FREEZE.json'), 'native_actor_replays': 0,
            'source_change': 'prospective optional bank boundary; original parent retains its actual historical source'}, True)
        return rows, profiles

    def author(self, profiles):
        from phiagent.harness.codex_runtime import run_codex_trial
        from scripts.gaia_native_bank_training_revision import decode_author_json
        folder = self.root/'author'; folder.mkdir()
        prompt = {'objective': 'Author ONE reusable native-app bank skill using only complete TRAIN32 public observations. '
            'Slow CoT emits literal future native tool banks; the charged one-token public-state judge selects or abstains. '
            'Target native success at least the capable parent and CoT-only, with40% fewer complete actor tokens against both. '
            'Complete each public task obligation, verify each mutation once, avoid irrelevant extra searches or redundant checks. '
            'Use bounded native retrieval without omitting needed matches. Only known literal arguments enter future candidates; '
            'unknown IDs or unresolved semantics require slow CoT. Honor native notifications and deadlines. '
            'Follow the supplied native compiler declaration exactly. Never hardcode scenario answers or use hidden state, '
            'validation/test feedback, arbitrary token/turn ceilings or shorter context as a substitute for Jev. '
            'Return only JSON with instruction (nonempty <=3000 characters) and supporting_task_ids (unique nonempty TRAIN subset).',
            'accepted_native_bank_contract': instruction({**POLICY, 'enabled': True}),
            'common_parent_prompt': self.cfg['shared_gaia2_prompt'], 'TRAIN': profiles,
            'validation_or_test_feedback': False}
        save(folder/'PUBLIC_TRAIN_PROMPT.json', prompt, True)
        result = run_codex_trial(output=folder/'codex', prompt=json.dumps(prompt,ensure_ascii=False),
            trial_id=self.root.name+'-slow-bank-author', provider_url=self.url,
            relay_secret_file=self.secret_file, codex_binary=Path(self.parent_cfg['codex_bin']),
            timeout_seconds=1800, mcp_servers={}, config_overrides=author_settings(self.url))
        save(folder/'RESULT.json', result, True)
        usage = result['provider_usage']
        if (result['status']!='COMPLETED' or usage['usage_status']!='MEASURED'
            or usage['known_usage']['total_tokens'] <= 0):
            raise ValueError('Paid slow author unresolved; retain without replay')
        self.author_tokens = usage['known_usage']['total_tokens']
        value = decode_author_json(result['final_answer']); text = value.get('instruction'); support = value.get('supporting_task_ids')
        if (set(value)!={'instruction','supporting_task_ids'} or not isinstance(text,str) or not text.strip()
            or len(text)>3000 or not isinstance(support,list) or not support or len(support)!=len(set(support))
            or not set(support)<=set(self.partition['splits']['train'])):
            raise ValueError('Paid slow skill violates the TRAIN declaration contract')
        self.skill = folder/'skill.txt'; self.skill.write_text(text.strip()+'\n')
        save(self.root/'TRAIN_FROZEN_CANDIDATE.json', {'skill':str(self.skill),'skill_sha256':sha(self.skill),
            'author_result_sha256':sha(folder/'RESULT.json'),'complete_author_tokens':self.author_tokens,
            'public_train_prompt_sha256':sha(folder/'PUBLIC_TRAIN_PROMPT.json'),'training_ids':self.partition['splits']['train'],
            'validation_or_test_feedback':False,'source_manifest_sha256':sha(self.root/'SOURCE_SHA256.json'),
            'policy':POLICY,'selection_rule':'one eligible TRAIN-frozen candidate; prospective all_outcomes',
            'shared_author_charged_once_per_bank_comparison':True},True)

    def execute(self, phase, ident, seed, arm):
        folder = self.root/'trials'/phase/arm/str(seed)/ident; folder.mkdir(parents=True,exist_ok=False)
        policy = None if arm=='native' else {**POLICY,'enabled':arm=='cot_jev'}
        freeze = deepcopy(self.base_freeze)
        freeze.update(frozen_at_utc=now(),source_files=self.hashes,native_bank_policy=policy,
            bridge_source_files={str(self.source/'scripts/serve_codex_deepseek.py'):self.hashes['scripts/serve_codex_deepseek.py']},
            test_authorized=phase=='test',validation_authorized=phase=='validation',parent_only=False,
            policies={'plain_codex':{'residual_sha256':None},'chisel_codex':{'residual_sha256':sha(self.skill)}},
            candidate_sha256=sha(self.root/'TRAIN_FROZEN_CANDIDATE.json'),
            selected_policy_sha256=sha(self.root/'FROZEN_POLICY.json') if phase=='test' else None)
        freeze_path=folder/'FREEZE.json';save(freeze_path,freeze,True)
        output=folder/'episode';scenario=self.partition_path.parent/'scenarios'/(ident+'.json')
        argv=[sys.executable,'-B',str(self.source/'scripts/run_codex_gaia2.py'),
            '--scenario',str(scenario),'--filesystem',self.parent_cfg['filesystem'],'--output',str(output),
            '--codex-bin',self.parent_cfg['codex_bin'],'--provider-url',self.url,'--relay-secret-file',str(self.secret_file),
            '--trial-id',self.root.name+'-'+phase+'-'+arm+'-'+str(seed)+'-'+ident,
            '--partition-manifest',str(self.partition_path),'--freeze-manifest',str(freeze_path)]
        if policy is not None:argv+=['--native-bank-policy',json.dumps(policy),'--residual-file',str(self.skill)]
        save(folder/'INTENT.json',{'phase':phase,'arm':arm,'task_id':ident,'registered_execution_seed':seed,
            'provider_sampling_seed_supported':False,'argv':argv,'source_manifest_sha256':sha(self.root/'SOURCE_SHA256.json'),
            'freeze_sha256':sha(freeze_path),'cumulative_token_limit':None},True)
        with(folder/'PROCESS.log').open('xb')as log:
            p=subprocess.Popen(argv,cwd=self.source,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                env={**os.environ,'PYTHONPATH':str(self.source),'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
            save(folder/'OWNER.json',{'pid':p.pid,'at':now()},True);rc=p.wait()
        result_path=output/'RESULT.json';result=load(result_path)
        if result.get('status')=='BLOCKED' and str(result.get('blocker','')).endswith(': Responses escaped the native tool capability boundary'):
            result_path=self.score_closed_model_failure(result_path,folder/'native-score-recovery',phase)
            result=load(result_path)
        actor=(result.get('provider_usage')or{}).get('actor')or{}
        valid=result['status']=='COMPLETED'and actor.get('complete')is True
        row={'phase':phase,'arm':arm,'task_id':ident,'seed':seed,'valid':valid,'native_score':result.get('native_score'),
            'actor_tokens':(actor.get('usage')or{}).get('total_tokens'),'result':str(result_path),
            'result_sha256':sha(result_path),'returncode':rc,'native_fast_actions':0,'actor_replays':0}
        if valid:
            try:
                audit=audit_bank_episode(result_path,arm);save(folder/'NATIVE_COT_JEV_AUDIT.json',audit,True)
                row['native_fast_actions']=audit['native_fast_actions']
            except Exception as error:
                row.update(valid=False,audit_unresolved=str(error))
                save(folder/'NATIVE_AUDIT_UNRESOLVED.json',{'error':str(error),'actor_replays':0,
                    'complete_paid_actor_tokens':row['actor_tokens'],'native_score':row['native_score']},True)
        save(folder/'OUTCOME.json',row,True);return row

    def score_closed_model_failure(self,result_path,folder,phase):
        from scripts.recover_codex_gaia_native_judge import verified_native_model_failure, admitted_paid_actor
        old=Path(result_path).parent;result=load(result_path);actor=load(old/'private/codex-process.json')
        proof=verified_native_model_failure(old,result,actor)
        admitted_paid_actor(result,actor,load(old/'ADMISSION.json'),native_model_failure=True,phase=phase)
        retained=self.cfg.get('retained_native_model_scores',{}).get(sha(result_path))
        if retained:
            receipt_path=Path(retained['receipt']);receipt=load(receipt_path)
            recovered_path=Path(receipt['result']);recovered=load(recovered_path)
            recovery=recovered['native_scorer_recovery'];native=load(Path(recovered['outputs']['native_result']))
            if (sha(receipt_path)!=retained['sha256'] or receipt['status']!='SCORED'
                or sha(recovered_path)!=receipt['result_sha256'] or receipt['new_actor_calls']!=0
                or recovery['new_actor_calls']!=0 or recovery['original_trial']!=str(old)
                or recovery['paid_result_sha256']!=sha(result_path)
                or recovery['retained_trace_sha256']!=sha(Path(load(old/'private/native-result.json')['export_path']))
                or recovered['split']!=phase or recovered['status']!='COMPLETED'
                or recovered['provider_usage']['actor']!=result['provider_usage']['actor']
                or type(native['success'])is not bool or native['exception']is not None
                or int(native['success'])!=recovered['native_score']):
                raise ValueError('Previously executed native scorer does not bind the unchanged paid actor')
            return recovered_path
        if not self.capacity():
            while not self.capacity():time.sleep(15)
        folder.mkdir(parents=True,exist_ok=False);trace=Path(load(old/'private/native-result.json')['export_path'])
        save(folder/'CONFIG.json',{'paid_trial':str(old),'paid_result_sha256':sha(result_path),
            'retained_trace_sha256':sha(trace),'credential_file':self.parent_cfg['credential_file'],
            'allow_closed_native_model_failure':True,'phase':phase,'model_failure_proof':proof,
            'cumulative_token_limit':None,'new_actor_calls':0},True)
        save(folder/'SOURCE_SHA256.json',self.hashes,True);save(folder/'POOL.json',load(self.root/'POOL.json'),True)
        argv=[sys.executable,'-B',str(self.source/'scripts/recover_codex_gaia_native_judge.py'),'--root',str(folder)]
        with(folder/'PROCESS.log').open('xb')as log:
            proc=subprocess.Popen(argv,cwd=self.source,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                env={**os.environ,'PYTHONPATH':str(self.source),'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
            save(folder/'SCORER_LAUNCH.json',{'pid':proc.pid,'new_actor_calls':0,'at':now()},True);code=proc.wait()
        receipt=load(folder/'RECOVERY.json') if(folder/'RECOVERY.json').exists() else {}
        if code or receipt.get('status')!='SCORED' or receipt.get('new_actor_calls')!=0:
            raise RuntimeError('Retained native model-failure scorer unresolved; do not replay paid actor')
        recovered=Path(receipt['result'])
        if sha(recovered)!=receipt['result_sha256']:raise ValueError('Scored native result changed')
        return recovered

    def adopt_bank_train(self):
        predecessor=Path(self.cfg['adopt_bank_train_from']);old_cfg=load(predecessor/'CONFIG.json')
        hashes=load(predecessor/'SOURCE_SHA256.json')
        for n in ('phiagent/harness/codex_gaia2.py','phiagent/harness/codex_native_app_bank.py',
                  'scripts/serve_codex_deepseek.py','phiagent/harness/codex_runtime.py',
                  'integrations/deepseek_harness/cot_jev.mjs','integrations/deepseek_harness/codex_bank_contract.mjs'):
            if hashes[n]!=self.hashes[n]:raise ValueError('Retained actor execution changed: '+n)
        if old_cfg['shared_gaia2_prompt']!=self.cfg['shared_gaia2_prompt']:
            raise ValueError('Retained common native prompt changed')
        candidate=load(predecessor/'TRAIN_FROZEN_CANDIDATE.json');self.skill=Path(candidate['skill'])
        author=load(predecessor/'author/RESULT.json')
        if (sha(self.skill)!=candidate['skill_sha256'] or sha(predecessor/'author/RESULT.json')!=candidate['author_result_sha256']
            or author['status']!='COMPLETED' or author['provider_usage']['usage_status']!='MEASURED'
            or author['provider_usage']['known_usage']['total_tokens']!=candidate['complete_author_tokens']
            or candidate['validation_or_test_feedback']is not False
            or set(candidate['training_ids'])!=set(self.partition['splits']['train'])):
            raise ValueError('Paid TRAIN-derived bank authorship changed')
        self.author_tokens=candidate['complete_author_tokens']
        save(self.root/'TRAIN_FROZEN_CANDIDATE.json',{**candidate,'source_manifest_sha256':sha(self.root/'SOURCE_SHA256.json'),
            'original_candidate_sha256':sha(predecessor/'TRAIN_FROZEN_CANDIDATE.json'),
            'retained_paid_author_once':True,'new_author_calls':0,'adopted_from':str(predecessor)},True)
        rows=[]
        for old in load(predecessor/'TRAIN_OUTCOMES.json'):
            if old['phase']!='train' or sha(old['result'])!=old['result_sha256']:
                raise ValueError('Retained paid TRAIN result changed')
            row=deepcopy(old)
            if not old['valid']:
                folder=self.root/'retained-native-scores'/old['arm']/str(old['seed'])/old['task_id']
                result_path=self.score_closed_model_failure(Path(old['result']),folder,'train');result=load(result_path)
                audit=audit_bank_episode(result_path,old['arm']);save(folder/'NATIVE_COT_JEV_AUDIT.json',audit,True)
                if result['provider_usage']['actor']['usage']['total_tokens']!=old['actor_tokens']:
                    raise ValueError('Retained original actor costs changed')
                row.update(valid=True,native_score=result['native_score'],result=str(result_path),result_sha256=sha(result_path),
                    native_fast_actions=audit['native_fast_actions'],original_result_sha256=old['result_sha256'],new_actor_calls=0)
            row['retained_paid']=True;rows.append(row)
        # Population/duplicate guards execute before launching any unopened actor.
        aggregate(rows,self.partition['splits']['train'],[20260928],self.author_tokens)
        save(self.root/'ADOPTED_BANK_TRAIN.json',{'rows':rows,'predecessor':str(predecessor),
            'new_author_calls':0,'actor_replays':0,'source_manifest_sha256':sha(predecessor/'SOURCE_SHA256.json')},True)
        return rows

    def stage(self,phase,ids,seeds,retained=()):
        self.phase=phase;rows=list(retained);self.rows+=rows
        expected=[{'phase':phase,'task_id':i,'seed':s,'arm':a}for i in ids for s in seeds for a in ARMS]
        done_keys={(r['task_id'],r['seed'],r['arm'])for r in rows}
        pending=[r for r in expected if(r['task_id'],r['seed'],r['arm'])not in done_keys];blocked=False
        with ThreadPoolExecutor(max_workers=self.cfg['executor_workers'])as ex:
            while pending or self.active:
                while pending and len(self.active)<self.cfg['executor_workers']and not blocked and self.capacity():
                    request=pending.pop(0);future=ex.submit(self.execute,phase,request['task_id'],request['seed'],request['arm']);self.active[future]=request
                self.state(stage=aggregate(rows,ids,seeds,self.author_tokens))
                if not self.active:
                    if blocked:raise RuntimeError('Paid native actor unresolved; preserve unopened slots and repair without replay')
                    time.sleep(15);continue
                completed,_=wait(self.active,timeout=15,return_when=FIRST_COMPLETED)
                for f in completed:
                    r=f.result();self.active.pop(f);rows.append(r);self.rows.append(r);blocked|=not r['valid']
                    save(self.root/(phase.upper()+'_OUTCOMES.json'),rows)
        report=aggregate(rows,ids,seeds,self.author_tokens);save(self.root/(phase.upper()+'_COMPARISON.json'),report,True)
        if not report['complete']:raise RuntimeError('Complete native population required')
        return report

    def run(self):
        lock=(self.root/'OWNER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(self.root/'OWNER.json').exists():raise ValueError('Adopt paid owner; do not restart')
        save(self.root/'OWNER.json',{'pid':os.getpid(),'at':now(),'controller_sha256':sha(__file__)},True)
        parent,profiles=self.adopt_parent()
        self.secret_file=self.root/'relay.secret';self.secret_file.write_text(secrets.token_hex(32)+'\n');self.secret_file.chmod(0o600)
        provider=self.root/'provider';provider.mkdir()
        relay=make_server(provider,read_credential(Path(self.parent_cfg['credential_file'])),read_credential(self.secret_file),
            bind='127.0.0.1',port=0,allow_native_fast=True)
        threading.Thread(target=relay.serve_forever,daemon=True).start();self.url='http://127.0.0.1:'+str(relay.server_port)
        try:
            while not self.capacity():self.state(status='WAITING_FRESH_POOL');time.sleep(15)
            if self.cfg.get('adopt_bank_train_from'):
                self.phase='retained_native_train_scoring';self.state();retained=self.adopt_bank_train()
            else:
                self.phase='train_author';self.state();self.author(profiles);retained=parent
            train=self.stage('train',self.partition['splits']['train'],[20260928],retained)
            if train['arms']['cot_jev']['native_fast_actions']==0:
                raise RuntimeError('Complete TRAIN has no native fast execution; revise from TRAIN before protected evaluation')
            validation=self.stage('validation',self.partition['splits']['validation'],[20260928])
            save(self.root/'FROZEN_POLICY.json',{'at':now(),'candidate_sha256':sha(self.root/'TRAIN_FROZEN_CANDIDATE.json'),
                'train_comparison_sha256':sha(self.root/'TRAIN_COMPARISON.json'),
                'validation_comparison_sha256':sha(self.root/'VALIDATION_COMPARISON.json'),
                'selection_rule':'single eligible TRAIN-frozen native-bank candidate; all_outcomes reporting',
                'source_manifest_sha256':sha(self.root/'SOURCE_SHA256.json'),'partition_sha256':sha(self.partition_path),
                'test_execution_seeds':[20260928,20260929,20260930],'policy':POLICY,'cumulative_token_limit':None},True)
            test=self.stage('test',self.partition['splits']['test'],[20260928,20260929,20260930])
            save(self.root/'COMPLETE_TEST.json',test,True);self.state(status='COMPLETE_NATIVE_TEST',stage=test)
        finally:relay.shutdown();relay.server_close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:Campaign(a.root.resolve()).run()
    except BaseException as error:
        save(a.root/'BLOCKED.json',{'error':str(error),'traceback':traceback.format_exc(),'actor_replays':0});raise


if __name__=='__main__':main()
