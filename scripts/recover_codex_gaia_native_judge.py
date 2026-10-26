#!/usr/bin/env python3
"""Run the unchanged native judge on an already settled Codex TRAIN trace."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
import re
from pathlib import Path
import secrets
import socket
import sys
import tempfile
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phiagent.harness import codex_gaia2 as native
from scripts.serve_codex_deepseek import make_server, read_credential


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admitted_paid_actor(result, actor, admission, *, native_model_failure=False, phase='train'):
    """An evaluator failure cannot authorize another actor execution."""
    judge_failure = str(result.get('blocker', '')).startswith('judge request ')
    model_failure = (native_model_failure is True
        and re.fullmatch(r'actor request [1-9][0-9]*: Responses escaped the native tool capability boundary',
                         str(result.get('blocker', ''))) is not None
        and actor.get('status') == 'STOPPED' and actor.get('stop_reason') == 'infrastructure_failure')
    if (result.get('status') != 'BLOCKED'
            or not (judge_failure or model_failure)
            or phase not in ('train','validation','test') or admission.get('split') != phase
            or actor.get('status') not in ('COMPLETED', 'STOPPED')):
        raise ValueError('Require a settled TRAIN actor blocked only by its judge')
    measured = actor.get('provider_usage', {})
    accounted = result.get('provider_usage', {}).get('actor', {})
    usage = measured.get('known_usage') or {}
    if (measured.get('usage_status') != 'MEASURED'
            or measured.get('unresolved_call_ids')
            or measured.get('failed_calls')
            or measured.get('provider_models') != ['deepseek-flash']
            or accounted.get('complete') is not True
            or measured.get('known_usage') != accounted.get('usage')
            or any(type(usage.get(field)) is not int or usage[field] < 0
                   for field in ('input_tokens', 'output_tokens', 'total_tokens'))
            or usage['input_tokens'] + usage['output_tokens'] != usage['total_tokens']):
        raise ValueError('Paid actor must have complete matching native provider accounting')
    return accounted


def verified_native_model_failure(old, result, actor):
    """A closed, paid undeclared model tool has no executed native mutation."""
    match = re.fullmatch(r'actor request ([1-9][0-9]*): Responses escaped the native tool capability boundary',
                         str(result.get('blocker', '')))
    if match is None:
        raise ValueError('Only a declared native model boundary failure is eligible')
    native_result = read(old/'private/native-result.json')
    if native_result['exception'] != result['blocker'] or native_result['success'] not in (None, False):
        raise ValueError('Original native failure does not match the settled actor')
    folder = old/'private/provider/actor'/('call-'+match[1].zfill(6))
    receipt = read(folder/'receipt.json'); request = folder/'request.json'
    response = folder/('response.sse' if read(request).get('stream') else 'response.json')
    if (receipt['status']!='FAILED' or receipt.get('http_status')!=200
        or receipt.get('native_fast_decision') is True or not receipt.get('usage')
        or sha(request)!=receipt['request_sha256']):
        raise ValueError('Native model failure lacks a complete original paid response')
    raw = native.completed_response(response.read_bytes()) if response.suffix=='.sse' else read(response)
    if raw.get('status')!='completed' or native.normalized_usage(raw['usage'],'actor')!=receipt['usage']:
        raise ValueError('The native model failure is a provider or accounting failure')
    physical=[(Path(p),read(Path(p))) for p in actor['provider_usage']['receipts']]
    physical=[(p,r) for p,r in physical if r.get('response_id')==raw.get('id')
              and r.get('request_sha256')==sha(request) and r.get('response_sha256')==sha(response)
              and r.get('usage_status')=='MEASURED']
    if len(physical)!=1:
        raise ValueError('The rejected model response lacks its unique original physical provider binding')
    from phiagent.harness.codex_native_app_bank import native_name
    tools = {t['name'] for t in read(old/'private/native-tools.json')['tools']}
    calls = [x for x in raw.get('output',[]) if x.get('type')=='function_call']
    rejected = [x for x in calls if native_name(x.get('name'),x.get('namespace')) not in tools]
    if not rejected: raise ValueError('No undeclared model-generated tool was rejected')
    events = [json.loads(x) for x in (old/'private/public-observations.jsonl').read_text().splitlines()]
    if any(e.get('tool') in {x.get('name') for x in rejected} for e in events):
        raise ValueError('An undeclared tool execution cannot enter judge-only recovery')
    return {'original_paid_response_sha256':sha(response), 'original_receipt_sha256':sha(folder/'receipt.json'),
        'physical_provider_receipt':str(physical[0][0]),'physical_provider_receipt_sha256':sha(physical[0][0]),
        'rejected_model_tool_names':[x.get('name') for x in rejected],
        'undeclared_tools_executed':False,'new_actor_calls':0,'actor_tokens':actor['provider_usage']['known_usage']['total_tokens']}


def combined_evaluator_usage(before, after):
    if any(row.get('complete') is not True for row in (before, after)):
        raise ValueError('Prior and recovered evaluator fees must both be complete')
    usage = {field: sum(row['usage'][field] for row in (before, after))
        for field in ('input_tokens', 'output_tokens', 'total_tokens')}
    if usage['input_tokens'] + usage['output_tokens'] != usage['total_tokens']:
        raise ValueError('Evaluator fees do not reconcile')
    return {'complete': True, 'usage': usage, 'known_usage_subtotal': usage,
        'requests': before['requests'] + after['requests'], 'cost_usd': None,
        'cost_status': 'not_reported', 'components': [before, after],
        'prior_failed_calls_retained': before.get('failed_calls', []),
        'actor_tokens_included': False}


def run(root):
    if sys.platform != 'linux' or socket.gethostname() != native.REMOTE_HOST:
        raise RuntimeError('Native recovery is remote-only on the authorized A800 host')
    if any(name.startswith('are.simulation') for name in sys.modules):
        raise RuntimeError('A fresh native scorer process is required')
    cfg = read(root / 'CONFIG.json')
    old = Path(cfg['paid_trial'])
    previous, actor, admission = (read(old / 'RESULT.json'),
        read(old / 'private/codex-process.json'), read(old / 'ADMISSION.json'))
    original = read(old / 'CONFIG.json')
    failure_proof = (verified_native_model_failure(old, previous, actor)
                     if cfg.get('allow_closed_native_model_failure') is True else None)
    phase=cfg.get('phase','train')
    if phase=='test' and read(old/'private/FREEZE.json').get('test_authorized') is not True:
        raise ValueError('Retained TEST requires its original pre-TEST execution freeze')
    paid = admitted_paid_actor(previous, actor, admission, native_model_failure=failure_proof is not None, phase=phase)
    trace = Path(read(old / 'private/native-result.json')['export_path'])
    if sha(trace) != cfg['retained_trace_sha256'] or sha(old / 'RESULT.json') != cfg['paid_result_sha256']:
        raise ValueError('Frozen paid evaluator input changed')
    if sha(Path(original['partition_manifest'])) != admission['partition_sha256']:
        raise ValueError('Frozen TRAIN partition changed')
    if sha(Path(original['scenario'])) != admission['scenario_sha256']:
        raise ValueError('Frozen original native scenario changed')
    pool = read(root / 'POOL.json')
    if ({r['alias'] for r in pool['snapshots']} !=
        {'h200-1', 'h200-2', 'h20-1', 'h20-2', 'h20-3', 'h20-4', 'a800-4'}
        or not 0 <= datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(pool['at']).timestamp() <= 300):
        raise ValueError('Require a fresh complete authorized compute-pool audit')
    for name, digest in read(root / 'SOURCE_SHA256.json').items():
        if sha(ROOT / name) != digest:
            raise ValueError('Scorer recovery source changed: ' + name)
    os.umask(0o077)
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHON_DOTENV_DISABLED='1')
    secret_file = root / 'relay.secret'
    with secret_file.open('x') as stream:
        stream.write(secrets.token_hex(32) + '\n')
    server = make_server(root / 'provider', read_credential(Path(cfg['credential_file'])),
        read_credential(secret_file), bind='127.0.0.1', port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    config = native.RunConfig(
        scenario=Path(original['scenario']), filesystem=Path(original['filesystem']),
        output=root, codex_bin=Path(original['codex_bin']),
        provider_url='http://127.0.0.1:' + str(server.server_address[1]),
        freeze_manifest=Path(original['freeze_manifest']),
        partition_manifest=Path(original['partition_manifest']), model=original['model'],
        residual_file=Path(original['residual_file']) if original.get('residual_file') else None,
        relay_secret_file=secret_file, trial_id='native-scorer-' + root.name,
        native_bank_policy=original.get('native_bank_policy'))
    recorder = native.ProviderRecorder(root / 'private/provider', config)
    storage = root / 'private/native-storage'
    storage.mkdir(parents=True)
    tempfile.tempdir = str(storage)
    os.environ.update(TMPDIR=str(storage), FS_PATH=str(storage / 'fs-states'),
        DEMO_FS_PATH=str(config.filesystem.resolve()))
    logging.basicConfig(filename=root / 'private/native.log', level=logging.INFO, force=True)
    try:
        if importlib.metadata.version('meta-agents-research-environments') != '1.1.0':
            raise ValueError('The retained scorer requires the pinned ARE 1.1.0')
        from are.simulation.agents.agent_config_builder import AgentConfigBuilder
        from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMEngine, LiteLLMModelConfig
        from are.simulation.benchmark.scenario_loader import load_scenario
        from are.simulation.scenario_runner import ScenarioRunner
        from are.simulation.scenarios.config import ScenarioRunnerConfig
        from are.simulation.scenarios.scenario_imported_from_json.utils import get_scenario_duration, preprocess_scenario
        from are.simulation.validation.configs import GraphPerEventJudgeConfig

        class Judge(LiteLLMEngine):
            def chat_completion(self, messages, stop_sequences=(), **_kwargs):
                converted = [self._convert_message_to_litellm_format(item) for item in messages]
                body = {'model': config.model, 'stream': False, 'temperature': 0,
                    'max_output_tokens': 4096, 'input': native.judge_responses_input(converted)}
                _, raw, usage = native.metered_native_judge_request(recorder, body)
                text = native.judge_response_text(native.responses_text(raw), stop_sequences)
                return text, {'prompt_tokens': usage['input_tokens'],
                    'completion_tokens': usage['output_tokens'], 'total_tokens': usage['total_tokens']}

        class NoActorBuilder:
            def build(self, *_args, **_kwargs):
                raise RuntimeError('Judge-only recovery must never instantiate an actor')

        class NativeConfigBuilder(AgentConfigBuilder):
            def build(self, agent_name, **kwargs):
                if agent_name != 'codex_mcp':
                    raise RuntimeError('Retained harness identity changed')
                return super().build(agent_name='default', **kwargs)

        retained_scenario, replay = load_scenario(trace.read_text(), str(trace), load_completed_events=True)
        # Exported traces contain initialized CONDITION events. Build the native
        # oracle from the original frozen definition, and import only the exact
        # completed actor events from the export. No tool mutation is replayed.
        scenario, unused = load_scenario(config.scenario.read_text(), str(config.scenario),
            load_completed_events=False)
        if (scenario is None or retained_scenario is None or replay is None or unused
                or scenario.scenario_id != admission['scenario_id']
                or retained_scenario.scenario_id != scenario.scenario_id):
            raise ValueError('Retained native trace does not identify the frozen scenario')
        runner_config = ScenarioRunnerConfig(agent='codex_mcp', model=config.model,
            model_provider='local', export=True, output_dir=str(root / 'private/native-traces'),
            simulated_generation_time_mode='measured', max_turns=None, judge_only=True)
        judge = Judge(LiteLLMModelConfig(model_name=config.model, provider='local',
            endpoint=recorder.trial_url('judge')))
        preprocess_scenario(scenario, judge_config=GraphPerEventJudgeConfig(engine=judge),
            max_scenario_duration=get_scenario_duration(scenario,
                runner_config.max_time_scenario_duration, runner_config.max_scenario_duration),
            offline_validation=True)
        with (root / 'private/native-stdout.log').open('x') as log, redirect_stdout(log):
            scored = ScenarioRunner(agent_builder=NoActorBuilder(),
                agent_config_builder=NativeConfigBuilder()).run(runner_config, scenario,
                    completed_events=replay)
        if scored.exception is not None or type(scored.success) is not bool or recorder.failure:
            raise RuntimeError('Native retained judge did not yield a valid verdict: ' +
                str(scored.exception or recorder.failure))
        costs = recorder.snapshot()
        if costs['actor']['requests'] != 0 or not costs['judge']['complete']:
            raise ValueError('Recovery incurred actor calls or unresolved evaluator fees')
        result = {**previous, 'status': 'COMPLETED', 'native_score': int(scored.success),
            'native_success': scored.success, 'split': phase, 'scenario_id': scenario.scenario_id,
            'blocker': None, 'error_type': None,
            'outputs': {'native_result': str(root / 'NATIVE_RESULT.json'),
                'native_traces': str(root / 'private/native-traces'),
                'actor_events': actor['events_path'], 'public_observations': str(old / 'private/public-observations.jsonl'),
                'actor_run': actor['run_directory'], 'actor_workspace': actor['workspace'],
                'provider_receipts': str(old / 'private/provider')},
            'provider_usage': {'actor': paid,
                'judge': combined_evaluator_usage(previous['provider_usage']['judge'], costs['judge'])},
            'native_scorer_recovery': {'original_trial': str(old), 'paid_result_sha256': sha(old / 'RESULT.json'),
                'retained_trace': str(trace), 'retained_trace_sha256': sha(trace),
                'new_actor_calls': 0, 'new_judge_usage': costs['judge'],
                'prior_failed_judge_usage_retained': previous['provider_usage']['judge'],
                'judge_prompt_parser_model_and_output_limit_unchanged': True,
                'native_model_boundary_failure':failure_proof}}
        native.write_json(root / 'NATIVE_RESULT.json', {'success': scored.success,
            'exception': None, 'rationale': scored.rationale, 'export_path': scored.export_path})
        native.write_json(root / 'RECOVERED_RESULT.json', result)
        native.write_json(root / 'RECOVERY.json', {'status': 'SCORED', 'native_score': int(scored.success),
            'result': str(root / 'RECOVERED_RESULT.json'), 'result_sha256': sha(root / 'RECOVERED_RESULT.json'),
            'actor_tokens': paid['usage']['total_tokens'], 'new_actor_calls': 0,
            'new_judge_usage': costs['judge'], 'retained_trace_sha256': sha(trace),
            'source_sha256': sha(root / 'SOURCE_SHA256.json')})
    finally:
        native.write_json(root / 'PROVIDER_ACCOUNTING.json', recorder.snapshot())
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    with (root / 'OWNER.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'OWNER.json').exists():
            raise RuntimeError('Adopt the existing native scorer rather than restarting it')
        native.write_json(root / 'OWNER.json', {'pid': os.getpid(), 'at': native.now()})
        try:
            run(root)
        except BaseException as error:
            native.write_json(root / 'FAILURE.json', {'error': str(error),
                'traceback': traceback.format_exc(), 'new_actor_calls': 0})
            raise


if __name__ == '__main__':
    main()
