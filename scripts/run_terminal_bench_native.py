#!/usr/bin/env python3
"""Freeze and launch the matched full89 campaign through native Harbor trials."""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_terminal_bench_solpi_campaign import build_trial_plan, validate_protocol


MAX_REQUESTS = 76000
MAX_TOTAL_TOKENS = 2500000000
MAX_REQUEST_TOKENS = 32768
MAX_SECONDS = 172800
PREINFERENCE_RUNTIME_SOURCES = {
    'renderers/params.py': '3a551bcbda5d18c3258ab4a4c17db953e97c733badab9bd2157ff62726c73442',
    'entrypoints/openai/chat_completion/serving.py': '6239fae503211193942d0e2037f7b0edcf71ed70271604701283bfc0453d202b',
    'renderers/online_renderer.py': '0bf1a5620fe090a305ac21f7987a77f41a591c24186e5fe521abed432e44cded',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(target):
    value = hashlib.sha256()
    with Path(target).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write(target, value):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, prefix=target.name + '.', suffix='.pending', delete=False) as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        pending = Path(stream.name)
    pending.replace(target)


def trial_config(row, *, root: Path, source: Path, binding: Path, binding_sha: str, task_metadata: dict,
                 verifier_https_proxy=None, verifier_artifact_cache=None, verifier_apt_cache=None):
    timeout = task_metadata['official_task_config']['agent']['timeout_sec']
    identity = hashlib.sha256(f"{root}:{row['arm']}:{row['task_id']}:{row['repeat_seed']}".encode()).hexdigest()[:12]
    return {
        'task': {'path': str(source / row['task_id']), 'source': 'terminal-bench@2.0'},
        'trial_name': row['task_id'][:40] + '__' + identity + '-s' + str(row['repeat_seed']),
        'trials_dir': str(root / 'jobs' / row['arm']), 'timeout_multiplier': 1.0,
        'agent': {'import_path': 'scripts.harbor_terminal_agent:TerminalAgent',
                  'model_name': 'Qwen/Qwen3.8-27B', 'kwargs': {
                      'binding_path': str(binding), 'binding_sha256': binding_sha, 'arm': row['arm'],
                      'repeat_seed': row['repeat_seed'], 'task_id': row['task_id'],
                      'official_agent_timeout': timeout}},
        'environment': {'type': 'docker', 'force_build': False, 'delete': True,
                        'cpu_enforcement_policy': 'limit', 'memory_enforcement_policy': 'limit'},
        'verifier': {'env': verifier_proxy_environment(verifier_https_proxy, verifier_artifact_cache, verifier_apt_cache)} if verifier_https_proxy else {},
    }


def verifier_proxy_environment(proxy, artifact_cache=None, apt_cache=None):
    if proxy != 'http://172.17.0.1:18994':
        raise ValueError('verifier HTTPS egress must use the campaign-owned Docker bridge')
    environment = {'HTTPS_PROXY': proxy, 'https_proxy': proxy,
                   'NO_PROXY': 'localhost,127.0.0.1,::1', 'no_proxy': 'localhost,127.0.0.1,::1'}
    if artifact_cache is not None:
        if artifact_cache != 'http://172.17.0.1:18996':
            raise ValueError('verifier artifacts must use the hash-pinned campaign cache')
        environment.update(UV_INSTALLER_GITHUB_BASE_URL=artifact_cache + '/github',
                           UV_PYTHON_INSTALL_MIRROR=artifact_cache + '/github/astral-sh/python-build-standalone/releases/download',
                           UV_FIND_LINKS=artifact_cache + '/wheels/')
        environment['NO_PROXY'] += ',172.17.0.1'
        environment['no_proxy'] = environment['NO_PROXY']
    if apt_cache is not None:
        if apt_cache != 'http://172.17.0.1:18997':
            raise ValueError('APT downloads must use the campaign-owned private cache')
        environment.update(http_proxy=apt_cache, HTTP_PROXY=apt_cache)
    return environment


def verify_apt_cache_receipt(target):
    receipt = json.loads(Path(target).read_text())
    if (receipt.get('status') != 'READY_OFFICIAL_APT_CACHE' or receipt.get('bind') != '172.17.0.1:18997'
            or receipt.get('source_sha256') != sha256(ROOT / 'scripts/serve_terminal_apt_cache.py')
            or receipt.get('upstream_tls_verified') is not True
            or receipt.get('apt_signature_checks_disabled') is not False):
        raise ValueError('APT cache identity or signature policy is not admitted')
    return receipt


def verify_dependency_cache_receipt(receipt_path):
    receipt = json.loads(Path(receipt_path).read_text())
    if (receipt.get('status') != 'READY_HASH_VERIFIED_OFFICIAL_ARTIFACTS'
            or receipt.get('bind') != '172.17.0.1:18996'
            or receipt.get('source_sha256') != sha256(ROOT / 'scripts/serve_terminal_dependency_cache.py')
            or not receipt.get('artifacts')):
        raise ValueError('dependency cache identity is not admitted')
    for record in receipt['artifacts'].values():
        target = Path(record['path'])
        if (target.is_symlink() or not target.is_file() or target.stat().st_size != record['size']
                or sha256(target) != record['sha256']):
            raise ValueError('dependency cache artifact changed: ' + str(target))
    return receipt


def verifier_infrastructure_diagnostics(native_result_path):
    root = Path(native_result_path).parent
    patterns = ('uvx: command not found', 'uv: command not found', 'failed to download https://github.com/astral-sh/',
                'Verifier execution timed out', 'No matching distribution found', 'Failed to download cpython-',
                'Failed to download distribution due to network timeout')
    diagnostics = []
    for relative in ('verifier/test-stdout.txt', 'verifier/test-stderr.txt', 'trial.log'):
        target = root / relative
        if target.is_file():
            text = target.read_text(errors='replace')
            for pattern in patterns:
                if pattern in text:
                    diagnostics.append({'path': str(target), 'sha256': sha256(target), 'matched_error': pattern})
    return diagnostics


def record_has_verified_reward(recorded):
    return scorable_native_result(recorded.get('result', {})) and not recorded.get('infrastructure_diagnostics')


def scorable_native_result(native):
    verifier = native.get('verifier_result')
    rewards = verifier.get('rewards') if isinstance(verifier, dict) else None
    if not isinstance(rewards, dict) or not rewards:
        return False
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in rewards.values()):
        return False
    exception = native.get('exception_info')
    return not exception or exception.get('exception_type') == 'AgentTimeoutError'


def preserve_completed_trials(plan, completed, *, allow_unscored=False):
    expected = {row['trial_id']: row for row in plan}
    if len(expected) != len(plan):
        raise ValueError('frozen trial plan contains duplicate identities')
    retained = {}
    for row in completed:
        trial_id = row.get('trial_id')
        if trial_id not in expected or trial_id in retained:
            raise ValueError('completed trial identity is missing, duplicated or outside the plan')
        if any(row.get(key) != expected[trial_id][key] for key in ('arm', 'task_id', 'repeat_seed')):
            raise ValueError('completed trial metadata differs from the frozen plan')
        native = row.get('result')
        if allow_unscored and not isinstance(native, dict):
            error = row.get('controller_error')
            if not isinstance(error, dict) or not isinstance(error.get('type'), str) or not isinstance(error.get('error'), str):
                raise ValueError('unscored trial must retain its native result or explicit controller failure')
        if not allow_unscored and not scorable_native_result(native or {}):
            raise ValueError('completed trial lacks a scorable official result; do not omit or silently rerun it')
        retained[trial_id] = deepcopy(row)
    return [retained[row['trial_id']] for row in plan if row['trial_id'] in retained]


def load_preserved_results(prior_root, plan, binding, protocol, *, allow_unscored=False):
    if json.loads((prior_root / 'actor-binding.json').read_text()) != binding:
        raise ValueError('continuation must retain the exact actor and provider configuration')
    if json.loads((prior_root / 'PROTOCOL.json').read_text()) != protocol:
        raise ValueError('continuation cannot change the frozen benchmark protocol')
    completed = []
    existing = prior_root / 'PRESERVED_RESULTS.json'
    if existing.is_file():
        completed.extend(json.loads(existing.read_text())['rows'])
    result_lines = prior_root / 'TRIAL_RESULTS.jsonl'
    if result_lines.is_file():
        completed.extend(json.loads(line) for line in result_lines.read_text().splitlines() if line.strip())
    recorded_count = json.loads((prior_root / 'RESULT.json').read_text())['completed_trials']
    if recorded_count != len(completed):
        raise ValueError('prior result coverage differs from its recorded completion count')
    rows = preserve_completed_trials(plan, completed, allow_unscored=allow_unscored)
    sources = []
    for row in rows:
        if not row.get('native_result_path') and row.get('controller_error'):
            sources.append({'path': str(result_lines), 'sha256': sha256(result_lines)})
            continue
        target = Path(row['native_result_path'])
        value = json.loads(target.read_text())
        for key in ('id', 'trial_name', 'task_name', 'verifier_result', 'exception_info', 'agent_result'):
            if value.get(key) != row['result'].get(key):
                raise ValueError('recorded native trial differs from its saved result artifact')
        sources.append({'path': str(target), 'sha256': sha256(target)})
        diagnostics = verifier_infrastructure_diagnostics(target)
        if diagnostics:
            row['infrastructure_diagnostics'] = diagnostics
    return {'prior_root': str(prior_root), 'rows': rows, 'source_artifacts': sources,
            'official_rewards_unchanged': True, 'completed_trials_reexecuted': 0,
            'unscored_trials_retained': sum(not record_has_verified_reward(row) for row in rows)}


def trial_blocks_queue(recorded, *, continuous=False):
    return not continuous and not scorable_native_result(recorded.get('result', {}))


def campaign_result_status(results, expected_count, *, stopped):
    if stopped or len(results) != expected_count:
        return 'BLOCKED'
    if any(not record_has_verified_reward(row) for row in results):
        return 'COMPLETED_WITH_UNRESOLVED_TRIALS'
    return 'COMPLETE'


def validate_preinference_contract(target):
    contract = json.loads(Path(target).read_text())
    if (contract.get('status') != 'PINNED_PREINFERENCE_CONTEXT_VALIDATION_VERIFIED'
            or contract.get('model') != 'qwen38-27b-dense-ms-e823e888'
            or contract.get('revision') != '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
            or contract.get('vllm_version') != '0.25.1+cu129'
            or contract.get('source_files_sha256') != PREINFERENCE_RUNTIME_SOURCES
            or contract.get('rejection_before_engine_generate') is not True):
        raise ValueError('preinference rejection requires the verified pinned runtime contract')
    return contract


def preinference_context_rejection(failure, request):
    if (not isinstance(failure, dict) or failure.get('type') != 'HTTPError'
            or failure.get('upstream_status') != 400
            or request.get('model') != 'qwen38-27b-dense-ms-e823e888'):
        return None
    try:
        error = json.loads(failure['body'])['error']
    except (KeyError, TypeError, ValueError):
        return None
    if (not isinstance(error, dict) or not isinstance(error.get('message'), str)
            or error.get('type') != 'BadRequestError'
            or error.get('param') != 'input_tokens' or error.get('code') != 400):
        return None
    match = re.fullmatch(
        r"This model's maximum context length is (\d+) tokens\. However, you requested (\d+) "
        r'output tokens and your prompt contains (at least )?(\d+) input tokens, for a total of '
        r'(at least )?(\d+) tokens\. Please reduce the length of the input prompt or the number '
        r'of requested output tokens\. \(parameter=input_tokens, value=(\d+)\)',
        error.get('message', ''))
    if match is None:
        return None
    context, output, qualifier, prompt, total_qualifier, total, value = match.groups()
    context, output, prompt, total, value = map(int, (context, output, prompt, total, value))
    if (context != 32768 or output != request.get('max_tokens') or prompt != value
            or prompt + output != total or total <= context or qualifier != total_qualifier):
        return None
    return {'category': 'pinned_vllm_preinference_context_rejection', 'model_inference_tokens': 0,
            'submitted_prompt_tokens_lower_bound': prompt, 'prompt_count_is_exact': qualifier is None,
            'requested_output_tokens': output, 'native_usage_returned': False,
            'raw_failure_preserved': True}


class UsageLedger:
    def __init__(self, prior=None, *, bounded_unknown_attempts=0):
        prior = prior or {}
        if type(bounded_unknown_attempts) is not int or bounded_unknown_attempts < 0:
            raise ValueError('bounded historical unknown attempts must be a nonnegative integer')
        if bounded_unknown_attempts and (
                prior.get('unknown_usage_attempts') != bounded_unknown_attempts
                or prior.get('inflight') != 0
                or prior.get('attempts') != prior.get('responses_with_counts', 0)
                    + prior.get('preinference_rejections', 0) + bounded_unknown_attempts
                or prior.get('total_tokens') != prior.get('input_tokens', 0) + prior.get('output_tokens', 0)):
            raise ValueError('bounded historical usage requires an exact closed prior ledger')
        if prior and not bounded_unknown_attempts and (
                not prior.get('complete') or prior.get('inflight') or prior.get('unknown_usage_attempts')):
            raise ValueError('prior usage must be resolved before resubmission')
        self.lock = threading.Lock()
        self.drained = threading.Event()
        self.drained.set()
        self.attempts = prior.get('attempts', 0)
        self.prior_attempts = self.attempts
        self.inflight = 0
        self.responses = prior.get('responses_with_counts', 0)
        self.input_tokens = prior.get('input_tokens', 0)
        self.output_tokens = prior.get('output_tokens', 0)
        self.unknown = bounded_unknown_attempts
        self.bounded_unknown_attempts = bounded_unknown_attempts
        self.preinference_rejections = prior.get('preinference_rejections', 0)
        self.rejected_prompt_lower_bound = prior.get('rejected_prompt_tokens_lower_bound', 0)

    def reserve(self):
        with self.lock:
            if self.unknown > self.bounded_unknown_attempts or self.attempts >= MAX_REQUESTS:
                raise RuntimeError('campaign provider admission closed')
            if self.input_tokens + self.output_tokens + (
                    self.inflight + self.preinference_rejections + self.bounded_unknown_attempts + 1
            ) * MAX_REQUEST_TOKENS > MAX_TOTAL_TOKENS:
                raise RuntimeError('campaign token reservation exhausted')
            self.attempts += 1
            self.inflight += 1
            self.drained.clear()
            return self.attempts

    def finish(self, response, *, rejection=None):
        with self.lock:
            self.inflight -= 1
            if self.inflight == 0:
                self.drained.set()
            if rejection is not None:
                if response is not None or rejection.get('category') != 'pinned_vllm_preinference_context_rejection':
                    self.unknown += 1
                    return
                self.preinference_rejections += 1
                self.rejected_prompt_lower_bound += rejection['submitted_prompt_tokens_lower_bound']
                return
            usage = response.get('usage') if isinstance(response, dict) else None
            if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0
                                                  for key in ('prompt_tokens', 'completion_tokens')):
                self.unknown += 1
                return
            self.responses += 1
            self.input_tokens += usage['prompt_tokens']
            self.output_tokens += usage['completion_tokens']
            if 'total_tokens' in usage and usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
                self.unknown += 1

    def snapshot(self):
        with self.lock:
            return {'at_utc': now(), 'attempts': self.attempts, 'inflight': self.inflight,
                    'prior_attempts': self.prior_attempts,
                    'responses_with_counts': self.responses, 'input_tokens': self.input_tokens,
                    'output_tokens': self.output_tokens, 'total_tokens': self.input_tokens + self.output_tokens,
                    'preinference_rejections': self.preinference_rejections,
                    'rejected_prompt_tokens_lower_bound': self.rejected_prompt_lower_bound,
                    'rejection_budget_reservation_tokens': self.preinference_rejections * MAX_REQUEST_TOKENS,
                    'bounded_unknown_usage_attempts': self.bounded_unknown_attempts,
                    'unknown_budget_reservation_tokens': self.bounded_unknown_attempts * MAX_REQUEST_TOKENS,
                    'unbounded_unknown_usage_attempts': self.unknown - self.bounded_unknown_attempts,
                    'budget_accounting_complete': self.unknown == self.bounded_unknown_attempts and not self.inflight,
                    'all_attempts_returned_native_usage': not self.preinference_rejections and not self.unknown and not self.inflight,
                    'unknown_usage_attempts': self.unknown, 'complete': not self.unknown and not self.inflight}


def reconcile_provider_usage(prior_root):
    original = json.loads((prior_root / 'PROVIDER_FINAL.json').read_text())
    previous = json.loads((prior_root / 'EXECUTION.json').read_text()).get('prior_provider_usage')
    ledger = UsageLedger(previous)
    evidence = []
    rejected = []
    for directory in sorted((prior_root / 'provider').glob('call-*')):
        call_id = ledger.reserve()
        if directory.name != f'call-{call_id:06d}':
            raise ValueError('provider raw call coverage is not consecutive')
        request_path = directory / 'request.json'
        request = json.loads(request_path.read_text())
        evidence.append({'path': str(request_path), 'sha256': sha256(request_path)})
        response_path = directory / 'response.json'
        if response_path.is_file():
            ledger.finish(json.loads(response_path.read_text()))
            evidence.append({'path': str(response_path), 'sha256': sha256(response_path)})
        else:
            failure_path = directory / 'failure.json'
            failure = json.loads(failure_path.read_text())
            rejection = preinference_context_rejection(failure, request)
            if rejection is None:
                raise ValueError('provider failure is not a verified preinference context rejection')
            ledger.finish(None, rejection=rejection)
            evidence.append({'path': str(failure_path), 'sha256': sha256(failure_path)})
            rejected.append({'call_id': call_id, **rejection})
    resolved = ledger.snapshot()
    fields = ('attempts', 'inflight', 'responses_with_counts', 'input_tokens', 'output_tokens', 'total_tokens')
    newly_resolved = resolved['preinference_rejections'] - original.get('preinference_rejections', 0)
    if (any(resolved[key] != original[key] for key in fields) or not resolved['complete']
            or original['unknown_usage_attempts'] != newly_resolved):
        raise ValueError('provider reconciliation changed a known debit or left unresolved usage')
    return {'status': 'PRIOR_PROVIDER_USAGE_RECONCILED', 'at_utc': now(),
            'original_final_path': str(prior_root / 'PROVIDER_FINAL.json'),
            'original_final_sha256': sha256(prior_root / 'PROVIDER_FINAL.json'),
            'original_usage': original, 'resolved_usage': resolved,
            'preinference_rejections': rejected, 'source_artifacts': evidence,
            'raw_responses_modified': False, 'model_requests_repeated': 0}


def bound_prior_timeout_usage(prior_root, call_ids):
    prior_root = Path(prior_root)
    original_path = prior_root / 'PROVIDER_FINAL.json'
    original = json.loads(original_path.read_text())
    execution_path = prior_root / 'EXECUTION.json'
    previous = json.loads(execution_path.read_text()).get('prior_provider_usage') or {}
    if (not call_ids or len(set(call_ids)) != len(call_ids)
            or any(type(call_id) is not int or call_id < 1 for call_id in call_ids)
            or original.get('inflight') != 0 or original.get('unknown_usage_attempts') != len(call_ids)
            or previous.get('unknown_usage_attempts', 0)):
        raise ValueError('timeout carry requires exactly the authorized closed unknown attempts')
    UsageLedger(original, bounded_unknown_attempts=len(call_ids))
    totals = {key: previous.get(key, 0) for key in
              ('input_tokens', 'output_tokens', 'responses_with_counts', 'preinference_rejections')}
    sources = [{'path': str(target), 'sha256': sha256(target)} for target in (original_path, execution_path)]
    timeouts = []
    sequence = previous.get('attempts', 0)
    for directory in sorted((prior_root / 'provider').glob('call-*')):
        sequence += 1
        if directory.name != f'call-{sequence:06d}':
            raise ValueError('timeout carry requires every consecutive prior call')
        request_path = directory / 'request.json'
        payload = json.loads(request_path.read_text())
        if (payload.get('model') != 'qwen38-27b-dense-ms-e823e888'
                or type(payload.get('max_tokens')) is not int or not 1 <= payload['max_tokens'] <= 3072
                or payload.get('stream')):
            raise ValueError('timeout request is outside the pinned inference bound')
        sources.append({'path': str(request_path), 'sha256': sha256(request_path)})
        response_path = directory / 'response.json'
        if response_path.is_file():
            response = json.loads(response_path.read_text())
            usage = response.get('usage', {})
            if sequence in call_ids or any(type(usage.get(key)) is not int or usage[key] < 0
                    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
                raise ValueError('timeout carry cannot overwrite returned native usage')
            if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
                raise ValueError('native usage arithmetic differs during timeout carry')
            totals['input_tokens'] += usage['prompt_tokens']
            totals['output_tokens'] += usage['completion_tokens']
            totals['responses_with_counts'] += 1
            sources.append({'path': str(response_path), 'sha256': sha256(response_path)})
            continue
        failure_path = directory / 'failure.json'
        failure = json.loads(failure_path.read_text())
        sources.append({'path': str(failure_path), 'sha256': sha256(failure_path)})
        if sequence in call_ids:
            if failure.get('type') != 'TimeoutError' or failure.get('message') != 'timed out':
                raise ValueError('only authorized response timeouts may remain bounded unknowns')
            timeouts.append({'call_id': sequence, 'native_usage': None,
                             'reservation_tokens': MAX_REQUEST_TOKENS, 'failure_path': str(failure_path)})
        elif preinference_context_rejection(failure, payload) is not None:
            totals['preinference_rejections'] += 1
        else:
            raise ValueError('an additional unknown provider failure is not authorized')
    if (sequence != original['attempts'] or {row['call_id'] for row in timeouts} != set(call_ids)
            or any(totals[key] != original[key] for key in totals)):
        raise ValueError('timeout carry changed known usage or omitted a request')
    return {'status': 'AUTHORIZED_BOUNDED_HISTORICAL_TIMEOUT_USAGE', 'at_utc': now(),
            'original_final_path': str(original_path), 'original_final_sha256': sha256(original_path),
            'original_usage': original, 'timeouts': timeouts, 'source_artifacts': sources,
            'unknown_budget_reservation_tokens': len(timeouts) * MAX_REQUEST_TOKENS,
            'raw_responses_modified': False, 'model_requests_repeated': 0,
            'unknown_usage_resolved': False}


def timeout_carry_allowance(root, execution):
    expected = execution.get('prior_timeout_carry_sha256')
    if not expected:
        return 0
    target = Path(root) / 'PRIOR_TIMEOUT_CARRY.json'
    if sha256(target) != expected:
        raise ValueError('historical timeout carry changed')
    receipt = json.loads(target.read_text())
    prior = execution['prior_provider_usage']
    count = len(receipt['timeouts'])
    if (receipt['status'] != 'AUTHORIZED_BOUNDED_HISTORICAL_TIMEOUT_USAGE'
            or receipt['original_usage'] != prior or receipt['unknown_usage_resolved'] is not False
            or count != prior['unknown_usage_attempts'] or len({row['call_id'] for row in receipt['timeouts']}) != count
            or any(row['native_usage'] is not None or row['reservation_tokens'] != MAX_REQUEST_TOKENS for row in receipt['timeouts'])
            or receipt['unknown_budget_reservation_tokens'] != count * MAX_REQUEST_TOKENS
            or sha256(receipt['original_final_path']) != receipt['original_final_sha256']):
        raise ValueError('historical timeout carry is not bound to the original unknown debit')
    for source in receipt['source_artifacts']:
        if sha256(source['path']) != source['sha256']:
            raise ValueError('historical timeout source artifact changed')
    UsageLedger(prior, bounded_unknown_attempts=count)
    return count


def start_proxy(root: Path, upstream: str, port: int, prior=None, *, verified_preinference=False,
                bounded_unknown_attempts=0, durable_provider=False, capture_recovery_seconds=None):
    if capture_recovery_seconds is not None and (
            not durable_provider or type(capture_recovery_seconds) not in (int, float)
            or not math.isfinite(capture_recovery_seconds) or capture_recovery_seconds <= 0):
        raise ValueError('Capture recovery requires a durable provider and a finite request lifetime')
    ledger = UsageLedger(prior, bounded_unknown_attempts=bounded_unknown_attempts)
    opener = build_opener(ProxyHandler({}))
    provider = root / 'provider'
    provider.mkdir(exist_ok=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_error(404)
                return
            reserved = False
            response = None
            rejection = None
            try:
                count = int(self.headers.get('Content-Length', '0'))
                if not 0 < count <= 8 * 1024 * 1024:
                    raise ValueError('request size outside frozen bounds')
                payload = json.loads(self.rfile.read(count))
                if payload.get('stream'):
                    raise ValueError('this capture boundary requires non-streaming requests')
                call_id = ledger.reserve()
                reserved = True
                directory = provider / f'call-{call_id:06d}'
                directory.mkdir()
                write(directory / 'request.json', payload)
                write(directory / 'started.json', {'at_utc': now(), 'upstream': upstream, 'path': self.path})
                write(provider / 'LEDGER.json', ledger.snapshot())
                headers = {'Content-Type': 'application/json'}
                if durable_provider:
                    request_id = hashlib.sha256(str(root).encode()).hexdigest()[:16] + f'-{call_id:06d}'
                    headers['X-PhiAgent-Request-Id'] = request_id
                    write(directory / 'durable-request.json', {'request_id': request_id,
                          'capture_lookup': upstream + '/captures/' + request_id})
                request = Request(upstream + self.path, data=json.dumps(payload).encode(),
                                  headers=headers, method='POST')
                recovery_deadline = time.monotonic() + (capture_recovery_seconds or 0)
                try:
                    timeout = (min(60, capture_recovery_seconds) if capture_recovery_seconds
                               else 1230 if durable_provider else 310)
                    with opener.open(request, timeout=timeout) as http_response:
                        raw = http_response.read()
                except HTTPError:
                    raise
                except (TimeoutError, ConnectionError, URLError):
                    if capture_recovery_seconds is None:
                        raise
                    from scripts.terminal_durable_recovery import recover_captured_response

                    status, body, receipt = recover_captured_response(
                        opener, upstream, request_id, payload['model'], deadline=recovery_deadline,
                    )
                    write(directory / 'CAPTURE_RECOVERY.json', receipt)
                    raw = json.dumps(body, ensure_ascii=False).encode()
                    if status >= 400:
                        raise HTTPError(upstream + self.path, status, 'captured native failure',
                                        {}, io.BytesIO(raw))
                response = json.loads(raw)
                write(directory / 'response.json', response)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except Exception as error:
                detail = {'type': type(error).__name__, 'message': str(error), 'at_utc': now()}
                if isinstance(error, HTTPError):
                    detail['upstream_status'] = error.code
                    detail['body'] = error.read().decode(errors='replace')[:10000]
                    error.close()
                if reserved:
                    write(directory / 'failure.json', detail)
                    rejection = preinference_context_rejection(detail, payload) if verified_preinference else None
                    if rejection is not None:
                        write(directory / 'preinference-rejection.json', rejection)
                try:
                    self.send_error(502, 'captured provider failure')
                except (BrokenPipeError, ConnectionResetError):
                    pass
            finally:
                if reserved:
                    ledger.finish(response, rejection=rejection)
                    write(provider / 'LEDGER.json', ledger.snapshot())

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = not durable_provider
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, ledger


def provider_checks(root: Path, endpoint: str, model: str):
    from scripts.run_terminal_bench_qwen35_4b_direct import EXEC_TOOL
    from scripts.terminal_solpi import TerminalSoLPiSession

    directory = root / 'provider-admission'
    directory.mkdir()
    session = TerminalSoLPiSession({'observation_pack': True, 'action_fusion': True}, directory)
    tools = session.tools([EXEC_TOOL])
    requests = [
        ('exec_in_sandbox', {'command': 'printf synthetic', 'validation_command': 'test 1 = 1'}),
        ('ObsRecall', {'observation_id': 'obs_' + 'a' * 64, 'offset': 0, 'limit': 10}),
    ]
    receipts = []
    opener = build_opener(ProxyHandler({}))
    for index, (name, arguments) in enumerate(requests):
        payload = {'model': model, 'messages': [{'role': 'user', 'content':
            'This is a synthetic interface check. Call ' + name + ' with exactly these arguments: ' + json.dumps(arguments)}],
            'tools': tools, 'tool_choice': 'auto', 'parallel_tool_calls': False,
            'max_tokens': 512, 'temperature': 0, 'seed': 0, 'chat_template_kwargs': {'enable_thinking': False}}
        with opener.open(Request(endpoint, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'}, method='POST'), timeout=300) as response:
            raw = json.load(response)
        write(directory / f'response-{index}.json', raw)
        calls = raw['choices'][0]['message'].get('tool_calls', [])
        if len(calls) != 1 or calls[0]['function']['name'] != name or json.loads(calls[0]['function']['arguments']) != arguments:
            raise RuntimeError('provider does not preserve the frozen tool interface: ' + name)
        if raw.get('model') != model or not all(type(raw.get('usage', {}).get(key)) is int
                                              for key in ('prompt_tokens', 'completion_tokens')):
            raise RuntimeError('provider model or exact token usage admission failed')
        receipts.append({'tool': name, 'arguments_equal': True, 'usage': raw['usage']})
    write(directory / 'PASS.json', {'status': 'PASS', 'requests': receipts, 'benchmark_calls': 0})


def prepare(args):
    from harbor.models.trial.config import TrialConfig

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cache_receipt = getattr(args, 'verifier_cache_receipt', None)
    if bool(args.verifier_artifact_cache) != bool(cache_receipt):
        raise ValueError('dependency cache URL and verified receipt must be bound together')
    if cache_receipt:
        verify_dependency_cache_receipt(cache_receipt)
        shutil.copy2(cache_receipt, root / 'DEPENDENCY_CACHE_READY.json')
    apt_cache = getattr(args, 'verifier_apt_cache', None)
    apt_receipt = getattr(args, 'verifier_apt_receipt', None)
    if bool(apt_cache) != bool(apt_receipt):
        raise ValueError('APT cache URL and verified receipt must be bound together')
    if apt_receipt:
        verify_apt_cache_receipt(apt_receipt)
        shutil.copy2(apt_receipt, root / 'APT_CACHE_READY.json')
    rejection_receipt = getattr(args, 'provider_rejection_receipt', None)
    if rejection_receipt:
        validate_preinference_contract(rejection_receipt)
        shutil.copy2(rejection_receipt, root / 'PROVIDER_REJECTION_CONTRACT.json')
    prior = None
    bounded_unknown_attempts = 0
    timeout_ids = getattr(args, 'retain_timed_out_usage', None)
    if args.prior_run:
        prior_result = json.loads((args.prior_run / 'RESULT.json').read_text())
        if prior_result['status'] != 'BLOCKED':
            raise ValueError('only an explicitly blocked infrastructure attempt may be superseded')
        prior = json.loads((args.prior_run / 'PROVIDER_FINAL.json').read_text())
        if timeout_ids:
            if not rejection_receipt or not args.preserve_completed or not getattr(args, 'durable_provider', False):
                raise ValueError('timeout recovery requires preserved trials and durable response capture')
            carry = bound_prior_timeout_usage(args.prior_run, timeout_ids)
            write(root / 'PRIOR_TIMEOUT_CARRY.json', carry)
            bounded_unknown_attempts = len(carry['timeouts'])
        elif not prior.get('complete') and rejection_receipt:
            reconciliation = reconcile_provider_usage(args.prior_run)
            reconciliation['runtime_contract_sha256'] = sha256(root / 'PROVIDER_REJECTION_CONTRACT.json')
            write(root / 'PRIOR_USAGE_RECONCILIATION.json', reconciliation)
            prior = reconciliation['resolved_usage']
        UsageLedger(prior, bounded_unknown_attempts=bounded_unknown_attempts)
        write(root / 'PRIOR_ATTEMPT.json', {'root': str(args.prior_run), 'result_sha256': sha256(args.prior_run / 'RESULT.json'),
              'provider_final_sha256': sha256(args.prior_run / 'PROVIDER_FINAL.json'), 'usage': prior,
              'reused_trials': 0, 'failed_trials_preserved': True})
    parent = args.preparation.resolve()
    protocol = json.loads((parent / 'frozen/protocol.json').read_text())
    validate_protocol(protocol)
    actors = json.loads((parent / 'actor-variants.json').read_text())
    manifest = json.loads((parent / 'frozen/manifest.json').read_text())
    images = json.loads(args.images_receipt.read_text())
    service = json.loads(args.service_receipt.read_text())
    if service.get('status') != 'READY' or service.get('revision') != protocol['model']['revision']:
        raise ValueError('new pinned model service is not admitted for configuration binding')
    if rejection_receipt and service.get('preinference_validation_source_sha256') != PREINFERENCE_RUNTIME_SOURCES:
        raise ValueError('live service is not bound to the verified preinference runtime')
    if getattr(args, 'durable_provider', False) and (
            service.get('durable_capture', {}).get('schema') != 'terminal-provider-durable-response/v1'
            or service['durable_capture'].get('response_saved_before_delivery') is not True):
        raise ValueError('durable provider capture was not attested by the live service')
    if images['official_images_present'] != 89 or not images['all_new_images_digest_and_rootfs_rechecked']:
        raise ValueError('complete official images are not attested')
    if sha256(parent / 'frozen/manifest.json') != images['full89_manifest_sha256']:
        raise ValueError('source and installed image inventories differ')
    for row in manifest['files']:
        if sha256(args.task_source / row['path']) != row['sha256']:
            raise ValueError('official source changed: ' + row['path'])
    for row in images['images']:
        observed = json.loads(subprocess.check_output(['docker', 'image', 'inspect', row['image']], text=True))[0]
        if observed['Id'] != row['id']:
            raise ValueError('official installed image changed: ' + row['image'])
    model = json.loads((parent / 'frozen/no_skill.json').read_text())['model']
    model.update(chat_completions_url=f'http://127.0.0.1:{args.port}/v1/chat/completions',
                 models_url=args.upstream + '/v1/models', request_timeout_seconds=330)
    for key in ('service_dir', 'service_files_sha256', 'service_kind', 'tool_transport'):
        model.pop(key, None)
    binding = root / 'actor-binding.json'
    write(binding, {'model': model, 'actors': actors, 'identity': 'Native Harbor adapter, frozen selected parents; no new outer-loop search',
                    'parent_preparation_sha256': sha256(parent / 'PREPARATION.json')})
    task_metadata = {row['task_id']: row for row in manifest['tasks']}
    plan = build_trial_plan(manifest['expected_task_ids'], protocol)
    preserved = {'rows': [], 'source_artifacts': []}
    maximum_seconds = MAX_SECONDS
    campaign_deadline = None
    if args.preserve_completed:
        if args.prior_run is None:
            raise ValueError('preserving completed trials requires an explicit prior run')
        preserved = load_preserved_results(args.prior_run, plan, json.loads(binding.read_text()), protocol,
                          allow_unscored=args.continuous)
        previous_execution = json.loads((args.prior_run / 'EXECUTION.json').read_text())
        previous_start = datetime.fromisoformat(json.loads((args.prior_run / 'STARTED.json').read_text())['at_utc'])
        campaign_deadline = previous_execution.get('campaign_deadline_utc') or (
            previous_start + timedelta(seconds=previous_execution['maximum_seconds'])).isoformat()
        deadline_receipt = args.prior_run.parent / 'SERVICE_DEADLINE_COVERAGE.json'
        if deadline_receipt.is_file():
            original_deadline = json.loads(deadline_receipt.read_text())['original_campaign_deadline_utc']
            campaign_deadline = min(datetime.fromisoformat(campaign_deadline),
                                    datetime.fromisoformat(original_deadline)).isoformat()
            shutil.copy2(deadline_receipt, root / 'PRIOR_DEADLINE_COVERAGE.json')
        maximum_seconds = math.floor((datetime.fromisoformat(campaign_deadline) - datetime.now(timezone.utc)).total_seconds())
        if maximum_seconds <= 0:
            raise ValueError('original campaign wall budget is exhausted')
        previous = json.loads((root / 'PRIOR_ATTEMPT.json').read_text())
        previous.update(reused_trials=len(preserved['rows']), completed_trials_reexecuted=0)
        write(root / 'PRIOR_ATTEMPT.json', previous)
    write(root / 'PRESERVED_RESULTS.json', preserved)
    binding_sha = sha256(binding)
    configs = []
    for row in plan:
        config = TrialConfig.model_validate(trial_config(row, root=root, source=args.task_source,
            binding=binding, binding_sha=binding_sha, task_metadata=task_metadata[row['task_id']],
            verifier_https_proxy=args.verifier_https_proxy, verifier_artifact_cache=args.verifier_artifact_cache,
            verifier_apt_cache=apt_cache))
        path = root / 'trial-configs' / row['arm'] / (row['task_id'] + f"__seed-{row['repeat_seed']}.json")
        write(path, config.model_dump(mode='json'))
        configs.append({**row, 'config_path': str(path), 'config_sha256': sha256(path)})
    write(root / 'TRIAL_PLAN.json', configs)
    write(root / 'PROTOCOL.json', protocol)
    write(root / 'PACKAGES.json', {name: importlib.metadata.version(name) for name in ('harbor','litellm','pydantic','terminal-bench')})
    shutil.copy2(args.images_receipt, root / 'IMAGE_VERIFICATION.json')
    shutil.copy2(args.service_receipt, root / 'SERVICE_READY.json')
    write(root / 'EXECUTION.json', {'status': 'FROZEN_BEFORE_EXECUTION', 'created_at_utc': now(),
          'hostname': socket.gethostname(), 'seed_schedule': protocol['model']['repeat_seeds'],
          'upstream': args.upstream, 'proxy_port': args.port, 'concurrency': 8,
          'maximum_requests': MAX_REQUESTS, 'maximum_provider_tokens': MAX_TOTAL_TOKENS,
          'maximum_seconds': maximum_seconds, 'planned_trials': len(configs), 'per_arm': 445,
          'campaign_deadline_utc': campaign_deadline,
          'preserved_trials': len(preserved['rows']), 'remaining_trials': len(configs) - len(preserved['rows']),
          'preserved_results_sha256': sha256(root / 'PRESERVED_RESULTS.json'),
          'trial_plan_sha256': sha256(root / 'TRIAL_PLAN.json'), 'actor_binding_sha256': binding_sha,
          'official_manifest_sha256': sha256(parent / 'frozen/manifest.json'), 'argv': sys.argv,
          'service_receipt_sha256': sha256(root / 'SERVICE_READY.json'),
          'prior_provider_usage': prior,
          'prior_timeout_carry_sha256': sha256(root / 'PRIOR_TIMEOUT_CARRY.json') if timeout_ids else None,
          'durable_provider': getattr(args, 'durable_provider', False),
          'provider_rejection_contract_sha256': sha256(root / 'PROVIDER_REJECTION_CONTRACT.json') if rejection_receipt else None,
          'prior_usage_reconciliation_sha256': sha256(root / 'PRIOR_USAGE_RECONCILIATION.json') if (root / 'PRIOR_USAGE_RECONCILIATION.json').exists() else None,
          'continuous': args.continuous,
          'verifier_https_proxy': args.verifier_https_proxy,
          'verifier_artifact_cache': args.verifier_artifact_cache,
          'verifier_apt_cache': apt_cache,
          'apt_cache_receipt_sha256': sha256(root / 'APT_CACHE_READY.json') if apt_receipt else None,
          'dependency_cache_receipt_sha256': sha256(root / 'DEPENDENCY_CACHE_READY.json') if cache_receipt else None,
          'verifier_source_or_metric_modified': False,
          'trial_failure_policy': 'retain every result, continue remaining trials, never impute missing rewards' if args.continuous else 'stop on infrastructure failure',
          'native_evaluator': 'Harbor Trial', 'native_metric': 'harbor.metrics.mean.Mean',
          'task_timeouts_and_resource_overrides': None, 'retries': 0,
          'source_files_sha256': {name: sha256(ROOT / name) for name in (
              'scripts/run_terminal_bench_native.py', 'scripts/harbor_terminal_agent.py',
              'scripts/run_terminal_bench_qwen35_4b_direct.py', 'scripts/terminal_solpi.py',
              'phiagent/harness/observation_pack.py')}})
    print(json.dumps({'status': 'FROZEN_NATIVE_TRIALS', 'trials': len(configs), 'root': str(root)}), flush=True)


async def run_campaign(root: Path):
    from harbor.metrics.mean import Mean
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial

    execution = json.loads((root / 'EXECUTION.json').read_text())
    if (root / 'STARTED.json').exists():
        raise FileExistsError('use a fresh execution; do not silently replay official trials')
    if sha256(root / 'TRIAL_PLAN.json') != execution['trial_plan_sha256']:
        raise ValueError('trial plan changed after freeze')
    for name, expected in execution['source_files_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('source changed after freeze: ' + name)
    if execution.get('dependency_cache_receipt_sha256'):
        cache_receipt = root / 'DEPENDENCY_CACHE_READY.json'
        if sha256(cache_receipt) != execution['dependency_cache_receipt_sha256']:
            raise ValueError('dependency cache receipt changed after freeze')
        verify_dependency_cache_receipt(cache_receipt)
    if execution.get('apt_cache_receipt_sha256'):
        apt_receipt = root / 'APT_CACHE_READY.json'
        if sha256(apt_receipt) != execution['apt_cache_receipt_sha256']:
            raise ValueError('APT cache receipt changed after freeze')
        verify_apt_cache_receipt(apt_receipt)
    verified_preinference = bool(execution.get('provider_rejection_contract_sha256'))
    if verified_preinference:
        contract = root / 'PROVIDER_REJECTION_CONTRACT.json'
        if sha256(contract) != execution['provider_rejection_contract_sha256']:
            raise ValueError('provider rejection contract changed after freeze')
        validate_preinference_contract(contract)
        service_path = root / 'SERVICE_READY.json'
        if (sha256(service_path) != execution['service_receipt_sha256'] or
                json.loads(service_path.read_text()).get('preinference_validation_source_sha256') != PREINFERENCE_RUNTIME_SOURCES):
            raise ValueError('live service preinference validation changed')
    if execution.get('prior_usage_reconciliation_sha256'):
        reconciliation_path = root / 'PRIOR_USAGE_RECONCILIATION.json'
        if sha256(reconciliation_path) != execution['prior_usage_reconciliation_sha256']:
            raise ValueError('provider reconciliation changed after freeze')
        reconciliation = json.loads(reconciliation_path.read_text())
        if (not verified_preinference or reconciliation['resolved_usage'] != execution['prior_provider_usage']
            or reconciliation['runtime_contract_sha256'] != execution['provider_rejection_contract_sha256']
            or sha256(reconciliation['original_final_path']) != reconciliation['original_final_sha256']):
            raise ValueError('provider reconciliation is not bound to the admitted runtime and debit')
        for source in reconciliation['source_artifacts']:
            if sha256(source['path']) != source['sha256']:
                raise ValueError('provider reconciliation source artifact changed')
    plan = json.loads((root / 'TRIAL_PLAN.json').read_text())
    binding = json.loads((root / 'actor-binding.json').read_text())
    preserved = {'rows': [], 'source_artifacts': []}
    if execution.get('preserved_results_sha256'):
        if sha256(root / 'PRESERVED_RESULTS.json') != execution['preserved_results_sha256']:
            raise ValueError('preserved results changed after freeze')
        preserved = json.loads((root / 'PRESERVED_RESULTS.json').read_text())
        for source in preserved['source_artifacts']:
            if sha256(source['path']) != source['sha256']:
                raise ValueError('preserved native result artifact changed')
    continuous = execution.get('continuous', False)
    bounded_unknown_attempts = timeout_carry_allowance(root, execution)
    results = preserve_completed_trials(plan, preserved['rows'], allow_unscored=continuous)
    completed_ids = {row['trial_id'] for row in results}
    pending_plan = [row for row in plan if row['trial_id'] not in completed_ids]
    retained_count = len(results)
    proxy_options = {'verified_preinference': True} if verified_preinference else {}
    if bounded_unknown_attempts:
        proxy_options['bounded_unknown_attempts'] = bounded_unknown_attempts
    if execution.get('durable_provider'):
        proxy_options['durable_provider'] = True
    server, ledger = start_proxy(root, execution['upstream'], execution['proxy_port'], execution.get('prior_provider_usage'), **proxy_options)
    write(root / 'STARTED.json', {'at_utc': now(), 'pid': os.getpid(), 'hostname': socket.gethostname(),
                                 'execution_sha256': sha256(root / 'EXECUTION.json')})
    started = time.monotonic()
    stopped = False
    running = {}
    cursor = 0

    async def run_one(row):
        path = Path(row['config_path'])
        if sha256(path) != row['config_sha256']:
            raise ValueError('frozen trial config changed')
        config = TrialConfig.model_validate_json(path.read_text())
        trial = await Trial.create(config)
        result = await trial.run()
        value = result.model_dump(mode='json')
        result_path = Path(config.trials_dir) / config.trial_name / 'result.json'
        return {'arm': row['arm'], 'task_id': row['task_id'], 'repeat_seed': row['repeat_seed'],
                'trial_id': row['trial_id'], 'result': value,
            'native_result_path': str(result_path),
            'infrastructure_diagnostics': verifier_infrastructure_diagnostics(result_path)}

    try:
        await asyncio.to_thread(provider_checks, root, binding['model']['chat_completions_url'],
                                binding['model']['expected_response_model_id'])
        write(root / 'ADMITTED.json', {'at_utc': now(), 'status': 'NATIVE_FULL89_RUNNING',
              'synthetic_provider_calls': 2, 'planned_trials': len(plan), 'preserved_trials': retained_count,
              'remaining_trials': len(pending_plan), 'provider_usage': ledger.snapshot()})
        print('NATIVE_FULL89_RUNNING', flush=True)
        while running or cursor < len(pending_plan):
            deadline_passed = execution.get('campaign_deadline_utc') and (
                datetime.now(timezone.utc) >= datetime.fromisoformat(execution['campaign_deadline_utc']))
            if ((root / 'STOP').exists() or deadline_passed or time.monotonic() - started > execution['maximum_seconds']
                    or ledger.snapshot()['unbounded_unknown_usage_attempts']):
                stopped = True
            while not stopped and len(running) < execution['concurrency'] and cursor < len(pending_plan):
                row = pending_plan[cursor]
                running[asyncio.create_task(run_one(row))] = row
                cursor += 1
            write(root / 'HEARTBEAT.json', {'at_utc': now(), 'status': 'DRAINING' if stopped else 'RUNNING',
                  'started_trials': retained_count + cursor, 'completed_trials': len(results), 'expected_trials': len(plan),
                  'preserved_trials': retained_count, 'newly_started_trials': cursor,
                  'valid_reward_trials': sum(record_has_verified_reward(row) for row in results),
                  'unresolved_trials': sum(not record_has_verified_reward(row) for row in results),
                  'continuous': continuous,
                  'active': list(running.values()), 'provider_usage': ledger.snapshot()})
            if not running:
                break
            done, _pending = await asyncio.wait(running, timeout=30, return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                row = running.pop(future)
                try:
                    recorded = future.result()
                except Exception as error:
                    recorded = {'arm': row['arm'], 'task_id': row['task_id'], 'repeat_seed': row['repeat_seed'],
                                'trial_id': row['trial_id'], 'controller_error': {'type': type(error).__name__, 'error': str(error)}}
                native = recorded.get('result', {})
                exception = native.get('exception_info')
                if trial_blocks_queue(recorded, continuous=continuous):
                    stopped = True
                results.append(recorded)
                with (root / 'TRIAL_RESULTS.jsonl').open('a') as stream:
                    stream.write(json.dumps(recorded, sort_keys=True) + '\n')
                print(json.dumps({'finished_trial': row['trial_id'], 'completed': len(results),
                                  'exception': exception, 'stopped': stopped}), flush=True)
        if execution.get('durable_provider'):
            write(root / 'PROVIDER_DRAINING.json', {'at_utc': now(), 'usage': ledger.snapshot()})
            if not await asyncio.to_thread(ledger.drained.wait, 1260):
                stopped = True
        scores = {}
        for arm in binding['actors']:
            rows = [row for row in results if row['arm'] == arm]
            rewards = [(row.get('result', {}).get('verifier_result') or {}).get('rewards') for row in rows]
            scores[arm] = {'completed': len(rows), 'expected': 445,
                           'valid_rewards': sum(record_has_verified_reward(row) for row in rows),
                           'native_mean': Mean().compute(rewards) if len(rows) == 445 and all(record_has_verified_reward(row) for row in rows) else None}
        write(root / 'UNRESOLVED_TRIALS.json', {'trial_ids': [row['trial_id'] for row in results
                  if not record_has_verified_reward(row)], 'missing_rewards_imputed': False})
        write(root / 'RESULT.json', {'status': campaign_result_status(results, len(plan), stopped=stopped),
              'at_utc': now(), 'expected_trials': len(plan), 'completed_trials': len(results), 'arms': scores,
              'usage': ledger.snapshot(), 'new_claim_admitted': False})
    except Exception as error:
        write(root / 'FAILURE.json', {'at_utc': now(), 'type': type(error).__name__, 'error': str(error),
                                     'usage': ledger.snapshot()})
        raise
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        write(root / 'PROVIDER_FINAL.json', ledger.snapshot())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'monitor'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--preparation', type=Path)
    parser.add_argument('--images-receipt', type=Path)
    parser.add_argument('--service-receipt', type=Path)
    parser.add_argument('--prior-run', type=Path)
    parser.add_argument('--preserve-completed', action='store_true')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--verifier-https-proxy')
    parser.add_argument('--verifier-artifact-cache')
    parser.add_argument('--verifier-cache-receipt', type=Path)
    parser.add_argument('--verifier-apt-cache')
    parser.add_argument('--verifier-apt-receipt', type=Path)
    parser.add_argument('--provider-rejection-receipt', type=Path)
    parser.add_argument('--retain-timed-out-usage', type=int, action='append')
    parser.add_argument('--durable-provider', action='store_true')
    parser.add_argument('--task-source', type=Path)
    parser.add_argument('--upstream', default='http://127.0.0.1:18991')
    parser.add_argument('--port', type=int, default=18992)
    parser.add_argument('--remote-root')
    parser.add_argument('--service-root')
    args = parser.parse_args()
    if args.mode == 'monitor':
        return monitor(args)
    if socket.gethostname() != 'harnessmm-api-worker':
        parser.error('all native benchmark execution and real-data preparation is restricted to a800-4')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.mode == 'prepare':
        prepare(args)
    else:
        asyncio.run(run_campaign(args.root.resolve()))
    return 0


def monitor(args):
    remote_root = Path(args.remote_root)
    service_root = Path(args.service_root)
    if (not remote_root.is_relative_to('/path/to/data/phiagent/work/terminal-solpi-native')
            or not service_root.is_relative_to('/opt/phiagent/runs')
            or not service_root.name.startswith('tb0921s')):
        raise ValueError('monitor roots are outside this campaign')
    args.root.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    start = time.monotonic()
    previous = None
    while not stop.is_set() and time.monotonic() - start < MAX_SECONDS + 1800:
        code = ('import json;from pathlib import Path;root=Path(' + repr(str(remote_root)) + ');'
                'print(json.dumps({name:json.loads((root/name).read_text()) for name in '
                '("HEARTBEAT.json","RESULT.json","FAILURE.json","ADMITTED.json") if (root/name).is_file()}))')
        command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                   'a800-4', shlex.join(['python3', '-c', code])]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=40)
        if completed.returncode == 0:
            value = json.loads(completed.stdout)
            write(args.root / 'REMOTE_STATUS.json', value)
            heartbeat = value.get('HEARTBEAT.json', {})
            progress = (heartbeat.get('completed_trials'), heartbeat.get('started_trials'), bool(value.get('RESULT.json')), bool(value.get('FAILURE.json')))
            if progress != previous:
                print(json.dumps({'at_utc': now(), 'completed': progress[0], 'started': progress[1],
                                  'result': progress[2], 'failure': progress[3]}), flush=True)
                previous = progress
            if 'RESULT.json' in value or 'FAILURE.json' in value:
                shutdown = ('from pathlib import Path;root=Path(' + repr(str(service_root)) + ');'
                            'assert (root/"OWNER.json").is_file();(root/"STOP").write_text("native campaign finished\\n")')
                stopped = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                                         'h200-1', shlex.join(['python3', '-c', shutdown])], capture_output=True, text=True, timeout=40)
                write(args.root / 'SERVICE_STOP_REQUEST.json', {'at_utc': now(), 'returncode': stopped.returncode,
                      'service_root': str(service_root), 'diagnostic': stopped.stderr})
                if stopped.returncode:
                    raise RuntimeError('owned model-service stop request failed')
                return 0
        else:
            write(args.root / 'CONNECTION_FAILURE.json', {'at_utc': now(), 'returncode': completed.returncode,
                                                        'diagnostic': completed.stderr})
        stop.wait(30)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
