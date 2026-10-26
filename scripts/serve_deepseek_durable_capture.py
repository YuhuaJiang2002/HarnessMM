#!/usr/bin/env python3
"""Preserve native DeepSeek wire responses even after an actor disconnects.

This opt-in transport belongs in a new source-frozen run. It never regenerates a
request, changes a request body, or extends a native task's execution time.
"""
from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import uuid
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from phiagent.harness.loopback_http import FetchCompatibleHTTPServer

SCHEMA = "deepseek-native-durable-capture/v1"
MODEL = "deepseek-flash"
UPSTREAM_TIMEOUT_SECONDS = 1860
ROUTES = {"anthropic/v1/messages": "https://api.deepseek.com/anthropic/v1/messages",
          "chat/completions": "https://api.deepseek.com/chat/completions"}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,159}\Z")


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def native_usage(raw, route, streaming):
    """Read provider counters; truncated/ambiguous responses remain unknown."""
    def count(value):
        return type(value) is int and value >= 0

    if route == "chat/completions":
        if streaming:
            raise ValueError("Fast decisions require a complete non-streaming response")
        value = json.loads(raw)
        if value.get("model") != MODEL:
            raise ValueError("Provider model differs from the frozen model")
        usage = value.get("usage") or {}
        keys = ("prompt_tokens", "completion_tokens", "total_tokens",
                "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
        if not all(count(usage.get(key)) for key in keys):
            raise ValueError("Missing native fast input, output or cache counters")
        if (usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
                or usage["prompt_tokens"] != usage["prompt_cache_hit_tokens"]
                + usage["prompt_cache_miss_tokens"]):
            raise ValueError("Native fast counters do not reconcile")
        return {"inputTokens": usage["prompt_cache_miss_tokens"],
                "outputTokens": usage["completion_tokens"],
                "cacheReadTokens": usage["prompt_cache_hit_tokens"], "cacheWriteTokens": 0,
                "totalTokens": usage["total_tokens"]}
    if route != "anthropic/v1/messages":
        raise ValueError("Unknown native route")
    if streaming:
        events = []
        for frame in raw.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
            lines = [line[5:].lstrip(" ") for line in frame.splitlines()
                     if line.startswith("data:")]
            if lines:
                events.append(json.loads("\n".join(lines)))
        starts = [e for e in events if e.get("type") == "message_start"]
        stops = [e for e in events if e.get("type") == "message_stop"]
        if len(starts) != 1 or len(stops) != 1 or any(e.get("type") == "error" for e in events):
            raise ValueError("Native stream did not finish exactly once")
        start = events.index(starts[0])
        stop = events.index(stops[0])
        if start >= stop:
            raise ValueError("Native stream stop precedes its start")
        message = starts[0].get("message") or {}
        deltas = [e for e in events[start + 1:stop] if e.get("type") == "message_delta"]
        if not deltas or not any(e.get("delta", {}).get("stop_reason") for e in deltas):
            raise ValueError("Native stream lacks a terminal usage update")
        usage = dict(message.get("usage") or {})
        for event in deltas:
            update = event.get("usage") or {}
            if "output_tokens" in update and (not count(update["output_tokens"])
                    or update["output_tokens"] < usage.get("output_tokens", 0)):
                raise ValueError("Native output counters are malformed or decrease")
            usage.update(update)
    else:
        message = json.loads(raw)
        if message.get("type") != "message" or not message.get("stop_reason"):
            raise ValueError("Native message did not finish")
        usage = message.get("usage") or {}
    if message.get("model") != MODEL:
        raise ValueError("Provider model differs from the frozen model")
    keys = ("input_tokens", "output_tokens", "cache_read_input_tokens",
            "cache_creation_input_tokens")
    if not all(count(usage.get(key)) for key in keys):
        raise ValueError("Missing native slow input, output or cache counters")
    result = dict(zip(("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens"),
                      (usage[key] for key in keys)))
    result["totalTokens"] = sum(result.values())
    return result


class DurableCaptureServer(FetchCompatibleHTTPServer):
    daemon_threads = False  # server_close waits for dispatched upstream calls.
    block_on_close = True

    def __init__(self, root, api_key, relay_secret, *, port=0, opener=None, attribution=None, execution_guards=None):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.api_key, self.relay_secret = api_key, relay_secret
        self.opener = opener or build_opener(ProxyHandler({}))
        self.sequence_lock = threading.Lock()
        self.sequence = 0
        self.attribution = attribution
        self.execution_guards = execution_guards
        self.actor_counts = {'physical': 0, 'slow': 0, 'fast': 0, 'input': 0, 'tokens': 0}
        super().__init__(("127.0.0.1", port), DurableCaptureHandler)
        write_json(self.root / "CONFIG.json", {
            "schema": SCHEMA, "model": MODEL, "bind": "127.0.0.1",
            "port": self.server_port, "automatic_retries": 0,
            "native_request_body_rewritten": False, "cumulative_token_limit": None,
            "upstream_timeout_seconds": UPSTREAM_TIMEOUT_SECONDS,
            "capture_continues_after_client_disconnect": True,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


class DurableCaptureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *_args):
        pass

    def persist_usage_event(self, receipt, raw, payload, attempt, ident, sequence, usage=None):
        """Persist one physical request before its completion reaches the actor."""
        metadata = self.server.attribution or {}
        provider_id = None
        try:
            if payload.get('stream'):
                for line in raw.decode().splitlines():
                    if line.startswith('data:'):
                        event = json.loads(line[5:])
                        if event.get('type') == 'message_start':
                            provider_id = event['message'].get('id')
                            break
            else:
                provider_id = json.loads(raw).get('id')
        except (ValueError, KeyError, TypeError):
            pass
        row = {'schema':'physical-actor-usage/v1', 'run_id':metadata.get('run_id', str(self.server.root)),
               'task_id':metadata.get('task_id'), 'arm':metadata.get('arm'),
               'request_seq':sequence, 'physical_request_id':ident, 'provider_request_id':provider_id,
               'model':payload.get('model'), 'input_tokens':usage['totalTokens']-usage['outputTokens'] if usage else None,
               'output_tokens':usage['outputTokens'] if usage else None,
               'cached_input_tokens':usage.get('cacheReadTokens',0)+usage.get('cacheWriteTokens',0) if usage else None,
               'status':'COMPLETE' if usage else 'AUDIT_INCOMPLETE',
               'slow_or_fast':'fast' if receipt['route']=='chat/completions' else 'slow',
               'candidate_id':None, 'native_action_hash':None, 'native_receipt_hash':None,
               'binding_status':'native action/certificate joins are separate append-only events',
               'timestamp':datetime.now(timezone.utc).isoformat(), 'attempt':attempt,
               'request_sha256':receipt['request_sha256']}
        with self.server.sequence_lock:
            with (self.server.root/'ACTOR_USAGE.jsonl').open('a') as log:
                log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno())

    def authorized(self):
        supplied = self.headers.get("Authorization", "")
        if supplied.startswith("Bearer "):
            supplied = supplied[7:]
        else:
            supplied = self.headers.get("X-Api-Key", "")
        return hmac.compare_digest(supplied, self.server.relay_secret)

    def send_json(self, status, value):
        body = json.dumps(value).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if not self.authorized():
            self.send_json(401, {"error": "owner authentication required"})
            return
        pieces = self.path.removeprefix("/").split("/", 1)
        if len(pieces) != 2 or not IDENTIFIER.fullmatch(pieces[0]) or pieces[1] not in ROUTES:
            self.send_json(404, {"error": "unregistered native route"})
            return
        attempt, route = pieces
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32 * 1024 * 1024:
                raise ValueError("Invalid native request length")
            body = self.rfile.read(length)
            payload = json.loads(body)
            if payload.get("model") != MODEL or (route == "chat/completions" and payload.get("stream")):
                raise ValueError("Use the pinned native model and non-streaming fast route")
        except (ValueError, TypeError):
            self.send_json(400, {"error": "native request contract rejected before dispatch"})
            return
        with self.server.sequence_lock:
            if attempt == 'actor' and self.server.execution_guards:
                pairs = [('physical','max_physical_requests'), ('slow','max_slow_requests'),
                         ('fast','max_fast_requests'), ('input','max_total_input_tokens'), ('tokens','max_total_tokens')]
                active = dict(self.server.actor_counts)
                branch = 'fast' if route == 'chat/completions' else 'slow'
                reason = next((limit for key, limit in pairs
                               if key in ('physical','input','tokens',branch)
                               and active[key] >= self.server.execution_guards[limit]), None)
                if reason:
                    with (self.server.root / 'LOCAL_GUARD_DENIALS.jsonl').open('a') as log:
                        log.write(json.dumps({'reason':reason,'route':route,'physical_provider_dispatched':False,
                           'request_sha256':hashlib.sha256(body).hexdigest(),'counts':active})+'\n')
                    self.send_json(409, {'error':'CHISEL_EXECUTION_GUARD', 'reason':reason})
                    return
            if attempt == 'actor':
                self.server.actor_counts['physical'] += 1
                self.server.actor_counts['fast' if route == 'chat/completions' else 'slow'] += 1
            self.server.sequence += 1
            sequence = self.server.sequence
            ident = f"{self.server.sequence:08d}-{uuid.uuid4().hex}"
        directory = self.server.root / attempt / ident
        directory.mkdir(mode=0o700, parents=True)
        (directory / "request.body").write_bytes(body)
        receipt = {"schema": SCHEMA, "attempt_id": attempt, "call_id": ident, "route": route,
                   "request_sha256": hashlib.sha256(body).hexdigest(), "body_rewritten": False,
                   "started_at_utc": datetime.now(timezone.utc).isoformat(),
                   "status": "DISPATCHED", "usage": None, "client_disconnected": False}
        write_json(directory / "RECEIPT.json", receipt)
        if self.server.attribution:
            from scripts.audit_jev_cost_train import context_sections
            write_json(directory / 'CONTEXT_ATTRIBUTION.json', {
                **self.server.attribution, 'physical_request_id':ident,
                'component':'jev_fast_selector' if route == 'chat/completions' else 'slow_pending_response_classification',
                'context':context_sections(payload), 'request_sha256':receipt['request_sha256'],
                'section_tokens_exact':False})
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        if route.startswith("anthropic/"):
            headers["X-Api-Key"] = self.server.api_key
            headers["Anthropic-Version"] = self.headers.get("Anthropic-Version", "2023-06-01")
            if self.headers.get("Anthropic-Beta"):
                headers["Anthropic-Beta"] = self.headers["Anthropic-Beta"]
        else:
            headers["Authorization"] = "Bearer " + self.server.api_key
        request = Request(ROUTES[route], data=body, headers=headers, method="POST")
        raw = bytearray()
        usage_persisted = False
        try:
            try:
                upstream = self.server.opener.open(request, timeout=UPSTREAM_TIMEOUT_SECONDS)
            except HTTPError as error:
                upstream = error
            with upstream:
                receipt["upstream_status"] = upstream.status
                content_type = upstream.headers.get("Content-Type", "application/json")
                try:
                    self.send_response(upstream.status)
                    self.send_header("Content-Type", content_type)
                    self.end_headers()
                except (BrokenPipeError, ConnectionResetError):
                    receipt["client_disconnected"] = True
                with (directory / "response.body").open("xb") as stream:
                    for chunk in iter(upstream.readline, b""):
                        stream.write(chunk)
                        stream.flush()
                        raw.extend(chunk)
                        if not usage_persisted and upstream.status == 200 and (not payload.get('stream') or b'message_stop' in chunk):
                            try:
                                known = native_usage(bytes(raw), route, bool(payload.get('stream')))
                            except (ValueError, TypeError, KeyError):
                                pass
                            else:
                                self.persist_usage_event(receipt, raw, payload, attempt, ident, sequence, known)
                                usage_persisted = True
                        if not receipt["client_disconnected"]:
                            try:
                                self.wfile.write(chunk)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                receipt["client_disconnected"] = True
            receipt["status"] = "RESPONSE_CAPTURED"
            if receipt["upstream_status"] == 200:
                try:
                    receipt["usage"] = native_usage(bytes(raw), route, bool(payload.get("stream")))
                    receipt["status"] = "MEASURED"
                except (ValueError, TypeError, KeyError):
                    receipt["usage_error"] = "Native final model/usage counters unavailable or inconsistent"
        except Exception as error:
            receipt["status"] = "UPSTREAM_INCOMPLETE"
            receipt["error_type"] = type(error).__name__
            # Exception text may include credentials from an injected request implementation.
        finally:
            receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            receipt["response_sha256"] = hashlib.sha256(raw).hexdigest()
            receipt["response_bytes"] = len(raw)
            if attempt == 'actor' and receipt.get('usage'):
                with self.server.sequence_lock:
                    self.server.actor_counts['tokens'] += receipt['usage']['totalTokens']
                    self.server.actor_counts['input'] += receipt['usage']['totalTokens'] - receipt['usage']['outputTokens']
            write_json(directory / "RECEIPT.json", receipt)
            if not usage_persisted:
                self.persist_usage_event(receipt, raw, payload, attempt, ident, sequence, receipt.get('usage'))


def start_capture(root, api_key, relay_secret, *, port=0, opener=None, attribution=None, execution_guards=None):
    server = DurableCaptureServer(root, api_key, relay_secret, port=port, opener=opener,
                                 attribution=attribution, execution_guards=execution_guards)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def audited_physical_accounting(root, sdk_accounting):
    """Rehash every closed native wire receipt; never infer missing final usage."""
    root = Path(root)
    values, attempts, errors, call_ids = [], [], [], set()
    for path in sorted((root / 'actor').glob('*/RECEIPT.json')):
        receipt = json.loads(path.read_text())
        body, response = path.with_name('request.body'), path.with_name('response.body')
        try:
            if receipt.get('call_id') in call_ids:
                raise ValueError('Duplicate physical provider call identity')
            call_ids.add(receipt.get('call_id'))
            if (receipt.get('status') != 'MEASURED' or receipt.get('body_rewritten') is not False
                    or hashlib.sha256(body.read_bytes()).hexdigest() != receipt['request_sha256']
                    or hashlib.sha256(response.read_bytes()).hexdigest() != receipt['response_sha256']):
                raise ValueError('Native physical response is incomplete or changed')
            if json.loads(body.read_text()).get('model') != MODEL:
                raise ValueError('Original physical request model differs')
            usage = native_usage(response.read_bytes(), receipt['route'],
                                 bool(json.loads(body.read_text()).get('stream')))
            if usage != receipt['usage']:
                raise ValueError('Native raw counters differ from receipt')
            values.append(usage)
            attempts.append({'attempt_id': receipt['call_id'], 'ended': True, 'usage': usage,
                'start': {'purpose': 'actor', 'model': MODEL},
                'finish': {'reason': 'native-wire-complete'}, 'native_route': receipt['route'],
                'wire_receipt': str(path), 'wire_receipt_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        except (OSError, KeyError, ValueError) as error:
            errors.append(str(path) + ':' + type(error).__name__)
    keys = ('inputTokens', 'outputTokens', 'cacheReadTokens', 'cacheWriteTokens', 'totalTokens')
    known = {key: sum(row[key] for row in values) for key in keys}
    expected = sdk_accounting.get('model_call_attempt_count', sdk_accounting.get('attempt_count'))
    if type(expected) is not int or expected != len(attempts):
        errors.append('Physical requests do not reconcile with SDK logical fast/slow counts')
    if sdk_accounting.get('complete') is True and sdk_accounting.get('usage') != known:
        errors.append('SDK complete counters differ from raw physical provider counters')
    return {'complete': not errors, 'usage': known if not errors else None,
            'known_reported_usage': known, 'attempts': attempts,
            'attempt_count': len(attempts), 'model_call_attempt_count': len(attempts),
            'source': 'rehash-native-physical-provider-responses', 'unknown_reasons': errors,
            'sdk_accounting_preserved': True, 'unknown_is_zero': False}
