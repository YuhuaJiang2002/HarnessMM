/**
 * Host-shape checks: drive lib/index.js through a fake cordis context so the
 * stream boundary, the dispatch guard and the certificate sink are exercised
 * the way the desktop agent loop exercises them.
 */

import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { apply, Config } from '../lib/index.js';
import { nativeCallId, hashJson, parseBank } from '../lib/controller.mjs';

const TOOLS = [
  { name: 'pwsh', parameters: { properties: { description: { type: 'string' }, command: { type: 'string' } } } },
];

const CAND = {
  candidate_id: 'A',
  predecessor_id: 'root',
  tool: 'pwsh',
  arguments: { description: 'Run the full test suite', command: 'pytest -q tests/' },
  guard: { exit_code: 0 },
  required_receipt_fields: ['exit_code'],
  expected_local_cost: { selector_upper_tokens: 8192 },
  dependency_ids: [],
};

/** Build a minimal cordis-like context and return the registered handlers. */
function fakeHost(config, selectorStream = stubSelector('A')) {
  const handlers = new Map();
  const sections = [];
  const tools = new Map();
  const ctx = {
    tools: { schemas: () => TOOLS, register: tool => tools.set(tool.name, tool) },
    systemPrompt: { section: entry => sections.push(entry) },
    // The selector answers a single label; the deterministic path never asks it.
    get: service => (service === 'llm' ? { stream: selectorStream } : undefined),
    on: (name, handler) => handlers.set(name, handler),
  };
  const api = apply(ctx, config);
  return { ctx, handlers, sections, tools, api };
}

function stubSelector(label) {
  return () => (async function* () {
    yield { type: 'text-delta', index: 0, delta: label };
    yield { type: 'usage', usage: { inputTokens: 8, outputTokens: 1, totalTokens: 9 } };
    yield { type: 'finish', reason: { kind: 'stop' } };
  })();
}

function config(enabled, receiptsFile, selector = { enabled: true }, policy = {}) {
  return Config({
    policy: { enabled, maximum_candidates: 16, maximum_fast_steps: 8, maximum_public_bytes: 65536, ...policy },
    selector,
    shellTools: ['pwsh'],
    planSection: true,
    receiptsFile,
  });
}

function slowStreamWithBank() {
  return slowStreamWithPlan([CAND]);
}

function slowStreamWithPlan(candidates) {
  const plan = JSON.stringify({ candidates });
  const chunks = [
    { type: 'block-start', index: 0, blockType: 'reasoning' },
    { type: 'reasoning-delta', index: 0, delta: 'I should run the suite.' },
    { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'I should run the suite.' } },
    { type: 'block-start', index: 1, blockType: 'text' },
    { type: 'text-delta', index: 1, delta: `<CHISEL_PLAN>${plan}</CHISEL_PLAN>` },
    { type: 'block-end', index: 1, block: { type: 'text', text: `<CHISEL_PLAN>${plan}</CHISEL_PLAN>` } },
    { type: 'block-start', index: 2, blockType: 'tool-call' },
    { type: 'tool-call-delta', index: 2, id: 'root_call', name: 'pwsh', argumentsDelta: '{"description":"Bootstrap","command":"ls"}' },
    {
      type: 'block-end',
      index: 2,
      block: { type: 'tool-call', id: 'root_call', name: 'pwsh', arguments: '{"description":"Bootstrap","command":"ls"}' },
    },
    { type: 'usage', usage: { inputTokens: 900, outputTokens: 400, totalTokens: 1300 } },
    { type: 'finish', reason: { kind: 'tool-calls' } },
  ];
  return async function* stream() {
    for (const chunk of chunks) yield chunk;
  };
}

const SESSION = 'session-1';
// The agent loop hands `llm/stream` a frozen request with `sessionId` and no
// agent object; tool events carry `agent: { id }` — the same string.
const agent = { id: SESSION };

test('the plan section is empty while the policy is disabled and populated when it is on', () => {
  const off = fakeHost(config(false, ''));
  assert.equal(off.sections.length, 1);
  assert.equal(off.sections[0].text(), '');
  assert.equal(off.sections[0].name, 'harnessmm:plan');

  const on = fakeHost(config(true, ''));
  const text = on.sections[0].text();
  assert.match(text, /Executable action graph/);
  assert.match(text, /CHISEL_PLAN/);
  assert.match(text, /`pwsh`/);
  assert.match(text, /bare assistant text/);
});

test('a slow turn records its bank and a later step commits deterministically', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'harnessmm-'));
  const receipts = join(dir, 'receipts.jsonl');
  const host = fakeHost(config(true, receipts));
  const stream = host.handlers.get('llm/stream');

  // Step 1 �?the slow request. Not projected, so the inner stream is passed through.
  const first = await collect(stream({ sessionId: SESSION, messages: [{ role: 'user', content: [{ type: 'text', text: 'run the tests' }] }], tools: TOOLS }, slowStreamWithBank()));
  assert.equal(first.at(-1).type, 'finish');

  // Step 2 �?bound to the same agent, the request now ends in the tool result.
  const rootArgs = '{"description":"Bootstrap","command":"ls"}';
  const assistant = {
    role: 'assistant',
    content: [
      { type: 'text', text: '<CHISEL_PLAN>' + JSON.stringify({ candidates: [CAND] }) + '</CHISEL_PLAN>' },
      { type: 'tool-call', id: 'root_call', name: 'pwsh', arguments: rootArgs },
    ],
  };
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'run the tests' }] },
    assistant,
    { role: 'tool', toolCallId: 'root_call', content: [{ type: 'text', text: '(no output)' }] },
  ];
  let slowCalled = false;
  const second = await collect(stream({ sessionId: SESSION, messages, tools: TOOLS }, () => (async function* () {
    slowCalled = true;
    yield { type: 'finish', reason: { kind: 'stop' } };
  })()));

  assert.equal(slowCalled, false, 'the slow actor must not be called for a deterministic commit');
  const call = second.find(chunk => chunk.type === 'block-end').block;
  assert.equal(call.type, 'tool-call');
  assert.equal(call.name, 'pwsh');
  assert.equal(call.arguments, JSON.stringify(CAND.arguments));
  assert.equal(second.at(-1).reason.kind, 'tool-calls');
  assert.ok(second.some(chunk => chunk.type === 'usage'));

  const bank = parseBank(assistant, TOOLS, { maxCandidates: 16 });
  assert.equal(call.id, nativeCallId(bank.sha256, 'A'));

  const entries = readFileSync(receipts, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  assert.ok(entries.some(entry => entry.type === 'bank'));
  assert.ok(entries.some(entry => entry.type === 'decision' && entry.kind === 'commit'));
  assert.ok(entries.some(entry => entry.type === 'substitution-issued'));
});

test('the configured maximum_candidates bounds the live parse path', async () => {
  // Regression: `policy.maximum_candidates` was documented but never handed to
  // parseBank, so the parser always allowed its own default of 16.
  const dir = mkdtempSync(join(tmpdir(), 'harnessmm-'));
  const user = [{ role: 'user', content: [{ type: 'text', text: 'run the tests' }] }];
  const second = { ...CAND, candidate_id: 'B', predecessor_id: 'A', arguments: { description: 'Then list', command: 'ls' } };
  const third = { ...CAND, candidate_id: 'C', predecessor_id: 'B', arguments: { description: 'Then print', command: 'pwd' } };

  const strict = fakeHost(config(true, join(dir, 'r.jsonl'), { enabled: true }, { maximum_candidates: 2 }));
  await collect(strict.handlers.get('llm/stream')(
    { sessionId: SESSION, messages: user, tools: TOOLS },
    slowStreamWithPlan([CAND, second, third]),
  ));
  const entries = readFileSync(join(dir, 'r.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line));
  assert.ok(entries.some(entry => entry.type === 'invalid-bank' && /candidate contract/.test(entry.reason)));
  assert.equal(entries.some(entry => entry.type === 'bank'), false);

  // The same authored bank fits once the configured cap allows it.
  const roomyDir = mkdtempSync(join(tmpdir(), 'harnessmm-'));
  const roomy = fakeHost(config(true, join(roomyDir, 'r.jsonl')));
  await collect(roomy.handlers.get('llm/stream')(
    { sessionId: SESSION, messages: user, tools: TOOLS },
    slowStreamWithPlan([CAND, second, third]),
  ));
  const roomyEntries = readFileSync(join(roomyDir, 'r.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line));
  assert.equal(roomyEntries.filter(entry => entry.type === 'bank').length, 1);
});

test('a restart mid-turn restores the executable graph, not just the bank hash', async () => {
  // Regression (audit P0-2): a projected turn whose per-session store was lost
  // (host restart, late plugin mount) was given a bare record with no
  // pendingBank/pendingCandidates, so the dispatch guard rejected the very
  // substitution the boundary had just issued. The projection re-derives the
  // authored graph, so the resumed turn must carry it.
  const dir = mkdtempSync(join(tmpdir(), 'harnessmm-'));
  const receipts = join(dir, 'r.jsonl');
  const user = [{ role: 'user', content: [{ type: 'text', text: 'run the tests' }] }];
  const authored = fakeHost(config(true, receipts));
  await collect(authored.handlers.get('llm/stream')(
    { sessionId: SESSION, messages: user, tools: TOOLS },
    slowStreamWithBank(),
  ));

  // A fresh host with an empty per-session store models a restart between the
  // authoring turn and the next projected request.
  const restarted = fakeHost(config(true, receipts));
  const rootArgs = '{"description":"Bootstrap","command":"ls"}';
  const messages = [
    ...user,
    {
      role: 'assistant',
      content: [
        { type: 'text', text: '<CHISEL_PLAN>' + JSON.stringify({ candidates: [CAND] }) + '</CHISEL_PLAN>' },
        { type: 'tool-call', id: 'root_call', name: 'pwsh', arguments: rootArgs },
      ],
    },
    { role: 'tool', toolCallId: 'root_call', content: [{ type: 'text', text: '(no output)' }] },
  ];
  const issued = await collect(restarted.handlers.get('llm/stream')(
    { sessionId: SESSION, messages, tools: TOOLS },
    () => (async function* () { yield { type: 'finish', reason: { kind: 'stop' } }; })(),
  ));
  const call = issued.find(chunk => chunk.type === 'block-end').block;
  const bank = parseBank({ role: 'assistant', content: messages[1].content }, TOOLS, { maxCandidates: 16 });
  assert.equal(call.id, nativeCallId(bank.sha256, 'A'));

  let ran = false;
  await restarted.handlers.get('tools/execute')(
    { agent, callId: call.id, name: 'pwsh', arguments: CAND.arguments, messages, tools: TOOLS },
    async () => { ran = true; return { isError: false, content: [] }; },
  );
  assert.equal(ran, true, 'the issued call must survive the dispatch guard after a restart');

  restarted.handlers.get('tools/result')(
    { agent, callId: call.id, name: 'pwsh', arguments: CAND.arguments, messages },
    { isError: false, value: { kind: 'foreground', exitCode: 0 }, content: [{ type: 'text', text: 'ok' }] },
  );
  const entries = readFileSync(receipts, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  const certificate = entries.filter(entry => entry.type === 'substitution').at(-1);
  assert.equal(certificate.bank_sha256, bank.sha256);
  assert.equal(certificate.exact_native_tool, 'pwsh');
});

test('the dispatch guard accepts the authored call and rejects a tampered one', async t => {
  const receipts = testReceipts(t);
  const host = fakeHost(config(true, receipts));
  const stream = host.handlers.get('llm/stream');
  await collect(stream({ sessionId: SESSION, messages: [{ role: 'user', content: [{ type: 'text', text: 'run the tests' }] }], tools: TOOLS }, slowStreamWithBank()));

  const rootArgs = '{"description":"Bootstrap","command":"ls"}';
  const plan = '<CHISEL_PLAN>' + JSON.stringify({ candidates: [CAND] }) + '</CHISEL_PLAN>';
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'run the tests' }] },
    { role: 'assistant', content: [{ type: 'text', text: plan }, { type: 'tool-call', id: 'root_call', name: 'pwsh', arguments: rootArgs }] },
    { role: 'tool', toolCallId: 'root_call', content: [{ type: 'text', text: '(no output)' }] },
  ];
  const issued = await collect(stream({ sessionId: SESSION, messages, tools: TOOLS }, () => (async function* () { yield { type: 'finish', reason: { kind: 'stop' } }; })()));
  const callId = issued.find(chunk => chunk.type === 'block-end').block.id;
  const guard = host.handlers.get('tools/execute');
  let ran = false;
  await guard(
    { agent, callId, name: 'pwsh', arguments: CAND.arguments, messages, tools: TOOLS },
    async () => { ran = true; return { isError: false, content: [] }; },
  );
  assert.equal(ran, true, 'the authored call must reach the native executor');

  await assert.rejects(
    () => guard(
      { agent, callId, name: 'pwsh', arguments: { ...CAND.arguments, command: 'echo tampered' }, messages, tools: TOOLS },
      async () => assert.fail('a tampered call must never reach the native executor'),
    ),
    /HarnessMM rejected a substituted call/,
  );
  assert.equal(readsReceipts(receipts).find(entry => entry.type === 'dispatch-rejected').call_id, callId);
  const status = host.tools.get('harnessmm_status');
  assert.deepEqual(await status.execute({}, { agent }), {
    enabled: true,
    banks_authored: 1,
    substitutions_issued: 1,
    substitutions_certified: 0,
    boundary_errors: 0,
    dispatch_rejections: 1,
  });
  assert.deepEqual(await status.execute({}, { agent: { id: 'another-session' } }), {
    enabled: true,
    banks_authored: 0,
    substitutions_issued: 0,
    substitutions_certified: 0,
    boundary_errors: 0,
    dispatch_rejections: 0,
  });
});

test('a single surviving candidate commits even while the selector is enabled', async () => {
  // Regression: `deterministic` used to be derived from `selector.enabled === false`,
  // so with the shipped config (selector on) a one-candidate bank still paid for a
  // selector call and then failed the net-saving threshold, and nothing was ever
  // substituted. The property belongs to the surviving set, not to the config.
  const dir = mkdtempSync(join(tmpdir(), 'harnessmm-'));
  const host = fakeHost(config(true, join(dir, 'r.jsonl')));
  const stream = host.handlers.get('llm/stream');
  await collect(stream({ sessionId: SESSION, messages: [{ role: 'user', content: [{ type: 'text', text: 'run the tests' }] }], tools: TOOLS }, slowStreamWithBank()));

  const rootArgs = '{"description":"Bootstrap","command":"ls"}';
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'run the tests' }] },
    {
      role: 'assistant',
      content: [
        { type: 'text', text: '<CHISEL_PLAN>' + JSON.stringify({ candidates: [CAND] }) + '</CHISEL_PLAN>' },
        { type: 'tool-call', id: 'root_call', name: 'pwsh', arguments: rootArgs },
      ],
    },
    { role: 'tool', toolCallId: 'root_call', content: [{ type: 'text', text: '(no output)' }] },
  ];
  let slowCalled = false;
  // The selector would be reached here only through the slow path, so a stream that
  // records the call proves the commit path took over instead.
  const issued = await collect(stream({ sessionId: SESSION, messages, tools: TOOLS }, () => (async function* () {
    slowCalled = true;
    yield { type: 'finish', reason: { kind: 'stop' } };
  })()));

  assert.equal(slowCalled, false, 'a single candidate must not need any selector round trip');
  const call = issued.find(chunk => chunk.type === 'block-end').block;
  assert.equal(call.name, 'pwsh');
  assert.equal(call.arguments, JSON.stringify(CAND.arguments));

  const entries = readFileSync(join(dir, 'r.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line));
  assert.ok(entries.some(entry => entry.type === 'decision' && entry.kind === 'commit'));
});

test('the certificate carries the native result, not just the execution object', async () => {
  // Regression: the listener used to read only the first argument. The host
  // contract is `tools/result(exec, result)`, so the receipt decoded from `exec`
  // always advertised an empty stdout and a successful exit — a failing
  // substitution looked exactly like a passing one.
  const dir = mkdtempSync(join(tmpdir(), 'harnessmm-'));
  const host = fakeHost(config(true, join(dir, 'r.jsonl')));
  const stream = host.handlers.get('llm/stream');
  await collect(stream({ sessionId: SESSION, messages: [{ role: 'user', content: [{ type: 'text', text: 'run the tests' }] }], tools: TOOLS }, slowStreamWithBank()));

  const rootArgs = '{"description":"Bootstrap","command":"ls"}';
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'run the tests' }] },
    {
      role: 'assistant',
      content: [
        { type: 'text', text: '<CHISEL_PLAN>' + JSON.stringify({ candidates: [CAND] }) + '</CHISEL_PLAN>' },
        { type: 'tool-call', id: 'root_call', name: 'pwsh', arguments: rootArgs },
      ],
    },
    { role: 'tool', toolCallId: 'root_call', content: [{ type: 'text', text: '(no output)' }] },
  ];
  const issued = await collect(stream({ sessionId: SESSION, messages, tools: TOOLS }, () => (async function* () {
    yield { type: 'finish', reason: { kind: 'stop' } };
  })()));
  const call = issued.find(chunk => chunk.type === 'block-end').block;

  const certificateOf = async (sessionId, result) => {
    await collect(stream({ sessionId, messages, tools: TOOLS }, () => (async function* () {
      yield { type: 'finish', reason: { kind: 'stop' } };
    })()));
    host.handlers.get('tools/result')(
      { agent: { id: sessionId }, callId: call.id, name: 'pwsh', arguments: CAND.arguments },
      result,
    );
    const entries = readFileSync(join(dir, 'r.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line));
    return entries.filter(entry => entry.type === 'substitution').at(-1);
  };

  const passed = await certificateOf(SESSION, {
    isError: false,
    value: { kind: 'foreground', exitCode: 0 },
    content: [{ type: 'text', text: 'collected 3 items' }],
  });
  assert.equal(passed.native_receipt.status, 'PARSED');
  assert.equal(passed.native_receipt.is_error, false);
  assert.equal(passed.native_receipt.value.ok, true);
  assert.equal(passed.native_receipt.value.exit_code, 0);
  assert.equal(passed.native_receipt.value.stdout, 'collected 3 items');
  // Fix #1 (certificate unification): the live receipt must be the one shape the
  // pure controller defines, plus the two evidence fields. Without the hashes the
  // certificate cannot prove which native action/result it binds.
  assert.equal(passed.schema, 'harnessmm-desktop-certificate/v1');
  assert.equal(passed.native_receipt_hash, hashJson(passed.native_receipt));
  assert.equal(passed.native_action_hash, hashJson(['pwsh', JSON.stringify(CAND.arguments)]));
  assert.deepEqual(passed.control_usage, { inputTokens: 0, outputTokens: 0, totalTokens: 0 });
  assert.equal(passed.slow_request_tokens_at_authoring, 1300);
  assert.equal(passed.selector, 'deterministic');
  assert.equal(passed.valid_substitution, true);
  assert.equal(passed.amount_substituted_slow_callbacks, 1);
  assert.ok('bypassed_slow_callback_id' in passed);

  const failed = await certificateOf('session-2', {
    isError: true,
    error: { message: 'exit 1' },
    content: [{ type: 'text', text: 'boom' }],
  });
  assert.equal(failed.native_receipt.is_error, true);
  assert.equal(failed.native_receipt.value.ok, false);
  assert.equal(failed.native_receipt.value.stderr, 'boom');
});

test('the boundary decodes the frozen root receipt, not the rendered text', async () => {
  // Regression (audit P0-1): `canonicalOfMessage` could not see structured
  // data, so a failed root whose rendered text carried no `[exit code: N]`
  // trailer was decoded as exit 0 and could authorize a substitution.
  const dir = mkdtempSync(join(tmpdir(), 'harnessmm-'));
  const receipts = join(dir, 'r.jsonl');
  const host = fakeHost(config(true, receipts));
  const stream = host.handlers.get('llm/stream');
  await collect(stream({ sessionId: SESSION, messages: [{ role: 'user', content: [{ type: 'text', text: 'run the tests' }] }], tools: TOOLS }, slowStreamWithBank()));

  // The root really failed: the frozen result says exit 1, while the rendered
  // text says only "boom" — no renderer trailer, no exit marker.
  host.handlers.get('tools/result')(
    { agent, callId: 'root_call', name: 'pwsh', arguments: '{"description":"Bootstrap","command":"ls"}' },
    { isError: false, value: { kind: 'foreground', exitCode: 1 }, content: [{ type: 'text', text: 'boom' }] },
  );

  const rootArgs = '{"description":"Bootstrap","command":"ls"}';
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'run the tests' }] },
    {
      role: 'assistant',
      content: [
        { type: 'text', text: '<CHISEL_PLAN>' + JSON.stringify({ candidates: [CAND] }) + '</CHISEL_PLAN>' },
        { type: 'tool-call', id: 'root_call', name: 'pwsh', arguments: rootArgs },
      ],
    },
    { role: 'tool', toolCallId: 'root_call', content: [{ type: 'text', text: 'boom' }] },
  ];
  let slowCalled = false;
  await collect(stream({ sessionId: SESSION, messages, tools: TOOLS }, () => (async function* () {
    slowCalled = true;
    yield { type: 'finish', reason: { kind: 'stop' } };
  })()));

  assert.equal(slowCalled, true, 'a guard for exit 0 must reject a structured exit 1 even when the text has no marker');
  const entries = readFileSync(receipts, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  const decision = entries.filter(entry => entry.type === 'decision').at(-1);
  assert.equal(decision.kind, 'slow');
  assert.deepEqual(decision.rejected, [{ id: 'A', reason: 'guard_false' }]);

  // A structured success still commits deterministically: the fallback is not
  // weakened, only made authoritative when a frozen result is available.
  host.handlers.get('tools/result')(
    { agent, callId: 'root_call', name: 'pwsh', arguments: '{"description":"Bootstrap","command":"ls"}' },
    { isError: false, value: { kind: 'foreground', exitCode: 0 }, content: [{ type: 'text', text: 'ok' }] },
  );
  let slowAgain = false;
  const committed = await collect(stream({ sessionId: SESSION, messages, tools: TOOLS }, () => (async function* () {
    slowAgain = true;
    yield { type: 'finish', reason: { kind: 'stop' } };
  })()));
  assert.equal(slowAgain, false);
  assert.equal(committed.find(chunk => chunk.type === 'block-end').block.name, 'pwsh');
});

test('unrelated tool calls bypass the guard untouched', async () => {
  const host = fakeHost(config(true, ''));
  const guard = host.handlers.get('tools/execute');
  let ran = false;
  await guard({ agent, callId: 'call_ordinary', name: 'pwsh', arguments: { command: 'ls' } }, async () => { ran = true; });
  assert.equal(ran, true);
});

test('Config accepts the shipped defaults and rejects out-of-range policy numbers', () => {
  // schemastery resolves a declared group into a plain object whose fields are
  // live sources, so a field is read through its `get()`.
  const resolve = source => (typeof source?.get === 'function' ? source.get() : source);
  const resolved = Config({});
  assert.equal(resolve(resolved.policy.enabled), false);
  assert.equal(resolve(resolved.policy.maximum_candidates), 16);
  assert.equal(resolve(resolved.policy.maximum_fast_steps), 8);
  assert.equal(resolve(resolved.policy.maximum_public_bytes), 65536);
  assert.equal(resolve(resolved.selector.maxTokens), 1024);

  const switchedOn = Config({ policy: { enabled: true, maximum_candidates: 4 } });
  assert.equal(resolve(switchedOn.policy.enabled), true);
  assert.equal(resolve(switchedOn.policy.maximum_candidates), 4);

  assert.throws(() => Config({ policy: { maximum_candidates: 99 } }));
  assert.throws(() => Config({ policy: { maximum_fast_steps: 0 } }));
  assert.throws(() => Config({ policy: { maximum_public_bytes: 8 } }));
});

/* ------------------------------------------------------------------------- *
 * Multi-candidate steps: the selector path and its failure fallback
 * ------------------------------------------------------------------------- */

const OTHER = {
  ...CAND,
  candidate_id: 'B',
  arguments: { description: 'Run the linter', command: 'ruff check .' },
};
const ROOT_ARGS = '{"description":"Bootstrap","command":"ls"}';
const ROOT_CALL = { type: 'tool-call', id: 'root_call', name: 'pwsh', arguments: ROOT_ARGS };

/** A slow stream that declares a bank of `candidates` and makes one root call. */
function slowWithPlan(candidates) {
  const text = `<CHISEL_PLAN>${JSON.stringify({ candidates })}</CHISEL_PLAN>`;
  return async function* stream() {
    yield { type: 'block-start', index: 0, blockType: 'text' };
    yield { type: 'text-delta', index: 0, text };
    yield { type: 'block-end', index: 0, block: { type: 'text', text } };
    yield { type: 'block-start', index: 1, blockType: 'tool-call' };
    yield { type: 'block-end', index: 1, block: ROOT_CALL };
    // Large enough that the net-saving admission accepts a non-deterministic step.
    yield { type: 'usage', usage: { inputTokens: 20000, outputTokens: 400, totalTokens: 20400 } };
    yield { type: 'finish', reason: { kind: 'tool-calls' } };
  };
}

/** The request the loop makes once the root call is armed and answered. */
function projectedMessages(candidates) {
  const text = `<CHISEL_PLAN>${JSON.stringify({ candidates })}</CHISEL_PLAN>`;
  return [
    { role: 'user', content: [{ type: 'text', text: 'run the tests' }] },
    { role: 'assistant', content: [{ type: 'text', text }, ROOT_CALL] },
    { role: 'tool', toolCallId: 'root_call', content: [{ type: 'text', text: '(no output)' }] },
  ];
}

function readsReceipts(path) {
  return readFileSync(path, 'utf8').trim().split('\n').map(line => JSON.parse(line));
}

test('an asynchronous controller fault falls back to the stock stream once', async t => {
  const receipts = testReceipts(t);
  const candidates = [CAND, OTHER];
  const host = fakeHost(config(true, receipts));
  const stream = host.handlers.get('llm/stream');
  await collect(stream({ sessionId: SESSION, tools: TOOLS }, slowWithPlan(candidates)));
  host.ctx.get = () => { throw new Error('LLM service lookup failed'); };

  const finish = { type: 'finish', reason: { kind: 'stop' } };
  let requests = 0;
  const chunks = await collect(stream(
    { sessionId: SESSION, messages: projectedMessages(candidates), tools: TOOLS },
    () => {
      requests++;
      return (async function* () { yield finish; })();
    },
  ));

  assert.equal(requests, 1);
  assert.deepEqual(chunks, [finish]);
  const entries = readsReceipts(receipts);
  assert.equal(entries.filter(entry => entry.type === 'boundary-error').length, 1);
  assert.equal(entries.find(entry => entry.type === 'boundary-error').reason, 'LLM service lookup failed');
  assert.equal(entries.some(entry => entry.type === 'substitution-issued'), false);
});

test('bank bookkeeping errors preserve the stock response already sent to the host', async t => {
  const receipts = testReceipts(t);
  const stream = fakeHost(config(true, receipts)).handlers.get('llm/stream');
  const options = { sessionId: SESSION, tools: TOOLS };
  const expected = await collect(slowStreamWithBank()());
  let requests = 0;
  const chunks = await collect(stream(options, async function* () {
    requests++;
    yield* expected;
    Object.defineProperty(options, 'tools', {
      get() { throw new Error('tool catalog unavailable after streaming'); },
    });
  }));

  assert.equal(requests, 1);
  assert.deepEqual(chunks, expected);
  assert.deepEqual(readsReceipts(receipts).map(entry => [entry.type, entry.reason]), [
    ['boundary-error', 'tool catalog unavailable after streaming'],
  ]);
});

test('a stock stream that throws mid-response is never requested again', async t => {
  const receipts = testReceipts(t);
  const stream = fakeHost(config(true, receipts)).handlers.get('llm/stream');
  const upstreamError = new Error('provider connection lost');
  const start = { type: 'block-start', index: 0, blockType: 'text' };
  let requests = 0;
  const iterator = stream({ sessionId: SESSION, tools: TOOLS }, () => {
    requests++;
    return (async function* () {
      yield start;
      throw upstreamError;
    })();
  });

  assert.deepEqual(await iterator.next(), { value: start, done: false });
  await assert.rejects(iterator.next(), error => error === upstreamError);
  assert.equal(requests, 1);
  assert.equal(existsSync(receipts), false);
});

function testReceipts(t) {
  const dir = mkdtempSync(join(tmpdir(), 'harnessmm-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return join(dir, 'receipts.jsonl');
}

test('a two-candidate step asks the selector and runs the letter it answered', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'harnessmm-'));
  const receipts = join(dir, 'r.jsonl');
  const candidates = [CAND, OTHER];
  let selectorCalls = 0;
  const host = fakeHost(config(true, receipts), () => {
    selectorCalls += 1;
    return (async function* () {
      yield { type: 'block-start', index: 0, blockType: 'text' };
      yield { type: 'text-delta', index: 0, text: 'B' };
      yield { type: 'block-end', index: 0, block: { type: 'text', text: 'B' } };
      yield { type: 'usage', usage: { inputTokens: 40, outputTokens: 1, totalTokens: 41 } };
      yield { type: 'finish', reason: { kind: 'stop' } };
    })();
  });
  const stream = host.handlers.get('llm/stream');

  await collect(stream(
    { sessionId: SESSION, messages: [{ role: 'user', content: [{ type: 'text', text: 'run the tests' }] }], tools: TOOLS },
    slowWithPlan(candidates),
  ));

  let slowCalled = false;
  const issued = await collect(stream(
    { sessionId: SESSION, messages: projectedMessages(candidates), tools: TOOLS },
    () => (async function* () {
      slowCalled = true;
      yield { type: 'finish', reason: { kind: 'stop' } };
    })(),
  ));

  assert.equal(slowCalled, false, 'the slow actor must not be called once the selector answered');
  assert.equal(selectorCalls, 1, 'one neighbour is exactly one selector call');
  const call = issued.find(chunk => chunk.type === 'block-end').block;
  assert.equal(call.name, 'pwsh');
  assert.equal(call.arguments, JSON.stringify(OTHER.arguments));
  assert.equal(issued.at(-1).reason.kind, 'tool-calls');

  const entries = readsReceipts(receipts);
  const decision = entries.find(entry => entry.type === 'decision');
  assert.equal(decision.kind, 'select');
  assert.deepEqual(decision.eligible, ['A', 'B']);
  const issuedEntry = entries.find(entry => entry.type === 'substitution-issued');
  assert.equal(issuedEntry.selector, 'fast-selector');
  assert.equal(issuedEntry.candidate_id, 'B');
  assert.equal(issuedEntry.control_usage.totalTokens, 41);
});

test('a failed selector call is recorded and the slow actor keeps control', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'harnessmm-'));
  const receipts = join(dir, 'r.jsonl');
  const candidates = [CAND, OTHER];
  let selectorCalls = 0;
  const host = fakeHost(config(true, receipts), () => {
    selectorCalls += 1;
    return (async function* () {
      yield { type: 'finish', reason: { kind: 'error', failure: { code: 'BAD_REQUEST', message: 'unsupported temperature' } } };
    })();
  });
  const stream = host.handlers.get('llm/stream');

  await collect(stream(
    { sessionId: SESSION, messages: [{ role: 'user', content: [{ type: 'text', text: 'run the tests' }] }], tools: TOOLS },
    slowWithPlan(candidates),
  ));

  let slowCalled = false;
  await collect(stream(
    { sessionId: SESSION, messages: projectedMessages(candidates), tools: TOOLS },
    () => (async function* () {
      slowCalled = true;
      yield { type: 'finish', reason: { kind: 'stop' } };
    })(),
  ));

  assert.equal(slowCalled, true, 'a selector that cannot answer must hand the turn back');
  assert.equal(selectorCalls, 2, 'the conservative request shape is tried once more');
  const abstain = readsReceipts(receipts).find(entry => entry.type === 'selector-abstain');
  assert.equal(abstain.reason, 'selector-error');
  assert.equal(abstain.failure.code, 'BAD_REQUEST');
  assert.equal(abstain.failure.message, 'unsupported temperature');
});

test('the selector reuses the boundary route and inherits its reasoning level', async () => {
  // Regression: the shipped selector route answered `400 ... "The value of the
  // enable_thinking parameter is restricted to True."` — a thinking-only gateway
  // refuses a call that carries no reasoning control at all. The selector now
  // rides the route the boundary request itself proves answerable and reuses the
  // level that request just used.
  const dir = mkdtempSync(join(tmpdir(), 'harnessmm-'));
  const receipts = join(dir, 'r.jsonl');
  const candidates = [CAND, OTHER];
  const asked = [];
  const host = fakeHost(config(true, receipts), options => {
    asked.push(options);
    return (async function* () {
      yield { type: 'block-start', index: 0, blockType: 'text' };
      yield { type: 'text-delta', index: 0, text: 'A' };
      yield { type: 'block-end', index: 0, block: { type: 'text', text: 'A' } };
      yield { type: 'finish', reason: { kind: 'stop' } };
    })();
  });
  const stream = host.handlers.get('llm/stream');

  const boundary = {
    sessionId: SESSION,
    provider: 'qwen-token-plan-cn',
    model: 'deepseek-v4.1-flash',
    reasoningEffort: 'max',
    tools: TOOLS,
  };
  await collect(stream({ ...boundary, messages: [{ role: 'user', content: [{ type: 'text', text: 'run the tests' }] }] }, slowWithPlan(candidates)));
  const issued = await collect(stream({ ...boundary, messages: projectedMessages(candidates) }, () => (async function* () {
    yield { type: 'finish', reason: { kind: 'stop' } };
  })()));

  assert.equal(asked.length, 1, 'one neighbour is exactly one selector call');
  assert.equal(asked[0].provider, 'qwen-token-plan-cn', 'the selector rides the boundary route');
  assert.equal(asked[0].model, 'deepseek-v4.1-flash');
  assert.equal(asked[0].reasoningEffort, 'max', 'a thinking-only route must receive a reasoning control');
  assert.equal(asked[0].maxTokens, 1024);
  assert.equal(asked[0].messages[0].role, 'system');

  const call = issued.find(chunk => chunk.type === 'block-end').block;
  assert.equal(call.arguments, JSON.stringify(CAND.arguments));
});

test('a disabled selector hands a multi-candidate step back to the slow actor', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'harnessmm-'));
  const receipts = join(dir, 'r.jsonl');
  const candidates = [CAND, OTHER];
  let selectorCalls = 0;
  const host = fakeHost(config(true, receipts, { enabled: false }), () => {
    selectorCalls += 1;
    return (async function* () {
      yield { type: 'finish', reason: { kind: 'stop' } };
    })();
  });
  const stream = host.handlers.get('llm/stream');

  await collect(stream(
    { sessionId: SESSION, messages: [{ role: 'user', content: [{ type: 'text', text: 'run the tests' }] }], tools: TOOLS },
    slowWithPlan(candidates),
  ));

  let slowCalled = false;
  await collect(stream(
    { sessionId: SESSION, messages: projectedMessages(candidates), tools: TOOLS },
    () => (async function* () {
      slowCalled = true;
      yield { type: 'finish', reason: { kind: 'stop' } };
    })(),
  ));

  assert.equal(slowCalled, true);
  assert.equal(selectorCalls, 0, 'a switched-off selector must not spend a model call');
  const abstain = readsReceipts(receipts).find(entry => entry.type === 'selector-abstain');
  assert.equal(abstain.reason, 'selector-disabled');
});

async function collect(iterable) {
  const chunks = [];
  for await (const chunk of iterable) chunks.push(chunk);
  return chunks;
}