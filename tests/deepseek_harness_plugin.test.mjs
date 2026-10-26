import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import test from 'node:test';
import {
  apply, nativeObservationText, notificationContent,
} from '../integrations/deepseek_harness/bridge_plugin.mjs';

function fixture(t, responses, {
  context, contextHash, arm = 'baseline', parameters = { type: 'object', properties: {} },
  observationEncoding,
} = {}) {
  const root = resolve(`.pytest_cache/deepseek-plugin-unit-${randomUUID()}`);
  mkdirSync(root, { recursive: true, mode: 0o700 });
  t.after(() => rmSync(root, { recursive: true }));
  const originalFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = originalFetch; });
  const requests = [];
  globalThis.fetch = async (url, options) => {
    assert.equal(options.headers.Authorization, 'Bearer SYNTHETIC_BRIDGE_SECRET');
    requests.push({ path: url.pathname, payload: JSON.parse(options.body), signal: options.signal });
    return { ok: true, json: async () => responses(url.pathname) };
  };
  writeFileSync(`${root}/bridge.secret`, 'SYNTHETIC_BRIDGE_SECRET', { mode: 0o600 });
  writeFileSync(`${root}/job.json`, JSON.stringify({
    arm, benchmark_context: context,
    native_observation_encoding: observationEncoding,
    session_id: 'synthetic-session',
    tools: [{ name: 'native_wait', description: 'Synthetic native turn boundary',
      parameters }],
    bridge: { url: 'http://127.0.0.1:12345', secret_file: `${root}/bridge.secret`,
      notifications: true, generation_timing: true },
  }));
  writeFileSync(`${root}/message.mjs`, 'export const createUserMessage = value => value;\n');
  const hooks = new Map();
  const tools = [];
  const sections = [];
  let benchmarkContext;
  if (context !== undefined) {
    writeFileSync(`${root}/frozen-benchmark-context.txt`, context);
    benchmarkContext = {
      path: `${root}/frozen-benchmark-context.txt`,
      sha256: contextHash ?? createHash('sha256').update(context).digest('hex'),
    };
  }
  const ctx = {
    tools: { guard() {}, register(tool) { tools.push(tool); } },
    systemPrompt: { section(value) { sections.push(value); } },
    on(name, handler) { hooks.set(name, handler); },
  };
  apply(ctx, {
    jobFile: `${root}/job.json`, receiptsFile: `${root}/receipts.jsonl`,
    messageModule: `${root}/message.mjs`,
    benchmarkContext,
  });
  return { hooks, tools, sections, requests, receipts: () => readFileSync(`${root}/receipts.jsonl`, 'utf8')
    .trim().split('\n').map(line => JSON.parse(line)) };
}

test('shared public benchmark context is literal, additive, and independent of arm', t => {
  const context = 'Public native time: 2030-01-02.\nLiteral {{clock}} and 初始时间.';
  for (const arm of ['baseline', 'chisel']) {
    const f = fixture(t, () => ({}), { context, arm });
    assert.deepEqual(f.sections, [{
      name: 'benchmark:context', order: 10200, text: context, interpolate: false,
    }]);
    assert.equal(f.requests.length, 0);
  }
});

test('empty shared context preserves the original official system prompt', t => {
  for (const context of [undefined, '']) {
    const f = fixture(t, () => ({}), { context });
    assert.deepEqual(f.sections, []);
  }
});

test('changed frozen common context fails before any provider or bridge request', t => {
  assert.throws(() => fixture(t, () => assert.fail('network must not be called'), {
    context: 'public native context', contextHash: '0'.repeat(64),
  }), /benchmark context SHA256 mismatch/);
});

test('native optional JSON Schema and omitted/null arguments reach execution unchanged', async t => {
  const parameters = {
    type: 'object', title: 'NativeAppArguments', additionalProperties: false,
    properties: {
      value: { type: 'string', title: 'Value' },
      limit: { anyOf: [{ type: 'integer', minimum: 1 }, { type: 'null' }], default: null },
      label: { anyOf: [{ type: 'string' }, { type: 'null' }], default: null },
    },
    required: ['value'],
  };
  const f = fixture(t, () => ({ observation: 'synthetic result', conclude_turn: false }), { parameters });
  assert.deepEqual(f.tools[0].parameters, parameters);
  for (const args of [{ value: 'probe' }, { value: 'probe', limit: null, label: null }]) {
    const value = await f.tools[0].execute(args, {
      agent: { session: { id: 'same-session' } }, callId: `call-${f.requests.length}`,
      signal: new AbortController().signal, concludeTurn() { assert.fail('unexpected turn conclusion'); },
    });
    assert.deepEqual(f.requests.at(-1).payload.arguments, args);
    assert.deepEqual(f.tools[0].output.render(args, value), [{ type: 'text', text: '"synthetic result"' }]);
  }
});

test('native text encoding preserves original observations rather than JSON-string quoting', async t => {
  const observation = "Contact ID: native-id\nPhone: +123\nLiteral \"quoted\" text";
  assert.equal(nativeObservationText(observation, 'native_text'), observation);
  assert.equal(nativeObservationText(observation), JSON.stringify(observation));
  assert.throws(() => nativeObservationText({ native: 'record' }, 'native_text'), /original string/);
  assert.throws(() => nativeObservationText(observation, 'unknown'), /declared encoding/);
  for (const arm of ['baseline', 'chisel']) {
    const f = fixture(t, () => ({ observation, conclude_turn: false }), {
      arm, observationEncoding: 'native_text',
    });
    const value = await f.tools[0].execute({}, {
      agent: { session: { id: 'synthetic' } }, callId: 'synthetic-call',
      signal: new AbortController().signal, concludeTurn() { assert.fail('unexpected conclusion'); },
    });
    assert.equal(value, observation);
    assert.deepEqual(f.tools[0].output.render({}, value), [{ type: 'text', text: observation }]);
    assert.equal(f.receipts().at(-1).result.observation, observation);
  }
});

test('invalid native text policy fails before any native request', t => {
  assert.throws(() => fixture(t, () => assert.fail('native request not permitted'), {
    observationEncoding: 'unknown',
  }), /Unsupported native observation encoding/);
});

test('official notification content preserves text and image parts literally', () => {
  const parts = [{ type: 'text', text: 'native notice' },
    { type: 'image', data: Buffer.from('synthetic image').toString('base64'), mimeType: 'image/png' }];
  assert.equal(notificationContent(parts), parts);
  assert.deepEqual(notificationContent('plain notice'), [{ type: 'text', text: 'plain notice' }]);
  for (const invalid of [[], null, {}, [{ type: 'audio', data: 'YWJj' }],
    [{ type: 'image_url', image_url: { url: 'https://invalid.example/' } }],
    [{ type: 'image', data: 'https://invalid.example/', mimeType: 'image/png' }],
    [{ type: 'image', data: 'YWJj', mimeType: 'application/pdf' }],
    [{ type: 'text', text: 4 }]]) {
    assert.throws(() => notificationContent(invalid));
  }
});

test('notification hook forwards inline images without creating URLs or dropping parts', async t => {
  const image = { type: 'image', data: 'c3ludGhldGlj', mimeType: 'image/jpeg' };
  const parts = [{ type: 'text', text: 'native attachment' }, image];
  const f = fixture(t, () => ({ messages: ['plain text', parts], stop_requested: false }));
  const original = { kind: 'enter', messages: [{ content: [{ type: 'text', text: 'user turn' }] }] };
  const result = await f.hooks.get('agent/pre-step')({
    agent: { session: { id: 'same-session' } }, turn: 2, step: 1,
  }, async () => original);
  assert.equal(result.messages.length, 3);
  assert.deepEqual(result.messages[2].content, parts);
  assert.deepEqual(f.requests[0].payload, { session_id: 'same-session' });
  assert.deepEqual(f.receipts()[0].messages[1][1], image);
});

test('environment stop rejects the next official step without a model call', async t => {
  const f = fixture(t, () => ({ messages: [], stop_requested: true }));
  const result = await f.hooks.get('agent/pre-step')({
    agent: { session: { id: 'same-session' } }, turn: 3, step: 1,
  }, async () => ({ kind: 'enter', messages: [] }));
  assert.deepEqual(result, { kind: 'reject' });
  assert.deepEqual(f.requests.map(item => item.path), ['/notifications']);
  assert.equal(f.receipts().at(-1).kind, 'bridge/environment-stop');
});

test('unsupported notification types fail instead of silently becoming text', async t => {
  const f = fixture(t, () => ({ messages: [[{ type: 'audio', data: 'YWJj' }]], stop_requested: false }));
  await assert.rejects(f.hooks.get('agent/pre-step')({
    agent: { session: { id: 'same-session' } }, turn: 1, step: 1,
  }, async () => ({ kind: 'enter', messages: [] })), /Unsupported/);
});

test('native wait concludes only its turn; later native calls retain the session and signal', async t => {
  const f = fixture(t, () => ({ observation: 'waiting', conclude_turn: true, is_error: false }));
  const signal = new AbortController().signal;
  let conclusions = 0;
  for (const callId of ['call-1', 'call-2']) {
    await f.tools[0].execute({}, { agent: { session: { id: 'same-session' } },
      callId, signal, concludeTurn() { conclusions++; } });
  }
  assert.equal(conclusions, 2);
  assert.deepEqual(f.requests.map(item => item.payload), [
    { session_id: 'same-session', call_id: 'call-1', name: 'native_wait', arguments: {} },
    { session_id: 'same-session', call_id: 'call-2', name: 'native_wait', arguments: {} },
  ]);
  assert(f.requests.every(item => item.signal === signal));
});

test('all stream chunks are forwarded unchanged and generation clocks stay paired', async t => {
  const f = fixture(t, () => ({ ok: true }));
  const chunks = [{ type: 'text-delta', text: 'synthetic' },
    { type: 'usage', usage: { inputTokens: 1, outputTokens: 2, cacheReadTokens: 0, cacheWriteTokens: 0 } },
    { type: 'finish', reason: { kind: 'completed' }, replayState: { opaque: 'provider-data' } }];
  const stream = f.hooks.get('llm/stream')({ sessionId: 'same-session' },
    async function* () { yield* chunks; });
  const received = [];
  for await (const chunk of stream) received.push(chunk);
  assert(received.every((chunk, index) => chunk === chunks[index]));
  assert.deepEqual(f.requests.map(item => item.path), ['/generation/start', '/generation/end']);
  assert.equal(f.requests[0].payload.attempt_id, f.requests[1].payload.attempt_id);
  assert(f.requests[1].payload.elapsed_seconds >= 0);
  assert.equal(f.receipts().filter(item => item.kind === 'llm/chunk').length, chunks.length);
});

test('aborted streams still resume native generation clocks without the aborted signal', async t => {
  const f = fixture(t, () => ({ ok: true }));
  const abort = new AbortController();
  const stream = f.hooks.get('llm/stream')({ sessionId: 'same-session', signal: abort.signal },
    async function* () { abort.abort(); throw new Error('synthetic abort'); });
  await assert.rejects(async () => { for await (const _chunk of stream) {} }, /synthetic abort/);
  assert.equal(f.requests.at(-1).path, '/generation/end');
  assert.equal(f.requests.at(-1).signal, undefined);
  assert.equal(f.receipts().at(-1).completed, false);
});
