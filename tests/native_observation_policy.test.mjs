import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import test from 'node:test';
import {
  NativeObservationPolicy, RECEIPT_PREFIX, RECALL_PREFIX, apply,
  completePaidUsage, validatePolicy,
} from '../integrations/deepseek_harness/observation_policy.mjs';

const POLICY = {
  version: 1, minimum_characters: 4096, full_exposures: 2, keep_recent: 2,
  preview_characters: 160, page_characters: 8000,
};
const USAGE = { inputTokens: 1000, outputTokens: 25, cacheReadTokens: 5000, cacheWriteTokens: 0 };
const freeze = value => Object.freeze(structuredClone(value));
const meter = { estimateMessage: message => message.content[0].text.length / 4 };
const message = (id, text, extra = {}) => ({
  id: `message-${id}`, role: 'tool', toolCallId: id, source: { kind: 'tool', callId: id },
  content: [{ type: 'text', text }], ...extra,
});

function fixture(t, changes = {}) {
  const root = resolve(`.pytest_cache/native-context-unit-${randomUUID()}`);
  mkdirSync(root, { recursive: true });
  t.after(() => rmSync(root, { recursive: true }));
  const records = [];
  const policy = new NativeObservationPolicy({
    directory: join(root, 'archive'), policy: { ...POLICY, ...changes },
    sessionId: 'synthetic-session', record: value => records.push(value),
  });
  return { policy, records, root };
}

function session(messages) {
  const events = new Map(messages.map((item, seq) => [seq, {
    seq, type: 'tool/result', data: { message: freeze(item), unrelated: 'preserved' },
  }]));
  let sequence = events.size;
  return {
    id: 'synthetic-session', surface: { nodes: [...events.keys()] }, events, operations: [],
    eventAt(seq) { return events.get(seq); },
    deriveEventMessage(event) { return event.data.message; },
    append(type, data, options) {
      const event = { seq: sequence++, type, data };
      events.set(event.seq, event);
      this.operations.push({ type, data, options });
      if (options) {
        const index = this.surface.nodes.indexOf(options.surfaceOp.startSeq);
        assert(index >= 0);
        this.surface.nodes[index] = event.seq;
      }
      return event;
    },
  };
}

function fullHistory(text = 'unaltered synthetic output\n'.repeat(500), extra = {}) {
  return [message('old', text, extra), message('recent-1', 'recent output'),
    message('recent-2', 'latest output')];
}

test('policy requires a frozen complete schema and protects first exposures', () => {
  assert.deepEqual(validatePolicy(POLICY), POLICY);
  for (const bad of [null, {}, { ...POLICY, extra: true }, { ...POLICY, full_exposures: 1 },
    { ...POLICY, page_characters: 9000 }, { ...POLICY, keep_recent: true },
    { ...POLICY, preview_characters: 2048 }]) {
    assert.throws(() => validatePolicy(bad), /Invalid frozen/);
  }
});

test('cache buckets are counted once and incomplete paid usage is not an exposure', () => {
  assert(completePaidUsage(USAGE));
  assert(completePaidUsage({ inputTokens: 1000, outputTokens: 25, totalTokens: 6025 }));
  for (const bad of [null, {}, { ...USAGE, totalTokens: 1025 }, { ...USAGE, inputTokens: true },
    { inputTokens: 1, outputTokens: 1 }, { ...USAGE, cacheReadTokens: -1 }]) {
    assert.equal(completePaidUsage(bad), false);
  }
});

test('two settled full exposures precede replay-safe replacement; native history survives', t => {
  const f = fixture(t);
  const history = fullHistory();
  const original = structuredClone(history);
  const s = session(history);
  f.policy.compact(s, meter, freeze);
  assert.equal(s.operations.length, 0);
  f.policy.observe(history, USAGE);
  f.policy.compact(s, meter, freeze);
  assert.equal(s.operations.length, 0);
  f.policy.observe(history, { inputTokens: 1, outputTokens: 1 });
  f.policy.compact(s, meter, freeze);
  assert.equal(s.operations.length, 0);
  f.policy.observe([...history, history[0]], USAGE);
  f.policy.compact(s, meter, freeze);
  assert.equal(s.operations.length, 2);
  assert.equal(s.operations[0].type, 'compaction/prune');
  const change = s.operations[1];
  assert.deepEqual(change.options, {
    surfaceOp: { op: 'replace', startSeq: 0, endSeq: 0 }, sourceEventSeqs: [0],
  });
  assert(change.data.message.content[0].text.startsWith(RECEIPT_PREFIX));
  assert.equal(change.data.message.id, history[0].id);
  assert.deepEqual(change.data.message.source, history[0].source);
  assert.equal(change.data.unrelated, 'preserved');
  assert.deepEqual(s.eventAt(0).data.message, history[0]);
  assert.deepEqual(history, original);
  assert.equal(f.records.at(-1).full_paid_exposures, 2);
  assert.equal(f.records.at(-1).native_request_rewritten, false);
  f.policy.compact(s, meter, freeze);
  assert.equal(s.operations.length, 2);
});

test('exact recall follows code-point offsets, includes middle evidence and verifies hashes', t => {
  const f = fixture(t, { page_characters: 17 });
  const source = 'header\n' + 'evidence \u{1F680} \u4E2D\u6587\n'.repeat(20) + 'tail\n';
  const id = f.policy.archive(source);
  let recovered = '';
  let offset = 0;
  do {
    const result = f.policy.recall({ observation_id: id, offset }, 'recall-call');
    assert(result.startsWith(RECALL_PREFIX));
    const page = JSON.parse(result.split('\n').slice(1).join('\n'));
    recovered += page.text;
    offset = page.next_offset;
  } while (offset !== null);
  assert.equal(recovered, source);
  for (const args of [
    { observation_id: id, offset: null }, { observation_id: id, limit: null },
    { observation_id: id, offset: -1 }, { observation_id: id, limit: 18 },
    { observation_id: id, offset: 10000 }, { observation_id: '../secret' },
    { observation_id: id, arbitrary: true },
  ]) assert.throws(() => f.policy.recall(args, 'bad-call'));
  const archive = join(f.root, 'archive', `${id}.json`);
  const damaged = JSON.parse(readFileSync(archive, 'utf8'));
  damaged.text += 'changed';
  writeFileSync(archive, JSON.stringify(damaged));
  assert.throws(() => f.policy.recall({ observation_id: id }, 'bad-call'), /checksum mismatch/);
});

test('errors, recent results, rich content and recalled observations remain intact', t => {
  const f = fixture(t);
  for (const history of [
    fullHistory(undefined, { isError: true }),
    fullHistory(JSON.stringify(JSON.stringify({ stdout: 'log'.repeat(4000), return_code: 1 }))),
    fullHistory('FAILED test_x\n' + 'log'.repeat(4000)),
    fullHistory(RECALL_PREFIX + 'log'.repeat(4000)),
    fullHistory(undefined, { content: [{ type: 'image', data: 'evidence' }] }),
    fullHistory().map((item, index) => index === 0 ? message('recall-call', item.content[0].text) : item),
  ]) {
    f.policy.recalledCalls.add('recall-call');
    const s = session(history);
    f.policy.observe(history, USAGE);
    f.policy.observe(history, USAGE);
    f.policy.compact(s, meter, freeze);
    assert.equal(s.operations.length, 0);
  }
  const recent = session([message('one', 'text'.repeat(4000))]);
  f.policy.observe([recent.eventAt(0).data.message], USAGE);
  f.policy.observe([recent.eventAt(0).data.message], USAGE);
  f.policy.compact(recent, meter, freeze);
  assert.equal(recent.operations.length, 0);
});

test('archive failure preserves the original and writes an explicit audit event', t => {
  const f = fixture(t);
  const history = fullHistory();
  const s = session(history);
  f.policy.observe(history, USAGE);
  f.policy.observe(history, USAGE);
  f.policy.archive = () => { throw new Error('synthetic archive unavailable'); };
  f.policy.compact(s, meter, freeze);
  assert.equal(s.operations.length, 0);
  assert.equal(f.records.at(-1).reason, 'archive_error');
  assert.equal(f.records.at(-1).error.message, 'synthetic archive unavailable');
});

test('incompatible native API and invalid meter values fail explicitly', t => {
  const f = fixture(t);
  assert.throws(() => f.policy.compact({}, meter, freeze), /surface API/);
  const history = fullHistory();
  f.policy.observe(history, USAGE);
  f.policy.observe(history, USAGE);
  assert.throws(() => f.policy.compact(session(history), {
    estimateMessage: () => Number.NaN,
  }, freeze), /token-meter estimate/);
});

test('the plugin forwards frozen requests and all stream chunks unchanged', async t => {
  const f = fixture(t);
  const job = { session_id: 'synthetic-session', observation_policy: POLICY };
  writeFileSync(join(f.root, 'job.json'), JSON.stringify(job));
  writeFileSync(join(f.root, 'message.mjs'),
    'export const freezeMessage = value => Object.freeze(structuredClone(value));\n');
  const hooks = new Map();
  const tools = [];
  const initialized = apply({
    tokenMeter: meter, tools: { register: tool => tools.push(tool) },
    on: (name, callback) => hooks.set(name, callback),
  }, { jobFile: join(f.root, 'job.json'), messageModule: join(f.root, 'message.mjs'),
    archiveDirectory: join(f.root, 'plugin-archive'), receiptsFile: join(f.root, 'audit.jsonl') });
  assert.equal(initialized, undefined);
  assert.equal(tools.length, 1, 'The recall tool must exist before native agent/created fires');
  const options = freeze({ sessionId: 'synthetic-session', messages: fullHistory() });
  const before = JSON.stringify(options);
  const chunks = [{ type: 'text', text: 'native output' }, { type: 'usage', usage: USAGE }];
  for (let round = 0; round < 2; round++) {
    const received = [];
    for await (const chunk of hooks.get('llm/stream')(options, async function* () {
      yield* chunks;
    })) received.push(chunk);
    assert(received.every((chunk, index) => chunk === chunks[index]));
  }
  assert.equal(JSON.stringify(options), before);
  assert.equal(tools.length, 1);
  const s = session(options.messages);
  const decision = { kind: 'enter', messages: [] };
  assert.equal(await hooks.get('agent/pre-step')({
    agent: { session: s }, signal: new AbortController().signal,
  }, async () => decision), decision);
  assert.equal(s.operations.length, 2);
  await assert.rejects(async () => tools[0].execute({ observation_id: 'obs_' + '0'.repeat(64) }, {
    agent: { session: { id: 'different-session' } }, callId: 'recall',
  }), /session mismatch/);
});
