import assert from 'node:assert/strict';
import test from 'node:test';
import {
  NativeCotJev, addControlUsage, chatDecision, decisionState, parseBank,
  planInstruction, publicReceipt, validateCotJev,
} from '../integrations/deepseek_harness/cot_jev.mjs';

const policy = {
  version: 1, enabled: true, maximum_candidates: 16,
  maximum_fast_steps: 16, maximum_public_bytes: 32768,
};
const candidate = {
  label: 'A', after: 'root', condition: 'The root returned READY.',
  tool: 'terminal', arguments: { command: 'printf DONE', cwd: '/tmp', timeout_sec: 10 },
};
const root = { type: 'tool-call', id: 'native-root', name: 'terminal',
  arguments: '{"command":"printf READY"}' };

function slowMessage(candidates = [candidate]) {
  return { role: 'assistant', content: [
    { type: 'text', text: `<CHISEL_PLAN>${JSON.stringify({ candidates })}</CHISEL_PLAN>` },
    { ...root },
  ] };
}

function options(candidates) {
  return {
    model: 'deepseek-flash', provider: 'deepseek-official',
    messages: [
      { role: 'user', content: [{ type: 'text', text: 'Create and check the synthetic marker.' }] },
      slowMessage(candidates),
      { role: 'tool', toolCallId: root.id,
        content: [{ type: 'text', text: '{"exit_code":0,"stdout":"READY"}' }] },
    ],
    tools: [{ name: 'terminal', description: 'Unchanged native terminal',
      parameters: { type: 'object', properties: { command: { type: 'string' } },
        required: ['command'] } }],
  };
}

function response(label = 'A') {
  return {
    model: 'deepseek-flash',
    usage: { prompt_tokens: 120, completion_tokens: 1, total_tokens: 121,
      prompt_cache_hit_tokens: 20, prompt_cache_miss_tokens: 100 },
    choices: [{ message: { content: label },
      logprobs: { content: [{ token: label, logprob: Math.log(0.9) }] } }],
  };
}

function selector(label = 'A', overrides = {}) {
  const records = [];
  const requests = [];
  const instance = new NativeCotJev({
    policy, model: 'deepseek-flash', apiKey: 'synthetic-not-a-credential',
    record: value => records.push(value),
    request: async (url, settings) => {
      requests.push({ url, ...settings });
      return { ok: true, status: 200, text: async () => JSON.stringify(response(label)) };
    },
    ...overrides,
  });
  return { instance, records, requests };
}

async function collect(stream) {
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return chunks;
}

const slowUsage = {
  inputTokens: 30, cacheReadTokens: 10, cacheWriteTokens: 0,
  outputTokens: 5, totalTokens: 45,
};
async function* fallback() {
  yield { type: 'usage', usage: slowUsage };
  yield { type: 'finish', reason: { kind: 'stop' } };
}

test('closed prospective policy permits sixteen declared candidates without a cumulative budget', () => {
  assert.deepEqual(validateCotJev(policy), policy);
  assert.match(planInstruction(policy), /at most 16/);
  assert.match(planInstruction(policy), /A through P/);
  assert.match(planInstruction(policy), /At every slow terminal step/);
  assert.match(planInstruction(policy), /reasoning note about a plan does not declare it/);
  for (const patch of [
    { maximum_candidates: 17 }, { maximum_fast_steps: true },
    { maximum_public_bytes: 1 }, { enabled: 1 }, { token_budget: 1000 },
    { version: 99 }, { version: true }, { version: '2' },
  ]) assert.throws(() => validateCotJev({ ...policy, ...patch }), /policy/);
});

test('only an actual slow assistant bank can declare unchanged native actions', () => {
  const message = slowMessage();
  const copy = structuredClone(message);
  const bank = parseBank(message, policy);
  assert.deepEqual(bank.candidates, [candidate]);
  assert.deepEqual(message, copy);
  assert.equal(parseBank({ ...message, role: 'tool' }, policy), null);
  assert.equal(parseBank({ ...message, role: 'user' }, policy), null);
});

test('compiled native actions retain the exact bank-author reasoning required by thinking mode', async () => {
  const input = options();
  const reasoning = { type: 'reasoning', text: 'Synthetic CoT-authored reasoning retained verbatim.' };
  input.messages[1].content.unshift(reasoning);
  const original = structuredClone(input);
  const instance = selector();
  const chunks = await collect(instance.instance.stream(input, fallback, 'synthetic-native'));
  const blocks = chunks.filter(chunk => chunk.type === 'block-end').map(chunk => chunk.block);
  assert.deepEqual(blocks[0], reasoning);
  assert.equal(blocks[1].type, 'tool-call');
  assert.equal(chunks.find(chunk => chunk.type === 'usage').usage.totalTokens, 121);
  assert.deepEqual(input, original);
  const record = instance.records.find(item => item.kind === 'jev/inherited-native-reasoning');
  assert.equal(record.new_reasoning_generated, false);
});

test('an empty native reasoning block is retained without disabling a declared action', async () => {
  const input = options();
  const reasoning = { type: 'reasoning', text: '' };
  input.messages[1].content.unshift(reasoning);
  const original = structuredClone(input);
  const instance = selector();
  const chunks = await collect(instance.instance.stream(input, fallback, 'synthetic-native'));
  const blocks = chunks.filter(chunk => chunk.type === 'block-end').map(chunk => chunk.block);
  assert.deepEqual(blocks[0], reasoning);
  assert.equal(blocks[1].type, 'tool-call');
  assert.deepEqual(input, original);
  assert.equal(instance.records.some(row => row.kind === 'jev/invalid-bank'), false);
});

test('banks reject future dependencies, duplicates, unknown native arguments and extra tools', () => {
  for (const candidates of [
    [{ ...candidate, after: 'B' }],
    [candidate, candidate],
    [{ ...candidate, tool: 'invented-tool' }],
    [{ ...candidate, arguments: { command: 'printf OK', invented: true } }],
    [{ ...candidate, arguments: { command: 'printf OK', timeout_sec: true } }],
    [{ ...candidate, label: 'AB' }],
  ]) assert.throws(() => parseBank(slowMessage(candidates), policy), /candidate/);
});

test('fresh state retains the full native tool schema, exact receipt and original task', () => {
  const input = options();
  const copy = structuredClone(input);
  const state = decisionState(input, policy);
  assert.equal(state.steps, 0);
  assert.deepEqual(state.state.native_tool_contract, input.tools);
  assert.deepEqual(state.state.native_receipts, [input.messages[2]]);
  assert.deepEqual(state.state.public_user_requests, [input.messages[0]]);
  assert.deepEqual(input, copy);
});

test('single encoding removes only the duplicate receipt and preserves every decoded value', () => {
  const input = options();
  input.messages[2].content[0].text = JSON.stringify(JSON.stringify({
    stdout: 'x'.repeat(10000), stderr: 'an unresolved warning', exit_code: 0,
    files: [{ name: 'keep.txt', size: 13 }], pending: true,
  }));
  const copy = structuredClone(input);
  const original = decisionState(input, policy);
  const compactPolicy = { ...policy, version: 2 };
  assert.deepEqual(validateCotJev(compactPolicy), compactPolicy);
  const compact = decisionState(input, compactPolicy);
  const { native_receipts: receipts, ...once } = original.state;
  assert.deepEqual(compact.state, once);
  assert.equal(compact.receiptBinding.native_receipt_count, receipts.length);
  assert.equal(compact.receiptBinding.state_encoding, 'single-decoded-native-receipts/v1');
  assert.equal(compact.receiptBinding.native_receipts_sha256.length, 64);
  const oldBytes = Buffer.byteLength(JSON.stringify(original.state));
  const newBytes = Buffer.byteLength(JSON.stringify(compact.state));
  assert.ok(newBytes * 100 <= oldBytes * 60);
  assert.equal(decisionState(input, { ...policy, maximum_public_bytes: newBytes }), null);
  assert.ok(decisionState(input, { ...compactPolicy, maximum_public_bytes: newBytes }));
  assert.equal(decisionState(input, { ...compactPolicy, maximum_public_bytes: newBytes - 1 }), null);
  assert.equal(planInstruction(compactPolicy), planInstruction(policy));
  assert.deepEqual(input, copy);
});

test('single encoding declines numeric rewrites and retains numbers inside exact text', () => {
  for (const text of [
    '{"exit_code":9007199254740993}', '{"elapsed":0.1234567890123456789}',
    '{"elapsed":1.0}', '{"elapsed":1e999}',
  ]) {
    const input = options();
    input.messages[2].content[0].text = text;
    assert.equal(decisionState(input, { ...policy, version: 2 }), null);
    assert.equal(publicReceipt(input.messages[2], true).reason, 'noncanonical_native_number');
  }
  const input = options();
  input.messages[2].content[0].text = JSON.stringify({
    stdout: 'Exact literal 9007199254740993 and 0.1234567890123456789',
    elapsed: 1.25, exit_code: 0,
  });
  const compact = decisionState(input, { ...policy, version: 2 });
  assert.equal(compact.state.public_results[0].value.stdout,
    'Exact literal 9007199254740993 and 0.1234567890123456789');
});

test('single encoding still charges one native decision and the complete abstention fallback', async () => {
  for (const label of ['A', '?']) {
    const fixture = selector(label, { policy: { ...policy, version: 2 } });
    const input = options();
    const original = structuredClone(input);
    const chunks = await collect(fixture.instance.stream(input, fallback, `v2-${label}`));
    const request = fixture.records.find(row => row.kind === 'jev/request');
    assert.equal(request.policy_version, 2);
    assert.equal(request.native_receipt_count, 1);
    assert.equal(Object.hasOwn(JSON.parse(request.request.messages[1].content), 'native_receipts'), false);
    assert.equal(chunks.find(chunk => chunk.type === 'usage').usage.totalTokens,
      label === '?' ? 166 : 121);
    assert.deepEqual(input, original);
  }
});

test('native JSON-string observations decode without changing their original receipt', () => {
  const value = { stdout: 'READY', stderr: '', exit_code: 0, timed_out: false };
  const message = { role: 'tool', toolCallId: 'native-root', isError: false,
    content: [{ type: 'text', text: JSON.stringify(JSON.stringify(value)) }] };
  const original = structuredClone(message);
  assert.deepEqual(publicReceipt(message), {
    call_id: 'native-root', is_error: false, status: 'PARSED', value,
  });
  assert.deepEqual(message, original);
  const input = options();
  input.messages[2] = message;
  assert.deepEqual(decisionState(input, policy).state.public_results[0].value, value);
});

test('ambiguous native result shapes abstain without fabricating record fields', () => {
  for (const text of ['READY', '"READY"', 'null', '[]', '{"stdout":']) {
    const message = { role: 'tool', toolCallId: 'native-root',
      content: [{ type: 'text', text }] };
    assert.equal(publicReceipt(message).status, 'UNKNOWN');
    const input = options();
    input.messages[2] = message;
    assert.equal(decisionState(input, policy), null);
  }
});

test('unseen, mismatched, changed-user or oversized state takes complete slow CoT', () => {
  const unseen = options();
  unseen.messages.pop();
  assert.equal(decisionState(unseen, policy), null);
  const mismatched = options();
  mismatched.messages[2].toolCallId = 'not-the-native-action';
  assert.equal(decisionState(mismatched, policy), null);
  const changed = options();
  changed.messages.push({ role: 'user', content: [{ type: 'text', text: 'Change the task.' }] });
  assert.equal(decisionState(changed, policy), null);
  const oversized = options();
  oversized.messages[2].content[0].text = JSON.stringify({ stdout: 'x'.repeat(32769), exit_code: 0 });
  assert.equal(decisionState(oversized, policy), null);
});

test('one control token compiles the exact slow-authored action into native stream chunks', async () => {
  const fixture = selector();
  const input = options();
  const original = structuredClone(input);
  const chunks = await collect(fixture.instance.stream(
    input, () => assert.fail('No new slow generation for an admitted typed action'), 'native-attempt-1',
  ));
  assert.equal(fixture.requests.length, 1);
  const body = JSON.parse(fixture.requests[0].body);
  assert.equal(body.model, 'deepseek-flash');
  assert.equal(body.max_tokens, 1);
  assert.equal(body.logprobs, true);
  const action = chunks.find(chunk => chunk.type === 'block-end').block;
  assert.equal(action.name, 'terminal');
  assert.deepEqual(JSON.parse(action.arguments), candidate.arguments);
  assert.equal(chunks.find(chunk => chunk.type === 'usage').usage.totalTokens, 121);
  assert.deepEqual(chunks.at(-1), { type: 'finish', reason: { kind: 'tool-calls' } });
  assert.deepEqual(input, original);
  assert.equal(fixture.records.find(row => row.kind === 'jev/request').native_attempt_id, 'native-attempt-1');
  assert.equal(fixture.records.find(row => row.kind === 'jev/compiled-native-action').native_execution_claimed, false);
});

test('explicit abstention charges fast input/control plus complete slow fallback exactly once', async () => {
  const fixture = selector('?');
  const chunks = await collect(fixture.instance.stream(options(), fallback, 'attempt-fallback'));
  assert.equal(fixture.requests.length, 1);
  const total = chunks.find(chunk => chunk.type === 'usage').usage;
  assert.deepEqual(total, {
    inputTokens: 130, cacheReadTokens: 30, cacheWriteTokens: 0,
    outputTokens: 6, totalTokens: 166,
  });
  const receipt = fixture.records.find(row => row.kind === 'jev/fallback-usage');
  assert.deepEqual(receipt.slow_native_usage, slowUsage);
  assert.equal(receipt.control_native_usage.totalTokens, 121);
  assert.equal(receipt.combined_usage.totalTokens, 166);
});

test('successive fast actions require actual matching native tool receipts and a declared edge', async () => {
  const second = { ...candidate, label: 'B', after: 'A',
    arguments: { command: 'printf CHECKED' } };
  const input = options([candidate, second]);
  const fixture = selector();
  const chunks = await collect(fixture.instance.stream(input, fallback, 'attempt-first'));
  const action = chunks.find(chunk => chunk.type === 'block-end').block;
  input.messages.push({ role: 'assistant', content: [action] });
  assert.equal(decisionState(input, policy), null);
  input.messages.push({ role: 'tool', toolCallId: action.id,
    content: [{ type: 'text', text: '{"exit_code":0,"stdout":"DONE"}' }] });
  const next = decisionState(input, policy);
  assert.equal(next.steps, 1);
  assert.deepEqual(next.state.candidates, [second]);
  assert.equal(decisionState(input, { ...policy, maximum_fast_steps: 1 }), null);
  action.arguments = '{"command":"a different action"}';
  assert.equal(decisionState(input, policy), null);
});

test('all sixteen declared fast steps execute once before returning to slow CoT', async () => {
  const labels = 'ABCDEFGHIJKLMNOP';
  const candidates = [...labels].map((label, index) => ({
    ...candidate, label, after: index ? labels[index - 1] : 'root',
    arguments: { command: `printf STEP${index + 1}` },
  }));
  const input = options(candidates);
  let calls = 0;
  const fixture = selector('A', {
    request: async (_url, settings) => {
      calls++;
      const body = JSON.parse(settings.body);
      const state = JSON.parse(body.messages[1].content);
      return { ok: true, status: 200,
        text: async () => JSON.stringify(response(state.candidates[0].label)) };
    },
  });
  const identities = new Set();
  for (let index = 0; index < 16; index++) {
    const chunks = await collect(fixture.instance.stream(
      input, () => assert.fail('A declared fast step was lost'), `attempt-${index}`,
    ));
    const action = chunks.find(chunk => chunk.type === 'block-end').block;
    assert.deepEqual(JSON.parse(action.arguments), candidates[index].arguments);
    assert.equal(identities.has(action.id), false);
    identities.add(action.id);
    input.messages.push(
      { role: 'assistant', content: [action] },
      { role: 'tool', toolCallId: action.id, isError: false,
        content: [{ type: 'text', text: '{"stdout":"READY","exit_code":0}' }] },
    );
  }
  assert.equal(calls, 16);
  assert.equal(identities.size, 16);
  const chunks = await collect(fixture.instance.stream(input, fallback, 'after-sixteen'));
  assert.deepEqual(chunks[0].usage, slowUsage);
  assert.equal(calls, 16);
});

test('disabled matched CoT control makes no fast calls and retains native usage bytes', async () => {
  for (const version of [1, 2]) {
    const disabled = { ...policy, version, enabled: false };
    const fixture = selector('A', { policy: disabled });
    const chunks = await collect(fixture.instance.stream(options(), fallback, 'control'));
    assert.equal(fixture.requests.length, 0);
    assert.deepEqual(chunks[0], { type: 'usage', usage: slowUsage });
    assert.equal(planInstruction(policy), planInstruction(disabled));
  }
});

test('malformed CoT banks retain an explicit receipt and full slow fallback', async () => {
  const fixture = selector();
  const input = options([{ ...candidate, after: 'undeclared' }]);
  const chunks = await collect(fixture.instance.stream(input, fallback, 'invalid-bank'));
  assert.equal(fixture.requests.length, 0);
  assert.equal(chunks[0].usage.totalTokens, 45);
  assert.equal(fixture.records[0].kind, 'jev/invalid-bank');
});

test('unsettled failed provider calls are not converted into free successful fallbacks', async () => {
  const fixture = selector('A', {
    request: async () => ({ ok: false, status: 429, text: async () => '{"error":"rate limit"}' }),
  });
  await assert.rejects(collect(fixture.instance.stream(
    options(), () => assert.fail('Unknown provider cost cannot masquerade as a free fallback'), 'failed',
  )), /429/);
  assert.equal(fixture.records.at(-1).usage_complete, false);
});

test('invalid paid labels fall back with their full cost and no invented confidence', async () => {
  const fixture = selector('Z');
  const chunks = await collect(fixture.instance.stream(options(), fallback, 'invalid-decision'));
  const decision = fixture.records.find(row => row.kind === 'jev/decision');
  assert.equal(decision.label, '?');
  assert.equal(decision.invalid_decision, true);
  assert.equal(decision.probability, null);
  assert.equal(chunks.find(chunk => chunk.type === 'usage').usage.totalTokens, 166);
  assert.equal(fixture.records.some(row => row.kind === 'jev/compiled-native-action'), false);
});

test('different models, multiple output tokens and incomplete cache usage are rejected', () => {
  for (const patch of [
    value => { value.model = 'different-model'; },
    value => { value.usage.completion_tokens = 2; value.usage.total_tokens = 122; },
    value => { delete value.usage.prompt_cache_hit_tokens; },
    value => { value.choices[0].message.content = 'A\n'; },
  ]) {
    const value = response();
    patch(value);
    assert.throws(() => chatDecision(value, ['A'], 'deepseek-flash'));
  }
});

test('fallback accounting rejects contradictory cache breakdowns without dropping cached inputs', () => {
  const fast = chatDecision(response(), ['A'], 'deepseek-flash').usage;
  assert.throws(() => addControlUsage({ ...slowUsage, totalTokens: 44 }, fast), /contradicts/);
  assert.throws(() => addControlUsage({ inputTokens: 10, outputTokens: 2 }, fast), /exact/);
  const mixed = addControlUsage({ inputTokens: 30, outputTokens: 5, totalTokens: 45 }, fast);
  assert.equal(mixed.totalTokens, 166);
  assert.equal(Object.hasOwn(mixed, 'cacheReadTokens'), false);
});
