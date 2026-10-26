import assert from 'node:assert/strict';
import test from 'node:test';
import { decisionState, parseBank, planInstruction, validateCotJev } from '../integrations/deepseek_harness/cot_jev.mjs';

const policy = { version: 3, enabled: true, maximum_candidates: 16,
  maximum_fast_steps: 16, maximum_public_bytes: 32768 };
const schema = { type: 'object', properties: { path: { type: 'string' } },
  required: ['path'], additionalProperties: false };
const tools = ['Files__exists', 'Files__cat', 'Files__rm'].map(name => ({ name, parameters: schema }));
const candidate = { label: 'A', after: 'root', condition: 'The returned file exists.',
  tool: 'Files__cat', arguments: { path: '/public/example.txt' } };
const root = { type: 'tool-call', id: 'native-root', name: 'Files__exists',
  arguments: '{"path":"/public/example.txt"}' };
function options(item = candidate) {
  return { tools, messages: [{ role: 'user', content: [{ type: 'text', text: 'Read the file.' }] },
    { role: 'assistant', content: [{ type: 'text',
      text: `<CHISEL_PLAN>${JSON.stringify({ candidates: [item] })}</CHISEL_PLAN>` }, root] },
    { role: 'tool', toolCallId: root.id, content: [{ type: 'text', text: "{'exists': True}" }] }] };
}

test('native app banks preserve original Python-style receipt bytes and exact schemas', () => {
  assert.equal(validateCotJev(policy).version, 3);
  assert.match(planInstruction(policy), /Never guess IDs/);
  const state = decisionState(options(), policy);
  assert.deepEqual(state.state.public_results[0].value, { native_text: "{'exists': True}" });
  assert.deepEqual(state.state.native_tool_contract.map(t => t.name), ['Files__exists', 'Files__cat']);
  assert.equal(state.receiptBinding.full_native_catalog_sha256.length, 64);
  assert.deepEqual(state.state.candidates[0].arguments, candidate.arguments);
});

test('unknown native app tools, missing required arguments and changed types take slow CoT', () => {
  for (const item of [{ ...candidate, tool: 'HiddenOracle__answer' },
    { ...candidate, arguments: {} }, { ...candidate, arguments: { path: 1 } },
    { ...candidate, arguments: { path: '/public/example.txt', answer: 'invented' } }]) {
    const input = options(item);
    assert.throws(() => parseBank(input.messages[1], policy, tools), /candidate/);
  }
});

test('unseen user updates, unmatched receipts and images return to full native reasoning', () => {
  const updated = options();
  updated.messages.push({ role: 'user', content: [{ type: 'text', text: 'Cancel reading.' }] });
  assert.equal(decisionState(updated, policy), null);
  const mismatch = options(); mismatch.messages.at(-1).toolCallId = 'another-root';
  assert.equal(decisionState(mismatch, policy), null);
  const image = options(); image.messages.at(-1).content = [{ type: 'image', data: 'AA==' }];
  assert.equal(decisionState(image, policy), null);
});

test('prospective native policy rejects repeating the action already executed at root', () => {
  const next = { ...policy, version: 4 };
  assert.match(planInstruction(next), /AFTER that call has completed/);
  assert.ok(decisionState(options(), next));
  const repeated = options({ ...candidate, tool: root.name });
  assert.throws(() => parseBank(repeated.messages[1], next, tools), /already executed/);
  assert.ok(parseBank(repeated.messages[1], policy, tools));
  const twoArguments = options({ ...candidate, tool: root.name,
    arguments: { kwargs: {}, path: '/public/example.txt' } });
  twoArguments.tools = tools.map(tool => ({ ...tool, parameters: { type: 'object',
    properties: { path: { type: 'string' }, kwargs: { type: 'object' } },
    required: ['path', 'kwargs'], additionalProperties: false } }));
  twoArguments.messages[1].content[1] = { ...root,
    arguments: '{"path":"/public/example.txt","kwargs":{}}' };
  assert.throws(() => parseBank(twoArguments.messages[1], next, twoArguments.tools), /already executed/);
  twoArguments.messages[1].content[1].arguments = '{"path":"/different.txt","kwargs":{}}';
  assert.ok(parseBank(twoArguments.messages[1], next, twoArguments.tools));
});
