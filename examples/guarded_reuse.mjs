// Synthetic public receipts and an in-memory dispatcher. No model or shell calls.
import assert from 'node:assert/strict';
import {ExecutableSubdag} from '../integrations/deepseek_harness/sar_v2.mjs';

const policy = {version: 6, enabled: true, maximum_candidates: 16,
  maximum_fast_steps: 8, maximum_public_bytes: 65536};
const tool = {name: 'terminal', parameters: {type: 'object',
  properties: {command: {type: 'string'}}, required: ['command'],
  additionalProperties: false}};

// A slow actor would author these literal actions before the root tool call.
const commands = ['compile example.py', 'test example.py', 'report example.py'];
const nodes = commands.map((command, index) => ({
  candidate_id: 'ABC'[index], predecessor_id: index ? 'ABC'[index - 1] : 'root',
  tool: 'terminal', arguments: {command}, guard: {return_code: 0},
  required_receipt_fields: ['return_code'],
  expected_local_cost: {selector_upper_tokens: 8192}, dependency_ids: [],
}));
const rootCall = {type: 'tool-call', id: 'root-call', name: 'terminal',
  arguments: JSON.stringify({command: 'prepare example.py'})};
const authorMessage = {role: 'assistant', content: [
  {type: 'text', text: `<CHISEL_PLAN>${JSON.stringify({candidates: nodes})}</CHISEL_PLAN>`},
  rootCall,
]};
const receipt = callId => ({role: 'tool', toolCallId: callId, content: [
  {type: 'text', text: JSON.stringify({return_code: 0, stdout: 'OK', stderr: null})},
]});
const events = [];
let slowCalls = 0;
const controller = new ExecutableSubdag({policy, model: 'synthetic-no-provider',
  record: event => events.push(event),
  request: () => {throw new Error('This deterministic example needs no selector call');},
});
// Synthetic prior used only to exercise the declared admission rule.
controller.slowTokens = 40000;
const options = {model: 'synthetic-no-provider', tools: [tool], messages: [
  {role: 'user', content: [{type: 'text', text: 'Compile, test and report the example.'}]},
  authorMessage, receipt(rootCall.id),
]};
const originalSlowStream = async function* () {slowCalls += 1;};

for (let index = 0; index < nodes.length; index += 1) {
  const chunks = [];
  for await (const chunk of controller.stream(options, originalSlowStream, `callback-${index}`)) {
    chunks.push(chunk);
  }
  const call = chunks.find(chunk => chunk.type === 'block-end' && chunk.block.type === 'tool-call')?.block;
  assert.ok(call, 'Expected one stored action at the native boundary');
  const args = JSON.parse(call.arguments);
  assert.deepEqual(args, nodes[index].arguments);
  // The native dispatcher executes the tool. Here it returns a synthetic receipt.
  const result = receipt(call.id);
  controller.observeTool({call_id: call.id, name: call.name, arguments: args,
    result: {observation: result.content[0].text}});
  options.messages.push({role: 'assistant', content: [call]}, result);
}
assert.equal(slowCalls, 0);
for await (const chunk of controller.stream(options, originalSlowStream, 'after-exhaustion')) {
  void chunk;
}
assert.equal(slowCalls, 1);
const certificates = events.filter(event => event.kind === 'sar/substitution-certificate');
assert.equal(certificates.length, 3);
console.log(JSON.stringify({
  scope: 'Synthetic controller demonstration; no benchmark or measured token savings',
  executed_actions: certificates.map(event => event.exact_arguments),
  certified_substitutions: certificates.length,
  slow_calls_after_graph_exhaustion: slowCalls,
  bypassed_callbacks: certificates.map(event => event.bypassed_slow_callback_id),
}, null, 2));
