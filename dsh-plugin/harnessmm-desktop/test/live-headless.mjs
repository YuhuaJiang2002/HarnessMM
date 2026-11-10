// Drive the real DSH agent loop against a scripted, local Messages endpoint.
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtempSync, mkdirSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const [executable] = process.argv.slice(2);
assert.ok(executable, 'Usage: node test/live-headless.mjs /path/to/dsh');
const dsh = resolve(executable);
const plugin = fileURLToPath(new URL('../lib/index.js', import.meta.url));
const verifier = fileURLToPath(new URL('./live-session.mjs', import.meta.url));
const run = promisify(execFile);
const root = mkdtempSync(join(tmpdir(), 'harnessmm-live-'));
const home = join(root, 'home');
const receipts = join(root, 'receipts.jsonl');
const sessions = join(root, 'sessions');
const cwd = join(root, 'workspace');
mkdirSync(cwd);
const bash = 'bash';
const rootArgs = { description: 'Print root marker', command: 'echo HMM-ROOT' };
const subArgs = { description: 'Print and append substitution marker', command: 'echo HMM-SUBSTITUTED; echo HMM-SUBSTITUTED >> hmm-smoke.txt' };
const endArgs = { description: 'Print final marker', command: 'echo HMM-DONE' };
const plan = '<CHISEL_PLAN>' + JSON.stringify({ candidates: [{
  candidate_id: 'A',
  predecessor_id: 'root',
  tool: bash,
  arguments: subArgs,
  guard: { exit_code: 0, stdout_equals: 'HMM-ROOT' },
  required_receipt_fields: ['exit_code', 'stdout'],
  expected_local_cost: { selector_upper_tokens: 0 },
  dependency_ids: [],
}] }) + '</CHISEL_PLAN>';
const toolUse = (id, name, input) => ({ type: 'tool_use', id, name, input });
const responses = [
  [{ type: 'text', text: plan }, toolUse('call_root', bash, rootArgs)],
  [toolUse('call_done', bash, endArgs)],
  [toolUse('call_status', 'harnessmm_status', {})],
  [{ type: 'text', text: 'Smoke test complete.' }],
];
const expectedResults = [null, 'HMM-SUBSTITUTED', 'HMM-DONE', 'substitutions_certified'];
let requests = 0;
let fixtureError;
const server = createServer(async (req, res) => {
  try {
    assert.equal(req.method, 'POST');
    assert.match(req.url, /\/v1\/messages$/);
    assert.equal(req.headers['x-api-key'], 'hmm-test');
    const data = [];
    for await (const chunk of req) data.push(chunk);
    const body = JSON.parse(Buffer.concat(data).toString());
    const index = requests++;
    assert.ok(responses[index], 'Unexpected model request');
    if (index > 0) {
      const last = body.messages.at(-1);
      assert.ok(JSON.stringify(last.content).includes(expectedResults[index]),
        'Expected preceding result ' + expectedResults[index] + ' in request ' + index);
    }
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    const emit = (type, value = {}) => res.write('event: ' + type + '\ndata: ' + JSON.stringify({ type, ...value }) + '\n\n');
    emit('message_start', { message: { id: 'msg_' + index, role: 'assistant', type: 'message',
      model: body.model, content: [], usage: { input_tokens: 1300, output_tokens: 0 } } });
    const response = responses[index];
    response.forEach((block, index) => {
      if (block.type === 'text') {
        emit('content_block_start', { index, content_block: { type: 'text', text: '' } });
        emit('content_block_delta', { index, delta: { type: 'text_delta', text: block.text } });
      } else {
        emit('content_block_start', { index, content_block: { ...block, input: {} } });
        emit('content_block_delta', { index, delta: { type: 'input_json_delta', partial_json: JSON.stringify(block.input) } });
      }
      emit('content_block_stop', { index });
    });
    emit('message_delta', { delta: { stop_reason: response.at(-1).type === 'tool_use' ? 'tool_use' : 'end_turn', stop_sequence: null },
      usage: { output_tokens: 100 } });
    emit('message_stop');
    res.end();
  } catch (error) {
    fixtureError = error;
    res.writeHead(400);
    res.end('{}');
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const baseURL = 'http://127.0.0.1:' + server.address().port;
const patch = join(root, 'patch.json');
writeFileSync(patch, JSON.stringify([
  { id: 'session-log-deepseek', disabled: true },
  { id: 'session-title-llm', disabled: true },
  { id: 'session-telemetry-otel', disabled: true },
  { id: 'agent-instructions', disabled: true },
  { id: 'session-persistence-jsonl', config: { root: sessions, compression: 'none' } },
  { id: 'agent-default-model', config: { provider: 'deepseek-official', model: 'deepseek-v4-flash' } },
  { id: 'llm-deepseek', config: { baseURL } },
  { insert: [{ id: 'harnessmm', name: plugin,
    config: { policy: { enabled: true }, receiptsFile: receipts } }] },
], null, 2));
const { PATH, HOME } = process.env;
const env = { PATH, HOME, DSH_HOME: home, DEEPSEEK_API_KEY: 'hmm-test', DEEPSEEK_BASE_URL: baseURL };
try {
  const result = await run(dsh, ['headless', '--patch', patch, '--json', 'Run the HarnessMM smoke test.'],
    { cwd, env, timeout: 45000, maxBuffer: 4000000 });
  writeFileSync(join(root, 'stdout.jsonl'), result.stdout);
  writeFileSync(join(root, 'stderr.log'), result.stderr);
  if (fixtureError) throw fixtureError;
  assert.equal(requests, responses.length, 'The substituted step must skip the model request');
  const logs = readdirSync(sessions, { recursive: true }).filter(path => path.endsWith('.jsonl'));
  assert.equal(logs.length, 1, 'Expected one persisted DSH session');
  const checked = await run(process.execPath, [verifier, join(sessions, logs[0]), receipts, join(cwd, 'hmm-smoke.txt')]);
  console.log(checked.stdout.trim());
  rmSync(root, { recursive: true, force: true });
} catch (error) {
  console.error(`Smoke test artifacts: ${root}`);
  if (error.stderr) console.error(error.stderr);
  throw fixtureError ?? error;
} finally {
  server.closeAllConnections();
  server.close();
}
