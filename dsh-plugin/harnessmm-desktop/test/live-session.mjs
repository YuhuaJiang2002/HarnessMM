/** Verify a fresh DSH desktop smoke session. See live-session.md. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { CERTIFICATE_SCHEMA, hashJson } from '../lib/controller.mjs';

const [sessionPath, receiptsPath, markerPath] = process.argv.slice(2);
assert.ok(sessionPath && receiptsPath && markerPath,
  'Usage: node test/live-session.mjs SESSION.jsonl RECEIPTS.jsonl hmm-smoke.txt');

const readJsonl = path => readFileSync(path, 'utf8').trim().split(/\r?\n/).map(line => JSON.parse(line));
const [{ id: sessionId }, ...events] = readJsonl(sessionPath);
assert.ok(sessionId, 'Expected a DSH session header');
const receipts = readJsonl(receiptsPath);
const issued = receipts.filter(entry => entry.type === 'substitution-issued'
  && entry.turn_id.startsWith(`${sessionId}:`));
assert.equal(issued.length, 1, 'Expected one substitution issued in this session');
const [action] = issued;
const turn = receipts.filter(entry => entry.turn_id === action.turn_id);
assert.ok(['bash', 'pwsh'].includes(action.tool), 'Expected a native shell action');
assert.equal(action.candidate_id, 'A');
assert.equal(action.selector, 'deterministic');
assert.deepEqual(turn.map(entry => entry.type).slice(0, 4),
  ['bank', 'decision', 'substitution-issued', 'substitution']);
const [bank, decision, , certificate] = turn;
assert.equal(bank.candidates, 1);
assert.equal(decision.kind, 'commit');
assert.deepEqual(decision.eligible, ['A']);
assert.equal(certificate.schema, CERTIFICATE_SCHEMA);
assert.equal(certificate.bank_sha256, bank.bank_sha256);
assert.equal(certificate.decision_id, action.decision_id);
assert.equal(certificate.candidate_id, action.candidate_id);
assert.equal(certificate.exact_native_tool, action.tool);
assert.equal(certificate.valid_substitution, true);
assert.equal(certificate.amount_substituted_slow_callbacks, 1);
assert.equal(certificate.native_receipt_hash, hashJson(certificate.native_receipt));
assert.equal(certificate.native_receipt.value.exit_code, 0);
assert.equal(certificate.native_receipt.value.stdout, 'HMM-SUBSTITUTED');

const calls = events.filter(event => event.type === 'tool/call').map(event => event.data);
const shellCalls = calls.filter(call => call.name === action.tool);
assert.deepEqual(shellCalls.map(call => JSON.parse(call.arguments).command), [
  'echo HMM-ROOT',
  'echo HMM-SUBSTITUTED; echo HMM-SUBSTITUTED >> hmm-smoke.txt',
  'echo HMM-DONE',
]);
assert.match(action.call_id, /^chisel_/);
assert.equal(calls.filter(call => call.callId === action.call_id).length, 1);
assert.equal(shellCalls[1].callId, action.call_id);
assert.equal(shellCalls[1].arguments, certificate.exact_arguments);
assert.deepEqual(readFileSync(markerPath, 'utf8').split(/\r?\n/), ['HMM-SUBSTITUTED', ''],
  'The substituted shell command must append exactly one line');

const statusCalls = calls.filter(call => call.name === 'harnessmm_status');
assert.equal(statusCalls.length, 1, 'Expected one in-session harnessmm_status call');
const statusResult = events.find(event => event.type === 'tool/result'
  && event.data.message.toolCallId === statusCalls[0].callId)?.data.message;
assert.ok(statusResult && !statusResult.isError, 'Expected a successful status result');
const status = JSON.parse(statusResult.content.find(block => block.type === 'text').text);
assert.deepEqual(status, {
  enabled: true,
  banks_authored: 1,
  substitutions_issued: 1,
  substitutions_certified: 1,
  boundary_errors: 0,
  dispatch_rejections: 0,
});

console.log('Verified one native substitution, one shell side effect, and the in-session status result.');
