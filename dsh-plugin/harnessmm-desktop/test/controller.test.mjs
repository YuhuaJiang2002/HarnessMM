/**
 * Offline checks for the HarnessMM desktop core. No host needed:
 * `node --test test/` (or `node test/controller.test.mjs`).
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  BankContractError,
  bytes,
  decide,
  hashJson,
  localGuard,
  nativeCallId,
  parseBank,
  projectTurn,
  publicReceipt,
  violateDispatch,
} from '../lib/controller.mjs';

const POLICY = {
  enabled: true,
  version: 6,
  maximum_candidates: 16,
  maximum_fast_steps: 8,
  maximum_public_bytes: 65536,
};

const TOOLS = [
  { name: 'pwsh', parameters: { description: { type: 'string' }, command: { type: 'string' } } },
  { name: 'read', parameters: { file_path: { type: 'string' } } },
];

const CANDIDATE_A = {
  candidate_id: 'A',
  predecessor_id: 'root',
  tool: 'pwsh',
  arguments: { description: 'Run the full test suite', command: 'pytest -q tests/' },
  guard: { exit_code: 0 },
  required_receipt_fields: ['exit_code'],
  expected_local_cost: { selector_upper_tokens: 8192 },
  dependency_ids: [],
};

const CANDIDATE_B = {
  candidate_id: 'B',
  predecessor_id: 'A',
  tool: 'read',
  arguments: { file_path: 'D:\\temp\\HarnessMM\\README.md' },
  guard: { exit_code: 0, stdout_contains: 'HarnessMM' },
  required_receipt_fields: ['exit_code', 'stdout'],
  expected_local_cost: { selector_upper_tokens: 4096 },
  dependency_ids: ['A'],
};

function assistantWithBank(candidates, rootCall) {
  return {
    role: 'assistant',
    content: [
      { type: 'reasoning', text: 'I will run the suite then read the readme.' },
      {
        type: 'text',
        text: `Plan:\n<CHISEL_PLAN>${JSON.stringify({ candidates })}</CHISEL_PLAN>`,
      },
      rootCall,
    ],
  };
}

const ROOT_CALL = {
  type: 'tool-call',
  id: 'call_root_1',
  name: 'pwsh',
  arguments: JSON.stringify({ description: 'Run the test suite', command: 'pytest -q' }),
};

function toolMessage(id, name, text, isError = false) {
  return { role: 'tool', toolCallId: id, name, isError, content: [{ type: 'text', text }] };
}

/* ------------------------------------------------------------------ bank */

test('parseBank accepts a v6 bank and derives a stable hash', () => {
  const message = assistantWithBank([CANDIDATE_A], ROOT_CALL);
  const bank = parseBank(message, TOOLS, POLICY);
  assert.ok(bank);
  assert.equal(bank.candidates.length, 1);
  assert.equal(bank.root.name, 'pwsh');
  assert.match(bank.sha256, /^[0-9a-f]{64}$/);
  // Deterministic across calls.
  assert.equal(parseBank(message, TOOLS, POLICY).sha256, bank.sha256);
});

test('parseBank returns null when no plan is present', () => {
  const message = { role: 'assistant', content: [ROOT_CALL] };
  assert.equal(parseBank(message, TOOLS, POLICY), null);
});

test('parseBank ignores a fenced example but still reads a bare declaration', () => {
  // The prompt itself shows the dialect inside a fence. A model quoting that
  // example (or a transcript quoting it) must not turn the documentation into
  // a live, executable action bank.
  const plan = JSON.stringify({ candidates: [CANDIDATE_A] });
  const example = `<CHISEL_PLAN>${plan}</CHISEL_PLAN>`;
  const quotedOnly = {
    role: 'assistant',
    content: [
      { type: 'text', text: `The section says:\n\`\`\`\n${example}\n\`\`\`\nDone.` },
      ROOT_CALL,
    ],
  };
  assert.equal(parseBank(quotedOnly, TOOLS, POLICY), null);

  // Quoting the fenced example while declaring one bare plan is one declaration,
  // not two: the fenced copy must not trigger the one-bank contract.
  const withBare = {
    role: 'assistant',
    content: [
      { type: 'text', text: `Example:\n\`\`\`\n${example}\n\`\`\`\nMy plan:\n<CHISEL_PLAN>${plan}</CHISEL_PLAN>` },
      ROOT_CALL,
    ],
  };
  assert.equal(parseBank(withBare, TOOLS, POLICY).candidates.length, 1);

  // Only the fence marks documentation; an unfenced copy is still a declaration
  // (reasoning models legitimately draft the plan while thinking).
  const unfenced = {
    role: 'assistant',
    content: [{ type: 'reasoning', text: `I will reuse the documented plan: ${example}` }, ROOT_CALL],
  };
  assert.equal(parseBank(unfenced, TOOLS, POLICY).candidates.length, 1);
});

test('parseBank enforces its configured maximum_candidates', () => {
  const message = assistantWithBank([CANDIDATE_A, CANDIDATE_B], ROOT_CALL);
  assert.equal(parseBank(message, TOOLS, { maxCandidates: 2 }).candidates.length, 2);
  // The cap is a bound on the authored list, not a warning: the bank is refused
  // whole rather than silently truncated, so the slow turn keeps control.
  assert.throws(() => parseBank(message, TOOLS, { maxCandidates: 1 }), BankContractError);
});

test('parseBank rejects two banks, unknown tools and imperative guards', () => {
  const twoDistinct = {
    role: 'assistant',
    content: [
      { type: 'text', text: `<CHISEL_PLAN>${JSON.stringify({ candidates: [CANDIDATE_A] })}</CHISEL_PLAN>` },
      { type: 'reasoning', text: `<CHISEL_PLAN>${JSON.stringify({ candidates: [CANDIDATE_B] })}</CHISEL_PLAN>` },
      ROOT_CALL,
    ],
  };
  assert.throws(() => parseBank(twoDistinct, TOOLS, POLICY), BankContractError);

  const empty = assistantWithBank([], ROOT_CALL);
  assert.throws(() => parseBank(empty, TOOLS, POLICY), BankContractError);

  // A reasoning model that drafts the bank while thinking and restates the same
  // bytes in the answer has still declared one bank; the copies collapse.
  const onePlan = JSON.stringify({ candidates: [CANDIDATE_A] });
  const echoed = {
    role: 'assistant',
    content: [
      { type: 'reasoning', text: `<CHISEL_PLAN>${onePlan}</CHISEL_PLAN>` },
      { type: 'text', text: `<CHISEL_PLAN>${onePlan}</CHISEL_PLAN>` },
      ROOT_CALL,
    ],
  };
  assert.equal(parseBank(echoed, TOOLS, POLICY).candidates.length, 1);

  const unknownTool = assistantWithBank([{ ...CANDIDATE_A, tool: 'terminal' }], ROOT_CALL);
  assert.throws(() => parseBank(unknownTool, TOOLS, POLICY), BankContractError);

  const imperative = assistantWithBank([{ ...CANDIDATE_A, guard: { output_looks_correct: true } }], ROOT_CALL);
  assert.throws(() => parseBank(imperative, TOOLS, POLICY), BankContractError);

  const extraKey = assistantWithBank([{ ...CANDIDATE_A, confidence: 0.9 }], ROOT_CALL);
  assert.throws(() => parseBank(extraKey, TOOLS, POLICY), BankContractError);

  const repeatsRoot = assistantWithBank(
    [{ ...CANDIDATE_A, arguments: JSON.parse(ROOT_CALL.arguments) }],
    ROOT_CALL,
  );
  assert.throws(() => parseBank(repeatsRoot, TOOLS, POLICY), BankContractError);
});

test('parseBank rejects a dangling dependency', () => {
  const broken = assistantWithBank([{ ...CANDIDATE_B }], ROOT_CALL);
  assert.throws(() => parseBank(broken, TOOLS, POLICY), BankContractError);
});

/* --------------------------------------------------------------- receipt */

test('publicReceipt decodes a shell result from the canonical value', () => {
  const receipt = publicReceipt({
    name: 'pwsh',
    isError: false,
    text: '3 passed\n',
    canonical: {
      kind: 'foreground',
      exitCode: 0,
      signal: null,
      timedOut: false,
      aborted: false,
      stdout: { text: '3 passed\n', truncated: false },
      stderr: { text: '', truncated: false },
    },
  });
  assert.equal(receipt.status, 'PARSED');
  assert.equal(receipt.is_error, false);
  assert.equal(receipt.value.exit_code, 0);
  assert.equal(receipt.value.stdout, '3 passed');
  assert.equal(receipt.value.stderr, '');
});

test('publicReceipt reads a nonzero exit from the rendered text when canonical is absent', () => {
  const receipt = publicReceipt({
    name: 'bash',
    isError: false,
    text: 'boom\n[stderr]\nTraceback\n[exit code: 2]',
  });
  assert.equal(receipt.value.exit_code, 2);
  assert.equal(receipt.value.stdout, 'boom');
  assert.match(receipt.value.stderr, /Traceback/);
});

test('publicReceipt drops the renderer status trailer from stdout', () => {
  // Observed on a real desktop host: the rendered content of a nonzero pwsh call
  // is the program output followed by the renderer's own `[exit code: N]` line.
  // Keeping it would make `stdout_equals` unable to match any failing command.
  const receipt = publicReceipt({
    name: 'pwsh',
    isError: false,
    text: "HMM13-FAIL-OUT\r\n[exit code: 3]",
    canonical: { kind: 'foreground', exitCode: 3 },
  });
  assert.equal(receipt.value.exit_code, 3);
  assert.equal(receipt.value.stdout, 'HMM13-FAIL-OUT');
  assert.equal(receipt.value.stderr, '');
});

test('publicReceipt refuses a background result and marks tool errors', () => {
  const background = publicReceipt({ name: 'pwsh', isError: false, text: '', canonical: { kind: 'background', jobId: 'j1', cwd: 'D:\\' } });
  assert.equal(background.status, 'UNKNOWN');
  assert.equal(background.value, null);

  const failed = publicReceipt({ name: 'read', isError: true, text: 'ENOENT', canonical: undefined });
  assert.equal(failed.is_error, true);
  assert.equal(failed.value.ok, false);
});

/* ----------------------------------------------------------------- guard */

test('localGuard evaluates receipt-local conditions only', () => {
  const receipt = publicReceipt({
    name: 'pwsh', isError: false, text: '{"total": 3}', canonical: { kind: 'foreground', exitCode: 0 },
  });
  assert.deepEqual(localGuard({ guard: { exit_code: 0 }, required_receipt_fields: ['exit_code'] }, receipt).eligible, true);
  assert.equal(localGuard({ guard: { exit_code: 1 }, required_receipt_fields: ['exit_code'] }, receipt).reason, 'guard_false');
  assert.equal(localGuard({ guard: { stdout_contains: 'total' }, required_receipt_fields: ['stdout'] }, receipt).eligible, true);
  assert.equal(
    localGuard({ guard: { json_equals: { path: ['total'], value: 3 } }, required_receipt_fields: ['stdout'] }, receipt).eligible,
    true,
  );
  assert.equal(
    localGuard({ guard: { json_equals: { path: ['__proto__'], value: 1 } }, required_receipt_fields: ['stdout'] }, receipt).known,
    false,
  );
  // A guard that reads a field the candidate did not declare is unknown, not true.
  assert.equal(localGuard({ guard: { exit_code: 0 }, required_receipt_fields: [] }, receipt).reason, 'required_field_missing');
});

/* ------------------------------------------------------------- admission */

function authoredTurn() {
  const message = assistantWithBank([CANDIDATE_A, CANDIDATE_B], ROOT_CALL);
  const bank = parseBank(message, TOOLS, POLICY);
  return { message, bank };
}

test('decide commits deterministically when exactly one candidate survives', () => {
  const { message, bank } = authoredTurn();
  const receiptMessage = toolMessage('call_root_1', 'pwsh', 'ok\n');
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'Run the tests then read the readme.' }] },
    message,
    receiptMessage,
  ];
  const projected = projectTurn(messages, TOOLS, POLICY);
  assert.ok(projected);
  assert.equal(projected.position, 'root');
  assert.equal(projected.steps, 0);

  const receipt = publicReceipt({ name: 'pwsh', isError: false, text: 'ok\n', canonical: { kind: 'foreground', exitCode: 0 } });
  const decision = decide({ projected, receipt, slowTokens: 30000, deterministic: true, policy: POLICY });
  assert.equal(decision.kind, 'commit');
  assert.equal(decision.candidate.candidate_id, 'A');
  assert.equal(decision.usage.totalTokens, 0);
});

test('parseBank ignores a reasoning fragment whose open tag swallows the real close', () => {
  const bankBody = JSON.stringify({ candidates: [CANDIDATE_A] });
  const message = {
    role: 'assistant',
    content: [
      { type: 'reasoning', text: 'I will declare <CHISEL_PLAN> and then the plan below, as the section asks.' },
      { type: 'text', text: `Plan:\n<CHISEL_PLAN>${bankBody}</CHISEL_PLAN>` },
      ROOT_CALL,
    ],
  };
  const bank = parseBank(message, TOOLS, POLICY);
  assert.ok(bank, 'a quoted open tag in reasoning must not poison the real declaration');
  assert.equal(bank.candidates.length, 1);
  assert.equal(bank.candidates[0].candidate_id, 'A');
});

test('projectTurn ignores history that only quotes a plan fragment', () => {
  const { message } = authoredTurn();
  const quoted = {
    role: 'assistant',
    content: [
      { type: 'text', text: 'The protocol is written <CHISEL_PLAN>see the docs</CHISEL_PLAN> for details.' },
      ROOT_CALL,
    ],
  };
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'keep going' }] },
    quoted,
    { role: 'tool', toolCallId: ROOT_CALL.id, content: [{ type: 'text', text: 'ok\n' }] },
    message,
    toolMessage('call_root_1', 'pwsh', 'ok\n'),
  ];
  const projected = projectTurn(messages, TOOLS, POLICY);
  assert.ok(projected, 'a quoted fragment must not disable projection for the session');
  assert.deepEqual(projected.candidates.map(c => c.candidate_id), ['A']);
  assert.equal(projected.bank.sha256.length, 64);
});

test('decide asks the selector when a whole sub-DAG is still live and it pays', () => {
  const { message } = authoredTurn();
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'keep going' }] },
    message,
    toolMessage('call_root_1', 'pwsh', 'ok\n'),
  ];
  const projected = projectTurn(messages, TOOLS, POLICY);
  const receipt = publicReceipt({ name: 'pwsh', isError: false, text: 'ok\n', canonical: { kind: 'foreground', exitCode: 0 } });
  // Only A descends from the root receipt; B waits for A, so the selector is
  // still asked (the graph is not deterministic) once the saving clears 1024.
  assert.deepEqual(projected.candidates.map(c => c.candidate_id), ['A']);
  const decision = decide({ projected, receipt, slowTokens: 30000, deterministic: false, policy: POLICY });
  assert.equal(decision.kind, 'select');
  assert.equal(decision.admission.eligible.length, 1);
  assert.equal(bytes(decision.payload) > 0, true);
});

test('decide commits without a selector when only one candidate can survive', () => {
  const { message } = authoredTurn();
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'keep going' }] },
    message,
    toolMessage('call_root_1', 'pwsh', 'ok\n'),
  ];
  const projected = projectTurn(messages, TOOLS, POLICY);
  const receipt = publicReceipt({ name: 'pwsh', isError: false, text: 'ok\n', canonical: { kind: 'foreground', exitCode: 0 } });
  const decision = decide({ projected, receipt, slowTokens: 30000, deterministic: true, policy: POLICY });
  assert.equal(decision.kind, 'commit');
  assert.equal(decision.admission.eligible.length, 1);
});

test('decide falls back to the slow actor when the saving is too small', () => {
  const { message } = authoredTurn();
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'keep going' }] },
    message,
    toolMessage('call_root_1', 'pwsh', 'ok\n'),
  ];
  const projected = projectTurn(messages, TOOLS, POLICY);
  const receipt = publicReceipt({ name: 'pwsh', isError: false, text: 'ok\n', canonical: { kind: 'foreground', exitCode: 0 } });
  const decision = decide({ projected, receipt, slowTokens: 200, deterministic: false, policy: POLICY });
  assert.equal(decision.kind, 'slow');
  assert.equal(decision.reason, 'no_eligible_candidate');
  assert.equal(decision.admission.rejected[0].reason, 'net_saving_threshold');
});

test('decide stops admitting a failed predecessor', () => {
  const { message } = authoredTurn();
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'keep going' }] },
    message,
    toolMessage('call_root_1', 'pwsh', 'boom\n[exit code: 2]'),
  ];
  const projected = projectTurn(messages, TOOLS, POLICY);
  const receipt = publicReceipt({ name: 'pwsh', isError: false, text: 'boom\n[exit code: 2]' });
  const decision = decide({ projected, receipt, slowTokens: 30000, deterministic: true, policy: POLICY });
  assert.equal(decision.kind, 'slow');
  assert.equal(decision.admission.rejected[0].reason, 'guard_false');
});

/* -------------------------------------------------------- turn projection */

test('projectTurn follows the authored substitution chain', () => {
  const { message, bank } = authoredTurn();
  const callIdA = nativeCallId(bank.sha256, 'A');
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'Run the tests then read the readme.' }] },
    message,
    toolMessage('call_root_1', 'pwsh', 'ok\n'),
    {
      role: 'assistant',
      content: [{ type: 'tool-call', id: callIdA, name: 'pwsh', arguments: JSON.stringify(CANDIDATE_A.arguments) }],
    },
    toolMessage(callIdA, 'pwsh', 'ok\n'),
  ];
  const projected = projectTurn(messages, TOOLS, POLICY);
  assert.ok(projected);
  assert.equal(projected.position, 'A');
  assert.equal(projected.steps, 1);
  assert.deepEqual(projected.candidates.map(c => c.candidate_id), ['B']);
});

test('projectTurn refuses an argument-mutated substitution', () => {
  const { message, bank } = authoredTurn();
  const callIdA = nativeCallId(bank.sha256, 'A');
  const tampered = { ...CANDIDATE_A.arguments, command: 'pytest -q --maxfail=1' };
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'Run the tests.' }] },
    message,
    toolMessage('call_root_1', 'pwsh', 'ok\n'),
    { role: 'assistant', content: [{ type: 'tool-call', id: callIdA, name: 'pwsh', arguments: JSON.stringify(tampered) }] },
    toolMessage(callIdA, 'pwsh', 'ok\n'),
  ];
  assert.equal(projectTurn(messages, TOOLS, POLICY), null);
});

test('projectTurn publishes only the current turn objective, not the whole transcript', () => {
  const { message } = authoredTurn();
  // A long-lived session carries earlier user text far beyond the public byte
  // budget. Only the newest turn's objective may reach the public receipt, so a
  // small current objective still projects successfully.
  const history = Array.from({ length: 6 }, (_, index) => ({
    role: 'user',
    content: [{ type: 'text', text: `earlier turn ${index}: ${'compacted history '.repeat(900)}` }],
  }));
  const messages = [
    ...history,
    { role: 'assistant', content: [{ type: 'text', text: 'noted' }] },
    { role: 'user', content: [{ type: 'text', text: 'Now run the tests.' }] },
    message,
    toolMessage('call_root_1', 'pwsh', 'ok\n'),
  ];
  const trace = [];
  const projected = projectTurn(messages, TOOLS, POLICY, trace);
  assert.ok(projected, `expected a projection, got trace ${trace.join(' | ')}`);
  assert.deepEqual(projected.objective, ['Now run the tests.']);
  assert.deepEqual(trace, [`anchor=${messages.length - 2}/${messages.length}`]);
});

test('projectTurn reports why a projection was refused', () => {
  const { message } = authoredTurn();
  const awaiting = [
    { role: 'user', content: [{ type: 'text', text: 'Run the tests.' }] },
    message,
  ];
  const awaitingTrace = [];
  assert.equal(projectTurn(awaiting, TOOLS, POLICY, awaitingTrace), null);
  assert.equal(awaitingTrace.at(-1), 'awaiting-receipt:role=assistant,id=call_root_1');

  const brokenChain = [
    ...awaiting,
    toolMessage('call_root_1', 'pwsh', 'ok\n'),
    { role: 'assistant', content: [{ type: 'text', text: 'the tests pass' }] },
  ];
  const chainTrace = [];
  assert.equal(projectTurn(brokenChain, TOOLS, POLICY, chainTrace), null);
  assert.equal(chainTrace.at(-1), 'chain-calls=0');
});

/* ------------------------------------------------------------- dispatch */

test('violateDispatch accepts only the exact authored action', () => {
  const { message, bank } = authoredTurn();
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'Run the tests.' }] },
    message,
    toolMessage('call_root_1', 'pwsh', 'ok\n'),
  ];
  const projected = projectTurn(messages, TOOLS, POLICY);
  const turn = { steps: 0 };
  const callId = nativeCallId(bank.sha256, 'A');

  assert.equal(violateDispatch({
    projected, turn, policy: POLICY,
    call: { id: callId, name: 'pwsh', arguments: JSON.stringify(CANDIDATE_A.arguments) },
  }), null);

  assert.match(violateDispatch({
    projected, turn, policy: POLICY,
    call: { id: callId, name: 'pwsh', arguments: JSON.stringify({ ...CANDIDATE_A.arguments, command: 'rm -rf /' }) },
  }).violation, /differ from the authored literal action/);

  assert.match(violateDispatch({
    projected, turn, policy: POLICY,
    call: { id: nativeCallId(bank.sha256, 'C'), name: 'pwsh', arguments: '{}' },
  }).violation, /not a live author-declared candidate/);

  assert.match(violateDispatch({
    projected, turn: { steps: 99 }, policy: POLICY,
    call: { id: callId, name: 'pwsh', arguments: JSON.stringify(CANDIDATE_A.arguments) },
  }).violation, /maximum_fast_steps/);

  assert.match(violateDispatch({
    projected: null, turn, policy: POLICY,
    call: { id: callId, name: 'pwsh', arguments: JSON.stringify(CANDIDATE_A.arguments) },
  }).violation, /no active executable decision graph/);
});

test('the certificate receipt hash binds the exact native result', () => {
  const receipt = publicReceipt({ name: 'pwsh', isError: false, text: 'ok\n', canonical: { kind: 'foreground', exitCode: 0 } });
  const other = publicReceipt({ name: 'pwsh', isError: false, text: 'ok\n', canonical: { kind: 'foreground', exitCode: 1 } });
  assert.notEqual(hashJson(receipt), hashJson(other));
});

/* -------------------------------------------------------------- admission */

function admittingTurn() {
  const { message } = authoredTurn();
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'keep going' }] },
    message,
    toolMessage('call_root_1', 'pwsh', 'ok\n'),
  ];
  return {
    projected: projectTurn(messages, TOOLS, POLICY),
    receipt: publicReceipt({ name: 'pwsh', isError: false, text: 'ok\n', canonical: { kind: 'foreground', exitCode: 0 } }),
  };
}

test('admission clamps the selector ceiling instead of trusting the author estimate', () => {
  const { projected, receipt } = admittingTurn();
  // The author declared 8192, but a selector call is a fast call: its ceiling
  // is the shipped FAST_CALL_UPPER_TOKENS, so 2048 tokens of slow work clears
  // the paper threshold with room to spare (2048 - 512 = 1536 > 1024).
  const decision = decide({ projected, receipt, slowTokens: 2048, deterministic: false, policy: POLICY });
  assert.equal(decision.kind, 'select');
  assert.equal(decision.admission.eligible.length, 1);
  assert.equal(decision.admission.eligible[0].fast_upper, 512);
  assert.equal(decision.admission.eligible[0].estimated_net_saving, 1536);

  // Without the clamp the declared 8192 would have vetoed the same step.
  const unclamped = decide({ projected, receipt, slowTokens: 2048, deterministic: false, policy: { ...POLICY, admission_threshold_tokens: 1024 } });
  assert.equal(unclamped.kind, 'select');
});

test('a raised admission threshold vetoes an otherwise admissible step', () => {
  const { projected, receipt } = admittingTurn();
  const decision = decide({
    projected, receipt, slowTokens: 2048, deterministic: false,
    policy: { ...POLICY, admission_threshold_tokens: 4096 },
  });
  assert.equal(decision.kind, 'slow');
  assert.equal(decision.admission.rejected[0].reason, 'net_saving_threshold');
  assert.equal(decision.admission.threshold_tokens, 4096);
});

test('the paper threshold stays the default', () => {
  const { projected, receipt } = admittingTurn();
  const decision = decide({ projected, receipt, slowTokens: 600, deterministic: false, policy: { ...POLICY, admission_threshold_tokens: undefined } });
  assert.equal(decision.admission.threshold_tokens, 1024);
  assert.equal(decision.kind, 'slow');
});

test('a multi-candidate step needs more than 1536 slow tokens at the paper threshold', () => {
  // With the 512-token fast-call ceiling, `net = slowTokens - 512`, so the paper
  // threshold of 1024 is only cleared above 1536 tokens. This is the boundary
  // that made every real desktop step abstain before the ceiling was clamped.
  const candidates = [
    { ...CANDIDATE_A, candidate_id: 'A', guard: { exit_code: 0 } },
    { ...CANDIDATE_A, candidate_id: 'B',
      arguments: { description: 'List the tree', command: 'ls -R' },
      guard: { exit_code: 0 } },
  ];
  const message = {
    role: 'assistant',
    content: [
      { type: 'text', text: `<CHISEL_PLAN>${JSON.stringify({ candidates })}</CHISEL_PLAN>` },
      { type: 'tool-call', id: 'call_root_1', name: 'pwsh',
        arguments: JSON.stringify({ description: 'Start', command: 'echo hi' }) },
    ],
  };
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'keep going' }] },
    message,
    { role: 'tool', toolCallId: 'call_root_1', content: [{ type: 'text', text: 'hi' }] },
  ];
  const projected = projectTurn(messages, TOOLS, POLICY);
  assert.equal(projected.candidates.length, 2);
  const receipt = publicReceipt({ name: 'pwsh', isError: false, text: 'hi', canonical: { kind: 'foreground', exitCode: 0 } });

  const justUnder = decide({ projected, receipt, slowTokens: 1536, deterministic: false, policy: POLICY });
  assert.equal(justUnder.kind, 'slow');
  assert.equal(justUnder.admission.rejected[0].estimated_net_saving, 1024);

  const justOver = decide({ projected, receipt, slowTokens: 1537, deterministic: false, policy: POLICY });
  assert.equal(justOver.kind, 'select');
  assert.equal(justOver.admission.eligible.length, 2);
});