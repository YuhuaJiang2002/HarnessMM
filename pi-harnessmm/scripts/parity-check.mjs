#!/usr/bin/env node
/**
 * Differential parity check against the reference implementation.
 *
 * Loads HarnessMM's own modules and this port side by side, feeds both the same
 * randomized inputs, and fails on the first disagreement. Run it after changing
 * anything in src/bank.ts, src/guard.ts or src/admission.ts:
 *
 *   node scripts/parity-check.mjs /path/to/HarnessMM
 */
import assert from "node:assert/strict";

const root = process.argv[2] ?? process.env.HARNESSMM_REFERENCE ?? "";
if (!root) {
  console.error("usage: node scripts/parity-check.mjs <path to HarnessMM checkout>");
  process.exit(2);
}

const ref = {
  cot: await import(`${root}/integrations/deepseek_harness/cot_jev.mjs`),
  sar: await import(`${root}/integrations/deepseek_harness/sar_v2.mjs`),
  admission: await import(`${root}/integrations/deepseek_harness/sar_admission.mjs`),
};
const port = {
  bank: await import("../src/bank.ts"),
  guard: await import("../src/guard.ts"),
  admission: await import("../src/admission.ts"),
};

const policy = { version: 6, enabled: true, maximum_candidates: 16, maximum_fast_steps: 8, maximum_public_bytes: 65536 };
const tool = { name: "terminal", parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"], additionalProperties: false } };

const receipts = [
  { return_code: 0, stdout: "READY", stderr: null },
  { return_code: 0, stdout: "NOT_READY", stderr: "" },
  { return_code: 1, stdout: "", stderr: "boom" },
  { return_code: 0, stdout: '{"count":3}', stderr: null },
  { return_code: 0, stdout: '{"count":4}', stderr: null },
  { return_code: 0, stdout: "", stderr: "  " },
];

const guards = [
  { return_code: 0 },
  { return_code: 1 },
  { stdout_contains: "READY" },
  { stdout_equals: "READY" },
  { stderr_empty: true },
  { stderr_empty: false },
  { json_equals: { path: ["count"], value: 3 } },
  { json_equals: { path: ["missing"], value: 3 } },
  { task_solved: true },
];

let checked = 0;
for (const guard of guards) {
  for (const receipt of receipts) {
    for (const isError of [false, true]) {
      const node = { guard, required_receipt_fields: ["return_code", "stdout", "stderr"] };
      const value = { is_error: isError, value: { native_text: JSON.stringify(receipt) } };
      const a = ref.sar.localGuard(node, value);
      const b = port.guard.localGuard(node, value);
      assert.deepEqual(
        { known: b.known, eligible: b.eligible },
        { known: a.known, eligible: a.eligible },
        `localGuard disagreed for ${JSON.stringify(guard)} on ${JSON.stringify(receipt)} isError=${isError}`,
      );
      checked += 1;
    }
  }
}

const reference = (candidates, receipt) => {
  const state = {
    public_user_requests: [{ role: "user", content: [{ type: "text", text: "task" }] }],
    candidates,
    public_results: [{ call_id: "root", is_error: !!receipt.is_error, status: "PARSED", value: { native_text: JSON.stringify(receipt) } }],
  };
  return { admission: ref.sar.deterministicAdmission({ state }, 40000, true), state };
};

for (const [index, receipt] of receipts.entries()) {
  const candidates = ["A", "B", "C"].map((label, offset) => ({
    label,
    tool: "terminal",
    arguments: { command: `printf ${label}` },
    node: {
      candidate_id: label,
      guard: { return_code: 0 },
      required_receipt_fields: ["return_code"],
      expected_local_cost: { selector_upper_tokens: 8192 },
      dependency_ids: index === 0 && offset === 2 ? ["A"] : [],
    },
  }));
  const a = reference(candidates, receipt).admission;
  const b = port.admission.deterministicAdmission({
    public_user_requests: [{ role: "user", content: [{ type: "text", text: "task" }] }],
    candidates,
    public_results: [{ call_id: "root", is_error: !!receipt.is_error, status: "PARSED", value: { native_text: JSON.stringify(receipt) } }],
  }, 40000, true);
  assert.deepEqual(
    { eligible: b.eligible.map(item => item.candidate.label), reasons: b.rejected.map(item => item.reason) },
    { eligible: a.eligible.map(item => item.candidate.label), reasons: a.rejected.map(item => item.reason) },
    `deterministicAdmission disagreed for ${JSON.stringify(receipt)}`,
  );
  checked += 1;
}

assert.equal(
  (await import("../src/policy.ts")).planInstruction(policy),
  ref.cot.planInstruction(policy),
  "authoring instruction drifted",
);
checked += 1;

console.log(`parity check passed: ${checked} comparisons, 0 disagreements`);
