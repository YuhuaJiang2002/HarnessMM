/**
 * Guard conformance.
 * Mirrors HarnessMM tests/sar_subdag.test.mjs ("unknown receipt fields, semantic
 * guards and future edges cannot invoke selector", "receipt-local stderr and exact
 * JSON values are deterministic") and tests/sar_v2.test.mjs ("guards reject
 * uncertain receipts and unexpected errors").
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { localGuard, receiptGuard } from "../src/guard.ts";

const node = (over: Record<string, unknown> = {}) => ({
  guard: { return_code: 0 },
  required_receipt_fields: ["return_code"],
  ...over,
});
const receipt = (value: unknown, isError = false) => ({ is_error: isError, value: { native_text: JSON.stringify(value) } });

test("a receipt field that was not required cannot be guarded on", () => {
  assert.equal(localGuard({ ...node({ guard: { stdout_equals: "OK" }, required_receipt_fields: ["return_code"] }) }, receipt({ return_code: 0, stdout: "OK" })).known, false);
  assert.equal(localGuard({ ...node({ guard: { return_code: 0 }, required_receipt_fields: [] }) }, receipt({ return_code: 0 })).known, false);
});

test("a semantic guard is unknown, never eligible", () => {
  const verdict = localGuard(node({ guard: { task_solved: true }, required_receipt_fields: ["return_code"] }), receipt({ return_code: 0 }));
  assert.equal(verdict.known, false);
  assert.equal(verdict.eligible, false);
});

test("stderr_empty and json_equals are deterministic and fail closed", () => {
  const guarded = node({
    guard: { stderr_empty: true, json_equals: { path: ["count"], value: 3 } },
    required_receipt_fields: ["stderr", "stdout"],
  });
  assert.equal(localGuard(guarded, receipt({ stderr: null, stdout: '{"count":3}' })).eligible, true);
  assert.equal(localGuard(guarded, receipt({ stderr: null, stdout: "{}" })).known, false);
  assert.equal(localGuard(guarded, receipt({ stderr: "boom", stdout: '{"count":3}' })).eligible, false);
  assert.equal(
    localGuard(node({ guard: { json_equals: { path: ["__proto__"], value: 1 } }, required_receipt_fields: ["stdout"] }), receipt({ stdout: "{}" })).known,
    false,
  );
});

test("a native error refuses admission but stays known", () => {
  const verdict = localGuard(node(), receipt({ return_code: 0 }, true));
  assert.deepEqual({ known: verdict.known, eligible: verdict.eligible }, { known: true, eligible: false });
  assert.equal(verdict.reason, "native_error");
});

test("a guard that does not hold is a definite negative", () => {
  const verdict = localGuard(node(), receipt({ return_code: 1 }));
  assert.deepEqual({ known: verdict.known, eligible: verdict.eligible, reason: verdict.reason }, { known: true, eligible: false, reason: "guard_false" });
});

test("an unreadable receipt is unknown", () => {
  assert.equal(localGuard(node(), { is_error: false, value: { native_text: "READY" } }).known, false);
  assert.equal(localGuard(node(), { is_error: false }).known, false);
  assert.equal(localGuard(node({ guard: {} }), receipt({ return_code: 0 })).known, false);
});

test("v5 condition strings check the same evidence", () => {
  const value = { return_code: 0, stdout: "READY" };
  assert.equal(receiptGuard('{"return_code":0,"stdout_contains":"READY"}', receipt(value)).eligible, true);
  assert.equal(receiptGuard('{"return_code":0}', receipt({ return_code: 1, stdout: "READY" })).eligible, false);
  assert.equal(receiptGuard("looks good", receipt(value)).known, false);
  assert.equal(receiptGuard('{"stdout_contains":"READY"}', receipt({ return_code: 0, stdout: "NOT_READY?" })).eligible, true);
  assert.equal(receiptGuard('{"stdout_equals":"READY"}', receipt({ return_code: 0, stdout: "NOT_READY" })).eligible, false);
  assert.equal(receiptGuard('{"return_code":0}', receipt(value, true)).eligible, false);
});
