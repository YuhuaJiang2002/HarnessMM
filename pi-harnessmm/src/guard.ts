/**
 * Guard evaluation: receiptGuard (v5 condition strings) and localGuard (v6+ nodes).
 *
 * Port of HarnessMM integrations/deepseek_harness/sar_v2.mjs. Every guard reads
 * observed native receipts only. `known` separates "the evidence says no" from
 * "the evidence cannot be read"; both refuse admission, but only the former is
 * a definite negative.
 */
import { hash } from "./policy.ts";

/** Verdict of a v5 condition string against one receipt. */
export interface ReceiptGuardVerdict {
  eligible: boolean;
  known: boolean;
  reason?: string;
  publicFields?: Record<string, unknown>;
  receipt_sha256?: string;
}

const RECEIPT_KEYS = ["return_code", "stdout_contains", "stdout_equals"];

/** v5 receipt guard: a JSON condition string checked against the immediate receipt. */
export function receiptGuard(condition: string, receipt: any): ReceiptGuardVerdict {
  if (receipt?.is_error) return { eligible: false, known: true };
  let guard: any;
  try {
    guard = JSON.parse(condition);
  } catch {
    return { eligible: false, known: false };
  }
  if (!guard || typeof guard !== "object" || Array.isArray(guard) || !Object.keys(guard).length || Object.keys(guard).some(key => !RECEIPT_KEYS.includes(key))) {
    return { eligible: false, known: false };
  }
  let value: any = receipt?.value?.native_text ?? receipt?.value;
  for (let i = 0; i < 3 && typeof value === "string"; i++) {
    try {
      value = JSON.parse(value);
    } catch {
      return { eligible: false, known: false };
    }
  }
  if (!value || typeof value !== "object" || !Number.isSafeInteger(value.return_code)) return { eligible: false, known: false };
  // Unexpected native errors always return to slow; failed-result recovery is authored anew.
  if (value.return_code !== 0 || receipt?.is_error) return { eligible: false, known: true };
  const publicFields: Record<string, unknown> = {};
  for (const [key, expected] of Object.entries(guard)) {
    if (key === "return_code") {
      if (!Number.isSafeInteger(expected) || expected !== 0) return { eligible: false, known: false };
      publicFields.return_code = value.return_code;
      if (value.return_code !== expected) return { eligible: false, known: true };
    } else {
      if (typeof expected !== "string" || !expected.length || typeof value.stdout !== "string") return { eligible: false, known: false };
      const ok = key === "stdout_contains" ? value.stdout.includes(expected) : value.stdout === expected;
      publicFields[key] = { literal: expected, matched: ok };
      if (!ok) return { eligible: false, known: true };
    }
  }
  return { eligible: true, known: true, publicFields, receipt_sha256: hash(receipt) };
}

/** Verdict of a v6+ node guard against the immediate predecessor receipt. */
export interface LocalGuardVerdict {
  known: boolean;
  eligible: boolean;
  reason?: string;
  fields?: Record<string, unknown>;
}

const FIELD_MAP: Record<string, string> = {
  return_code: "return_code",
  stdout_contains: "stdout",
  stdout_equals: "stdout",
  stderr_empty: "stderr",
  json_equals: "stdout",
};

/** v6+ node guard. Deterministic, closed, and fail-closed on unreadable evidence. */
export function localGuard(node: { guard?: unknown; required_receipt_fields?: unknown }, receipt: any): LocalGuardVerdict {
  const guard = node.guard;
  if (!guard || typeof guard !== "object" || Array.isArray(guard) || !Object.keys(guard).length) {
    return { known: false, eligible: false, reason: "invalid_guard" };
  }
  let r: any = receipt?.value?.native_text ?? receipt?.value;
  for (let i = 0; i < 3 && typeof r === "string"; i++) {
    try {
      r = JSON.parse(r);
    } catch {
      return { known: false, eligible: false, reason: "non_json_receipt" };
    }
  }
  if (!r || typeof r !== "object") return { known: false, eligible: false, reason: "missing_receipt" };
  if (receipt?.is_error) return { known: true, eligible: false, reason: "native_error" };
  const required = node.required_receipt_fields;
  if (
    !Array.isArray(required) ||
    required.some((key: string) => !Object.hasOwn(r, key)) ||
    Object.keys(guard).some(key => !FIELD_MAP[key] || !required.includes(FIELD_MAP[key]))
  ) {
    return { known: false, eligible: false, reason: "required_field_missing" };
  }
  for (const [key, value] of Object.entries(guard as Record<string, any>)) {
    let ok: boolean | undefined;
    if (key === "return_code") {
      if (!Number.isSafeInteger(value) || !Number.isSafeInteger(r.return_code)) return { known: false, eligible: false };
      ok = r.return_code === value;
    }
    if (key === "stdout_contains" || key === "stdout_equals") {
      if (typeof value !== "string" || !value.length || typeof r.stdout !== "string") return { known: false, eligible: false };
      ok = key === "stdout_contains" ? r.stdout.includes(value) : r.stdout === value;
    }
    if (key === "stderr_empty") {
      if (typeof value !== "boolean" || !(r.stderr === null || typeof r.stderr === "string")) return { known: false, eligible: false };
      ok = (r.stderr === null || r.stderr === "") === value;
    }
    if (key === "json_equals") {
      if (!value || !Array.isArray(value.path) || value.path.some((part: unknown) => typeof part !== "string" || ["__proto__", "prototype", "constructor"].includes(part)) || !Object.hasOwn(value, "value")) {
        return { known: false, eligible: false };
      }
      let parsed: any;
      try {
        parsed = JSON.parse(r.stdout);
      } catch {
        return { known: false, eligible: false };
      }
      for (const part of value.path) {
        if (!parsed || typeof parsed !== "object" || !Object.hasOwn(parsed, part)) return { known: false, eligible: false };
        parsed = parsed[part];
      }
      ok = hash(parsed) === hash(value.value);
    }
    if (!ok) return { known: true, eligible: false, reason: "guard_false" };
  }
  return { known: true, eligible: true, fields: Object.fromEntries((required as string[]).map(key => [key, r[key]])) };
}

/** Receipt shape both guards accept: native text, never a predicted value. */
export function guardReceipt(text: string, isError: boolean): { is_error: boolean; value: { native_text: string } } {
  return { is_error: isError, value: { native_text: text } };
}
