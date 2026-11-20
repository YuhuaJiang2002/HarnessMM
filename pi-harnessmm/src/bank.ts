/**
 * Action-bank parsing: the `<CHISEL_PLAN>` EDG contract.
 *
 * Port of HarnessMM integrations/deepseek_harness/cot_jev.mjs (publicReceipt,
 * terminalArguments, schemaMatches, parseBank, nativeCallId, decisionState).
 * Behaviour is preserved, including which violations throw and which inputs
 * yield `null` (no bank) instead of an error.
 */
import { isDeepStrictEqual } from "node:util";
import { BankContractError, LABELS, hash, object, type Policy } from "./policy.ts";

/** A normalized assistant turn, shaped like the upstream slow message. */
export interface AuthoringMessage {
  role: "assistant";
  content: Array<
    | { type: "text"; text: string }
    | { type: "reasoning"; text: string }
    | { type: "tool-call"; id: string; name: string; arguments: string }
  >;
}

/** One authored candidate, exactly as the frozen policy admits it. */
export interface Candidate {
  candidate_id: string;
  predecessor_id: string;
  tool: string;
  arguments: Record<string, unknown>;
  guard: Record<string, unknown>;
  required_receipt_fields: string[];
  expected_local_cost: { selector_upper_tokens: number };
  dependency_ids?: string[];
  args_resolved?: boolean;
  requires_new_reasoning?: boolean;
  depends_only_on_public_receipt?: boolean;
  native_action_is_literal?: boolean;
  /** Upstream v5 field names, filled in for versions before 6. */
  label?: string;
  after?: string;
  condition?: string;
  /** The v6+ node, used by guards and admission. */
  node?: Record<string, any>;
}

/** A parsed bank bound to the root call that authored it. */
export interface Bank {
  candidates: Candidate[];
  root: { type: "tool-call"; id: string; name: string; arguments: string };
  reasoning: Array<{ type: "reasoning"; text: string }>;
  sha256: string;
}

/** Receipt as the guards consume it: native text, never a predicted value. */
export type PublicResult =
  | { call_id?: string; is_error?: boolean | null; status: "PARSED"; value: any }
  | { call_id?: string; is_error?: boolean | null; status: "UNKNOWN"; reason: string };

/** Bank state the controller evaluates: receipts, candidates and user constraints. */
export interface BankState {
  public_user_requests: Array<{ role: string; content: unknown }>;
  candidates: Candidate[];
  public_results: PublicResult[];
  native_tool_contract?: unknown;
}

/** Decode one native tool message into the receipt a guard may read. */
export function publicReceipt(
  message: { toolCallId?: string; isError?: boolean | null; content?: unknown },
  singleEncoding = false,
): PublicResult {
  const identity = { call_id: message.toolCallId, is_error: message.isError ?? null };
  const content = message.content as Array<{ type: string; text?: string }> | undefined;
  if (!Array.isArray(content) || content.length !== 1 || content[0].type !== "text" || typeof content[0].text !== "string") {
    return { ...identity, status: "UNKNOWN", reason: "non_scalar_native_text" };
  }
  let value: any = content[0].text;
  for (let depth = 0; depth < 3 && typeof value === "string"; depth++) {
    try {
      if (singleEncoding) {
        const tokens = value.matchAll(/"(?:[^"\\]|\\.)*"|(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/g);
        for (const token of tokens) {
          if (token[1] !== undefined && JSON.stringify(Number(token[1])) !== token[1]) {
            return { ...identity, status: "UNKNOWN", reason: "noncanonical_native_number" };
          }
        }
      }
      value = JSON.parse(value);
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
      return { ...identity, status: "UNKNOWN", reason: "unparsed_native_observation" };
    }
  }
  if (!object(value)) return { ...identity, status: "UNKNOWN", reason: "non_object_native_result" };
  return { ...identity, status: "PARSED", value };
}

/** The closed terminal arguments the v1-v4 contract allows. */
function terminalArguments(value: unknown): boolean {
  return (
    object(value) &&
    Object.keys(value).every(key => ["command", "cwd", "timeout_sec"].includes(key)) &&
    typeof value.command === "string" &&
    value.command.trim().length > 0 &&
    !value.command.includes("\0") &&
    (!Object.hasOwn(value, "cwd") || (typeof value.cwd === "string" && value.cwd.length > 0 && !value.cwd.includes("\0"))) &&
    (!Object.hasOwn(value, "timeout_sec") || (Number.isSafeInteger(value.timeout_sec) && value.timeout_sec > 0))
  );
}

/** JSON-Schema subset check used to prove candidate arguments match the tool contract. */
export function schemaMatches(value: unknown, schema: any, root: any = schema, depth = 0): boolean {
  if (!object(schema) || depth > 16) return false;
  if (schema.$ref) {
    if (typeof schema.$ref !== "string" || !schema.$ref.startsWith("#/")) return false;
    let target: any = root;
    for (const part of schema.$ref.slice(2).split("/")) {
      const key = part.replace(/~1/g, "/").replace(/~0/g, "~");
      if (!object(target) || !Object.hasOwn(target, key)) return false;
      target = target[key];
    }
    return schemaMatches(value, target, root, depth + 1);
  }
  if (schema.enum && !schema.enum.some((item: unknown) => isDeepStrictEqual(item, value))) return false;
  if (Object.hasOwn(schema, "const") && !isDeepStrictEqual(schema.const, value)) return false;
  if (schema.allOf && !schema.allOf.every((item: unknown) => schemaMatches(value, item, root, depth + 1))) return false;
  if (schema.anyOf && !schema.anyOf.some((item: unknown) => schemaMatches(value, item, root, depth + 1))) return false;
  if (schema.oneOf && schema.oneOf.filter((item: unknown) => schemaMatches(value, item, root, depth + 1)).length !== 1) return false;
  const types = Array.isArray(schema.type) ? schema.type : [schema.type];
  const actual = value === null ? "null" : Array.isArray(value) ? "array" : typeof value === "object" ? "object" : typeof value;
  if (schema.type && !types.includes(actual) && !(actual === "number" && Number.isSafeInteger(value) && types.includes("integer"))) return false;
  if (actual === "number") {
    if (!Number.isFinite(value) || (Number.isInteger(value) && !Number.isSafeInteger(value)) || (schema.minimum !== undefined && (value as number) < schema.minimum) || (schema.maximum !== undefined && (value as number) > schema.maximum)) return false;
  } else if (actual === "string") {
    const text = value as string;
    if (text.includes("\0") || (schema.minLength !== undefined && text.length < schema.minLength) || (schema.maxLength !== undefined && text.length > schema.maxLength)) return false;
    if (schema.pattern && !new RegExp(schema.pattern).test(text)) return false;
  } else if (actual === "array") {
    const items = value as unknown[];
    if ((schema.minItems !== undefined && items.length < schema.minItems) || (schema.maxItems !== undefined && items.length > schema.maxItems) || (schema.items && !items.every(item => schemaMatches(item, schema.items, root, depth + 1)))) return false;
  } else if (actual === "object") {
    const record = value as Record<string, unknown>;
    const properties = schema.properties ?? {};
    if ((schema.required ?? []).some((key: string) => !Object.hasOwn(record, key))) return false;
    for (const [key, item] of Object.entries(record)) {
      if (Object.hasOwn(properties, key)) {
        if (!schemaMatches(item, properties[key], root, depth + 1)) return false;
      } else if (schema.additionalProperties === false) return false;
      else if (object(schema.additionalProperties) && !schemaMatches(item, schema.additionalProperties, root, depth + 1)) return false;
    }
  }
  return true;
}

/** Tool contract the bank is validated against. */
export interface ToolContract {
  name: string;
  parameters: any;
}

/**
 * Parse the single action bank a slow turn authored.
 * Returns null when the turn declares no bank; throws BankContractError when it
 * declares one that violates the frozen contract.
 */
export function parseBank(message: AuthoringMessage | null, policy: Policy, tools: ToolContract[] = []): Bank | null {
  if (message?.role !== "assistant" || !Array.isArray(message.content)) return null;
  const text = message.content
    .filter(part => part.type === "text" || (policy.version >= 8 && part.type === "reasoning"))
    .map(part => (part as { text: string }).text)
    .join("\n");
  const matches = [...text.matchAll(/<CHISEL_PLAN>([\s\S]*?)<\/CHISEL_PLAN>/g)];
  if (!matches.length) return null;
  if (matches.length !== 1) throw new BankContractError("A slow turn must declare only one action bank");
  const roots = message.content.filter((part): part is Extract<AuthoringMessage["content"][number], { type: "tool-call" }> => part.type === "tool-call");
  if (
    roots.length !== 1 ||
    typeof roots[0].id !== "string" ||
    (policy.version < 3 && roots[0].name !== "terminal") ||
    (policy.version >= 3 && !tools.some(tool => tool.name === roots[0].name))
  ) {
    throw new BankContractError("An action bank requires exactly one registered native root");
  }
  let bank: any;
  try {
    bank = JSON.parse(matches[0][1]);
  } catch (error) {
    if (!(error instanceof SyntaxError)) throw error;
    throw new BankContractError("Slow CoT emitted an invalid JSON action bank");
  }
  if (!object(bank) || Object.keys(bank).join() !== "candidates" || !Array.isArray(bank.candidates) || bank.candidates.length > policy.maximum_candidates) {
    throw new BankContractError("Slow CoT action bank exceeds its closed candidate contract");
  }
  const declared = new Set(["root"]);
  let rootArguments: any;
  if (policy.version >= 4) {
    try {
      rootArguments = JSON.parse(roots[0].arguments);
    } catch {
      throw new BankContractError("The native root needs literal JSON arguments");
    }
    if (!object(rootArguments)) throw new BankContractError("The native root needs object arguments");
  }
  for (let candidate of bank.candidates as any[]) {
    if (policy.version >= 6) {
      if (policy.version >= 8) {
        const flags = ["args_resolved", "requires_new_reasoning", "depends_only_on_public_receipt", "native_action_is_literal"];
        if (
          candidate?.args_resolved !== true ||
          candidate?.requires_new_reasoning !== false ||
          candidate?.depends_only_on_public_receipt !== true ||
          candidate?.native_action_is_literal !== true ||
          candidate?.guard?.return_code !== 0 ||
          !candidate?.required_receipt_fields?.includes("return_code")
        ) {
          throw new BankContractError("Candidate needs fully resolved literal public-receipt arguments");
        }
        candidate = bank.candidates[bank.candidates.indexOf(candidate)] = Object.fromEntries(
          Object.entries(candidate).filter(([key]) => !flags.includes(key)),
        );
      }
      const keys = object(candidate) ? Object.keys(candidate).sort().join() : "";
      const withDeps = "candidate_id,dependency_ids,expected_local_cost,guard,predecessor_id,required_receipt_fields,tool,arguments".split(",").sort().join();
      const withoutDeps = "candidate_id,expected_local_cost,guard,predecessor_id,required_receipt_fields,tool,arguments".split(",").sort().join();
      if (
        !object(candidate) ||
        ![withDeps, withoutDeps].includes(keys) ||
        !object(candidate.guard) ||
        !Object.keys(candidate.guard).length ||
        Object.keys(candidate.guard).some(key => !["return_code", "stdout_contains", "stdout_equals", "stderr_empty", "json_equals"].includes(key)) ||
        !Array.isArray(candidate.required_receipt_fields) ||
        !object(candidate.expected_local_cost) ||
        !Number.isSafeInteger(candidate.expected_local_cost.selector_upper_tokens) ||
        (candidate.dependency_ids && (!Array.isArray(candidate.dependency_ids) || candidate.dependency_ids.some((id: string) => !declared.has(id))))
      ) {
        throw new BankContractError("Nonlocal or incomplete executable sub-DAG node");
      }
      candidate.node = structuredClone(candidate);
      if (policy.version >= 8) {
        Object.assign(candidate.node, {
          args_resolved: true,
          requires_new_reasoning: false,
          depends_only_on_public_receipt: true,
          native_action_is_literal: true,
        });
      }
      candidate.label = candidate.candidate_id;
      candidate.after = candidate.predecessor_id;
      candidate.condition = JSON.stringify(candidate.guard);
    }
    if (
      !object(candidate) ||
      Object.keys(candidate).sort().join() !== (policy.version >= 6 ? Object.keys(candidate).sort().join() : "after,arguments,condition,label,tool") ||
      typeof candidate.label !== "string" ||
      candidate.label.length !== 1 ||
      !LABELS.slice(0, policy.maximum_candidates).includes(candidate.label) ||
      declared.has(candidate.label) ||
      !declared.has(candidate.after) ||
      typeof candidate.condition !== "string" ||
      !candidate.condition.trim() ||
      (policy.version < 3 && (candidate.tool !== "terminal" || !terminalArguments(candidate.arguments))) ||
      (policy.version >= 3 &&
        (!object(candidate.arguments) || !tools.some(tool => tool.name === candidate.tool && schemaMatches(candidate.arguments, tool.parameters))))
    ) {
      throw new BankContractError("Invalid, cyclic, duplicate or unsupported CoT-authored candidate");
    }
    if (policy.version >= 4 && candidate.after === "root" && candidate.tool === roots[0].name && isDeepStrictEqual(candidate.arguments, rootArguments)) {
      throw new BankContractError("A future candidate cannot repeat the already executed native root");
    }
    declared.add(candidate.label);
  }
  const reasoning = message.content.filter((part): part is { type: "reasoning"; text: string } => part.type === "reasoning");
  if (reasoning.some(part => typeof part.text !== "string")) {
    throw new BankContractError("Native authored reasoning must retain its exact text");
  }
  return {
    ...bank,
    candidates: bank.candidates as Candidate[],
    root: roots[0],
    reasoning: reasoning.map(part => structuredClone(part)),
    sha256: hash(JSON.stringify([roots[0].id, matches[0][1]])),
  };
}

/** Synthetic native call id for a committed candidate, identical upstream. */
export function nativeCallId(bank: Bank, label: string): string {
  return `chisel_${bank.sha256.slice(0, 24)}_${label}`;
}
