/**
 * Admission: decide whether a guard-satisfied candidate may replace a slow call.
 *
 * Port of HarnessMM integrations/deepseek_harness/sar_v2.mjs
 * (deterministicAdmission) and sar_admission.mjs (actionType, compactState,
 * admit). Both paths are here because the paper reports both: versions <= 7 use
 * the deterministic net-saving threshold, versions 8 and 10 require frozen TRAIN
 * priors and fail closed when they are absent.
 */
import { byteUpper, hash, SELECTOR_SYSTEM } from "./policy.ts";
import { localGuard } from "./guard.ts";

export { SELECTOR_SYSTEM };

/** A candidate plus its authored node, as the admission sees it. */
export interface AdmissionCandidate {
  label: string;
  tool: string;
  arguments: Record<string, unknown>;
  node: Record<string, any>;
}

/** One admission decision with the estimate that produced it. */
export interface AdmissionDecision {
  candidate: AdmissionCandidate;
  check: ReturnType<typeof localGuard>;
  estimated_net_saving: number;
  fast_upper: number;
  reason: string;
}

export interface AdmissionResult {
  eligible: AdmissionDecision[];
  rejected: AdmissionDecision[];
  threshold_tokens: number;
  identity: string;
}

/**
 * Deterministic admission for policy versions <= 7.
 * @param state `{ public_results, candidates, public_user_requests }` from the bank state.
 * @param slowTokens total tokens of the last paid slow call; the saving estimate.
 * @param fullSAR true for version 6, where the net-saving threshold applies.
 */
export function deterministicAdmission(state: any, slowTokens: number, fullSAR = true): AdmissionResult {
  const receipt = state.public_results.at(-1);
  // Each candidate is checked once and reused: the loop below needs every verdict
  // twice (own eligibility and the deterministic-commit test).
  const verdicts = new Map<AdmissionCandidate, ReturnType<typeof localGuard>>();
  const done = new Set<string>(["root", ...state.public_results.map((r: any) => r.call_id.split("_").at(-1))]);
  const eligible: AdmissionDecision[] = [];
  const rejected: AdmissionDecision[] = [];
  for (const candidate of state.candidates as AdmissionCandidate[]) {
    const node = candidate.node;
    const check = localGuard(node, receipt);
    verdicts.set(candidate, check);
    const missingDependency = (node.dependency_ids ?? []).some((id: string) => !done.has(id));
    // Provider calls are the same backbone. The estimate is deliberately visible,
    // uses only paid TRAIN/earlier slow usage, and is not a guaranteed bound.
    const locallyValid = (state.candidates as AdmissionCandidate[]).filter(x =>
      (verdicts.get(x) ?? localGuard(x.node, receipt)).eligible && (x.node.dependency_ids ?? []).every((id: string) => done.has(id)),
    );
    const deterministic = fullSAR && locallyValid.length === 1;
    const fastUpper = deterministic
      ? 0
      : Math.max(node.expected_local_cost.selector_upper_tokens, byteUpper({ objective: state.public_user_requests, candidates: state.candidates }));
    const net = slowTokens - fastUpper;
    const risk = receipt?.is_error || (check.known && check.fields?.return_code !== undefined && check.fields.return_code !== 0);
    const admitted = check.known && check.eligible && !missingDependency && !risk && (!fullSAR || net > 1024);
    (admitted ? eligible : rejected).push({
      candidate,
      check,
      estimated_net_saving: net,
      fast_upper: fastUpper,
      reason: missingDependency
        ? "dependency_missing"
        : risk
          ? "native_error"
          : !check.known
            ? "guard_unknown"
            : !check.eligible
              ? "guard_false"
              : net <= 1024
                ? "net_saving_threshold"
                : "admitted",
    });
  }
  return {
    eligible,
    rejected,
    threshold_tokens: 1024,
    identity: "prospective prior from preceding paid TRAIN slow usage; candidate estimate charged by complete wire ledger",
  };
}

/** Coarse action category used by the priors-based admission. */
export function actionType(node: any): string | null {
  const command = node?.arguments?.command;
  if (
    node?.tool !== "terminal" ||
    typeof command !== "string" ||
    /\/(?:mnt\/phiagent-data|data0\/phiagent|opt\/phiagent|Users)\b/.test(node.arguments.cwd ?? "") ||
    /\/(?:mnt\/phiagent-data|data0\/phiagent|opt\/phiagent|tests|solution)\b/.test(command) ||
    /(?:rm\s+-[a-z]*r|sudo\b|docker\b|chmod\s+777|curl.*\|.*(?:sh|bash))/i.test(command)
  ) {
    return null;
  }
  if (/\b(?:pytest|unittest|npm test|cargo test|go test|make check|ctest)\b/.test(command)) return "verify";
  if (/^(?:\s*cd\s+[^;&]+\s*&&\s*)?(?:cat|head|tail|ls|find|pwd|rg|grep|sed\s+-n|stat|wc|file)\b/.test(command)) return "inspect";
  if (/\b(?:python|python3|make|gcc|cc|cargo|npm|git|printf|echo|cp|mkdir|patch|sed)\b/.test(command)) return "edit_or_build";
  return null;
}

/** The compact selector state: task constraints and guard results, nothing else. */
export function compactState(state: any, candidates: AdmissionCandidate[], episodeId: string): Record<string, unknown> {
  const immediate = state.public_results.at(-1);
  return {
    task: episodeId,
    receipt_hash: hash(immediate),
    candidates: candidates.map(candidate => ({
      id: candidate.label,
      action: { tool: candidate.tool, arguments: candidate.arguments },
      guard: candidate.node.guard,
      public_guard_result: localGuard(candidate.node, immediate).eligible,
    })),
    schema: { terminal: ["command", "cwd", "timeout_sec"] },
  };
}

/** Priors file the v8/v10 admission requires; absent means "fail closed". */
export interface TrainPriors {
  phase: string;
  source_sha256?: string;
  categories: Record<string, { phase: string; median_tokens: number }>;
}

/**
 * Priors-based admission for policy versions 8 and 10.
 * Every selectable candidate must have a frozen TRAIN median for its category,
 * and the fast path must stay under a quarter of the replaced slow call.
 */
export function admit(state: any, priors: TrainPriors | null, episodeId: string, fullSAR = true) {
  const receipt = state.public_results.at(-1);
  const done = new Set<string>(["root", ...state.public_results.map((r: any) => r.call_id.split("_").at(-1))]);
  const eligible: any[] = [];
  const rejected: any[] = [];
  const locallyValid = (state.candidates as AdmissionCandidate[]).filter(candidate => localGuard(candidate.node, receipt).eligible);
  const selectorState = compactState(state, locallyValid, episodeId);
  // UTF-8 byte count is a conservative serialization estimate, not measured usage.
  const fastUpper = byteUpper(SELECTOR_SYSTEM) + byteUpper(selectorState) + 1 + 128;
  for (const candidate of state.candidates as AdmissionCandidate[]) {
    const node = candidate.node;
    const check = localGuard(node, receipt);
    const type = actionType(node);
    const prior = priors?.categories?.[type as string];
    const resolved =
      node.args_resolved === true &&
      node.requires_new_reasoning === false &&
      node.depends_only_on_public_receipt === true &&
      node.native_action_is_literal === true;
    const reason = !resolved
      ? "unresolved_arguments"
      : !type
        ? "not_allowlisted_or_high_risk"
        : !check.known
          ? "unknown_public_receipt"
          : !check.eligible
            ? "guard_false"
            : receipt.is_error || (check.fields?.return_code !== undefined && check.fields.return_code !== 0)
              ? "native_error"
              : (node.dependency_ids ?? []).some((id: string) => !done.has(id))
                ? "dependency_missing"
                : !prior || prior.phase !== "train" || !Number.isSafeInteger(prior.median_tokens) || prior.median_tokens <= 0
                  ? "missing_train_estimate"
                  : fullSAR && fastUpper * 4 > prior.median_tokens
                    ? "fast_cost_above_quarter_of_slow"
                    : null;
    const decision = {
      candidate,
      reason,
      estimated_fast_total_tokens: fastUpper,
      expected_slow_calls_saved: 1,
      estimated_replaced_slow_tokens: prior?.median_tokens ?? null,
      type,
    };
    (reason ? rejected : eligible).push(decision);
  }
  return { eligible, rejected, state: selectorState, fastUpper };
}
