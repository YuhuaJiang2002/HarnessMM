/**
 * Substitution controller: Guarded Decision Reuse + Certified Callback Substitution.
 *
 * Port of HarnessMM integrations/deepseek_harness/sar_v2.mjs (ExecutableSubdag)
 * with the same admission, the same certificate schema and the same event kinds.
 * The controller is host-neutral: it decides, the host executes, and it records.
 */
import { admit, compactState, deterministicAdmission, SELECTOR_SYSTEM, type AdmissionCandidate, type TrainPriors } from "./admission.ts";
import { byteUpper, hash, type Policy } from "./policy.ts";
import { nativeCallId, type Bank, type BankState, type Candidate } from "./bank.ts";
export type { BankState };

/** One ledger event. Kinds match the upstream controller so ledgers are comparable. */
export interface ControllerEvent {
  kind: string;
  [key: string]: unknown;
}

/** The selector's answer: one candidate label, or `?` to return to slow reasoning. */
export interface SelectorResult {
  label: string;
  usage?: { inputTokens: number; outputTokens: number; totalTokens: number };
}

export type Decision =
  | {
      kind: "commit";
      candidate: Candidate;
      call: { type: "tool-call"; id: string; name: string; arguments: string };
      deterministic: boolean;
      admission: AdmissionResultKind;
    }
  | { kind: "slow"; reason: string };

type AdmissionResultKind = ReturnType<typeof deterministicAdmission> | ReturnType<typeof admit>;

export interface ControllerOptions {
  policy: Policy;
  record: (event: ControllerEvent) => void;
  /** Frozen TRAIN priors; required for policy versions 8 and 10. */
  priors?: TrainPriors | null;
  /** Selector used when more than one candidate is eligible. */
  select?: (input: { system: string; state: unknown; labels: string[] }) => Promise<SelectorResult>;
  episodeId?: string;
}

/**
 * The controller owns admission, commitment and certification. It never
 * executes: the host dispatches the returned call through the native interface
 * and reports the receipt through {@link observeTool}.
 */
export class SubstitutionController {
  private readonly policy: Policy;
  private readonly record: (event: ControllerEvent) => void;
  private readonly priors: TrainPriors | null;
  private readonly select?: ControllerOptions["select"];
  private readonly episodeId: string;
  private readonly pending = new Map<string, { call: { name: string; arguments: string }; candidate: any; bank_sha256: string; boundary: any; slow_snapshot: number; actor_sequence: unknown }>();
  private readonly done = new Set<string>();
  private readonly slowSequence: string[] = [];
  private readonly actorSequence: Array<{ kind: "slow" | "fast"; id: string }> = [];
  private boundary: { bypassed_slow_callback_id: string; predecessor_callback_id?: string } | null = null;
  private authoringCallbackId = "unbound";
  private substitutions = 0;
  private authors = 0;
  slowTokens = 0;

  constructor(options: ControllerOptions) {
    this.policy = options.policy;
    this.record = event => {
      if (event.kind === "jev/request") this.actorSequence.push({ kind: "fast", id: String(event.decision_id) });
      if (event.kind === "jev/compiled-native-action") {
        const boundary = this.boundary;
        const call = event.call as { id: string };
        if (!boundary || this.pending.has(call.id) || this.done.has(call.id)) {
          throw new Error("Invalid substitution boundary");
        }
        this.pending.set(call.id, { ...(event as any), boundary, slow_snapshot: this.slowSequence.length, actor_sequence: structuredClone(this.actorSequence) });
      }
      options.record(event);
    };
    this.priors = options.priors ?? null;
    this.select = options.select;
    this.episodeId = options.episodeId ?? "native-task";
  }

  /** A paid slow call happened: the authoring turn of `bank`. */
  ownSlowCall(attemptId: string, tokens = 0): void {
    this.slowSequence.push(attemptId);
    this.actorSequence.push({ kind: "slow", id: attemptId });
    this.record({ kind: "sar/slow-call", native_attempt_id: attemptId });
    if (tokens > 0) this.slowTokens = tokens;
  }

  /** Record the bank the slow turn authored. */
  noteBank(bank: Bank, authorCallbackId: string): void {
    this.authors += 1;
    this.record({
      kind: "sar/bank-authored",
      bank_sha256: bank.sha256,
      root_id: bank.root.id,
      author_callback_id: authorCallbackId,
      candidates: bank.candidates.length,
      bank_author_call_index: this.authors,
    });
  }

  /**
   * Register the native boundary the next substitution bypasses: `attemptId` is the
   * authoring call, and each substitution derives its own omitted-callback identity
   * from it, so a certificate names the callback that did not happen.
   */
  beginBoundary(authoringCallbackId: string, predecessorCallbackId?: string): void {
    this.boundary = { bypassed_slow_callback_id: authoringCallbackId, predecessor_callback_id: predecessorCallbackId };
    this.authoringCallbackId = authoringCallbackId;
    this.substitutions = 0;
  }

  /**
   * Name the call whose receipt authorizes the next decision. The certificate binds
   * it as `predecessor_callback_id`, so a substitution always points at the exact
   * callback that produced its evidence.
   */
  precede(callbackId: string): void {
    if (this.boundary) this.boundary.predecessor_callback_id = callbackId;
  }

  /**
   * Decide the next action from the current bank state.
   * Admission runs first; only admitted candidates reach the selector, and the
   * selector may only pick a label it was given.
   */
  async decide(bank: Bank, state: BankState): Promise<Decision> {
    // Upstream refuses an oversized public state instead of paying for a selector over it.
    const bounded = this.policy.version >= 8 ? state.candidates : state;
    if (byteUpper(bounded) > this.policy.maximum_public_bytes) {
      this.record({ kind: "sar/defer", bank_sha256: bank.sha256, reason: "public_state_too_large", bytes: byteUpper(bounded), limit: this.policy.maximum_public_bytes });
      return { kind: "slow", reason: "public_state_too_large" };
    }
    const fullSAR = this.policy.version === 6;
    const priorsPath = this.policy.version === 8 || this.policy.version === 10;
    const admission: AdmissionResultKind = priorsPath
      ? admit(state, this.priors, this.episodeId, fullSAR)
      : deterministicAdmission(state, this.slowTokens, fullSAR);
    const decisionId = hash([bank.sha256, state.public_results.length, Date.now()]);
    this.record(
      priorsPath
        ? {
            kind: "sar/pre-admission",
            bank_sha256: bank.sha256,
            estimated_fast_total_tokens: (admission as ReturnType<typeof admit>).fastUpper,
            eligible: (admission as ReturnType<typeof admit>).eligible.map(item => ({ id: item.candidate.label, reason: item.reason, type: item.type })),
            rejected: (admission as ReturnType<typeof admit>).rejected.map(item => ({ id: item.candidate.label, reason: item.reason })),
            estimate_identity: "frozen same-parent TRAIN category median",
            threshold: "fast <= 0.25 * replaced slow",
            full_admission: fullSAR,
          }
        : {
            kind: "sar/pre-admission",
            bank_sha256: bank.sha256,
            eligible: (admission as ReturnType<typeof deterministicAdmission>).eligible.map(item => ({ id: item.candidate.label, estimated_net_saving: item.estimated_net_saving })),
            rejected: (admission as ReturnType<typeof deterministicAdmission>).rejected.map(item => ({ id: item.candidate.label, reason: item.reason })),
            threshold_tokens: (admission as ReturnType<typeof deterministicAdmission>).threshold_tokens,
            estimate_identity: (admission as ReturnType<typeof deterministicAdmission>).identity,
          },
    );
    if (admission.eligible.length === 0) {
      // Upstream names the two cases apart: an unreadable guard requests new CoT even
      // when another branch looked eligible, and an empty eligible set defers.
      const unknown = [...(admission as ReturnType<typeof deterministicAdmission>).rejected].some(item => item.reason === "guard_unknown");
      const reason = unknown ? "guard_unknown" : "no_eligible_or_unexpected_receipt";
      this.record({ kind: "sar/defer", bank_sha256: bank.sha256, reason });
      return { kind: "slow", reason };
    }

    const candidates = admission.eligible.map(item => item.candidate as unknown as AdmissionCandidate);
    const labels = candidates.map(candidate => candidate.label);
    let label: string;
    let deterministic = false;
    if (this.policy.version === 6 && candidates.length === 1) {
      label = labels[0];
      deterministic = true;
      this.record({ kind: "sar/deterministic-commit", decision_id: decisionId, native_attempt_id: this.boundary?.bypassed_slow_callback_id, bank_sha256: bank.sha256 });
    } else {
      const selectorState = priorsPath
        ? (admission as ReturnType<typeof admit>).state
        : compactState(state, candidates, this.episodeId);
      if (!this.select) return { kind: "slow", reason: "selector_unavailable" };
      this.record({ kind: "jev/request", decision_id: decisionId, native_attempt_id: this.boundary?.bypassed_slow_callback_id, bank_sha256: bank.sha256, step: state.public_results.length, request: { system: SELECTOR_SYSTEM, state: selectorState } });
      const answer = await this.select({ system: SELECTOR_SYSTEM, state: selectorState, labels });
      this.record({ kind: "jev/decision", decision_id: decisionId, ...answer });
      if (!labels.includes(answer.label)) {
        this.record({ kind: "jev/slow-fallback", decision_id: decisionId });
        return { kind: "slow", reason: "selector_abstained" };
      }
      label = answer.label;
    }

    const candidate = candidates.find(item => item.label === label)! as unknown as Candidate;
    const call = { type: "tool-call" as const, id: nativeCallId(bank, candidate.label!), name: candidate.tool, arguments: JSON.stringify(candidate.arguments) };
    this.record({ kind: "jev/compiled-native-action", decision_id: decisionId, bank_sha256: bank.sha256, candidate, call, native_execution_claimed: false });
    return { kind: "commit", candidate, call, deterministic, admission };
  }

  /**
   * Bind the native receipt to the certificate. Throws when the receipt does not
   * match the admitted action or when a slow request intervened.
   */
  observeTool(event: { call_id: string; name: string; arguments: unknown; result: { observation: string } }): void {
    const item = this.pending.get(event.call_id);
    if (!item) return;
    if (
      event.name !== item.call.name ||
      hash(event.arguments) !== hash(JSON.parse(item.call.arguments)) ||
      !Object.hasOwn(event.result ?? {}, "observation") ||
      this.slowSequence.length !== item.slow_snapshot
    ) {
      throw new Error("Substitution lacks exact native receipt or has intervening slow request");
    }
    this.pending.delete(event.call_id);
    this.done.add(event.call_id);
    this.substitutions += 1;
    let observation: any = event.result.observation;
    try {
      if (typeof observation === "string") observation = JSON.parse(observation);
    } catch {
      /* keep the raw text */
    }
    if (observation?.executed === false) {
      this.record({ kind: "sar/uncertified-execution", call_id: event.call_id, reason: "shared_parent_suppressed_action", certified_substitution: 0, executed_native_receipt: event.result });
      return;
    }
    this.record({
      kind: "sar/substitution-certificate",
      schema: "certified-native-substitution/v1",
      bank_hash: item.bank_sha256,
      bank_sha256: item.bank_sha256,
      candidate_id: item.candidate.label,
      decision_id: `${item.bank_sha256.slice(0, 12)}:${item.candidate.label}`,
      // The callback this execution replaced: the slow request that would have
      // produced it, identified per boundary rather than by the authoring call.
      bypassed_slow_callback_id: `${item.boundary.bypassed_slow_callback_id}:omitted:${this.substitutions}`,
      predecessor_callback_id: item.boundary.predecessor_callback_id,
      graph_hash: item.bank_sha256,
      exact_native_tool: event.name,
      exact_arguments: event.arguments,
      native_receipt_hash: hash(event.result),
      native_result_sha256: hash(event.result),
      call_id: event.call_id,
      native_receipt: event.result,
      guard: item.candidate.node?.guard,
      native_action_hash: hash([event.name, event.arguments]),
      actor_request_sequence: item.actor_sequence,
      slow_requests_at_boundary: this.slowSequence.length - item.slow_snapshot,
      slow_sequence_before: item.slow_snapshot,
      slow_sequence_after: this.slowSequence.length,
      valid_substitution: true,
      counterfactual_identity: "unchanged native SDK llm/stream boundary; no claim of paired-parent trajectory equality",
    });
  }
}
