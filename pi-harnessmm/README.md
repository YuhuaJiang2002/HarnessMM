# pi-harnessmm

HarnessMM ([arXiv 2610.32813](https://arxiv.org/abs/2610.32813)) as a [pi](https://github.com/earendil-works/pi) extension: a port of
[YuhuaJiang2002/HarnessMM](https://github.com/YuhuaJiang2002/HarnessMM) that removes repeated reasoning calls by executing
decisions the slow actor already resolved.

A slow turn authors an Executable Decision Graph in `<CHISEL_PLAN>`, then makes one root `terminal` call. Each
successor whose guard holds on the **immediately preceding native receipt** executes through the same terminal
interface, and every execution is certified as one bypassed reasoning callback. Control returns to the model on
exhaustion, unreadable evidence, a native error, or selector abstention.

```text
model turn N   ── <CHISEL_PLAN> A→B→C  +  root command
                ├─ root executes ──────────────────────────► receipt
                ├─ A: guard(return_code==0) holds ─────────► receipt   (no model call)
                ├─ B: guard(stdout_contains="PASS") holds ─► receipt   (no model call)
                └─ C: bank exhausted ──────────────────────► slow actor resumes
```

## Install

```bash
pi install git:github.com/jlon/pi-harnessmm
```

Or try it without adding it to settings:

```bash
pi -e ./pi-harnessmm
```

The extension registers the `terminal` tool. It replaces `bash` for work that benefits from a bank: one tool, one
receipt shape `{return_code, stdout, stderr}`, and no merged streams.

## The authoring contract

The instruction appended to the system prompt is the reference text for the configured policy version, byte for
byte. A bank is one `<CHISEL_PLAN>` block in the assistant text of the turn that makes the root call:

```text
<CHISEL_PLAN>{"candidates":[
  {"candidate_id":"A","predecessor_id":"root","tool":"terminal",
   "arguments":{"command":"npm run build"},
   "guard":{"return_code":0},"required_receipt_fields":["return_code"],
   "expected_local_cost":{"selector_upper_tokens":8192},"dependency_ids":[]},
  {"candidate_id":"B","predecessor_id":"A","tool":"terminal",
   "arguments":{"command":"npm test"},
   "guard":{"return_code":0,"stdout_contains":"PASS"},"required_receipt_fields":["return_code","stdout"],
   "expected_local_cost":{"selector_upper_tokens":8192},"dependency_ids":[]}
]}</CHISEL_PLAN>
```

Rules the parser enforces, exactly as upstream:

- labels are unique `A`..`P`; `predecessor_id` is `root` or an earlier label; `dependency_ids` must already be declared
- `guard` keys are limited to `return_code`, `stdout_contains`, `stdout_equals`, `stderr_empty`, `json_equals`
- every guard field must be listed in `required_receipt_fields`, so a guard never reads a field the receipt may omit
- `arguments` must match the terminal tool schema, be fully literal, and never repeat the root command
- exactly one bank and exactly one root tool call per turn

A bank argument with `${...}` or `{{...}}` is a contract violation: unresolved arguments must wait for a later turn.

## Admission

Two paths, as the paper reports them:

| Policy versions | Admission | Consequence |
| --- | --- | --- |
| 1–5, 7 | `deterministicAdmission`: net saving must beat the 1024-token threshold | selector cost is charged as `max(selector_upper_tokens, bytes(state))` |
| 6 | same as above, plus a single eligible candidate commits deterministically | no selector request at all |
| 8, 10 | priors-based `admit`: a frozen TRAIN category median must exist, and the fast path must stay under a quarter of it | **fails closed** without `HARNESSMM_TRAIN_PRIORS` |

Configuration:

```bash
# Frozen policy; validated before use, invalid values stop the extension.
export HARNESSMM_POLICY='{"version":6,"enabled":true,"maximum_candidates":16,"maximum_fast_steps":8,"maximum_public_bytes":65536}'

# Required by policy versions 8 and 10: {phase:"train", categories:{inspect:{phase:"train",median_tokens:12000}, ...}}
export HARNESSMM_TRAIN_PRIORS=/path/to/priors.json
```

## Certificates

Every substitution records `certified-native-substitution/v1`:

```json
{
  "schema": "certified-native-substitution/v1",
  "bank_sha256": "...", "candidate_id": "A",
  "bypassed_slow_callback_id": "call_abc", "predecessor_callback_id": "call_xyz",
  "exact_native_tool": "terminal", "exact_arguments": {"command": "npm test"},
  "native_receipt_hash": "...", "native_receipt": {"return_code": 0, "stdout": "...", "stderr": null},
  "guard": {"return_code": 0}, "native_action_hash": "...",
  "slow_requests_at_boundary": 0, "valid_substitution": true
}
```

The certificate carries the paper's six fields: `graph_hash` (the authored graph), `decision_id` (its node),
`predecessor_receipt`, `exact_arguments`, `native_receipt_hash` and `bypassed_slow_callback_id`, which names the slow
callback that did **not** happen (`<authoring-call>:omitted:<n>`), plus the audited counts
`slow_requests_at_boundary` / `slow_sequence_before` / `slow_sequence_after`. It binds the exact arguments, the exact
receipt and the absence of an intervening slow request; it never claims anything about task correctness or
counterfactual tokens. `/harnessmm` reports the ledger for the session:
banks authored, banks removed from the slow context, certified substitutions, deferrals and selector abstentions.

## Fidelity to the reference implementation

Ported from `integrations/deepseek_harness/{cot_jev,sar_v2,sar_admission}.mjs`:

| Mechanism | Where | Verified by |
| --- | --- | --- |
| Authoring instruction, selector prompts, policy validation | `src/policy.ts` | byte-identical to the reference export for all 11 versions (`tests/policy-parity.test.ts`) |
| `<CHISEL_PLAN>` bank contract, `publicReceipt` decoding | `src/bank.ts` | upstream bank-contract cases (`tests/bank.test.ts`) |
| `localGuard`, `receiptGuard` | `src/guard.ts` | 115 differential comparisons against the reference (`scripts/parity-check.mjs`) |
| `deterministicAdmission`, `admit`, `actionType`, `compactState` | `src/admission.ts` | differential comparisons + admission cases (`tests/admission.test.ts`) |
| `ExecutableSubdag` decisions, events, certificates | `src/controller.ts` | certificate, boundary, oversized-state and unknown-guard cases (`tests/admission.test.ts`) |
| Bank removal from the slow context (`compactSession`) | `src/pi-messages.ts` | projection preserves reasoning and native actions |

Reproduce:

```bash
npm test                                   # 30 contract tests
node scripts/parity-check.mjs ../HarnessMM  # differential parity, 0 disagreements
```

## Deliberate differences

pi is not a DSH/Chat-completions harness, so three things cannot be ported 1:1:

1. **No stream interception.** The reference replaces the next model stream with a compiled tool call. pi has no
   hook that emits an assistant tool call, so the controller runs *inside* the root `terminal` call: the root and every
   admitted successor execute there and their receipts are returned together. Same decisions, same certificates, same
   economics (N dependent steps per model turn), one fewer transcript entry per substitution. Each substitute is
   dispatched back through the registered `terminal` tool (`ctx.executeTool`), so pi's `tool_call`/`tool_result` hooks
   and every other extension's permission checks see it, and pi records it as a nested call on the result.
2. **No logprobs.** The reference selector requires a one-token reply with a native logprob. `pi` exposes no
   logprobs, so the selector asks the session model for one label, accepts only a label it offered, and treats
   everything else as abstention.
3. **No RSI loop.** Recursive self-improvement is an offline training pipeline over datasets and graders
   (`phiagent/harness/industrial_*`), not a runtime mechanism. This package implements the runtime adapter only;
   policy versions 8 and 10 consume its output through `HARNESSMM_TRAIN_PRIORS`.

Also not ported: the v5 `PersistentBank` (bank retention across turns), streaming chunk emission, and the
DeepSeek/Codex provider plumbing. `receiptGuard` is ported for v5 condition strings.

## Safety

The `terminal` tool runs shell commands with the permissions of the pi process and captures stdout and stderr in
separate pipes. The controller preserves the tool's permissions and action space; it never widens them. Guards are
data, never code, and a guard that cannot be read defers to the model instead of guessing. Priors-based admission
refuses high-risk commands (`rm -rf`, `sudo`, `docker`, `curl | sh`) and anything under tests/solution paths.

Hardening beyond the reference adapter:

- **Environment is scrubbed.** Variables whose name matches `key|password|secret|token|credential|passwd` are removed
  before a command runs, so a bank command cannot echo a credential into the transcript. The reference strips the same
  names for its worker shells.
- **Default timeout.** A command without `timeout_sec` is killed after 120 s (exit 124) instead of holding the turn.
- **Two views of one receipt.** Guards evaluate the exact receipt; the model sees at most 8 KiB of each stream, with
  the tail kept because that is what a guard or a person usually reads.
- **Prompt section, not prompt replacement.** The authoring instruction is injected as a prompt section, so pi keeps
  patching only changed sections instead of replacing the whole system prompt (and other extensions keep their own
  sections intact).

## License

Apache-2.0, matching the upstream project. HarnessMM is research code by Yuhua Jiang, Chang Xu, Feifei Gao and
Li Zhao; this port keeps its contracts and cites it in `NOTICE`.
