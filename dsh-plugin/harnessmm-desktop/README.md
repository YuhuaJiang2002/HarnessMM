# dsh-harnessmm

[English](./README.md) · [中文](./README.zh-CN.md)

HarnessMM for the DeepSeek Harness desktop app: guarded decision reuse over the
native tool surface.

The plugin brings the CHISEL / SAR protocol from
[`integrations/deepseek_harness/`](../../integrations/deepseek_harness/) into a
stock DSH desktop profile. It sits on the model-to-tool boundary: it never owns a
benchmark loop, never replaces the native executor and never scores a result.

## What it does

1. **Author once.** The slow actor is asked (through an injected system-prompt
   section) to declare a `<CHISEL_PLAN>` bank alongside its one root call. Each
   candidate is a fully literal native action plus a *receipt-local* guard, e.g.
   `{"exit_code": 0}`.
2. **Gate on the receipt.** After the root result arrives, `projectTurn` rebuilds
   the live decision state from the request messages and every candidate guard is
   evaluated against the public receipt only.
3. **Substitute the callback.** When a candidate survives and the net saving
   exceeds the admission threshold, the plugin emits that native tool call
   directly instead of asking the model for the next step. A one-character
   selector picks among several surviving candidates; with a single survivor the
   decision
   is deterministic and costs nothing.
4. **Certify.** Every substitution is written as a receipt that binds the bank
   hash, the candidate, the bypassed callback and the exact native result. The
   graph exhausts into the ordinary slow path.

A substituted turn is a turn no model authored, so it carries a short synthetic
`reasoning` block beside its tool call. Thinking-mode endpoints reject an
assistant turn whose tool call has no sibling thinking block — DeepSeek's
Anthropic-compatible endpoint answers `400 The content[].thinking in the
thinking mode must be passed back to the API` — and it accepted an unsigned
block for a substituted turn in a live session, so the note is accepted and
replayed verbatim in later history. The note states what happened and never
quotes a plan tag, so it cannot be mistaken for an authored bank.

`tools/execute` re-checks every `chisel_*` call against the authored literal
action, so a mutated argument cannot reach the native executor even if it appears
in the stream.

## Why this is not the paper configuration

The reference implementation talks to an SDK *terminal bridge*. A desktop profile
has no such bridge: its shell surface is the stock `bash` / `pwsh` tools, and the
only durable control points are `llm/stream`, `tools/execute`, `tools/result` and
the system prompt. This plugin therefore reproduces the protocol, not the
benchmark harness:

- guards are receipt-local (`ok`, `exit_code`, `stdout_contains`,
  `stdout_equals`, `stderr_empty`, `json_equals`), never semantic;
- the admission threshold (paper default `net > 1024` tokens) is configurable as
  `policy.admission_threshold_tokens`, and the token count comes from the host's
  own usage reporting;
- a selector call is capped at `FAST_CALL_UPPER_TOKENS` (512) tokens rather than
  at the author's own re-decision estimate, so a pessimistic `expected_local_cost`
  cannot veto every step;
- receipts are `harnessmm-desktop-certificate/v1` and claim only one bypassed
  callback inside the desktop loop — no episode-level counterfactual.

Do not compare these receipts against historical HumanEval run identities. The
repository's own README warns about exactly that.

## Configuration

The shipped row turns the controller on (`policy.enabled: true`). Set it to
`false` to fall back to the plain stock loop while keeping the row installed.
Edit the row in the profile patch layer:

```yaml
- insert:
    - id: harnessmm
      name: 'dsh-harnessmm'
      config:
        policy:
          enabled: true
          maximum_candidates: 16      # A-P labels, 1..16
          maximum_fast_steps: 8       # substitutions before the graph exhausts
          maximum_public_bytes: 65536 # selector payload ceiling
          admission_threshold_tokens: 1024 # paper default; see below
        selector:
          enabled: true
          provider: ''                # '' = the boundary request's own route
          model: ''                   # '' = the boundary request's own model
          maxTokens: 1024             # reasoning models think before the label
          timeoutMs: 20000
        shellTools: []                # [] = auto-detect bash/pwsh from the catalog
        planSection: true             # inject the CHISEL_PLAN authoring section
        receiptsFile: ''              # '' = <DSH_HOME or ~/.dsh>/harnessmm/receipts.jsonl
```

An empty `selector.provider` / `selector.model` reuses the route the boundary
request itself rides — the one route this host has just proven answerable. That
default matters on gateways that only serve thinking requests: a call carrying no
reasoning control at all is refused before the model ever reads the receipt
(`400 ... "The value of the enable_thinking parameter is restricted to True."`).
The selector therefore reuses the boundary request's own reasoning level (falling
back to the model's greatest declared effort when the boundary carries none), so a
real reasoning control reaches the wire. Setting `selector.enabled: false` never
spends that call: a multi-candidate step goes straight back to the slow actor and
the receipt records `selector-abstain` with `reason: "selector-disabled"`.

The desktop host does not export `DSH_HOME`, so an empty `receiptsFile` falls back
to `~/.dsh/harnessmm/receipts.jsonl` rather than silently disabling receipts. A
`receiptsFile` that cannot be written never breaks a turn: durability is
best-effort by design.

### Admission threshold

A substitution is admitted only when the slow turn it replaces would have cost
more than the fast call that replaces it, by at least
`policy.admission_threshold_tokens`. Concretely, with `n` the slow turn's own
usage in tokens:

- one surviving candidate → committed deterministically, `n` is irrelevant;
- several surviving candidates → admitted when `n - selector upper bound > threshold`,
  where the selector upper bound is clamped to 512 tokens.

The paper's 1024 is the default. Raising it is the conservative direction; lowering
it accepts substitutions with a smaller expected saving. A single surviving
candidate is unaffected — the deterministic commit is what makes the controller
useful on steps that cost less than the threshold.

The public receipt carries only the objective of the turn being decided, not the
whole transcript. A long-lived session accumulates far more user text than
`maximum_public_bytes`; projecting all of it would veto every late-turn
substitution on byte size alone. History is also read leniently: a message that
merely quotes a `<CHISEL_PLAN>` fragment (a document, a bug report) is not a
declaration and never poisons later turns.

## Verified on a real desktop host

One live session, plan of a single candidate `A` whose root call had already
succeeded, produced this receipt sequence (abridged):

```json
{"type":"bank","candidates":1,"public_bytes":337}
{"type":"decision","kind":"commit","eligible":["A"],"rejected":[]}
{"type":"substitution-issued","call_id":"chisel_e6c66019cfa84ddd97a91ba8_A","tool":"pwsh","selector":"deterministic"}
{"type":"substitution","schema":"harnessmm-desktop-certificate/v1",
 "exact_arguments":"{\"description\":\"HMM15 substituted step\",\"command\":\"Write-Output 'HMM15-GUARD-MATCHED'\"}",
 "native_receipt":{"status":"PARSED","is_error":false,
   "value":{"tool":"pwsh","ok":true,"exit_code":0,"stdout":"HMM15-GUARD-MATCHED"}},
 "valid_substitution":true,"amount_substituted_slow_callbacks":1}
```

That plan's guard was `{"exit_code":0,"stdout_equals":"HMM15-ROOT-MARK"}` — it matched
the root call's output exactly, which only happens because the receipt is
canonicalised (`HMM15-ROOT-MARK\r\n` from the shell would never equal it). Two
neighbouring live turns bound the behaviour from both sides:

- a plan whose guard compared against the wrong string was correctly refused with
  `{"kind":"slow","reason":"no_eligible_candidate","rejected":[{"id":"A","reason":"guard_false"}]}`
  and the slow actor kept control;
- a plan whose command exits nonzero recorded `"exit_code":3` with
  `"stdout":"HMM13-FAIL-OUT"` — the renderer's own `[exit code: 3]` trailer is
  read for status and then dropped, so guards see program output only.

The native call ran through the ordinary `pwsh` tool and its output came back to
the loop; the next turn then projected to `no-candidates:position=A` and handed
control back to the slow actor. That is the intended shape: substitution executes,
the graph exhausts, judgement returns.

### What has actually been exercised

Every row below was observed in this desktop host, not only in the fake-host
tests:

| behaviour | evidence |
| --- | --- |
| deterministic commit on one surviving candidate | `decision {"kind":"commit","eligible":["A"]}` → `chisel_…_A`, no selector call |
| a wrong guard refuses cleanly | `{"kind":"slow","reason":"no_eligible_candidate","rejected":[{"id":"A","reason":"guard_false"}]}` |
| receipt fidelity for nonzero exits | `"exit_code":3`, `"stdout":"HMM13-FAIL-OUT"` (renderer trailer dropped) |
| multi-candidate step decided by the selector | `{"kind":"select","eligible":["A","B"]}` → `substitution-issued` with `"selector":"fast-selector"` and `"control_usage":{"inputTokens":912,"outputTokens":39,"totalTokens":951}` |
| chained substitution inside one turn | same `turn_id`: commit `A` (`predecessor_id":"root"`) then commit `B` (`predecessor_id":"A"`), 4 ms apart, both `deterministic` |
| two concurrent sessions stay separate | two sessions' roots released at a shared wall-clock barrier (`…T01:48:00Z`); their commits landed 74 ms apart and their certificates kept their own session id and marker |

The concurrency case, abridged from one receipts file:

```json
{"at":"…01:48:00.340Z","type":"decision","turn_id":"ac3fcd41-…:…","kind":"commit"}
{"at":"…01:48:00.341Z","type":"substitution-issued","turn_id":"ac3fcd41-…:…","candidate_id":"A"}
{"at":"…01:48:00.414Z","type":"decision","turn_id":"232cf615-…:…","kind":"commit"}
{"at":"…01:48:00.415Z","type":"substitution-issued","turn_id":"232cf615-…:…","candidate_id":"A"}
{"at":"…01:48:00.851Z","type":"substitution","turn_id":"ac3fcd41-…:…","native_receipt":{"value":{"stdout":"V3-MARK-SEVEN"}}}
{"at":"…01:48:00.930Z","type":"substitution","turn_id":"232cf615-…:…","native_receipt":{"value":{"stdout":"V3-MARK-EIGHT"}}}
```

Turn state is keyed by `sessionId`, so overlapping sessions never share a
pending bank, a step counter or a certificate.

## Live verification

Run the [desktop smoke test](test/live-session.md) in a fresh DSH session to
check the native substitution, its certificate and the shell side effect.
Use `harnessmm_status` inside a session to read its in-memory bank, substitution
and error counters.

## Install

`dsh plugin add` (or the Plugin Manager) links this directory into a profile:

```sh
dsh plugin --profile desktop add /path/to/dsh-plugin/harnessmm-desktop
```

A patch layer replaces a row's whole `config`; it never deep-merges. To change
one knob, restate the full `config` block shown above.

## Layout

| path | role |
| --- | --- |
| `lib/controller.mjs` | pure protocol: bank parsing, receipts, guards, admission, projection, dispatch check |
| `lib/index.js` | host plugin: `systemPrompt` section, `llm/stream`, `tools/execute`, `tools/result`, receipts |
| `cordis.patch.yml` | the bundle row over the profile root |
| `locale/{en,zh}.json` | settings labels for the config fields |
| `test/*.test.mjs` | protocol tests plus a fake-host test that drives `lib/index.js` |

## Receipts

One JSON object per line:

| type | meaning |
| --- | --- |
| `bank` | a slow turn declared a valid action bank |
| `decision` | the gate received a receipt; lists eligible and rejected candidates with reasons |
| `substitution-issued` | a native call was emitted in place of the slow callback |
| `substitution` | the certificate, bound to the exact native result |
| `boundary-error` | the stream controller failed and recorded the exception |
| `dispatch-rejected` | `tools/execute` refused a mutated or out-of-graph call |
| `selector-abstain` / `selector-unmapped` | the selector declined; the slow path resumed |

A receipt's `stdout` and `stderr` are canonicalised before a guard sees them:
CRLF becomes LF and one trailing newline is dropped, and the renderer's own
status trailer (`[exit code: N]`, `[output truncated; …]`) is removed after the
exit code has been decoded. Guards therefore compare program output, not shells.

## Tests

From the repository root, run the controller suite with Node.js 22 or later.

```sh
node --test dsh-plugin/harnessmm-desktop/test/controller.test.mjs
```

The simulated-host suite also loads the plugin entry point and needs
`@deepseek-ai/schemastery`. From the repository root, install the dependency
and run both suites.

```sh
npm install --no-save --package-lock=false @deepseek-ai/schemastery@3.18.4
node --test dsh-plugin/harnessmm-desktop/test/*.test.mjs
```

The controller suite checks the protocol. The simulated host drives the
plugin's stream, dispatch and result hooks, including deterministic
substitution, fallback and certificate generation. After dependency setup,
both suites run offline without a DSH session or a provider account.
