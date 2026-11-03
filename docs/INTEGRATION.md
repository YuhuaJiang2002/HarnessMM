# Native harness integration

The controller sits at a model-to-tool boundary. It does not supply an
independent benchmark loop or replace the native scorer.

The [integration index](../integrations/README.md) lists the three harness
entry points. Import `VisualBaseline` from `integrations.baseline_harness`
for a fresh screenshot decision, or import `NativeBankBoundary`,
`TerminalBankProvider` and `CodexChiselController` from
`integrations.codex_harness` for the existing Codex implementations.
Use `harnessmm.harness` for direct implementation imports. Legacy
`phiagent.harness` imports remain supported; see [package migration](PACKAGE_MIGRATION.md).

1. Pass the actual public task, tool catalog and observed native receipts.
2. Let a slow actor author literal tool arguments and dependencies using the
   bank contract for the chosen version.
3. At the next registered slow boundary, ask the controller for an action.
4. Dispatch the returned tool call through the original native executor.
5. Call `observeTool` with that call's identifier, exact tool name, arguments
   and native result. Keep the certificate and the physical request ledger.
6. Resume the original slow callback on abstention, unsupported evidence,
   execution error or graph exhaustion.

`examples/guarded_reuse.mjs` gives a complete in-memory integration of this
sequence with the executable sub-DAG contract. The associated tests are
`tests/sar_subdag.test.mjs`, `tests/sar_v2.test.mjs` and
`tests/deepseek_cot_jev.test.mjs`.

## JavaScript interface

```javascript
import {ExecutableSubdag} from './integrations/deepseek_harness/sar_v2.mjs';

const controller = new ExecutableSubdag({
  policy: frozenPolicy,
  model: frozenModel,
  record: event => durableAudit(event),
  request: (...args) => meteredProviderRequest(...args),
});

// At a registered native reasoning boundary:
for await (const chunk of controller.stream(options, originalSlowStream, callbackId)) {
  await nativeConsume(chunk);
}

// After the native dispatcher executes a returned tool call:
controller.observeTool({call_id: call.id, name: call.name,
  arguments: JSON.parse(call.arguments), result: {observation: nativeReceipt}});
```

The host owns provider usage accounting and durable settlement. A deterministic
single eligible action can bypass an additional selector call in the declared
protocol. Ambiguous choices use the configured selector and its complete cost.
Preserve native user constraints, action schemas, deadlines and completion
semantics across both routes.

## Python interfaces

`harnessmm.harness.codex_native_app_bank.NativeBankBoundary` accepts callbacks
for `judge`, `execute`, `generation`, `finished` and `audit`. Register the slow
actor response with `register`, then invoke the original native tool via
`call`. See `tests/test_codex_native_app_bank.py` for callback shapes.

`harnessmm.harness.codex_terminal_bank.TerminalBankProvider` integrates with the
native Responses terminal boundary. Its tests cover request attribution,
complete usage and fallback behavior. Both Python bank adapters invoke the
packaged JavaScript contract with Node.js.

For RSI, `CodexChiselController` and `DeepSeekChiselController` accept real
executor and optimizer callbacks. Their module docstrings define the complete
callback schemas and source-freeze requirements. Synthetic callback tests
demonstrate the lifecycle without provider access.

## DeepSeek shell schemas and task completion

For prospective v10/11 shell integrations, call `structuredAuthorTool(tool)`
**when the host registers the tool**, before constructing the provider request.
It supports `terminal`, `bash` and `pwsh` object schemas, preserves required
native fields (including `description`), and adds optional `chisel_plan`
metadata. A stream listener does not change the schema already sent to the
model. `structuredAuthorStream` removes authored metadata before native
dispatch; absent supported shells take the ordinary slow path. Its audit
records whether the actual registered schema contains the field.

Use `planInstruction(policy, actualTools)` to bind shell examples and required
argument names to the host catalog. The one-argument form retains the historical
prompt. These library checks do not certify a stock-dsh plugin integration;
see [the compatibility notes](DEEPSEEK_REPRODUCTION.md).

The historical `run_with_completion_recovery` handles reasoning-only empty
finishes. For execution tasks, the optional
`run_with_task_completion_recovery(session, prompt, task_complete)` also handles
a visible final plan when a host-owned **public** completion check returns
`False`. For example, the host can check requested output files and recorded
native actions. `True` accepts completion; `None` leaves an unknown state alone.
Do not inspect protected benchmark tests or answers in this check.

The helper requests at most two additional native turns in the same session,
respects native stops, and never executes plan text or invents tool calls.
Charge every recovery call and keep the existing episode deadline. This helper
requires explicit host wiring; installing the package does not enable it.

## Experimental boundaries

Training can expose task traces to the optimizer. Validation selects among
training-derived candidates and returns selection statistics; it must not feed
held-out traces back into optimization. Testing begins after the selected
method, model identity and sources are frozen. Repetitions of a task remain in
the same partition.

Receipt guards establish local eligibility for a stored action. Native task
quality and complete measured costs determine whether the integrated harness
is beneficial. A certificate records a callback replacement, rather than an
estimate of an entire counterfactual trajectory's token consumption.
