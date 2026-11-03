# Codex harness

This package exposes the Codex integration already implemented under
`harnessmm.harness`. Its exports are the original classes and functions, so
existing imports and protocol versions keep their identities. New deployments
freeze the current source hashes; see [package migration](../../docs/PACKAGE_MIGRATION.md).

| Interface | Implementation | Use |
| --- | --- | --- |
| `codex_config`, `run_codex_trial` | [`codex_runtime.py`](../../harnessmm/harness/codex_runtime.py) | Pinned official CLI v0.142.5, isolated workspace and provider metering |
| `NativeBankBoundary` | [`codex_native_app_bank.py`](../../harnessmm/harness/codex_native_app_bank.py) | Guarded literal actions at the native MCP boundary |
| `TerminalBankProvider` | [`codex_terminal_bank.py`](../../harnessmm/harness/codex_terminal_bank.py) | Native Responses terminal history, action banks and metered selection |
| `ChiselConfig`, `CodexChiselController` | [`codex_chisel.py`](../../harnessmm/harness/codex_chisel.py) | Training, protected selection, source freeze and held-out evaluation |

```python
from integrations.codex_harness import NativeBankBoundary

# judge must meter every physical selector request. execute is the original
# native dispatcher; generation and finished come from the native runtime.
boundary = NativeBankBoundary(
    tools=native_tool_catalog,
    enabled=True,
    judge=metered_judge,
    execute=native_execute,
    generation=native_generation_context,
    finished=episode_finished,
    audit=durable_audit,
)
boundary.register(slow_response, public_task=[original_public_task])
result = boundary.call(native_tool_name, original_arguments)
```

The complete callback fixtures are in
[`test_codex_native_app_bank.py`](../../tests/test_codex_native_app_bank.py).
The original harness uses its ordinary dispatcher. Setting `enabled=False`
on this boundary defines the bank-authorship control: it retains the authored
bank and executes only the original root. That control has authoring overhead
and is distinct from the original harness without bank instructions.

## Research runners

- [`run_codex_terminal.py`](../../scripts/run_codex_terminal.py): original CLI terminal evaluation.
- [`run_codex_chisel_terminal.py`](../../scripts/run_codex_chisel_terminal.py): companion learning and native terminal execution.
- [`run_codex_native_app_bank_campaign.py`](../../scripts/run_codex_native_app_bank_campaign.py): native MCP bank campaign.
- [`run_codex_terminal_bank_campaign.py`](../../scripts/run_codex_terminal_bank_campaign.py): terminal bank campaign.
- [`run_codex_gaia2.py`](../../scripts/run_codex_gaia2.py): GAIA2 native adapter.

These runners require the upstream CLI, benchmark runtime, a metered provider
and a newly frozen deployment configuration. Their remote admission checks
remain active. The historical terminal configuration pins DeepSeek Flash;
the visual comparison uses a separately configured Qwen actor. Importing this
package does not change either model or enable a plugin in the desktop app.

The visual CLI/MCP deployment and benchmark-specific screenshot geometry need
their original experiment recipe. The exports here do not by themselves
reproduce the MiniWoB++ or VisualWebArena scores. See
[visual settings](../../docs/VISUAL_COMPARISON.md),
[integration](../../docs/INTEGRATION.md) and
[reproduction](../../docs/REPRODUCIBILITY.md).
