# Paper concepts and implementation

| Concept | Entry point | Role |
| --- | --- | --- |
| EDG authoring and validation | `integrations/deepseek_harness/cot_jev.mjs` | `planInstruction`, `parseBank`, structured authoring and typed bank contracts |
| GDR | `integrations/deepseek_harness/sar_v2.mjs` | `PersistentBank`, `localGuard` and `ExecutableSubdag` retain and activate prior decisions |
| CCS | `ExecutableSubdag.stream` and `ExecutableSubdag.observeTool` | Replace a registered slow boundary, then bind the exact action and native receipt in a certificate |
| Native action selection | `cot_jev.mjs`, `jev_v2.mjs`, `sar_admission.mjs` | Versioned deterministic and model-assisted selection with abstention |
| Codex native MCP integration | `phiagent/harness/codex_native_app_bank.py` | `NativeBankBoundary` with native execution, audit and metered selector callbacks |
| Codex terminal integration | `phiagent/harness/codex_terminal_bank.py` | Native Responses history, terminal bank boundary and metered provider integration |
| Shared RSI lifecycle | `phiagent/harness/codex_chisel.py` | Training, protected validation selection, source freeze and sealed testing |
| DeepSeek RSI adapter | `phiagent/harness/deepseek_chisel.py` | Preserve the shared lifecycle while adapting the official SDK request fields |
| Cost-aware RSI | `phiagent/harness/industrial_efficiency_rsi.py` | Candidate objective and complete quality/cost selection |
| Train-only industrial RSI | `phiagent/harness/industrial_train_rsi.py` | Procedural instruction optimization and cross-fitted routing |
| Grouped partition | `phiagent/harness/benchmark_partition.py` | Outcome-independent 20%/20%/60% split before optimization and repetitions |
| Complete token accounting | `phiagent/evaluation/deployment_token_accounting.py` | Aggregate declared physical calls without dropping unsuccessful executions |

Instruction optimization, context handling and finalization rules are companion
configurations. Enabling one does not imply that EDG execution or callback
substitution occurred. Use actual native substitution certificates to identify
that mechanism in a run.

## Naming

The public project name is HarnessMM. The extraction keeps the original
`phiagent` import namespace, `ChiselConfig`/`CodexChiselController` class names,
schema strings, event names and `<CHISEL_PLAN>` envelope. `SAR` and `Jev` occur
in implementation names for graph execution and selection. No scientific
protocol is renamed by changing the repository title.

The source tree contains successive research implementations. A benchmark run
must declare its exact modules, protocol version, model and configuration;
importing the whole package does not select a single deployment policy.
