# Harness integrations

| Harness | Public entry point | Scope |
| --- | --- | --- |
| Baseline harness | [`baseline_harness`](baseline_harness/README.md) | Fresh screenshot-based model decision at every native boundary |
| Codex Harness | [`codex_harness`](codex_harness/README.md) | Official CLI runtime, native MCP/terminal bank adapters and RSI lifecycle |
| DeepSeek Harness | [`deepseek_harness`](deepseek_harness/) | SDK streaming bridge, graph authoring, guards and native execution |

Codex exports refer to the existing `phiagent.harness` implementations.
Baseline action-contract definitions are extracted from the visual research
workspace. Both packages leave benchmark execution and physical usage
accounting with the host. Their READMEs distinguish integration code from
benchmark-specific reproduction recipes.

Start with the [integration guide](../docs/INTEGRATION.md). For the DeepSeek
SDK and archived HumanEval configuration, use
[DEEPSEEK_REPRODUCTION.md](../docs/DEEPSEEK_REPRODUCTION.md).
The [baseline index](baseline_harness/README.md#other-baseline-implementations)
locates other released comparison methods.
