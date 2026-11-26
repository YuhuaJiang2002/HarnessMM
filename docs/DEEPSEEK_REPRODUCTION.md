# DeepSeek Harness: reference configuration and compatibility

## HumanEval results

The paper's HumanEval comparison reports 71.43% to 85.71% success and
96.28 to 11.62 Tokens/task (k), an 87.93% reduction on the same 98 test tasks.
The reduction covers test execution with structured authoring, guarded reuse,
history compaction and finalization. Offline development costs are separate.

| Setting | HumanEval reference |
| --- | --- |
| Native host | DeepSeek Harness `dsh-v0.2.0-rc.1`, SDK profile |
| Pinned host source | `4878cdabd87d4041bdaff61d04c966883b9fd07a` |
| Actor | `Qwen/Qwen3.5-4B` |
| Actor revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Serving | Self-hosted Qwen endpoint through the SDK Messages-compatible provider route |
| Provider registration label | `deepseek-official` |
| Native execution tool | `terminal`, with literal `command` and optional `cwd`/`timeout_sec` |
| Controller policy | v6; at most 16 candidates and 8 fast steps; 65,536 public bytes |
| Authoring configuration | Experiment revision v23, `structured_bank_author.mjs`; distinct from controller policy v6 and prospective v10/11 |
| Authoring interface | One initial metered `author_bank` request per task, with an explicit root and two successors |
| Other components | Public-history compaction for fallback, public-receipt finalization with an additional paid model call when applicable |
| Archived request thinking | Enabled for author/native requests; disabled for dedicated finalization; no OpenRouter low/high effort setting |
| Output capacity | 32,768 tokens per request |
| Tasks | HumanEval release; custom 33 training / 33 validation / 98 test partition; seed 20261006; one attempt per task |
| Native limits | 300-second actor deadline and 30-second verifier deadline |

The actor sees only the public function specification and docstring examples.
Reference solutions and released verifier tests remain outside actor context.
The author writes the implementation and its public-example checks. Its two
stored successors compile the implementation and execute those public checks,
subject to native predecessor receipts. Unknown or failed guards return to slow
reasoning. Final success is determined separately by the native evaluator.

The original harness made 954 paid calls and the selected configuration made
415. Its 147 valid substitution certificates identify individual callbacks
replaced by native actions.

## Released reference modules

[`reproduction/humaneval_v23/`](../reproduction/humaneval_v23/) contains the
archived v23 authoring adapter and its JavaScript dependencies. Its manifest
records their digests and the digest of the source archive. Current integration
code lives in `integrations/deepseek_harness/`.

To rerun the benchmark, configure the pinned SDK, model endpoint, isolated
native executor, public task adapter, protected scorer, frozen partition and
provider metering. Assign a new run identity when sources change. The adapter
uses fixed Python commands through the terminal bridge.

## Stock dsh and community plugins

The [bundled dsh plugin](../dsh-plugin/harnessmm-desktop/README.md)
runs in the desktop host. A stock headless session exposes tools such as
`bash`, `read`, `write` and `edit`. The HumanEval run used the SDK terminal
bridge and the Qwen actor listed above.

For a v10/11 integration, register the augmented `terminal`, `bash` or `pwsh`
schema before the provider request. Keep the native required arguments and
strip plan metadata before dispatch. Read exit status from host metadata.
The library tests exercise this contract with synthetic tool calls.

The bank stores successors to a separate root call and rejects a repeated
root. Preserve native command chains such as `&&` in both comparison arms.
Measure quality and complete Tokens/task (k) across the full task population,
including authoring, fallback, finalization and recovery.

For tasks that require an output artifact, a host can check whether the agent
produced it after returning a final plan. The [integration guide](INTEGRATION.md)
describes the public completion callback and its recovery behavior.
