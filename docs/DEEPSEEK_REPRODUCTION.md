# DeepSeek Harness: reference configuration and compatibility

## Which result this describes

The paper's HumanEval comparison reports 71.43% to 85.71% success and
96.28 to 11.62 Tokens/task (k), an 87.93% reduction on the same 98 test tasks.
These are complete execution measurements of the selected configurations.
They are not a reuse-only ablation or a guarantee for another model or task.
The execution comparison does not measure all-in RSI development savings.

| Setting | HumanEval reference |
| --- | --- |
| Native host | DeepSeek Harness `dsh-v0.2.0-rc.1`, SDK profile |
| Pinned host source | `4878cdabd87d4041bdaff61d04c966883b9fd07a` |
| Actor | `Qwen/Qwen3.5-4B` |
| Actor revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Serving | Self-hosted Qwen endpoint through the SDK Messages-compatible provider route |
| Provider registration label | `deepseek-official`; this label does not identify the actor as a DeepSeek model |
| Native execution tool | `terminal`, with literal `command` and optional `cwd`/`timeout_sec` |
| Controller policy | v6; at most 16 candidates and 8 fast steps; 65,536 public bytes |
| Authoring configuration | Experiment revision v23, `structured_bank_author.mjs`; distinct from controller policy v6 and prospective v10/11 |
| Authoring interface | One initial metered `author_bank` request per task, with an explicit root and two successors |
| Other components | Public-history compaction for fallback, public-receipt finalization with an additional paid model call when applicable |
| Archived request thinking | Enabled for author/native requests; disabled for dedicated finalization; no OpenRouter low/high effort setting |
| Output capacity | 32,768 tokens per request, not a cumulative experiment budget |
| Tasks | HumanEval release; custom 33 training / 33 validation / 98 test partition; seed 20261006; one attempt per task |
| Native limits | 300-second actor deadline and 30-second verifier deadline |

The actor sees only the public function specification and docstring examples.
Reference solutions and released verifier tests remain outside actor context.
The author writes the implementation and its public-example checks. Its two
stored successors compile the implementation and execute those public checks,
subject to native predecessor receipts. Unknown or failed guards return to slow
reasoning. Final success is determined separately by the native evaluator.

The paper records 147 valid substitution certificates for this condition.
The observed paid-call difference is 954 versus 415, which is not the same
quantity as the certificate count. Overall token savings include the surrounding
configuration; certificates do not assign all saved tokens to guarded reuse.

## Released reference modules

[`reproduction/humaneval_v23/`](../reproduction/humaneval_v23/) contains the
archived JavaScript authoring adapter and its companion modules. The initial
public snapshot `c8f1994` omitted `structured_bank_author.mjs`; its v6 text
prompt and v10 metadata path were not the complete HumanEval configuration.
The reference manifest records the exact archived bytes and archive digest.
Current integration fixes live in `integrations/deepseek_harness/` and do not
change these reference modules.

These modules restore the authoring implementation, not a turnkey replay of a
paid run. A new benchmark deployment also needs the pinned SDK, configured
model serving, isolated native task executor, public task adapter, protected
scorer, frozen partitions and complete provider metering. Do not use historical
run identities for changed sources. The reference is terminal-specific and
contains fixed Python workflow commands; porting it to Node.js is a new adapter.

## Stock dsh and community plugins

The [community dsh plugin](https://github.com/DSH-Extensions-list/dsh-harnessmm)
is a separate integration. A stock headless tool catalogue with `bash`, `read`,
`write` and `edit` is different from the reference SDK terminal bridge. The
HumanEval measurements do not establish the same saving for stock dsh rc.5,
DeepSeek V4.1 Flash, OpenRouter or a Node.js load-balancer task.

The current library supports schema augmentation for `terminal`, `bash` and
`pwsh`. Register the augmented schema before the provider request, retain every
native required argument, and remove only metadata before dispatch. Prompt text
alone does not insert `chisel_plan` into the schema. Receipt guards also need
actual exit metadata from the host; never infer a return code from arbitrary
printed text. This update has synthetic contract coverage, not a live rc.5
plugin qualification.

A bank contains successors after a separate root call. Repeating that root is
rejected intentionally. Native command chaining such as `&&` is a valid
baseline behavior and can leave fewer callbacks to substitute. Do not split
commands merely to increase substitution counts. Preserve task quality and
measure complete Tokens/task (k), including authoring, fallback, finalization
and recovery, over the full declared population.

A visible final plan can finish a turn without executing the requested work.
The old reasoning-only recovery does not cover that case. Hosts can opt into
the public-artifact completion helper described in [INTEGRATION.md](INTEGRATION.md).
That check is host-owned and must not use protected benchmark outcomes.
