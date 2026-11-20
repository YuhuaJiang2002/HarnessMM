# Reproduction and deployment

## Source release

This release was assembled from the current local HarnessMM implementation on
2026-10-26. `SOURCE_MANIFEST.json` records SHA-256 hashes of the original files
and their published versions. The repository starts with a new Git history.

Release edits replace workstation paths and worker hostnames with deployment
placeholders, add packaging and documentation, and simplify namespace
initializers. The extracted algorithm implementations retain their model
identities, protocol constants and native interfaces. The Python package is
named `harnessmm`; legacy `phiagent` imports remain
supported. The [migration guide](PACKAGE_MIGRATION.md) describes source paths
and compatibility. Original experimental workspaces and paid records remain
outside this repository.

## Local checks

```bash
python -m pip install -e '.[dev,analysis]'
python -m pytest
node --test tests/*.test.mjs
node examples/guarded_reuse.mjs
python examples/partition.py
python -m build
```

Tests use synthetic fixtures and temporary workspaces. They check bank parsing,
guard behavior, native callback substitution, fallback, split boundaries, RSI
selection, source seals and accounting.

## Reproduction cases

[REPRODUCE.md](../REPRODUCE.md) maps released results and mechanisms to commands.
The [case index](../reproduction/README.md) includes MBPP aggregate recomputation,
Codex native-bank dispatch, GAIA2 public-receipt binding and the visual baseline.
Run `python reproduction/run_all.py` to verify the four JSON cases against their
checked-in digests and expected outputs, then run the dependency-free desktop
controller suite. The mechanism fixtures make no provider calls. The MBPP case
uses archived integer aggregates.

## Research runner entry points

The [integration index](../integrations/README.md) exposes Baseline, Codex and
DeepSeek harnesses. `python examples/harness_boundaries.py` checks the baseline
callback and Codex imports with synthetic inputs. The baseline package extracts
the original visual B0 prompt and action contract; it adds a host callback
boundary and the shared token-efficiency objective. Its source manifest records
the original visual workspace file digests. Codex exports retain their original
class/function identities.

The HumanEval DeepSeek Harness configuration is documented separately in
[DEEPSEEK_REPRODUCTION.md](DEEPSEEK_REPRODUCTION.md). The frozen v23 authoring adapter and its JavaScript dependencies are in
`reproduction/humaneval_v23/`.

| Study or component | Entry point |
| --- | --- |
| Codex residual learning with native terminal execution | `scripts/run_codex_chisel_terminal.py` |
| Codex native MCP action banks | `scripts/run_codex_native_app_bank_campaign.py` |
| DeepSeek SDK bridge and action graph execution | `integrations/deepseek_harness/runner.py` |
| Executable sub-DAG study | `scripts/run_sar_subdag_smoke.py` |
| ALFWorld instruction studies | `scripts/run_chisel_interactive_shared226.py` |
| ALFWorld RSI ablation | `scripts/run_chisel_core_ablation.py` |
| AIME companion policy | `scripts/run_aime_chisel_202060.py` |
| GAIA2 staged evaluation | `scripts/run_gaia2_202060.py` |
| SearchQA guarded pipeline | `scripts/run_searchqa_strict_jev.py` |
| Terminal-Bench guarded pipeline | `scripts/run_terminal_bench_jev.py` |
| Industrial train-only RSI | `scripts/run_industrial_train_rsi.py` |

These are research entry points with source-bound dependencies, native-runtime
requirements and deployment admission checks. A new deployment must supply
valid paths, upstream runtimes, datasets, endpoint configuration and a new
frozen experiment identity. `/path/to/...` values and `harnessmm-worker-*`
hostnames are placeholders. Host guards check those settings at launch.

The two files under `configs/` describe the historical protocols used by the
controller tests. Freeze your source and policy hashes under a new run ID.

## External dependencies

Obtain each benchmark and native harness from its upstream distribution and
follow its terms. Coding evaluation needs an isolated verifier environment;
ALFWorld needs its simulator and task data; GAIA2 needs its native task/tool
environment. Provider access and model serving are configured by the caller.
The optional `analysis` dependency supports numerical analysis; upstream
runtimes may require additional packages beyond the lightweight core.

Store provider credentials outside the repository and pass their paths through
the runner's documented configuration. Supply datasets and checkpoints through the native runtime. Keep provider logs
and run receipts with the experiment records.

## Cost and population

Split independent tasks or groups into 20% training, 20% validation and 60%
test before optimization and repetitions. Freeze identifiers and the rounding
policy for indivisible populations. Keep the same tasks and native evaluator
within each comparison.

Measure physical input plus output tokens, including shared authorship,
optimizer and selection work, control, retries, fallback and unsuccessful
executions within the stated scope. Count cached input once. Mark missing provider usage as unknown.

Report `Tokens/task (k) = complete input-plus-output tokens / declared complete
execution population / 1000`, rounded to two decimal places. Keep raw counts,
per-task scores, memberships, hashes and paid costs in the run's evidence
bundle. For an all-in RSI comparison, include development and selection costs
and declare the deployment population over which they are amortized.
