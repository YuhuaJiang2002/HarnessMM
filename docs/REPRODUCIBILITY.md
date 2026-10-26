# Reproduction and deployment

## Source release

This release was assembled from the current local HarnessMM implementation on
2026-10-26. `SOURCE_MANIFEST.json` records SHA-256 hashes of the original files
and their published versions. The repository starts with a new Git history.

Release edits replace workstation paths and worker hostnames with deployment
placeholders, add packaging and documentation, and simplify namespace
initializers. The extracted algorithm implementations retain their model
identities, protocol constants and native interfaces. Original experimental
workspaces and paid records remain outside this repository.

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
selection, source seals and accounting. They do not reproduce paper benchmark
scores or require a provider key.

## Research runner entry points

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
hostnames are placeholders. Host guards deliberately reject an unconfigured
machine. Do not disable them and assume that the original run was reproduced.

The two configurations under `configs/` support static regression tests for
the extracted controllers. Historical source and policy hashes describe the
original reference configuration; they do not seal the public extraction or
authorize reusing its old run identity. Freeze new sources before a new run.

## External dependencies

Obtain each benchmark and native harness from its upstream distribution and
follow its terms. Coding evaluation needs an isolated verifier environment;
ALFWorld needs its simulator and task data; GAIA2 needs its native task/tool
environment. Provider access and model serving are configured by the caller.
The optional `analysis` dependency supports numerical analysis; upstream
runtimes may require additional packages beyond the lightweight core.

Store provider credentials outside the repository and pass their paths through
the runner's documented configuration. The release contains no credential
files, datasets, model checkpoints, provider request logs or paid run receipts.
It does not run or resume an experiment during installation.

## Cost and population

Split independent tasks or groups into 20% training, 20% validation and 60%
test before optimization and repetitions. Freeze identifiers and the rounding
policy for indivisible populations. Keep the same tasks and native evaluator
within each comparison.

Measure physical input plus output tokens, including shared authorship,
optimizer and selection work, control, retries, fallback and unsuccessful
executions within the stated scope. Cached-input accounting must not add
cached tokens twice. Missing usage remains unknown. A local replacement
certificate cannot supply missing provider usage.

Report `Tokens/task (k) = complete input-plus-output tokens / declared complete
execution population / 1000`, rounded to two decimal places. Keep raw counts,
per-task scores, memberships, hashes and paid costs in the run's evidence
bundle. For an all-in RSI comparison, include development and selection costs
and declare the deployment population over which they are amortized.
