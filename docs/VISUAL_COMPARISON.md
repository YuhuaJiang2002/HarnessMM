# MiniWoB++ visual-extension comparison

HarnessMM-V extends guarded decision reuse to screenshot-based agents.
The ten-method MiniWoB++ comparison below follows Figure 3(b) of the
November 20, 2026 HarnessMM-V manuscript snapshot. The
[HarnessMM paper](https://arxiv.org/abs/2610.32813) reports the coding experiments.

## Shared protocol

All methods use Qwen3.5-9B on the same 96 held-out MiniWoB++ tasks, with
native screenshots, public screenshot OCR and geometry, mouse/keyboard actions
and native reward scoring. The partition contains 32 training tasks, 32 validation tasks and 96 test tasks;
task families are separated before optimization. Training supplies revisions,
validation selects a candidate, and testing evaluates the frozen configuration.
Each arm uses one execution per test task, a 15-action limit and a 180-second
episode allowance. All compared methods receive token-efficiency prompts and
the objective of reducing complete-task input-plus-output tokens while
preserving performance.

HarnessPro is a configured visual reproduction of the balanced Medium tier,
using tiered evolution and generated-test selection with eight optimization
rounds before protected validation and testing. SoL-Pi uses the released Pi
v0.85.1 runtime with action fusion, observation packing, evidence-preserving
reduction and online context compaction. Its primary actor and reducer use
the same Qwen3.5-9B backbone. Codex Harness retains its official CLI v0.142.5 core with
a configured native visual adapter; DeepSeek Harness retains its native
multimodal policy and tool loop.

## Full test execution comparison

| Method | Successes / tasks | Success (%) | Tokens/task (k) |
| --- | ---: | ---: | ---: |
| baseline harness | 58 / 96 | 60.42 | 5.65 |
| SkillOpt | 62 / 96 | 64.58 | 7.97 |
| AutoSaddler | 58 / 96 | 60.42 | 7.87 |
| SoL-Pi | 61 / 96 | 63.54 | 26.72 |
| Prompt Compaction | 63 / 96 | 65.63 | 4.41 |
| DeepSeek Harness | 59 / 96 | 61.46 | 20.03 |
| Codex Harness | 61 / 96 | 63.54 | 39.53 |
| RHO | 58 / 96 | 60.42 | 5.13 |
| HarnessPro | 59 / 96 | 61.46 | 10.03 |
| HarnessMM-V | 62 / 96 | 64.58 | 3.98 |

## Cost and comparison definitions

Tokens/task (k) equals the complete test execution input-plus-output token
count divided by all 96 test tasks and by 1,000. Every actor, author, selector,
control, reducer, compaction, fallback and unsuccessful execution call within
the test scope is counted. Token saving against a comparator is
`100 * (1 - HarnessMM-V execution tokens / comparator execution tokens)`.
Savings use exact totals before rounding. Offline setup, optimization,
training-checkpoint assessment and validation are accounted for separately.
HarnessPro has a complete test execution total and a partial pretest cost total.

HarnessMM-V had the lowest test execution cost at 3.98 Tokens/task (k).
Prompt Compaction solved the most tasks (63 of 96, or 65.63%).

## Why the repository values changed

The README and its chart used the October 29 comparison snapshot. Figure 3(b)
uses later frozen evaluations on the same 96 instances. Both snapshots report
test execution input and output per task. Training, optimization and
validation costs are accounted for separately.

- **HarnessMM-V:** The selected configuration preserves the native root-action
  instruction, validates the root before compiling future actions, and omits empty
  observation fields with a reversible encoding. Training image/action pairs
  calibrate the click-binding radius. The selected test run solved 62 tasks, up
  from 60 in the earlier run.
- **Baseline harness:** The figure uses the native baseline replication
  associated with the SoL-Pi review. It executes literal pixel coordinates.
  Both baseline runs solved 58 tasks, with different metered execution totals.
- **SoL-Pi:** The later run uses the original Pi core with a revised visual
  adapter. It pairs the latest observation with the current screenshot,
  removes duplicate instructions and schema descriptions, and clarifies
  coordinate units. The original reducer and compaction calls remain in
  the token count. The later run solved 61 tasks, up from 58.
- **Codex Harness:** The later visual adapter removes inaccessible host skill
  listings and filesystem paths from model context while keeping the official
  CLI, full screenshots and native GUI tools. The later run solved 61 tasks,
  up from 57.
- **Prompt Compaction and RHO:** Prompt Compaction uses updated screenshot
  measurements for square controls and their labels. RHO uses shorter
  screenshot-based guidance for square controls. The table uses the complete
  tests of those adapters.

The [figure data](assets/readme/visual-comparison.json) identify the source
record for each current row. Earlier totals and source references are
available in the Git history of that file.

## Figure data and provenance

[visual-comparison.json](assets/readme/visual-comparison.json) records the
complete integer execution-token totals, successes, task counts, source
record names and content bindings for all ten methods. The original evidence
uses the internal name VIGRACE for the visual method; its current public name
is HarnessMM-V.

The figures can be regenerated with
[generate.py](assets/readme/generate.py). The source evidence is the visual
study's `visual_rsi_comparisons_provenance.json`, reconciled against each
named native admission record.
