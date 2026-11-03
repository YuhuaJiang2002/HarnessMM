# Baseline harness

The visual paper's **baseline harness** makes a fresh model decision from the
current screenshot and public action history at every native action boundary.
It uses the original native tools and evaluator. It has no EDG authorship,
retained decision bank or certified callback substitution.

`VisualBaseline` exposes that decision loop through a metered actor callback.
Its prompt, JSON parser, coordinate compiler and action validator come from
the visual workspace's original `scripts/run_miniwob.py` B0 arm and
`src/vigrace/core.py`. The extracted definitions are in
[`_visual_contract.py`](_visual_contract.py), with source digests in
[`SOURCE_MANIFEST.json`](../../SOURCE_MANIFEST.json). The default instruction
also includes the shared token-efficiency objective used by comparison arms.

```python
from integrations.baseline_harness import VisualBaseline

baseline = VisualBaseline(actor=metered_actor.call, audit=durable_audit)
action = baseline.decide(
    png=current_full_png,
    image_size=(480, 630),
    history=public_executed_actions,
    callback_id=registered_slow_callback_id,
)
receipt = original_native_dispatch(action)
```

`actor(messages, purpose, callback_id)` returns model text and records each
physical request's input/output usage. A malformed response triggers one paid
fallback, as in the reference loop. Transport failures propagate to the host
for settlement. The adapter never creates replacement actions after a provider
error. The host retains both calls and all unsuccessful execution costs.

Coordinates are normalized by the model, then compiled once into literal
screenshot pixels. The host must use the corresponding native coordinate
mapping. The original MiniWoB++ renderer uses a 160 by 210 viewport at scale
three, producing a 480 by 630 PNG. Different benchmarks require their own
declared native mapping, prompt and public observation policy.

This is a reusable decision boundary. Exact paper reproduction also needs the
selected benchmark recipe, fixed task membership, model, episode limits,
native scorer and complete workflow ledger. This release does not launch or
re-evaluate those experiments. The callback example uses synthetic inputs:

```bash
python examples/harness_boundaries.py
```

## Other baseline implementations

The paper's baseline harness is one condition. The following compared methods
have separate implementations and deployment requirements:

| Method | Existing implementation |
| --- | --- |
| GAIA2 native baseline | [`gaia2_no_harness.py`](../../harnessmm/harness/gaia2_no_harness.py) |
| Codex Harness | [`integrations/codex_harness`](../codex_harness/README.md) |
| DeepSeek Harness | [`integrations/deepseek_harness`](../deepseek_harness/) and [configuration](../../docs/DEEPSEEK_REPRODUCTION.md) |
| SkillOpt | [`industrial_skillopt.py`](../../harnessmm/harness/industrial_skillopt.py) and [`cvpr_skillopt_native_bridge.py`](../../scripts/cvpr_skillopt_native_bridge.py) |
| AutoSaddler | [`run_chisel_autosaddler_corrected_loop.py`](../../scripts/run_chisel_autosaddler_corrected_loop.py) |
| SoL-Pi | [`gaia2_solpi.py`](../../harnessmm/harness/gaia2_solpi.py) and [`terminal_solpi.py`](../../scripts/terminal_solpi.py) |

These links identify released research code; benchmark-specific visual
adapters and upstream optimizers still need their own configurations. RHO,
HarnessPro and other unlisted visual recipes are not supplied by this package.
See [reproduction](../../docs/REPRODUCIBILITY.md) for deployment boundaries.
