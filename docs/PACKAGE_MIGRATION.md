# Python package migration

From version 0.2.0, use `harnessmm` for Python imports. The installable project,
source directory and import namespace now share the same name.

| Previous import | Current import |
| --- | --- |
| `phiagent.harness.codex_chisel` | `harnessmm.harness.codex_chisel` |
| `phiagent.learning.residual_meta_harness` | `harnessmm.learning.residual_meta_harness` |
| `phiagent.evaluation.deployment_token_accounting` | `harnessmm.evaluation.deployment_token_accounting` |

```python
from harnessmm.harness.codex_chisel import CodexChiselController
from harnessmm.harness.codex_native_app_bank import NativeBankBoundary
from harnessmm.evaluation.deployment_token_accounting import RecordedActorCall
```

The source lives under `harnessmm/`. Update local imports and source-file paths
to this directory. The public `integrations.codex_harness` and
`integrations.baseline_harness` entry points remain available.

## Existing applications

The `phiagent` package is a small compatibility bridge. Old and new imports
resolve to the same module and class objects, including implicit namespace
subpackages. Class checks, monkeypatches and shared module state work across
both spellings. Existing pickles that refer to old module paths can resolve
those paths through the bridge. New objects use `harnessmm` as their class
module name. The bridge does not install optional benchmark runtimes.

The compatibility guarantee covers Python imports. A frozen run also binds
source paths, file hashes, models and protocols. The renamed public sources
require a new source freeze; do not replace or relabel a historical run's
files or hashes to resume it. `SOURCE_MANIFEST.json` retains original source
digests, previous published digests and the path mapping for this migration.

## Historical identifiers

Existing schema strings, plugin identifiers, container labels, deployment
paths and the archived HumanEval implementation retain their registered
values. The provenance field for inspected design sources also keeps its
historical paths and digests. Those values identify existing records and
interfaces; the current executable source paths use `harnessmm/`.

## Verification

```bash
python -m pytest
python scripts/verify_release.py
node --test tests/*.test.mjs
python examples/harness_boundaries.py
python -m build
```

Namespace regression tests check both import orders in fresh interpreters,
module and class identity, shared state, reload and pickle round trips. The
release checks also import both namespaces from the built wheel outside the
source checkout.
