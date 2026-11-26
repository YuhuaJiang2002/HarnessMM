# Single-checkout source checks

The Python controllers, Codex and DeepSeek adapters, desktop plugin, Pi
extension, local examples and reproduction fixtures are included in this
repository. Installation and reference comparisons use these bundled files.
A second project checkout is not required.

From the repository root:

```sh
python scripts/check_self_contained.py
python scripts/verify_release.py
node pi-harnessmm/scripts/parity-check.mjs
python reproduction/run_all.py
```

The source check rejects missing project imports, links to another project
checkout, Git submodules, external symlinks and machine-local npm dependencies.
GitHub Actions runs the check and Pi reference comparison on each change.
The Pi parity command locates its reference modules relative to the script;
it also works when invoked from outside the checkout.

The core Python package has no third-party runtime dependencies. Running the
Pi or desktop integration requires its host application. Full benchmark and
robotics runs use separately installed native runtimes, datasets, checkpoints
and provider access as documented by each runner. Original license notices,
upstream API references and benchmark dependency sources remain attributed.
These dependencies do not supply missing HarnessMM project code.

## Verification on 2026-11-26

With Python 3.12 and Node.js 24, the local checks passed 722 Python tests,
92 core Node.js tests, 46 Pi extension tests, TypeScript checking and 115 Pi
reference comparisons. All local Markdown document links resolved.

The source distribution and wheel were built and checked in an isolated
temporary directory. The source archive ran the Pi reference comparison and
reproduction fixtures with no sibling checkout. Wheel imports and native Node
contract files resolved with Python site packages disabled. Injected missing
imports, cross-project links, external symlinks and machine-local lock entries
were each rejected by the source check.
