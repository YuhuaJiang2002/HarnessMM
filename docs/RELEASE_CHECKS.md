# Release checks

## 2026-11-03 Python namespace migration

- Version 0.2.0 uses `harnessmm` for source files, imports and documentation.
  The `phiagent` package provides a compatibility bridge to the same modules.
- 700 Python tests and 92 Node.js tests passed locally. New namespace tests
  cover both import orders, namespace subpackages, shared state, reload,
  legacy pickle paths and module execution in fresh interpreters.
- All 449 source-manifest entries and eight archived HumanEval modules pass
  digest verification. Original source paths and digests remain recorded;
  archived reproduction files are unchanged.
- The source distribution and wheel build successfully. A clean environment
  installs the wheel and imports both namespaces outside the checkout, including
  the native Node bank contract. All local examples and documentation links pass.
- This release changes the public Python namespace. Native benchmark scores
  were not re-evaluated, and new deployments require a new source freeze.

## 2026-11-03 Codex and baseline entry points

- 693 Python tests and 92 Node.js tests passed locally.
- The baseline checks preserve the complete screenshot and public history,
  require a new actor call at each boundary, retain both paid parsing calls,
  and propagate unsettled provider failures without replay.
- Extracted B0 prompt, parser, coordinate compiler and action validator match
  the original visual source definitions. All 447 manifest entries and eight
  archived HumanEval modules pass digest verification.
- The source distribution and wheel build successfully. Both integration
  packages and their documentation are present in the wheel. Imports and the
  Node bank contract work outside the checkout with Python site packages disabled.
- The synthetic example and README links pass. These are software checks;
  no benchmark evaluations or provider calls were made.

## 2026-11-03 integration and reproduction update

- 688 Python tests and 92 Node.js tests passed locally.
- New regression coverage checks absent shell schemas, `bash`/`pwsh` native
  required arguments, metadata removal, root-repeat rejection, and plan-only
  completion recovery through an explicit host check.
- Archived HumanEval authoring modules match their frozen source digests.
  Synthetic checks exercise the real `author_bank` schema and preserve the
  additional paid fallback usage on abstention.
- The source distribution includes the archived modules and configuration
  notes. Native benchmark scores and stock-dsh rc.5 compatibility were not
  re-evaluated by these software checks.

## Original extraction checks

The 2026-10-26 source extraction passed these local checks on macOS with Python
3.14.7:

- 680 Python tests passed.
- 82 Node.js tests passed.
- The guarded-reuse example emitted three native substitution certificates and
  returned to the slow callback after exhaustion.
- The partition example created disjoint groups with 20%/20%/60% membership.
- The Python source distribution and wheel built successfully.
- Credential-pattern and entropy scans were reviewed. Matches were source/model
  digests, deployment paths, runtime variable names or synthetic test values;
  no live provider credentials were identified in the published files.

GitHub Actions runs the test suite and build on Linux with Python 3.10, 3.12
and 3.14 and Node.js 22. Its current status is available in the repository's
Actions tab. These checks cover the released software contracts; measured
benchmark quality and cost require the separately configured native runs.

Verify extracted source hashes from the checkout with:

```bash
python scripts/verify_release.py
```
