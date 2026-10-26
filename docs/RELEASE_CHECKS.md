# Release checks

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
