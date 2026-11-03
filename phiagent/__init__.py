"""Compatibility import for the renamed :mod:`harnessmm` package."""

import sys

import harnessmm
from harnessmm._compat import enable_legacy_imports

enable_legacy_imports()
sys.modules[__name__] = harnessmm
