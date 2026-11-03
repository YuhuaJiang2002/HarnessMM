"""Resolve legacy Python imports to the single canonical HarnessMM module."""

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
import sys


class _LegacyLoader(Loader):
    def __init__(self, canonical_name):
        self.canonical_name = canonical_name

    def create_module(self, spec):
        module = import_module(self.canonical_name)
        self.canonical_spec = module.__spec__
        return module

    def exec_module(self, module):
        # Import machinery assigns the alias spec to an existing module. Restore
        # its canonical spec so reload, resources and pickling keep one identity.
        module.__spec__ = self.canonical_spec

    def get_code(self, fullname):
        spec = find_spec(self.canonical_name)
        getter = getattr(spec.loader, "get_code", None)
        if getter is None:
            raise ImportError("No executable code for " + fullname)
        return getter(self.canonical_name)


class _LegacyFinder(MetaPathFinder):
    _harnessmm_legacy_finder = True

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("phiagent."):
            return None
        canonical_name = "harnessmm" + fullname[len("phiagent"):]
        canonical_spec = find_spec(canonical_name)
        if canonical_spec is None:
            return None
        spec = ModuleSpec(fullname, _LegacyLoader(canonical_name),
                          origin=canonical_spec.origin,
                          is_package=canonical_spec.submodule_search_locations is not None)
        if canonical_spec.submodule_search_locations is not None:
            spec.submodule_search_locations = list(canonical_spec.submodule_search_locations)
        return spec


def enable_legacy_imports():
    """Install once, without importing optional runtimes or child modules."""
    if not any(getattr(finder, "_harnessmm_legacy_finder", False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyFinder())
