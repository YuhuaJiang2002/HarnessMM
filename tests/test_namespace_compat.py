"""Namespace migration contracts, including fresh-interpreter import order."""

from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def isolated(code):
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c",
         "import sys; sys.path.insert(0, sys.argv[1]);\n" + code, str(ROOT)],
        cwd=ROOT.parent, capture_output=True, text=True, check=True,
    )


@pytest.mark.parametrize("first", ["harnessmm", "phiagent"])
def test_import_order_keeps_one_module_class_and_namespace_package(first):
    isolated(f'''
import importlib
first = {first!r}
second = "phiagent" if first == "harnessmm" else "harnessmm"
modules = ["", ".harness", ".learning", ".evaluation", ".data_engine",
           ".harness.codex_chisel", ".learning.residual_meta_harness",
           ".evaluation.deployment_token_accounting", ".data_engine.provenance"]
for suffix in modules:
    a = importlib.import_module(first + suffix)
    b = importlib.import_module(second + suffix)
    assert a is b, suffix
    assert a.__spec__.name == "harnessmm" + suffix
from phiagent.harness.codex_chisel import ChiselConfig as legacy
from harnessmm.harness.codex_chisel import ChiselConfig as canonical
assert legacy is canonical
assert canonical.__module__ == "harnessmm.harness.codex_chisel"
''')


def test_legacy_and_current_imports_share_mutations_and_reload():
    isolated('''
import importlib
from phiagent.harness import benchmark_partition as old
from harnessmm.harness import benchmark_partition as new
old._migration_fixture = object()
assert new._migration_fixture is old._migration_fixture
assert importlib.reload(old) is new
assert old.__spec__.name == "harnessmm.harness.benchmark_partition"
assert importlib.import_module("phiagent.harness.benchmark_partition") is new
''')


def test_old_pickle_module_paths_resolve_and_new_pickles_use_current_name():
    isolated('''
import pickle
from harnessmm.harness.codex_chisel import ChiselConfig
legacy_class = pickle.loads(b"cphiagent.harness.codex_chisel\\nChiselConfig\\n.")
assert legacy_class is ChiselConfig
instance = ChiselConfig()
payload = pickle.dumps(instance)
assert b"harnessmm.harness.codex_chisel" in payload
assert pickle.loads(payload) == instance
''')


def test_import_bridge_does_not_load_optional_runtimes():
    isolated('''
import phiagent
import harnessmm
assert phiagent is harnessmm
assert not {"numpy", "torch", "are", "harbor", "playwright"} & set(sys.modules)
''')


def test_legacy_module_execution_uses_the_canonical_code():
    isolated('''
import runpy
old = runpy.run_module("phiagent.experiments", run_name="__main__")
assert "ExperimentReservation" in old
assert old["DIRECTORY_TIMESTAMP_FORMAT"] == "%Y%m%dT%H%M%SZ"
assert old["__spec__"].name == "phiagent.experiments"
''')


def test_missing_legacy_submodule_is_a_normal_import_error():
    isolated('''
import importlib
try:
    importlib.import_module("phiagent.harness.nonexistent_migration_fixture")
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("Missing legacy modules must not be fabricated")
''')
