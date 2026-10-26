"""Prospective, dependency-free startup hash-seed attestation.

Build an environment BEFORE spawning a worker, then attest inside that worker
before native initialization. This module never sets the current environment,
reseeds applications, draws RNG values, or changes public receipts.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from typing import Any


PROBES = (
    "phiagent/native-worker/startup-probe",
    "synthetic-participant-alpha",
    "synthetic-conversation-beta-with-a-longer-name",
    "model-seed-and-scenario-seed-are-distinct",
)
ENVIRONMENT_ALLOWLIST = (
    "PYTHONHASHSEED", "PYTHONNOUSERSITE", "PYTHONSAFEPATH", "PYTHONUTF8",
    "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING", "LANG", "LC_ALL", "TZ",
    "CUDA_VISIBLE_DEVICES",
)
HASH_FIELDS = (
    "width", "modulus", "inf", "nan", "imag", "algorithm", "hash_bits", "seed_bits", "cutoff",
)
REFERENCE_CODE = (
    "import sys\n"
    f"probes = {PROBES!r}\n"
    "print(repr({\n"
    " 'probe_values': [hash(value) for value in probes],\n"
    " 'version': sys.version,\n"
    " 'implementation': sys.implementation.name,\n"
    " 'cache_tag': sys.implementation.cache_tag,\n"
    " 'executable': sys.executable,\n"
    f" 'hash_info': {{name: getattr(sys.hash_info, name) for name in {HASH_FIELDS!r}}},\n"
    " 'ignore_environment': sys.flags.ignore_environment,\n"
    " 'no_site': sys.flags.no_site,\n"
    " 'module_roots': sorted({name.split('.')[0] for name in sys.modules}),\n"
    "}))\n"
)


class SeedAttestationError(RuntimeError):
    """A failed attestation with a JSON-safe receipt, never a success fallback."""

    def __init__(self, receipt: dict[str, Any]):
        self.receipt = receipt
        super().__init__("Worker seed attestation failed: " + ", ".join(receipt["errors"]))


def validate_hash_seed(seed: int | str) -> str:
    """Return canonical decimal seed text; reject random, booleans and coercions."""
    if type(seed) is int:
        value = seed
    elif isinstance(seed, str) and re.fullmatch(r"0|[1-9][0-9]{0,9}", seed):
        value = int(seed)
    else:
        raise ValueError("PYTHONHASHSEED must be an integer or canonical decimal string")
    if not 0 <= value <= 4294967295:
        raise ValueError("PYTHONHASHSEED must lie in [0, 4294967295]")
    return str(value)


def build_worker_environment(
    seed: int | str, parent_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a COPY for the next interpreter; all unrelated values are preserved."""
    expected = validate_hash_seed(seed)
    result = dict(os.environ if parent_environment is None else parent_environment)
    if any(not isinstance(key, str) or not key or "=" in key or "\0" in key
           or not isinstance(value, str) or "\0" in value for key, value in result.items()):
        raise ValueError("A valid string-to-string process environment is required")
    result["PYTHONHASHSEED"] = expected
    return result


def attest_startup_seed(
    expected_seed: int | str, *, role: str, reference_timeout: float = 10.0,
) -> dict[str, Any]:
    """Verify this interpreter against a fresh same-executable seeded child.

    The child uses ``-S`` and explicitly imports only the builtin ``sys`` module.
    Do not use ``-E``/``-I``: those would ignore the reference PYTHONHASHSEED.
    Finite hash probes attest consistency, not whole-environment determinism or
    historical seed recovery. Failed checks raise SeedAttestationError(receipt).
    """
    expected = validate_hash_seed(expected_seed)
    if not isinstance(role, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", role):
        raise ValueError("A short explicit process role is required")
    if (type(reference_timeout) not in (int, float) or not math.isfinite(reference_timeout)
            or reference_timeout <= 0):
        raise ValueError("Reference timeout must be positive and finite")
    if not sys.executable:
        raise ValueError("A same-executable reference interpreter is unavailable")
    executable = Path(sys.executable).resolve(strict=True)
    identity = {
        "version": sys.version, "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "executable": sys.executable, "real_executable": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "prefix": sys.prefix, "base_prefix": sys.base_prefix,
        "hash_info": {name: getattr(sys.hash_info, name) for name in HASH_FIELDS},
        "ignore_environment": sys.flags.ignore_environment,
        "hash_randomization": sys.flags.hash_randomization,
    }
    actual_environment = {key: os.environ.get(key) for key in ENVIRONMENT_ALLOWLIST}
    receipt: dict[str, Any] = {
        "format": "phiagent-native-worker-startup-seed-v1", "status": "FAIL",
        "role": role, "pid": os.getpid(), "ppid": os.getppid(), "hostname": socket.gethostname(),
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_hash_seed": expected, "allowlisted_environment": actual_environment,
        "interpreter": identity, "probe_strings": list(PROBES),
        "observed_probe_values": [hash(value) for value in PROBES],
        "helper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_code_sha256": hashlib.sha256(REFERENCE_CODE.encode()).hexdigest(),
        "reference_explicit_imports": ["sys"], "reference_flags": ["-S"],
        "app_seeds_read_or_predicted": False, "current_environment_modified": False,
        "native_initialization_or_rng_modified": False,
        "whole_environment_determinism_claimed": False,
        "historical_hash_seed_recovered": False, "errors": [],
    }
    checks = {
        "environment_matches_expected": actual_environment["PYTHONHASHSEED"] == expected,
        "current_interpreter_respects_environment": not sys.flags.ignore_environment,
        "hash_randomization_flag_consistent": bool(sys.flags.hash_randomization) == (expected != "0"),
    }
    reference_environment = build_worker_environment(expected)
    # Reference isolation must not import site hooks or application packages.
    reference_environment.pop("PYTHONPATH", None)
    reference_environment.pop("PYTHONSTARTUP", None)
    reference_environment.pop("PYTHONINSPECT", None)
    try:
        result = subprocess.run(
            [sys.executable, "-S", "-c", REFERENCE_CODE],
            env=reference_environment, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, check=True, timeout=reference_timeout,
        )
        reference = ast.literal_eval(result.stdout)
        if not isinstance(reference, dict):
            raise ValueError("Reference output is not a dictionary")
        receipt["reference"] = reference
        checks["same_interpreter_identity"] = all(
            reference[key] == identity[key]
            for key in ("version", "implementation", "cache_tag", "hash_info"))
        checks["same_executable"] = Path(reference["executable"]).resolve(strict=True) == executable
        checks["reference_site_disabled"] = reference["no_site"] == 1
        checks["reference_respects_environment"] = reference["ignore_environment"] == 0
        checks["reference_has_no_application_imports"] = not (
            {"are", "phiagent", "torch", "transformers", "numpy"} & set(reference["module_roots"]))
        checks["hash_probes_match"] = reference["probe_values"] == receipt["observed_probe_values"]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError,
            ValueError, SyntaxError, KeyError, TypeError) as error:
        receipt["reference_error"] = {
            "type": type(error).__name__,
            "returncode": getattr(error, "returncode", None),
        }
        stderr = getattr(error, "stderr", None)
        if stderr:
            data = stderr.encode() if isinstance(stderr, str) else stderr
            receipt["reference_error"]["stderr_sha256"] = hashlib.sha256(data).hexdigest()
        checks["reference_completed"] = False
    receipt["checks"] = checks
    receipt["errors"] = [key for key, passed in checks.items() if not passed]
    if receipt["errors"]:
        raise SeedAttestationError(receipt)
    receipt["status"] = "PASS"
    return receipt
