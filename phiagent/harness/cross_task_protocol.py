"""Shared validation and cost accounting for frozen cross-task benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_protocol(
    protocol: Mapping[str, Any],
    *,
    source_root: Path,
) -> None:
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"protocol schema_version must be {SCHEMA_VERSION}")
    if protocol.get("status") != "FROZEN_BEFORE_FIRST_RUN":
        raise ValueError("protocol must be frozen before the first run")
    if not protocol.get("frozen_at_utc"):
        raise ValueError("protocol must record frozen_at_utc")
    if not protocol.get("independent_unit"):
        raise ValueError("protocol must name its independent unit")

    required = protocol.get("required_frozen_source_sha256")
    if not isinstance(required, Mapping) or not required:
        raise ValueError("required_frozen_source_sha256 must be a non-empty object")
    for relative, expected in required.items():
        path = source_root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"frozen source is missing: {path}")
        actual = file_sha256(path)
        if actual != str(expected):
            raise ValueError(
                f"frozen source hash mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )

    auditor = protocol.get("auditor")
    if not isinstance(auditor, Mapping):
        raise ValueError("protocol must define an auditor object")
    if auditor.get("evolves_with_candidate") is not False:
        raise ValueError("the external auditor must not evolve with candidates")
    if auditor.get("aggregate_override_allowed") is not False:
        raise ValueError("aggregate scores may not override failed hard gates")

    accounting = protocol.get("cost_accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError("protocol must define cost_accounting")
    required_cost_fields = accounting.get("required_fields")
    if not isinstance(required_cost_fields, list) or not required_cost_fields:
        raise ValueError("cost_accounting.required_fields must be non-empty")
    if accounting.get("failed_attempt_cost_included") is not True:
        raise ValueError("failed-attempt cost must be included")
    if accounting.get("search_cost_separated") is not True:
        raise ValueError("search and frozen-confirmation cost must be separated")


@dataclass(frozen=True)
class EndToEndCost:
    model_calls: int
    deployment_model_calls: int
    smoke_model_calls: int
    wall_seconds: float
    gpu_seconds: float
    inference_gpu_seconds: float
    evaluator_wall_seconds: float
    prompt_utf8_bytes: int
    prompt_lexemes: int
    peak_gpu_memory_mib: float
    api_cost_usd: float

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.model_calls != self.deployment_model_calls + self.smoke_model_calls:
            raise ValueError("model_calls must equal deployment plus smoke calls")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def aggregate_costs(costs: Sequence[EndToEndCost]) -> EndToEndCost:
    return EndToEndCost(
        model_calls=sum(item.model_calls for item in costs),
        deployment_model_calls=sum(item.deployment_model_calls for item in costs),
        smoke_model_calls=sum(item.smoke_model_calls for item in costs),
        wall_seconds=sum(item.wall_seconds for item in costs),
        gpu_seconds=sum(item.gpu_seconds for item in costs),
        inference_gpu_seconds=sum(item.inference_gpu_seconds for item in costs),
        evaluator_wall_seconds=sum(item.evaluator_wall_seconds for item in costs),
        prompt_utf8_bytes=sum(item.prompt_utf8_bytes for item in costs),
        prompt_lexemes=sum(item.prompt_lexemes for item in costs),
        peak_gpu_memory_mib=max(
            (item.peak_gpu_memory_mib for item in costs),
            default=0.0,
        ),
        api_cost_usd=sum(item.api_cost_usd for item in costs),
    )


def cost_per_success(total: float, successes: int) -> float | None:
    if successes < 0:
        raise ValueError("successes must be non-negative")
    if successes == 0:
        return None
    return total / successes


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be in [0, total]")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {"low": center - radius, "high": center + radius}


def paired_bootstrap_difference_pp(
    candidate: Sequence[bool],
    reference: Sequence[bool],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("paired samples must have the same non-zero length")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    differences = [
        100.0 * (float(candidate_value) - float(reference_value))
        for candidate_value, reference_value in zip(candidate, reference, strict=True)
    ]
    generator = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(
            sum(differences[generator.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    means.sort()

    def percentile(probability: float) -> float:
        position = probability * (len(means) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return means[lower]
        weight = position - lower
        return means[lower] * (1.0 - weight) + means[upper] * weight

    return {
        "mean_difference_pp": sum(differences) / len(differences),
        "ci95_low_pp": percentile(0.025),
        "ci95_high_pp": percentile(0.975),
    }


__all__ = [
    "EndToEndCost",
    "SCHEMA_VERSION",
    "aggregate_costs",
    "canonical_sha256",
    "cost_per_success",
    "file_sha256",
    "paired_bootstrap_difference_pp",
    "validate_frozen_protocol",
    "wilson_interval",
]
