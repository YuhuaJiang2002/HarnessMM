"""Paper-faithful reimplementation of the evaluated HARBOR outer loop.

This module follows the *evaluated implementation* described in Section VII
and Appendix A of ``Harbor: Automated Harness Optimization`` rather than
claiming to reproduce an unpublished artifact.  In particular it uses:

* heteroscedastic cold-start correction from equations (3)--(5),
* weighted ridge main and cross-block interaction features,
* ``{-1, +1}`` Boolean embeddings and tensor-product interactions,
* per-candidate Monte-Carlo hypervolume improvement divided by search cost,
* fixed seeded-prefix fidelities, posterior chance constraints, Hamming trust
  regions, guarded block freezing, and silent-flag exclusion.

The benchmark evaluator is deliberately outside this module.  Callers must
execute candidates on a server, persist the returned observations, and then
feed those frozen observations to :class:`HarborOptimizer`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations, product
import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class HarborFlag:
    """One Boolean harness flag and its observability contract."""

    name: str
    block: str
    warm_state_dependent: bool = False
    telemetry_counter: str | None = None
    minimum_turns_to_fire: int = 1

    def __post_init__(self) -> None:
        if not self.name or not self.block:
            raise ValueError("flag name and block must be non-empty")
        if self.minimum_turns_to_fire < 1:
            raise ValueError("minimum_turns_to_fire must be positive")


@dataclass(frozen=True)
class HarborObservation:
    """One externally evaluated Boolean configuration."""

    config: tuple[bool, ...]
    fidelity: int
    passed: int
    deployment_cost: float
    search_cost: float
    warm_fraction: float = 1.0
    warm_fraction_variance: float = 0.0
    telemetry: Mapping[str, float] | None = None
    turns: int = 0
    evaluation_id: str = ""

    def __post_init__(self) -> None:
        if self.fidelity <= 0:
            raise ValueError("fidelity must be positive")
        if not 0 <= self.passed <= self.fidelity:
            raise ValueError("passed must lie in [0, fidelity]")
        if self.deployment_cost < 0 or self.search_cost <= 0:
            raise ValueError("costs must be non-negative and search_cost positive")
        if not 0 <= self.warm_fraction <= 1:
            raise ValueError("warm_fraction must lie in [0, 1]")
        if self.warm_fraction_variance < 0:
            raise ValueError("warm_fraction_variance must be non-negative")

    @property
    def observed_rate(self) -> float:
        return self.passed / self.fidelity


@dataclass(frozen=True)
class CorrectedObservation:
    mean: float
    variance: float
    raw_mean: float
    clipped: bool
    unidentified: bool


@dataclass(frozen=True)
class HarborPrediction:
    mean: float
    standard_deviation: float
    deployment_cost: float
    safety_lcb: float
    safe: bool


@dataclass(frozen=True)
class HarborProposal:
    config: tuple[bool, ...]
    fidelity: int
    acquisition: float
    prediction: HarborPrediction
    trust_region_center: tuple[bool, ...]
    trust_region_radius: int

    def to_dict(self, flags: Sequence[HarborFlag]) -> dict[str, Any]:
        return {
            "config": {
                flag.name: enabled for flag, enabled in zip(flags, self.config)
            },
            "fidelity": self.fidelity,
            "acquisition": self.acquisition,
            "prediction": asdict(self.prediction),
            "trust_region_center": {
                flag.name: enabled
                for flag, enabled in zip(flags, self.trust_region_center)
            },
            "trust_region_radius": self.trust_region_radius,
        }


@dataclass(frozen=True)
class PreflightFlagStatus:
    flag: str
    status: str
    counter_value: float
    turns: int
    minimum_turns_to_fire: int


@dataclass(frozen=True)
class TrustRegion:
    center: tuple[bool, ...]
    radius: int
    alive: bool = True
    successes: int = 0
    failures: int = 0


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in lean installs
        raise RuntimeError(
            "HARBOR fitting requires the optional numpy dependency"
        ) from exc
    return np


def seeded_prefix_task_order(task_ids: Sequence[str], seed: int) -> tuple[str, ...]:
    """Return one fixed shuffle from which every fidelity is a prefix."""

    import random

    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be non-empty and unique")
    ordered = list(task_ids)
    random.Random(seed).shuffle(ordered)
    return tuple(ordered)


def fidelity_prefix(task_order: Sequence[str], fidelity: int) -> tuple[str, ...]:
    if not 0 < fidelity <= len(task_order):
        raise ValueError("fidelity must be a valid task-order prefix length")
    return tuple(task_order[:fidelity])


def sobol_boolean_design(
    dimensions: int,
    count: int,
    *,
    seed: int,
    exclude: Sequence[Sequence[bool]] = (),
) -> tuple[tuple[bool, ...], ...]:
    """Generate the paper's scrambled Sobol Boolean initialization.

    SciPy is imported only at this adapter boundary so importing ``phiagent``
    remains independent of numerical experiment dependencies.
    """

    if dimensions <= 0 or count <= 0:
        raise ValueError("dimensions and count must be positive")
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover - lean package import path
        raise RuntimeError("Sobol initialization requires optional scipy") from exc
    forbidden = {tuple(bool(value) for value in row) for row in exclude}
    if any(len(row) != dimensions for row in forbidden):
        raise ValueError("excluded Sobol configurations have the wrong dimension")
    if count > (2**dimensions - len(forbidden)):
        raise ValueError("requested Sobol design exceeds the remaining Boolean space")
    required_power = max(1, math.ceil(math.log2(count + len(forbidden) + 1)))
    result: list[tuple[bool, ...]] = []
    while len(result) < count:
        sampler = qmc.Sobol(d=dimensions, scramble=True, seed=seed)
        points = sampler.random_base2(m=required_power)
        for point in points:
            config = tuple(bool(value >= 0.5) for value in point)
            if config in forbidden or config in result:
                continue
            result.append(config)
            if len(result) == count:
                break
        required_power += 1
        if required_power > dimensions + 12 and len(result) < count:
            raise RuntimeError("Boolean Sobol projection could not produce enough unique points")
    return tuple(result)


def compose_warm_fraction(
    flags: Sequence[HarborFlag],
    config: Sequence[bool],
    per_flag_warm_fraction: Mapping[str, float],
) -> float:
    """Apply HARBOR's pessimistic minimum composition over enabled warm flags."""

    if len(flags) != len(config):
        raise ValueError("config length differs from flag space")
    enabled = [
        float(per_flag_warm_fraction[flag.name])
        for flag, on in zip(flags, config)
        if on and flag.warm_state_dependent
    ]
    if any(not 0 <= value <= 1 for value in enabled):
        raise ValueError("warm fractions must lie in [0, 1]")
    return min(enabled) if enabled else 1.0


def cold_start_correct(
    observation: HarborObservation,
    *,
    baseline_rate: float,
    baseline_variance: float,
    prior_variance: float = 1.0,
) -> CorrectedObservation:
    """Apply HARBOR equations (4)--(5), including boundary inflation."""

    if not 0 <= baseline_rate <= 1:
        raise ValueError("baseline_rate must lie in [0, 1]")
    if baseline_variance < 0 or prior_variance <= 0:
        raise ValueError("variances must be non-negative and prior positive")
    p_obs = observation.observed_rate
    w = observation.warm_fraction
    if w == 0:
        return CorrectedObservation(
            mean=p_obs,
            variance=prior_variance,
            raw_mean=p_obs,
            clipped=False,
            unidentified=True,
        )

    raw = (p_obs - (1.0 - w) * baseline_rate) / w
    mean = min(1.0, max(0.0, raw))
    clipped = mean != raw
    obs_variance = p_obs * (1.0 - p_obs) / observation.fidelity
    # The paper inflates only the observation term at a clipping boundary.
    observation_term = 1.0 / (4.0 * w * w) if clipped else obs_variance / (w * w)
    baseline_term = ((1.0 - w) ** 2) * baseline_variance / (w * w)
    warm_term = (
        ((p_obs - baseline_rate) ** 2)
        * observation.warm_fraction_variance
        / (w**4)
    )
    return CorrectedObservation(
        mean=mean,
        variance=max(observation_term + baseline_term + warm_term, 1e-12),
        raw_mean=raw,
        clipped=clipped,
        unidentified=False,
    )


def preflight_flag_statuses(
    flags: Sequence[HarborFlag],
    config: Sequence[bool],
    *,
    telemetry: Mapping[str, float],
    turns: int,
    epsilon: float = 1e-12,
) -> tuple[PreflightFlagStatus, ...]:
    """Implement the paper's turn-aware GREEN/YELLOW/RED liveness gate."""

    if len(flags) != len(config):
        raise ValueError("config length differs from flag space")
    rows: list[PreflightFlagStatus] = []
    for flag, enabled in zip(flags, config):
        if not enabled or flag.telemetry_counter is None:
            continue
        value = float(telemetry.get(flag.telemetry_counter, 0.0))
        if value > epsilon:
            status = "GREEN"
        elif turns < flag.minimum_turns_to_fire:
            status = "YELLOW"
        else:
            status = "RED"
        rows.append(
            PreflightFlagStatus(
                flag=flag.name,
                status=status,
                counter_value=value,
                turns=turns,
                minimum_turns_to_fire=flag.minimum_turns_to_fire,
            )
        )
    return tuple(rows)


def preflight_passes(statuses: Sequence[PreflightFlagStatus]) -> bool:
    return all(row.status != "RED" for row in statuses)


def _pareto_points(points: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Maximize both coordinates and discard dominated points."""

    unique = sorted(set(points), key=lambda row: (row[0], row[1]), reverse=True)
    result: list[tuple[float, float]] = []
    best_second = -math.inf
    for first, second in unique:
        if second > best_second:
            result.append((first, second))
            best_second = second
    return tuple(result)


def hypervolume_2d(
    points: Sequence[tuple[float, float]], reference: tuple[float, float]
) -> float:
    """Exact dominated hypervolume for two maximized objectives."""

    eligible = [
        (first, second)
        for first, second in _pareto_points(points)
        if first > reference[0] and second > reference[1]
    ]
    if not eligible:
        return 0.0
    area = 0.0
    previous_first = reference[0]
    # Integrate from the reference along the first objective.
    for first, second in sorted(eligible):
        area += (first - previous_first) * max(0.0, second - reference[1])
        previous_first = first
    return area


class _WeightedRidgeSurrogate:
    def __init__(
        self,
        flags: Sequence[HarborFlag],
        *,
        alpha_main: float,
        alpha_cross: float,
    ) -> None:
        if alpha_main <= 0 or alpha_cross <= alpha_main:
            raise ValueError("require alpha_cross > alpha_main > 0")
        self.flags = tuple(flags)
        self.alpha_main = alpha_main
        self.alpha_cross = alpha_cross
        self.cross_pairs = tuple(
            (left, right)
            for left, right in combinations(range(len(flags)), 2)
            if flags[left].block != flags[right].block
        )
        self.coefficients = None
        self.covariance = None
        self.residual_variance = 1.0

    def features(self, config: Sequence[bool]):
        np = _require_numpy()
        if len(config) != len(self.flags):
            raise ValueError("config length differs from flag space")
        encoded = np.asarray([1.0 if value else -1.0 for value in config])
        cross = [encoded[left] * encoded[right] for left, right in self.cross_pairs]
        return np.asarray([1.0, *encoded.tolist(), *cross], dtype=float)

    def fit(
        self,
        configs: Sequence[Sequence[bool]],
        outcomes: Sequence[CorrectedObservation],
    ) -> None:
        np = _require_numpy()
        if len(configs) != len(outcomes) or not configs:
            raise ValueError("configs and outcomes must align and be non-empty")
        matrix = np.stack([self.features(config) for config in configs])
        target = np.asarray([row.mean for row in outcomes], dtype=float)
        variance = np.asarray([max(row.variance, 1e-12) for row in outcomes])
        weights = 1.0 / variance
        penalties = np.asarray(
            [0.0]
            + [self.alpha_main] * len(self.flags)
            + [self.alpha_cross] * len(self.cross_pairs),
            dtype=float,
        )
        precision = matrix.T @ (weights[:, None] * matrix) + np.diag(penalties)
        covariance = np.linalg.pinv(precision, hermitian=True)
        coefficients = covariance @ matrix.T @ (weights * target)
        residuals = target - matrix @ coefficients
        dof = max(1, len(target) - matrix.shape[1])
        residual_variance = float((weights * residuals * residuals).sum() / dof)
        self.coefficients = coefficients
        self.covariance = covariance
        self.residual_variance = max(residual_variance, 1e-9)

    def predict(self, config: Sequence[bool]) -> tuple[float, float]:
        np = _require_numpy()
        if self.coefficients is None or self.covariance is None:
            raise RuntimeError("surrogate has not been fit")
        features = self.features(config)
        mean = float(features @ self.coefficients)
        epistemic = float(features @ self.covariance @ features)
        variance = max(1e-12, epistemic * self.residual_variance)
        return min(1.0, max(0.0, mean)), math.sqrt(variance)

    def block_scales(self) -> dict[str, float]:
        if self.coefficients is None:
            raise RuntimeError("surrogate has not been fit")
        result: dict[str, float] = {}
        for index, flag in enumerate(self.flags):
            result[flag.block] = result.get(flag.block, 0.0) + float(
                self.coefficients[1 + index] ** 2
            )
        return result

    @property
    def cross_scale(self) -> float:
        if self.coefficients is None:
            raise RuntimeError("surrogate has not been fit")
        start = 1 + len(self.flags)
        return float(sum(value * value for value in self.coefficients[start:]))


class HarborOptimizer:
    """Ask/tell controller for the paper-evaluated Boolean HARBOR variant."""

    def __init__(
        self,
        flags: Sequence[HarborFlag],
        *,
        full_fidelity: int,
        fidelities: Sequence[int],
        deployment_budget: float,
        search_budget: float,
        safety_margin: float = 0.0,
        risk_level: float = 0.1,
        alpha_main: float = 1.0,
        alpha_cross: float = 100.0,
        block_freeze_min_observations: int = 8,
        block_freeze_scale: float = 1e-5,
        silent_min_observations: int = 3,
        silent_epsilon: float = 1e-12,
        trust_region_count: int = 3,
        trust_region_radius: int = 2,
        trust_region_success_tolerance: int = 3,
        trust_region_failure_tolerance: int = 3,
        monte_carlo_samples: int = 512,
        seed: int = 0,
    ) -> None:
        if not flags or len({flag.name for flag in flags}) != len(flags):
            raise ValueError("flags must be non-empty with unique names")
        if full_fidelity <= 0 or full_fidelity not in fidelities:
            raise ValueError("fidelities must include positive full_fidelity")
        if any(value <= 0 or value > full_fidelity for value in fidelities):
            raise ValueError("invalid fidelity")
        if deployment_budget < 0 or search_budget <= 0:
            raise ValueError("budgets are invalid")
        if not 0 < risk_level < 0.5:
            raise ValueError("risk_level must lie in (0, 0.5)")
        self.flags = tuple(flags)
        self.full_fidelity = full_fidelity
        self.fidelities = tuple(sorted(set(fidelities)))
        self.deployment_budget = deployment_budget
        self.search_budget = search_budget
        self.safety_margin = safety_margin
        self.risk_level = risk_level
        self.block_freeze_min_observations = block_freeze_min_observations
        self.block_freeze_scale = block_freeze_scale
        self.silent_min_observations = silent_min_observations
        self.silent_epsilon = silent_epsilon
        self.trust_region_count = trust_region_count
        self.initial_trust_region_radius = trust_region_radius
        self.trust_region_success_tolerance = trust_region_success_tolerance
        self.trust_region_failure_tolerance = trust_region_failure_tolerance
        self.monte_carlo_samples = monte_carlo_samples
        self.seed = seed
        self.observations: list[HarborObservation] = []
        self.baselines: dict[int, HarborObservation] = {}
        self.silent_flags: set[int] = set()
        self.frozen_blocks: dict[str, tuple[bool, ...]] = {}
        self.trust_regions: tuple[TrustRegion, ...] = ()
        self._surrogate = _WeightedRidgeSurrogate(
            self.flags, alpha_main=alpha_main, alpha_cross=alpha_cross
        )

    @property
    def spent_search_budget(self) -> float:
        return sum(row.search_cost for row in self.observations)

    @property
    def remaining_search_budget(self) -> float:
        return self.search_budget - self.spent_search_budget

    def register_baseline(self, observation: HarborObservation) -> None:
        expected = (False,) * len(self.flags)
        if observation.config != expected:
            raise ValueError("baseline must be the all-flags-off configuration")
        if observation.fidelity in self.baselines:
            raise ValueError("baseline fidelity already registered")
        self.baselines[observation.fidelity] = observation
        self.register(observation)

    def register(self, observation: HarborObservation) -> None:
        if len(observation.config) != len(self.flags):
            raise ValueError("observation config length differs from flag space")
        if observation.fidelity not in self.fidelities:
            raise ValueError("observation fidelity was not declared")
        if self.spent_search_budget + observation.search_cost > self.search_budget + 1e-12:
            raise ValueError("observation exceeds frozen search budget")
        prior = tuple(self.observations)
        self.observations.append(observation)
        self._update_trust_regions(observation, prior)
        self._update_silent_flags()

    def initial_design(self, count: int = 12) -> tuple[tuple[bool, ...], ...]:
        """Return Sobol points at the cheapest fidelity after the all-off anchor."""

        if self.full_fidelity not in self.baselines:
            raise RuntimeError("register the full-fidelity all-off baseline first")
        return sobol_boolean_design(
            len(self.flags),
            count,
            seed=self.seed,
            exclude=[row.config for row in self.observations],
        )

    def _baseline_stats(self, fidelity: int) -> tuple[float, float]:
        if self.full_fidelity not in self.baselines:
            raise RuntimeError("full-fidelity all-off baseline is required")
        baseline = self.baselines.get(fidelity, self.baselines[self.full_fidelity])
        rate = baseline.observed_rate
        variance = rate * (1.0 - rate) / baseline.fidelity
        return rate, max(variance, 1e-12)

    def corrected_observations(self) -> tuple[CorrectedObservation, ...]:
        corrected = []
        for row in self.observations:
            rate, variance = self._baseline_stats(row.fidelity)
            corrected.append(
                cold_start_correct(
                    row,
                    baseline_rate=rate,
                    baseline_variance=variance,
                )
            )
        return tuple(corrected)

    def fit(self) -> None:
        self._surrogate.fit(
            [row.config for row in self.observations],
            self.corrected_observations(),
        )
        self._update_frozen_blocks()
        self._place_trust_regions()

    def _update_silent_flags(self) -> None:
        for index, flag in enumerate(self.flags):
            if flag.telemetry_counter is None:
                continue
            on_rows = [row for row in self.observations if row.config[index]]
            if len(on_rows) < self.silent_min_observations:
                continue
            values = [
                float((row.telemetry or {}).get(flag.telemetry_counter, 0.0))
                for row in on_rows
            ]
            if all(value < self.silent_epsilon for value in values):
                self.silent_flags.add(index)

    def _distinct_block_projections(self, block: str) -> int:
        indices = [i for i, flag in enumerate(self.flags) if flag.block == block]
        return len({tuple(row.config[i] for i in indices) for row in self.observations})

    def _update_frozen_blocks(self) -> None:
        if not self.observations:
            return
        best = max(
            self.observations,
            key=lambda row: (row.observed_rate, -row.deployment_cost),
        ).config
        for block, scale in self._surrogate.block_scales().items():
            if (
                scale <= self.block_freeze_scale
                and self._distinct_block_projections(block)
                >= self.block_freeze_min_observations
            ):
                self.frozen_blocks[block] = tuple(
                    best[i] for i, flag in enumerate(self.flags) if flag.block == block
                )

    def _all_configs(self) -> tuple[tuple[bool, ...], ...]:
        if len(self.flags) > 20:
            raise RuntimeError(
                "exact Boolean enumeration is capped at 20 flags; provide a bounded "
                "candidate pool for larger nominal spaces"
            )
        configs = []
        for config in product((False, True), repeat=len(self.flags)):
            if any(config[index] for index in self.silent_flags):
                continue
            valid = True
            for block, frozen in self.frozen_blocks.items():
                projected = tuple(
                    config[i] for i, flag in enumerate(self.flags) if flag.block == block
                )
                if projected != frozen:
                    valid = False
                    break
            if valid:
                configs.append(tuple(config))
        return tuple(configs)

    @staticmethod
    def _hamming(left: Sequence[bool], right: Sequence[bool]) -> int:
        return sum(a != b for a, b in zip(left, right))

    def _update_trust_regions(
        self,
        observation: HarborObservation,
        prior: Sequence[HarborObservation],
    ) -> None:
        if not self.trust_regions:
            return
        alive = [region for region in self.trust_regions if region.alive]
        if not alive:
            return
        region = min(
            alive,
            key=lambda row: self._hamming(row.center, observation.config),
        )
        prior_same_fidelity = [
            row.observed_rate for row in prior if row.fidelity == observation.fidelity
        ]
        success = not prior_same_fidelity or observation.observed_rate > max(prior_same_fidelity)
        updated = []
        for current in self.trust_regions:
            if current != region:
                updated.append(current)
                continue
            if success:
                successes = current.successes + 1
                radius = current.radius
                if successes >= self.trust_region_success_tolerance:
                    radius = min(len(self.flags), max(1, radius * 2))
                    successes = 0
                updated.append(
                    TrustRegion(
                        center=observation.config,
                        radius=radius,
                        alive=True,
                        successes=successes,
                        failures=0,
                    )
                )
            else:
                failures = current.failures + 1
                radius = current.radius
                is_alive = current.alive
                if failures >= self.trust_region_failure_tolerance:
                    if radius <= 1:
                        is_alive = False
                    else:
                        radius = max(1, radius // 2)
                    failures = 0
                updated.append(
                    TrustRegion(
                        center=current.center,
                        radius=radius,
                        alive=is_alive,
                        successes=0,
                        failures=failures,
                    )
                )
        self.trust_regions = tuple(updated)

    def _prediction(self, config: tuple[bool, ...]) -> HarborPrediction:
        mean, standard_deviation = self._surrogate.predict(config)
        # Deployment cost is modeled by the closest observed configuration, a
        # conservative closed-form analogue when the evaluator has not supplied
        # a separate cost surrogate.
        nearest = min(
            self.observations,
            key=lambda row: (self._hamming(row.config, config), row.deployment_cost),
        )
        z = NormalDist().inv_cdf(1.0 - self.risk_level)
        lcb = mean - z * standard_deviation
        baseline_rate, _ = self._baseline_stats(self.full_fidelity)
        return HarborPrediction(
            mean=mean,
            standard_deviation=standard_deviation,
            deployment_cost=nearest.deployment_cost,
            safety_lcb=lcb,
            safe=lcb >= baseline_rate - self.safety_margin,
        )

    def _place_trust_regions(self) -> None:
        if self.trust_regions and any(region.alive for region in self.trust_regions):
            return
        configs = self._all_configs()
        ranked = sorted(
            configs,
            key=lambda config: self._prediction(config).mean,
            reverse=True,
        )
        centers: list[tuple[bool, ...]] = []
        for config in ranked:
            if all(self._hamming(config, prior) > 0 for prior in centers):
                centers.append(config)
            if len(centers) == self.trust_region_count:
                break
        self.trust_regions = tuple(
            TrustRegion(config, min(self.initial_trust_region_radius, len(self.flags)))
            for config in centers
        )

    def _frontier_predictions(self) -> tuple[tuple[float, float], ...]:
        points = []
        for config in self._all_configs():
            prediction = self._prediction(config)
            if prediction.safe and prediction.deployment_cost <= self.deployment_budget:
                points.append((prediction.mean, -prediction.deployment_cost))
        return _pareto_points(points)

    def _mc_ehvi(
        self,
        config: tuple[bool, ...],
        prediction: HarborPrediction,
        *,
        fidelity: int,
        evaluation_cost: float,
    ) -> float:
        import random

        frontier = self._frontier_predictions()
        reference = (0.0, -max(self.deployment_budget, prediction.deployment_cost, 1.0))
        base_hv = hypervolume_2d(frontier, reference)
        rng = random.Random(
            f"{self.seed}:{fidelity}:{''.join('1' if x else '0' for x in config)}"
        )
        improvement = 0.0
        fidelity_scale = math.sqrt(fidelity / self.full_fidelity)
        sample_sd = prediction.standard_deviation / max(fidelity_scale, 1e-12)
        for _ in range(self.monte_carlo_samples):
            sampled = min(1.0, max(0.0, rng.gauss(prediction.mean, sample_sd)))
            improvement += max(
                0.0,
                hypervolume_2d(
                    [*frontier, (sampled, -prediction.deployment_cost)], reference
                )
                - base_hv,
            )
        return improvement / self.monte_carlo_samples / evaluation_cost

    def propose(
        self,
        *,
        evaluation_cost_by_fidelity: Mapping[int, float],
        batch_size: int = 1,
    ) -> tuple[HarborProposal, ...]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.fit()
        seen = {(row.config, row.fidelity) for row in self.observations}
        candidates: list[HarborProposal] = []
        for region in self.trust_regions:
            if not region.alive:
                continue
            for config in self._all_configs():
                if self._hamming(config, region.center) > region.radius:
                    continue
                prediction = self._prediction(config)
                if not prediction.safe or prediction.deployment_cost > self.deployment_budget:
                    continue
                for fidelity in self.fidelities:
                    if (config, fidelity) in seen:
                        continue
                    cost = float(evaluation_cost_by_fidelity[fidelity])
                    if cost <= 0 or cost > self.remaining_search_budget:
                        continue
                    acquisition = self._mc_ehvi(
                        config,
                        prediction,
                        fidelity=fidelity,
                        evaluation_cost=cost,
                    )
                    candidates.append(
                        HarborProposal(
                            config=config,
                            fidelity=fidelity,
                            acquisition=acquisition,
                            prediction=prediction,
                            trust_region_center=region.center,
                            trust_region_radius=region.radius,
                        )
                    )
        candidates.sort(key=lambda row: row.acquisition, reverse=True)
        selected: list[HarborProposal] = []
        for candidate in candidates:
            # Paper's greedy diversity approximation for q > 1.
            diversity = 1.0
            if selected:
                diversity = min(
                    self._hamming(candidate.config, prior.config) for prior in selected
                ) / max(1, len(self.flags))
            adjusted = candidate.acquisition * (0.5 + 0.5 * diversity)
            candidate = replace(candidate, acquisition=adjusted)
            selected.append(candidate)
            if len(selected) == batch_size:
                break
        return tuple(selected)

    def safe_pareto_set(self) -> tuple[tuple[tuple[bool, ...], HarborPrediction], ...]:
        self.fit()
        rows = []
        for config in self._all_configs():
            prediction = self._prediction(config)
            if prediction.safe and prediction.deployment_cost <= self.deployment_budget:
                rows.append((config, prediction))
        frontier = set(
            _pareto_points(
                [(prediction.mean, -prediction.deployment_cost) for _, prediction in rows]
            )
        )
        return tuple(
            (config, prediction)
            for config, prediction in rows
            if (prediction.mean, -prediction.deployment_cost) in frontier
        )

    def audit(self) -> dict[str, Any]:
        self.fit()
        return {
            "implementation_label": "HARBOR paper-faithful reimplementation",
            "paper_evaluated_substitutions": {
                "surrogate": "heteroscedastic weighted ridge",
                "boolean_kernel": "linear +/-1 tensor-product features",
                "batch_acquisition": "per-candidate Monte-Carlo EHVI + greedy diversity",
            },
            "observations": len(self.observations),
            "spent_search_budget": self.spent_search_budget,
            "remaining_search_budget": self.remaining_search_budget,
            "silent_flags": [self.flags[index].name for index in sorted(self.silent_flags)],
            "frozen_blocks": sorted(self.frozen_blocks),
            "block_scales": self._surrogate.block_scales(),
            "cross_scale": self._surrogate.cross_scale,
            "trust_regions": [asdict(region) for region in self.trust_regions],
        }


__all__ = [
    "CorrectedObservation",
    "HarborFlag",
    "HarborObservation",
    "HarborOptimizer",
    "HarborPrediction",
    "HarborProposal",
    "PreflightFlagStatus",
    "TrustRegion",
    "cold_start_correct",
    "compose_warm_fraction",
    "fidelity_prefix",
    "hypervolume_2d",
    "preflight_flag_statuses",
    "preflight_passes",
    "seeded_prefix_task_order",
    "sobol_boolean_design",
]
