"""Thread-safe endpoint circuit breaker for infrastructure-sensitive model calls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitOpen(RuntimeError):
    """Raised when a shared endpoint has crossed its infrastructure-error limit."""


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 3
    backoff_seconds: tuple[float, ...] = (2.0, 5.0, 10.0)
    jitter_fraction: float = 0.2
    random_seed: int = 20260830

    def __post_init__(self) -> None:
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if not self.backoff_seconds or any(value < 0 for value in self.backoff_seconds):
            raise ValueError("backoff_seconds must be non-empty and non-negative")
        if not 0 <= self.jitter_fraction <= 1:
            raise ValueError("jitter_fraction must be in [0,1]")


class EndpointCircuitBreaker:
    """Coordinate retries across workers and open after global endpoint failures."""

    def __init__(
        self,
        endpoint: str,
        *,
        config: CircuitBreakerConfig | None = None,
        audit_path: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.endpoint = endpoint
        self.config = config or CircuitBreakerConfig()
        self.audit_path = audit_path
        self._sleep = sleep
        self._lock = threading.Lock()
        self._rng = random.Random(self.config.random_seed)
        self._state = "CLOSED"
        self._consecutive_failures = 0
        self._infra_failed_requests = 0
        self._infra_retry_count = 0
        self._infra_wallclock_seconds = 0.0
        self._events: list[dict[str, Any]] = []
        self._write_audit()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._state == "OPEN"

    def _snapshot_unlocked(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "endpoint": self.endpoint,
            "state": self._state,
            "consecutive_failures": self._consecutive_failures,
            "infra_failed_requests": self._infra_failed_requests,
            "infra_retry_count": self._infra_retry_count,
            "infra_wallclock_seconds": self._infra_wallclock_seconds,
            "failed_request_token_usage": "unknown",
            "config": asdict(self.config),
            "events": list(self._events),
            "updated_at_utc": utc_now(),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def _write_audit_unlocked(self) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.audit_path.with_suffix(self.audit_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                self._snapshot_unlocked(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.audit_path)

    def _write_audit(self) -> None:
        with self._lock:
            self._write_audit_unlocked()

    def _before_request(self, request_id: str, attempt: int) -> None:
        with self._lock:
            if self._state == "OPEN":
                raise EndpointCircuitOpen(
                    f"endpoint circuit is open before {request_id} attempt {attempt}"
                )
            if attempt > 1:
                self._infra_retry_count += 1
                self._write_audit_unlocked()

    def _record_success(
        self, request_id: str, attempt: int, elapsed_seconds: float
    ) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._events.append(
                {
                    "timestamp_utc": utc_now(),
                    "request_id": request_id,
                    "attempt": attempt,
                    "status": "SUCCESS",
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            self._write_audit_unlocked()

    def _record_failure(
        self,
        request_id: str,
        attempt: int,
        elapsed_seconds: float,
        error: Exception,
    ) -> bool:
        with self._lock:
            self._infra_failed_requests += 1
            self._consecutive_failures += 1
            self._infra_wallclock_seconds += elapsed_seconds
            if self._consecutive_failures >= self.config.failure_threshold:
                self._state = "OPEN"
            self._events.append(
                {
                    "timestamp_utc": utc_now(),
                    "request_id": request_id,
                    "attempt": attempt,
                    "status": "INFRASTRUCTURE_FAILURE",
                    "elapsed_seconds": elapsed_seconds,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "circuit_state_after": self._state,
                }
            )
            opened = self._state == "OPEN"
            self._write_audit_unlocked()
            return opened

    def call(
        self,
        request_id: str,
        operation: Callable[[], T],
        *,
        max_attempts: int = 4,
    ) -> T:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            self._before_request(request_id, attempt)
            started = time.monotonic()
            try:
                result = operation()
            except Exception as exc:  # noqa: BLE001
                elapsed = time.monotonic() - started
                last_error = exc
                opened = self._record_failure(request_id, attempt, elapsed, exc)
                if opened:
                    raise EndpointCircuitOpen(
                        f"endpoint circuit opened after {request_id} attempt {attempt}: {exc}"
                    ) from exc
                if attempt == max_attempts:
                    break
                delay_index = min(attempt - 1, len(self.config.backoff_seconds) - 1)
                base_delay = self.config.backoff_seconds[delay_index]
                with self._lock:
                    jitter = self._rng.uniform(
                        -base_delay * self.config.jitter_fraction,
                        base_delay * self.config.jitter_fraction,
                    )
                self._sleep(max(0.0, base_delay + jitter))
                continue
            self._record_success(
                request_id,
                attempt,
                time.monotonic() - started,
            )
            return result
        raise RuntimeError(
            f"endpoint call failed after {max_attempts} attempts: {last_error}"
        ) from last_error

    def close_after_readiness(self, readiness: dict[str, Any]) -> None:
        if (
            readiness.get("status") != "WORKING"
            or int(readiness.get("num_requests") or 0) < 20
            or int(readiness.get("num_failure") or 0) != 0
            or int(readiness.get("num_success") or 0)
            != int(readiness.get("num_requests") or 0)
        ):
            raise ValueError("readiness audit is not a stable 20/20 pass")
        with self._lock:
            self._state = "CLOSED"
            self._consecutive_failures = 0
            self._events.append(
                {
                    "timestamp_utc": utc_now(),
                    "status": "CIRCUIT_CLOSED_AFTER_READINESS",
                    "readiness_result_sha256": readiness.get("result_sha256"),
                }
            )
            self._write_audit_unlocked()


__all__ = [
    "CircuitBreakerConfig",
    "EndpointCircuitBreaker",
    "EndpointCircuitOpen",
]
