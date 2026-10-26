"""Explicit cumulative study-clock policy; native request and scenario timeouts stay separate."""
from __future__ import annotations

from datetime import datetime, timezone
import math


def study_deadline(limits, now):
    seconds = limits["maximum_wall_seconds"]
    if seconds is None:
        if limits.get("wall_time_mode") != "unlimited":
            raise ValueError("A null study wall limit requires explicit unlimited mode")
        deadline = math.inf
    else:
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
                or limits.get("wall_time_mode") not in (None, "finite")):
            raise ValueError("Invalid finite study wall limit")
        deadline = now + seconds
    absolute = limits.get("absolute_deadline_utc")
    if absolute:
        value = datetime.fromisoformat(absolute.replace("Z", "+00:00"))
        if value.tzinfo is None:
            raise ValueError("An explicit absolute deadline needs a timezone")
        deadline = min(deadline, value.timestamp())
    return deadline


def deadline_utc(deadline):
    if deadline == math.inf:
        return None
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("Invalid study deadline")
    return datetime.fromtimestamp(deadline, timezone.utc).isoformat()
