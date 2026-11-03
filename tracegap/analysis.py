"""Exact interval accounting across recorded GPU streams."""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction

from .reader import Activity, Trace, TraceError


def _ms(microseconds: Fraction) -> float:
    try:
        value = float(microseconds / 1000)
    except OverflowError as exc:
        raise TraceError("numeric_range", "trace time is too large to represent in JSON milliseconds") from exc
    if not math.isfinite(value):
        raise TraceError("numeric_range", "trace time is too large to represent in JSON milliseconds")
    return value


def _window(trace: Trace, window_ms) -> tuple[Fraction, Fraction]:
    if window_ms is None:
        return Fraction(0), trace.span_us
    try:
        start, end = (Fraction(str(value)) * 1000 for value in window_ms)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise TraceError("invalid_window", "window must contain two finite millisecond offsets") from exc
    if not 0 <= start < end <= trace.span_us:
        raise TraceError("invalid_window", "window must satisfy 0 <= START < END <= "
                         f"{_ms(trace.span_us):g} ms (the recorded GPU envelope)")
    return start, end


def _sweep(activities: list[Activity], start: Fraction, end: Fraction):
    changes = defaultdict(int, {start: 0, end: 0})
    for event in activities:
        changes[event.start_us] += 1
        changes[event.end_us] -= 1
    active = peak = 0
    busy = concurrent = Fraction(0)
    previous = start
    gaps = []
    for point, change in sorted(changes.items()):
        duration = point - previous
        if duration > 0:
            if active:
                busy += duration
                if active >= 2:
                    concurrent += duration
            elif gaps and gaps[-1][1] == previous:
                gaps[-1] = (gaps[-1][0], point)
            else:
                gaps.append((previous, point))
        # Aggregate same-time starts/ends before updating the concurrency peak.
        active += change
        peak = max(peak, active)
        previous = point
    return busy, concurrent, peak, gaps


def _device_report(pid: str, device: str, activities: list[Activity],
                   start: Fraction, end: Fraction, top: int) -> dict:
    clipped = [Activity(e.pid, e.device, e.name, e.category,
                        max(start, e.start_us), min(end, e.end_us))
               for e in activities if e.start_us < end and e.end_us > start]
    busy, concurrent, peak, gaps = _sweep(clipped, start, end)
    named = {}
    categories = {}
    total = Fraction(0)
    for event in clipped:
        duration = event.end_us - event.start_us
        total += duration
        for mapping, key in ((named, (event.name, event.category)), (categories, event.category)):
            group = mapping.setdefault(key, {"calls": 0, "duration": Fraction(0)})
            group["calls"] += 1
            group["duration"] += duration
    ranked = sorted(named.items(), key=lambda item: (-item[1]["duration"], item[0]))
    largest_gaps = sorted(gaps, key=lambda gap: (-(gap[1] - gap[0]), gap[0]))[:top]
    return {
        "pid": pid, "device": device, "activity_count": len(clipped),
        "busy_ms": _ms(busy), "idle_ms": _ms(end - start - busy),
        "busy_percent": float(100 * busy / (end - start)),
        "event_time_ms": _ms(total), "concurrent_ms": _ms(concurrent),
        "peak_concurrency": peak,
        "longest_gap_ms": _ms(max((b - a for a, b in gaps), default=Fraction(0))),
        "gaps": [{"start_ms": _ms(a), "end_ms": _ms(b), "duration_ms": _ms(b - a)}
                 for a, b in largest_gaps],
        "activities": [{"name": name, "category": kind, "calls": group["calls"],
                        "event_time_ms": _ms(group["duration"])}
                       for (name, kind), group in ranked[:top]],
        "categories": {kind: {"calls": group["calls"], "event_time_ms": _ms(group["duration"])}
                       for kind, group in sorted(categories.items())},
    }


def analyze(trace: Trace, window_ms=None, top: int = 5) -> dict:
    """Report per-device coverage, uncovered gaps and elapsed concurrency.

    Window offsets and output times are milliseconds relative to the first
    positive-duration GPU activity. Event time is a sum; busy time is a union.
    """
    if type(top) is not int or top < 0:
        raise ValueError("top must be a nonnegative integer")
    start, end = _window(trace, window_ms)
    groups = defaultdict(list)
    for event in trace.activities:
        groups[event.pid, event.device].append(event)
    return {
        "ok": True,
        "window": {"start_ms": _ms(start), "end_ms": _ms(end), "duration_ms": _ms(end - start)},
        "devices": [_device_report(pid, device, events, start, end, top)
                    for (pid, device), events in sorted(groups.items())],
        "ignored_events": trace.ignored_events,
        "zero_duration_events": trace.zero_duration_events,
        "notes": [
            "Coverage is for recorded kernel/copy/memset events; it is not hardware utilization or a bottleneck diagnosis.",
            "Times are milliseconds relative to the first positive-duration GPU activity; the default window excludes CPU-only preamble/tail.",
        ],
    }
