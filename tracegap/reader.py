"""Read complete GPU activity events from Chrome JSON traces."""

from __future__ import annotations

import gzip
import io
import json
import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


class TraceError(ValueError):
    """An input cannot be analyzed; code is exposed in CLI JSON diagnostics."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class Activity:
    pid: str
    device: str
    name: str
    category: str
    start_us: Fraction
    end_us: Fraction


@dataclass
class Trace:
    activities: list[Activity]
    origin_us: Fraction
    span_us: Fraction
    ignored_events: int
    zero_duration_events: int


def _number(value, field: str, index: int) -> Fraction:
    if (type(value) not in (int, float, Decimal)
            or (isinstance(value, float) and not math.isfinite(value))
            or (isinstance(value, Decimal) and not value.is_finite())):
        raise TraceError("invalid_event", f"event {index}: {field} must be a finite number")
    # json.load uses Decimal; str also gives intuitive values for API float input.
    return Fraction(str(value))


def _label(value, field: str, index: int) -> str:
    if type(value) not in (int, str) or value == "":
        raise TraceError("invalid_event", f"event {index}: {field} must be an integer or nonempty string")
    label = str(value)
    try:
        label.encode("utf-8")
    except UnicodeError as exc:
        raise TraceError("invalid_event", f"event {index}: invalid Unicode in {field}") from exc
    return label


def parse_trace(document) -> Trace:
    """Select complete kernel/copy/memset events and rebase exact times to zero.

    `displayTimeUnit` affects presentation in trace viewers, not the microsecond
    units of Chrome JSON timestamps/durations.
    """
    events = document.get("traceEvents") if isinstance(document, dict) else document
    if not isinstance(events, list):
        raise TraceError("invalid_trace", "expected an event array or an object with a traceEvents array")
    activities = []
    ignored = zero = 0
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise TraceError("invalid_event", f"event {index}: expected an object")
        category = event.get("cat", "")
        if not isinstance(category, str):
            raise TraceError("invalid_event", f"event {index}: cat must be a string")
        kinds = GPU_CATEGORIES.intersection(part.strip() for part in category.split(","))
        if not kinds:
            ignored += 1
            continue
        if len(kinds) != 1:
            raise TraceError("invalid_event", f"event {index}: multiple GPU activity categories")
        if event.get("ph") != "X":
            raise TraceError("unsupported_phase", f"event {index}: GPU phase {event.get('ph')!r} "
                             "is unsupported; export complete (ph='X') events")
        name = event.get("name")
        if not isinstance(name, str):
            raise TraceError("invalid_event", f"event {index}: name must be a string")
        try:
            name.encode("utf-8")
        except UnicodeError as exc:
            raise TraceError("invalid_event", f"event {index}: invalid Unicode in name") from exc
        pid = _label(event.get("pid"), "pid", index)
        args = event.get("args", {})
        if not isinstance(args, dict):
            raise TraceError("invalid_event", f"event {index}: args must be an object")
        device = _label(args.get("device", pid), "args.device", index)
        start = _number(event.get("ts"), "ts", index)
        duration = _number(event.get("dur"), "dur", index)
        if duration < 0:
            raise TraceError("invalid_event", f"event {index}: dur must be nonnegative")
        if duration == 0:
            zero += 1
            continue
        activities.append(Activity(pid, device, name, next(iter(kinds)), start, start + duration))
    if not activities:
        raise TraceError("no_gpu_events", "no positive-duration kernel, gpu_memcpy, or gpu_memset "
                         "events found; export a trace with GPU activity enabled")
    origin = min(event.start_us for event in activities)
    end = max(event.end_us for event in activities)
    for event in activities:
        event.start_us -= origin
        event.end_us -= origin
    return Trace(activities, origin, end - origin, ignored, zero)


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError(f"nonstandard JSON constant {value}")


def read_document(stream):
    """Decode JSON without first rounding decimal timestamps to binary floats."""
    try:
        return json.load(stream, parse_float=Decimal, parse_constant=_reject_constant,
                         object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise TraceError("invalid_trace", str(exc)) from exc


def load_trace(path: str | Path) -> Trace:
    """Read local JSON or gzip JSON; never alter the source file."""
    path = Path(path).expanduser()
    try:
        if not path.is_file():
            raise OSError("expected an existing regular trace file")
        with path.open("rb") as raw:
            magic = raw.read(2)
            raw.seek(0)
            source = gzip.GzipFile(fileobj=raw) if magic == b"\x1f\x8b" or path.suffix == ".gz" else raw
            with io.TextIOWrapper(source, encoding="utf-8-sig") as text:
                document = read_document(text)
    except (OSError, EOFError) as exc:
        raise TraceError("read_error", f"{path}: {exc}") from exc
    return parse_trace(document)
