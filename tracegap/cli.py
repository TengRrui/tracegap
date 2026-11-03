"""Text/JSON CLI for offline trace triage."""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from importlib import resources

from . import __version__
from .analysis import analyze
from .reader import TraceError, load_trace, parse_trace, read_document


def _top(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def _window(value: str):
    try:
        start, end = (Fraction(part) for part in value.split(":"))
        if not 0 <= start < end:
            raise ValueError("out of order")
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("use finite offsets START:END with 0 <= START < END") from exc
    return start, end


def _name(value: str) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= 100 else rendered[:97] + "..."


def render_text(data: dict) -> str:
    window = data["window"]
    lines = [f"tracegap: {data['source']!r}",
             (f"Window: {window['start_ms']:.3f}..{window['end_ms']:.3f} ms "
              f"({window['duration_ms']:.3f} ms)")]
    for device in data["devices"]:
        lines.extend([
            f"Device pid={device['pid']!r} id={device['device']!r}: {device['activity_count']} activities",
            (f"  Coverage: {device['busy_ms']:.3f} ms ({device['busy_percent']:.1f}%) | "
             f"Uncovered: {device['idle_ms']:.3f} ms | Event sum: {device['event_time_ms']:.3f} ms"),
            (f"  Concurrent: {device['concurrent_ms']:.3f} ms | Peak activities: {device['peak_concurrency']} | "
             f"Longest gap: {device['longest_gap_ms']:.3f} ms"),
        ])
        if device["gaps"]:
            lines.append("  Largest uncovered gaps:")
            for gap in device["gaps"]:
                lines.append(f"    {gap['start_ms']:.3f}..{gap['end_ms']:.3f} ms "
                             f"({gap['duration_ms']:.3f} ms)")
        if device["activities"]:
            lines.append("  Top activities by summed event time:")
            for activity in device["activities"]:
                lines.append(f"    {activity['event_time_ms']:.3f} ms | {activity['calls']} calls | "
                             f"{activity['category']} | {_name(activity['name'])}")
    lines.append(f"Ignored non-GPU events: {data['ignored_events']} | "
                 f"Zero-duration GPU events: {data['zero_duration_events']}")
    lines.extend(f"Note: {note}" for note in data["notes"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tracegap", description="Summarize recorded GPU coverage, concurrency and gaps in Chrome JSON traces.",
        epilog="Exit codes: 0 = report, 1 = input/analysis error, 2 = usage error.",
    )
    parser.add_argument("trace", nargs="?", help="local JSON or gzip JSON trace")
    parser.add_argument("--demo", action="store_true", help="analyze the bundled synthetic trace")
    parser.add_argument("--json", action="store_true", help="emit a JSON report or structured error")
    parser.add_argument("--window-ms", type=_window, metavar="START:END",
                        help="subwindow relative to the first positive GPU activity (milliseconds)")
    parser.add_argument("--top", type=_top, default=5, metavar="N",
                        help="show N largest gaps and activities per device (default: 5; 0 hides lists)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    if bool(args.trace) == args.demo:
        parser.error("provide either a trace path or --demo")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    try:
        if args.demo:
            with resources.files("tracegap").joinpath("demo.json").open("r", encoding="utf-8") as stream:
                trace = parse_trace(read_document(stream))
        else:
            trace = load_trace(args.trace)
        data = analyze(trace, args.window_ms, args.top)
        data["source"] = "<demo>" if args.demo else args.trace
    except TraceError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": {"code": exc.code, "message": str(exc)}},
                             ensure_ascii=True, allow_nan=False))
        else:
            print(f"tracegap: [{exc.code}] {exc}", file=sys.stderr)
        return 1
    try:
        print(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) if args.json else render_text(data))
    except BrokenPipeError:
        return 1
    return 0
