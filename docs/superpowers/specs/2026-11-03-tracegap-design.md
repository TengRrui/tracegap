# tracegap design

## Purpose and chosen approach

Summarize recorded GPU activity and gaps from a PyTorch/Kineto Chrome JSON
trace in a terminal, without installing PyTorch, a GPU driver, or a GUI.
An environment-comparison CLI and a research-run inventory CLI were considered;
tracegap has the clearest connection to inference-performance investigations.

Implement the interval accounting directly with Python's standard library.
Using a full trace-analysis framework would add substantial dependencies for a
small report; guessing bottleneck causes from heuristics would overstate the
evidence. This tool reports timings and where to inspect next.

## Inputs and semantics

- Python >=3.10; zero runtime dependencies; MIT; version 0.1.0.
- Read local JSON/JSON.gz, accepting a root event array or `traceEvents` array.
- Account for positive-duration complete (`ph: X`) events with category tokens
  `kernel`, `gpu_memcpy`, or `gpu_memset`. Reject unsupported phases in those
  categories instead of silently understating activity. Other categories are
  ignored and counted. Count zero-duration GPU events separately.
- Chrome `ts` and `dur` are microseconds regardless of `displayTimeUnit`.
  Parse decimal timestamps exactly and use rational arithmetic for interval
  operations, avoiding cancellation with epoch-sized timestamps.
- Group devices by the string pair `(pid, args.device)`, falling back to `pid`
  when `args.device` is absent. Do not infer hardware identity across processes.
- Default window is the common first-to-last positive GPU activity envelope.
  `--window-ms START:END` selects a subwindow relative to that first activity;
  reject windows outside the recorded envelope. Clip crossing events.
- Per device: report activity union (busy), uncovered time, busy percentage,
  summed event time, elapsed time with >=2 overlapping activities, peak
  concurrency, largest uncovered gaps, and activities ranked by summed time.
  Include leading/trailing gaps within the selected common window.
- A gap means no selected recorded GPU activity. It is not proof of hardware
  idleness, low SM utilization, or any particular bottleneck. No cross-device
  utilization sum or performance threshold is emitted.

## Structure and interfaces

- `tracegap/reader.py`: `parse_trace(document) -> Trace`,
  `load_trace(path) -> Trace`, `TraceError(code, message)`. Validate recognized
  records, finite numeric times, duration sign, names and device identifiers.
- `tracegap/analysis.py`: `analyze(trace, window_ms=None, top=5) -> dict`.
  A sweep over aggregated endpoints computes union, overlap and gaps without
  counting touching endpoints as simultaneous work. Sort results deterministically.
- `tracegap/cli.py`: installed `tracegap` and `python -m tracegap`; text or JSON,
  `--top N`, `--window-ms`, `--demo` (a bundled read-only synthetic example).
  Exit 0 for a report, 1 for input/analysis errors, 2 for invalid CLI arguments.
- Package the small demo as data so a wheel installation can be tried offline.
  No model execution, downloads, repairs, persistent state, hashes or gates.

## Validation and publication

Write stdlib unittest tests before each implementation. Use independently
calculated overlapping/nested/touching intervals, multiple devices, clipped
windows, decimal/large timestamps, empty and malformed input, gzip, and real
CLI subprocesses. Validate small generated interval sets against a discrete
occupancy oracle. Run the CLI on a pinned public HolisticTraceAnalysis sample
downloaded only to E:\codex\download; do not redistribute that trace.

Ship English README with definitions, installation, runnable examples/output,
and scope limits. Build/install the wheel in an isolated environment and run
outside the checkout. CI covers Linux, Windows, macOS and multiple Python
versions. Review code and tracked files before GitHub CLI publication to a new
public `TengRrui/tracegap` repository; leave existing repositories untouched.

JSON is loaded into memory, and endpoint sorting is O(n log n); this is a small
trace triage tool, not a streaming engine for multi-gigabyte captures.
