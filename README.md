# tracegap

[![CI](https://github.com/TengRrui/tracegap/actions/workflows/ci.yml/badge.svg)](https://github.com/TengRrui/tracegap/actions/workflows/ci.yml)

**Find recorded GPU activity, overlap, and the gaps between operations in a PyTorch trace.**

`tracegap` is a read-only Python CLI for local Chrome JSON profiler traces. It
works over SSH or on a laptop without PyTorch, CUDA, a GPU, a browser, or an API
key. There are **zero runtime dependencies**.

## The problem

A profiler's summed kernel durations do not tell you how much wall time had
recorded GPU work: operations on different streams can overlap. A long capture
can also hide large uncovered intervals among thousands of events. Opening a
full timeline viewer is not always convenient on a remote machine.

`tracegap` gives a small per-device report: coverage without double counting,
concurrency, the longest gaps with offsets, and activities ranked by summed
duration. Narrow the time window to a steady-state region before investigating
those offsets in a timeline viewer.

## Install

Requires **Python 3.10+**. Install the tagged GitHub version:

```sh
python -m pip install "git+https://github.com/TengRrui/tracegap.git@v0.1.0"
tracegap --version
```

Alternatively, clone the repository and install locally:

```sh
git clone https://github.com/TengRrui/tracegap.git
cd tracegap
python -m pip install .
```

`python -m tracegap` also works directly from the checkout without installation.
This is a GitHub release; the commands do not assume a PyPI publication.

## Quick start, without a trace download

The installed package includes a tiny synthetic two-device trace:

```sh
tracegap --demo --top 0
```

Actual example output:

```text
tracegap: '<demo>'
Window: 0.000..10.000 ms (10.000 ms)
Device pid='0' id='0': 3 activities
  Coverage: 8.000 ms (80.0%) | Uncovered: 2.000 ms | Event sum: 10.000 ms
  Concurrent: 2.000 ms | Peak activities: 2 | Longest gap: 2.000 ms
Device pid='1' id='1': 3 activities
  Coverage: 4.000 ms (40.0%) | Uncovered: 6.000 ms | Event sum: 4.000 ms
  Concurrent: 0.000 ms | Peak activities: 1 | Longest gap: 4.000 ms
Ignored non-GPU events: 2 | Zero-duration GPU events: 0
Note: Coverage is for recorded kernel/copy/memset events; it is not hardware utilization or a bottleneck diagnosis.
Note: Times are milliseconds relative to the first positive-duration GPU activity; the default window excludes CPU-only preamble/tail.
```

Try details, a clipped window, and JSON:

```sh
tracegap --demo --top 2
tracegap --demo --window-ms 3:9
tracegap --demo --json
```

Device 0's largest uncovered gap is `6.000..8.000 ms`. Its event sum is 10 ms,
but only 8 ms of the window is covered because two kernels overlap for 2 ms.

## Analyze your own profiler trace

```sh
tracegap trace.json
tracegap trace.json.gz --top 10
tracegap trace.json.gz --window-ms 100:250 --json > report.json
```

The last command requires that `100:250` lies within that trace's recorded GPU
envelope. Window offsets are relative to its **first positive-duration GPU
activity**, not Unix time or the start of CPU recording. Gaps retain these same
offsets after clipping. On Windows, ordinary Windows paths work too:

```powershell
python -m tracegap 'E:\traces\inference.json.gz' --top 3
```

To capture a trace in an environment that already has PyTorch and a GPU:

```python
import torch

# model and inputs are your existing GPU workload.
with torch.profiler.profile(activities=[
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
]) as prof:
    model(inputs)
    torch.cuda.synchronize()

prof.export_chrome_trace("trace.json")
```

Capture suitable warm-up/steady-state iterations for your workload. A
`ProfilerStep` schedule is not required by tracegap. See the
[PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html)
for recording and schedules.

## Metric definitions

Every device uses the same selected window. Device identity is the string pair
`(pid, args.device)`; if `args.device` is absent, it falls back to `pid`. This
keeps different trace processes separate instead of assuming their device
numbers refer to the same physical GPU.

| JSON field | Meaning |
| --- | --- |
| `busy_ms` | Time with at least one selected recorded GPU activity; interval union. |
| `idle_ms` | Window time with none of those activities; called **Uncovered** in text output. |
| `busy_percent` | `100 * busy_ms / window.duration_ms`. |
| `event_time_ms` | Sum of clipped activity durations; overlaps count once per activity. |
| `concurrent_ms` | Elapsed time with at least two activities; triple overlap still counts once. |
| `peak_concurrency` | Maximum simultaneous activity count, not GPU occupancy. |
| `longest_gap_ms` | Largest uncovered interval, including leading/trailing edges. |
| `activity_count` | Number of positive-overlap events after window clipping. |

`gaps` lists the largest uncovered intervals with `start_ms`, `end_ms`, and
`duration_ms`. `activities` groups by name/category and ranks summed
`event_time_ms`; `calls` counts events overlapping the selected window.
`categories` provides the corresponding per-category sums. `--top N` limits
both detail lists, not the summary metrics. Text abbreviates names longer than
100 characters; JSON retains full names. Text times are rounded to three
decimal places; JSON preserves the floating-point millisecond results.

## Supported inputs and JSON output

- A root event array or an object containing a `traceEvents` array.
- Plain UTF-8 JSON (including a BOM), or gzip JSON, detected by magic bytes or
  `.gz` suffix. Files are never rewritten.
- Complete duration events (`ph: "X"`) with category tokens `kernel`,
  `gpu_memcpy`, or `gpu_memset`. Other categories are ignored and counted.
- Zero-duration GPU events are counted separately and excluded from timings.
- Chrome JSON timestamps/durations are **microseconds** even when
  `displayTimeUnit` is `ms` or `ns`. Decimal timestamps and interval arithmetic
  stay exact internally; output is converted to floating-point milliseconds.

Supported GPU categories with B/E or other phases are rejected explicitly.
Malformed recognized GPU events, duplicate JSON keys, non-finite numbers,
negative durations, CPU-only traces, and windows outside the recorded envelope
produce diagnostics rather than a misleading successful report.

`--json` writes one object to stdout. A successful report has `ok: true`,
`source`, `window`, `devices`, `ignored_events`, `zero_duration_events`, and
`notes`. For example, `devices[0]` in the demo includes:

```json
{
  "pid": "0",
  "device": "0",
  "busy_ms": 8.0,
  "idle_ms": 2.0,
  "event_time_ms": 10.0,
  "concurrent_ms": 2.0,
  "peak_concurrency": 2
}
```

Analysis/input failures emit `{"ok": false, "error": {"code": "...", "message": "..."}}`
in JSON mode, or a stderr diagnostic in text mode. Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Report produced, regardless of how much uncovered time exists. |
| `1` | Input or analysis error. |
| `2` | Invalid CLI arguments; argparse writes usage to stderr. |

## Scope and limits

A gap is an absence of **selected recorded events**, not proof that the GPU was
physically idle. Other processes, missing instrumentation, profiler collection
boundaries, or other event categories may account for it. Coverage is not SM
utilization, throughput, a CPU bottleneck diagnosis, or a recommendation to
change batch size. The default envelope can include setup, pauses, or teardown
between GPU operations; choose a representative subwindow when needed.

The tool does not collect traces, join distributed ranks, analyze memory,
resolve CPU/kernel causality, or support Perfetto protobuf, Nsight captures,
zstd, or arbitrary Chrome categories. Use a full timeline viewer or
[HolisticTraceAnalysis](https://github.com/facebookresearch/HolisticTraceAnalysis)
for broader investigations. JSON is loaded into memory; endpoint sorting is
O(n log n). Very large multi-gigabyte captures need a different workflow.

Format references: [Perfetto's Chrome JSON guide](https://perfetto.dev/docs/getting-started/other-formats)
and [Kineto's JSON exporter](https://github.com/pytorch/kineto/blob/main/libkineto/src/output_json.cpp).

## Development

```sh
python -m unittest discover -v
python -m pip install build ruff
python -m ruff check .
python -m build
```

Tests use real temporary files, exact hand-calculated intervals, an independent
discrete occupancy oracle, and CLI subprocesses. A pinned public
[HTA AlexNet trace](https://github.com/facebookresearch/HolisticTraceAnalysis/blob/eb0b6d6ee288641abe06174216db920909a196b3/tests/data/critical_path/alexnet/benchmark_result_2869224_1695835535_trace.json.gz)
was also analyzed during development; that third-party capture is not bundled.
CI builds and installs the wheel and runs the bundled demo outside the checkout
on Linux, Windows and macOS. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE).
