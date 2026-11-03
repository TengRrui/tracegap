# Contributing

Report bugs with a minimal trace, exact command, diagnostic, Python version and
operating system. Avoid uploading private model names, filesystem paths or
application traces; a few synthetic events are usually enough to reproduce a
timing/accounting bug.

```sh
python -m venv .venv
# Activate using your shell's normal command, or invoke its Python directly.
python -m pip install -e . build ruff
python -m unittest discover -v
python -m ruff check .
python -m build
```

Write a failing regression before a bug fix. Keep expected times independent of
the implementation and preserve exact timestamp/interval arithmetic. Distinguish
event sums, interval union, and elapsed concurrency. Include clipping, touching
endpoints or device grouping when a change affects those behaviors.

Keep runtime dependencies empty. Analysis should remain read-only. New metrics
must define their denominator and supported event scope and must not claim
hardware utilization or a bottleneck cause without the necessary evidence.

Layout:
- `tracegap/reader.py`: JSON/gzip decoding and recognized GPU activities.
- `tracegap/analysis.py`: per-device clipping and endpoint sweep.
- `tracegap/cli.py`: arguments, structured errors, text/JSON presentation.
- `tracegap/demo.json`: the small hand-authored synthetic demonstration.
- `tests/`: reader, exact accounting/oracle, and real CLI processes.

Submit a focused pull request describing the problem and checks actually run.
The project is licensed under MIT.
