"""Small independent Chrome-trace fixtures; timestamps are microseconds."""


def gpu(ts=1000, dur=4000, name="gemm", pid=0, device=0, cat="kernel", **extra):
    return {"ph": "X", "cat": cat, "name": name, "pid": pid, "tid": 7,
            "ts": ts, "dur": dur, "args": {"device": device}, **extra}


def document(*events):
    return {"displayTimeUnit": "ms", "traceEvents": list(events)}
