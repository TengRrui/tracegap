import gzip
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from tests.helpers import document, gpu
from tracegap.reader import TraceError, load_trace, parse_trace


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_event_array_and_object_have_same_microsecond_envelope(self):
        for source in ([gpu()], document(gpu())):
            with self.subTest(source=type(source)):
                trace = parse_trace(source)
                self.assertEqual(trace.span_us, 4000)
                self.assertEqual(trace.activities[0].start_us, 0)
                self.assertEqual(trace.activities[0].end_us, 4000)
                self.assertEqual(trace.origin_us, 1000)

    def test_cpu_spans_do_not_enlarge_gpu_window(self):
        trace = parse_trace(document(gpu(), gpu(0, 90000, cat="cpu_op"),
                                     {"ph": "M", "name": "process_name", "pid": 0}))
        self.assertEqual(trace.span_us, 4000)
        self.assertEqual(trace.ignored_events, 2)

    def test_decimal_epoch_timestamps_keep_submicrosecond_spans(self):
        path = self.root / "trace.json"
        path.write_text('''{"traceEvents":[
          {"ph":"X","cat":"kernel","name":"a","pid":0,
           "ts":1000000000000000.001,"dur":0.003},
          {"ph":"X","cat":"kernel","name":"b","pid":0,
           "ts":1000000000000000.005,"dur":0.001}
        ]}''', encoding="utf-8")
        trace = load_trace(path)
        self.assertEqual(trace.span_us, Fraction(5, 1000))
        self.assertEqual(trace.activities[0].end_us, Fraction(3, 1000))
        self.assertEqual(trace.activities[1].start_us, Fraction(4, 1000))

    def test_reads_gzip_by_magic_even_without_gz_extension(self):
        path = self.root / "trace.data"
        path.write_bytes(gzip.compress(json.dumps(document(gpu())).encode()))
        self.assertEqual(load_trace(path).span_us, 4000)

    def test_reads_utf8_bom_and_unicode_name(self):
        path = self.root / "模型.json"
        path.write_text(json.dumps(document(gpu(name="矩阵乘法")), ensure_ascii=False),
                        encoding="utf-8-sig")
        self.assertEqual(load_trace(path).activities[0].name, "矩阵乘法")

    def test_recognizes_category_tokens_and_copies(self):
        trace = parse_trace(document(gpu(cat="other, kernel"),
                                     gpu(5000, 1000, cat="gpu_memcpy"),
                                     gpu(6000, 1000, cat="gpu_memset")))
        self.assertEqual([e.category for e in trace.activities],
                         ["kernel", "gpu_memcpy", "gpu_memset"])

    def test_missing_device_uses_pid_without_merging_other_processes(self):
        event = gpu(pid=7)
        del event["args"]
        trace = parse_trace(document(event, gpu(pid=8, device=7)))
        self.assertEqual([(e.pid, e.device) for e in trace.activities], [("7", "7"), ("8", "7")])

    def test_zero_duration_gpu_records_are_counted_but_not_measured(self):
        trace = parse_trace(document(gpu(0, 0), gpu(1000, 1000)))
        self.assertEqual(trace.zero_duration_events, 1)
        self.assertEqual(trace.span_us, 1000)
        self.assertEqual(len(trace.activities), 1)

    def test_cpu_only_empty_and_zero_only_traces_explain_missing_gpu_events(self):
        for events in ([], [gpu(cat="cpu_op")], [gpu(dur=0)]):
            with self.subTest(events=events):
                with self.assertRaises(TraceError) as caught:
                    parse_trace(events)
                self.assertEqual(caught.exception.code, "no_gpu_events")

    def test_gpu_begin_end_phases_are_rejected_instead_of_ignored(self):
        for phase in ("B", "E", "b", "e"):
            with self.subTest(phase=phase):
                with self.assertRaises(TraceError) as caught:
                    parse_trace(document(gpu(), gpu(ph=phase)))
                self.assertEqual(caught.exception.code, "unsupported_phase")

    def test_malformed_recognized_gpu_events_fail_instead_of_underreporting(self):
        patches = ({"ts": True}, {"dur": -1}, {"dur": "4000"}, {"dur": float("inf")},
                   {"ts": float("nan")}, {"name": []}, {"args": []},
                   {"args": {"device": []}}, {"pid": None}, {"cat": 7})
        for change in patches:
            with self.subTest(change=change):
                with self.assertRaises(TraceError) as caught:
                    parse_trace(document(gpu(**change)))
                self.assertEqual(caught.exception.code, "invalid_event")

    def test_malformed_root_or_event_is_rejected(self):
        for source in ({}, {"traceEvents": {}}, 7, ["not an event"]):
            with self.subTest(source=source), self.assertRaises(TraceError):
                parse_trace(source)

    def test_missing_file_and_corrupt_gzip_have_read_diagnostics(self):
        path = self.root / "bad.json.gz"
        path.write_bytes(b"\x1f\x8btruncated")
        for candidate in (path, self.root / "missing.json", self.root):
            with self.subTest(path=candidate):
                with self.assertRaises(TraceError) as caught:
                    load_trace(candidate)
                self.assertEqual(caught.exception.code, "read_error")

    def test_invalid_json_duplicate_keys_and_nonfinite_literals_are_rejected(self):
        path = self.root / "bad.json"
        for source in ('{', '{"traceEvents":[],"traceEvents":[]}', '{"traceEvents":NaN}'):
            with self.subTest(source=source):
                path.write_text(source, encoding="utf-8")
                with self.assertRaises(TraceError) as caught:
                    load_trace(path)
                self.assertEqual(caught.exception.code, "invalid_trace")


if __name__ == "__main__":
    unittest.main()
