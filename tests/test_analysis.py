import itertools
import random
import unittest

from tests.helpers import document, gpu
from tracegap.analysis import analyze
from tracegap.reader import TraceError, parse_trace


def three_spans():
    return document(gpu(0, 4000, "gemm"), gpu(2000, 4000, "relu"),
                    gpu(8000, 2000, "copy", cat="gpu_memcpy"))


class AnalysisTests(unittest.TestCase):
    def test_overlapping_streams_are_not_double_counted_as_busy(self):
        data = analyze(parse_trace(three_spans()))
        device = data["devices"][0]
        self.assertEqual(data["window"]["duration_ms"], 10)
        self.assertEqual(device["busy_ms"], 8)
        self.assertEqual(device["idle_ms"], 2)
        self.assertEqual(device["event_time_ms"], 10)
        self.assertEqual(device["concurrent_ms"], 2)
        self.assertEqual(device["peak_concurrency"], 2)
        self.assertEqual(device["busy_percent"], 80)
        self.assertEqual(device["gaps"], [{"start_ms": 6.0, "end_ms": 8.0, "duration_ms": 2.0}])

    def test_triple_overlap_reports_elapsed_concurrency_not_extra_event_time(self):
        trace = parse_trace(document(gpu(0, 10000), gpu(2000, 6000), gpu(4000, 2000)))
        device = analyze(trace)["devices"][0]
        self.assertEqual(device["event_time_ms"], 18)
        self.assertEqual(device["busy_ms"], 10)
        self.assertEqual(device["concurrent_ms"], 6)
        self.assertEqual(device["peak_concurrency"], 3)

    def test_touching_endpoints_are_not_concurrent_or_gaps(self):
        trace = parse_trace(document(gpu(0, 4000), gpu(4000, 2000)))
        device = analyze(trace)["devices"][0]
        self.assertEqual(device["busy_ms"], 6)
        self.assertEqual(device["concurrent_ms"], 0)
        self.assertEqual(device["peak_concurrency"], 1)
        self.assertEqual(device["gaps"], [])

    def test_window_clips_crossing_spans_and_preserves_common_offsets(self):
        data = analyze(parse_trace(three_spans()), window_ms=(3, 9))
        self.assertEqual(data["window"], {"start_ms": 3.0, "end_ms": 9.0, "duration_ms": 6.0})
        device = data["devices"][0]
        self.assertEqual(device["busy_ms"], 4)
        self.assertEqual(device["event_time_ms"], 5)
        self.assertEqual(device["concurrent_ms"], 1)
        self.assertEqual(device["activity_count"], 3)
        self.assertEqual(device["activities"][0]["name"], "relu")
        self.assertEqual(device["activities"][0]["event_time_ms"], 3)

    def test_fully_uncovered_subwindow_reports_zero_activity(self):
        device = analyze(parse_trace(three_spans()), window_ms=(6, 8))["devices"][0]
        self.assertEqual(device["activity_count"], 0)
        self.assertEqual(device["busy_ms"], 0)
        self.assertEqual(device["idle_ms"], 2)
        self.assertEqual(device["peak_concurrency"], 0)
        self.assertEqual(device["longest_gap_ms"], 2)

    def test_devices_share_window_but_leading_trailing_gaps_are_independent(self):
        trace = parse_trace(document(gpu(0, 2000, pid=0, device=0),
                                     gpu(8000, 2000, pid=1, device=1)))
        devices = analyze(trace)["devices"]
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]["gaps"][0], {"start_ms": 2.0, "end_ms": 10.0, "duration_ms": 8.0})
        self.assertEqual(devices[1]["gaps"][0], {"start_ms": 0.0, "end_ms": 8.0, "duration_ms": 8.0})

    def test_same_device_number_in_different_processes_is_not_merged(self):
        trace = parse_trace(document(gpu(0, 4000, pid=7), gpu(0, 4000, pid=8)))
        devices = analyze(trace)["devices"]
        self.assertEqual([(d["pid"], d["device"]) for d in devices], [("7", "0"), ("8", "0")])
        self.assertEqual([d["peak_concurrency"] for d in devices], [1, 1])

    def test_activity_ranking_combines_calls_by_name_and_category(self):
        trace = parse_trace(document(gpu(0, 2000, "same"), gpu(4000, 3000, "same"),
                                     gpu(9000, 1000, "same", cat="gpu_memcpy")))
        device = analyze(trace)["devices"][0]
        self.assertEqual(device["activities"][0],
                         {"name": "same", "category": "kernel", "calls": 2, "event_time_ms": 5.0})
        self.assertEqual(device["categories"]["gpu_memcpy"]["event_time_ms"], 1)

    def test_top_zero_hides_lists_without_changing_metrics(self):
        device = analyze(parse_trace(three_spans()), top=0)["devices"][0]
        self.assertEqual(device["gaps"], [])
        self.assertEqual(device["activities"], [])
        self.assertEqual(device["longest_gap_ms"], 2)
        self.assertEqual(device["busy_ms"], 8)

    def test_event_order_does_not_change_results(self):
        events = three_spans()["traceEvents"]
        expected = analyze(parse_trace(document(*events)))
        for permutation in itertools.permutations(events):
            self.assertEqual(analyze(parse_trace(document(*permutation))), expected)

    def test_invalid_windows_are_rejected_without_silently_extrapolating(self):
        trace = parse_trace(three_spans())
        for window in ((-1, 2), (1, 1), (0, 11), (2, 1), ("nan", 1)):
            with self.subTest(window=window):
                with self.assertRaises(TraceError) as caught:
                    analyze(trace, window_ms=window)
                self.assertEqual(caught.exception.code, "invalid_window")

    def test_fractional_durations_are_not_rounded_to_zero(self):
        trace = parse_trace(document(gpu(10**16, 0.003)))
        device = analyze(trace)["devices"][0]
        self.assertEqual(device["busy_ms"], 0.000003)
        self.assertEqual(device["busy_percent"], 100)

    def test_sweep_agrees_with_independent_discrete_occupancy_oracle(self):
        rng = random.Random(20261103)
        for _ in range(80):
            intervals = [(start, rng.randint(start + 1, 12))
                         for start in (rng.randrange(10) for _ in range(rng.randint(1, 8)))]
            origin = min(start for start, end in intervals)
            end = max(end for start, end in intervals)
            occupancy = [sum(start <= tick < stop for start, stop in intervals)
                         for tick in range(origin, end)]
            trace = parse_trace(document(*(gpu(start * 1000, (stop - start) * 1000)
                                           for start, stop in intervals)))
            device = analyze(trace, top=0)["devices"][0]
            self.assertEqual(device["busy_ms"], sum(count > 0 for count in occupancy))
            self.assertEqual(device["concurrent_ms"], sum(count >= 2 for count in occupancy))
            self.assertEqual(device["event_time_ms"], sum(occupancy))
            self.assertEqual(device["peak_concurrency"], max(occupancy))


if __name__ == "__main__":
    unittest.main()
