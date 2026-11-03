import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.helpers import document, gpu

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_cli(self, *args, env=None):
        return subprocess.run([sys.executable, "-m", "tracegap", *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                              check=False, env=env)

    def test_bundled_demo_has_independently_known_device_metrics(self):
        result = self.run_cli("--demo", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertTrue(data["ok"])
        self.assertEqual(data["window"]["duration_ms"], 10)
        self.assertEqual([d["busy_ms"] for d in data["devices"]], [8, 4])
        self.assertEqual([d["longest_gap_ms"] for d in data["devices"]], [2, 4])
        self.assertEqual(result.stderr, "")

    def test_text_report_labels_recorded_coverage_and_gaps(self):
        result = self.run_cli("--demo", "--top", 1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("80.0%", result.stdout)
        self.assertIn("6.000..8.000", result.stdout)
        self.assertIn("hardware utilization", result.stdout)
        self.assertIn("gemm", result.stdout)

    def test_reads_real_file_and_gzip_and_emits_source(self):
        for filename in ("trace.json", "trace.json.gz"):
            with self.subTest(filename=filename):
                path = self.root / filename
                raw = json.dumps(document(gpu())).encode()
                path.write_bytes(gzip.compress(raw) if filename.endswith("gz") else raw)
                result = self.run_cli(path, "--json")
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout)
                self.assertEqual(data["source"], str(path))
                self.assertEqual(data["devices"][0]["busy_ms"], 4)

    def test_window_option_clips_report(self):
        result = self.run_cli("--demo", "--json", "--window-ms", "3:9")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["window"]["duration_ms"], 6)
        self.assertEqual(data["devices"][0]["busy_ms"], 4)

    def test_missing_input_returns_json_error_and_exit_one(self):
        result = self.run_cli(self.root / "missing.json", "--json")
        self.assertEqual(result.returncode, 1)
        data = json.loads(result.stdout)
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "read_error")
        self.assertEqual(result.stderr, "")

    def test_cpu_only_trace_returns_actionable_text_error(self):
        path = self.root / "cpu.json"
        path.write_text(json.dumps(document(gpu(cat="cpu_op"))), encoding="utf-8")
        result = self.run_cli(path)
        self.assertEqual(result.returncode, 1)
        self.assertIn("no_gpu_events", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_window_outside_recorded_envelope_is_analysis_error(self):
        result = self.run_cli("--demo", "--json", "--window-ms", "0:11")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["error"]["code"], "invalid_window")

    def test_bad_arguments_exit_two(self):
        for args in ((), ("--demo", "--top", "-1"), ("--demo", "x.json"),
                     ("--demo", "--window-ms", "nan:2"),
                     ("--demo", "--window-ms", "2:1")):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_top_zero_hides_details_but_not_longest_gap(self):
        result = self.run_cli("--demo", "--json", "--top", 0)
        self.assertEqual(result.returncode, 0, result.stderr)
        device = json.loads(result.stdout)["devices"][0]
        self.assertEqual(device["gaps"], [])
        self.assertEqual(device["activities"], [])
        self.assertEqual(device["longest_gap_ms"], 2)

    def test_unicode_paths_work_in_ascii_redirected_output(self):
        path = self.root / "模型.json"
        path.write_text(json.dumps(document(gpu(name="矩阵乘法"))), encoding="utf-8")
        result = self.run_cli(path, env={**os.environ, "PYTHONIOENCODING": "ascii", "PYTHONUTF8": "0"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

    def test_help_and_version_need_no_trace(self):
        for option, expected in (("--help", "--window-ms"), ("--version", "0.1.0")):
            result = self.run_cli(option)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
