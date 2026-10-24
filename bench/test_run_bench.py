"""Checks that the benchmark driver fails closed: `python -m unittest bench/test_run_bench.py`.

Deterministic: misbehaving workers are scripted (bench/fake_worker.py) or psutil is
replaced by a stub; no test depends on sleeps or retries to win a race.
"""

import gc
import pathlib
import sys
import tempfile
import unittest
import warnings
from unittest import mock

sys.argv = sys.argv[:1]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import psutil  # noqa: E402

import run_bench as rb  # noqa: E402

REF = [0.5, 1.0, 2.0]
FAKE = str(HERE / "fake_worker.py")


def batches(vals):
    return {"1": list(vals), "7": list(vals), "64": list(vals)}


class ParityGate(unittest.TestCase):
    def test_mismatch_nonfinite_and_length_abort(self):
        for bad in ([0.5, 1.0, 2.01], [0.5, float("nan"), 2.0], [0.5, float("inf"), 2.0], [0.5, 1.0]):
            with self.assertRaises(rb.BenchError, msg=bad):
                rb.parity_error(batches(bad), REF, "t")

    def test_every_batch_size_is_checked(self):
        with self.assertRaises(rb.BenchError):
            rb.parity_error({"1": REF, "7": [0.5, 1.0, 2.5], "64": REF}, REF, "t")

    def test_a_missing_batch_size_aborts(self):
        # The previous driver iterated only the batches a worker chose to report.
        with self.assertRaises(rb.BenchError):
            rb.parity_error({"1": REF}, REF, "t")

    def test_empty_or_non_finite_reference_aborts(self):
        with self.assertRaises(rb.BenchError):
            rb.parity_error(batches([]), [], "t")
        with self.assertRaises(rb.BenchError):
            rb.parity_error(batches(REF), [0.5, float("nan"), 2.0], "t")

    def test_within_tolerance_reports_the_largest_error(self):
        err = rb.parity_error(batches([0.5, 1.0, 2.00001]), REF, "t")
        self.assertAlmostEqual(err, 1e-5, places=9)


class ParityFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.dir.name) / "parity.json"

    def tearDown(self):
        self.dir.cleanup()

    def load(self, text, rows=3, cols=1):
        self.path.write_text(text)
        return rb.load_parity(self.path, rows, cols, "t")

    def test_valid_file(self):
        got = self.load('{"outputs_per_row": 1, "batches": {"1": [1, 2, 3], "7": [1, 2, 3], "64": [1, 2, 3]}}')
        self.assertEqual(got["64"], [1.0, 2.0, 3.0])

    def test_rejected_files(self):
        cases = {
            "malformed JSON": "{not json",
            "not an object": "[1, 2]",
            "no batches": '{"outputs_per_row": 1}',
            "wrong width": '{"outputs_per_row": 2, "batches": {"1": [1,2,3], "7": [1,2,3], "64": [1,2,3]}}',
            "missing batch": '{"outputs_per_row": 1, "batches": {"1": [1,2,3], "7": [1,2,3]}}',
            "extra batch": '{"outputs_per_row": 1, "batches": {"1": [1,2,3], "7": [1,2,3], "64": [1,2,3], "2": [1,2,3]}}',
            "empty": '{"outputs_per_row": 1, "batches": {"1": [], "7": [], "64": []}}',
            "short": '{"outputs_per_row": 1, "batches": {"1": [1,2,3], "7": [1,2], "64": [1,2,3]}}',
            "not a list": '{"outputs_per_row": 1, "batches": {"1": [1,2,3], "7": 5, "64": [1,2,3]}}',
            "non-numeric": '{"outputs_per_row": 1, "batches": {"1": [1,2,3], "7": [1,"x",3], "64": [1,2,3]}}',
            "boolean": '{"outputs_per_row": 1, "batches": {"1": [1,2,3], "7": [1,true,3], "64": [1,2,3]}}',
            "NaN": '{"outputs_per_row": 1, "batches": {"1": [1,2,3], "7": [1,NaN,3], "64": [1,2,3]}}',
            "Infinity": '{"outputs_per_row": 1, "batches": {"1": [1,2,3], "7": [1,2,3], "64": [Infinity,2,3]}}',
        }
        for name, text in cases.items():
            with self.assertRaises(rb.BenchError, msg=name):
                self.load(text)

    def test_expected_count_comes_from_rows_and_width(self):
        text = '{"outputs_per_row": 2, "batches": {"1": [1,2,3,4], "7": [1,2,3,4], "64": [1,2,3,4]}}'
        self.assertEqual(len(self.load(text, rows=2, cols=2)["1"]), 4)
        with self.assertRaises(rb.BenchError):
            self.load(text, rows=3, cols=2)
        with self.assertRaises(rb.BenchError):
            self.load(text, rows=0, cols=2)

    def test_dataset_rows_and_model_width(self):
        csv_path = pathlib.Path(self.dir.name) / "d.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n")
        self.assertEqual(rb.count_rows(csv_path), 3)
        self.assertEqual(rb.output_cols({"task": {"kind": "binary"}}), 1)
        self.assertEqual(rb.output_cols({"task": {"kind": "regression"}}), 1)
        self.assertEqual(rb.output_cols({"task": {"kind": "multiclass", "classes": 3}}), 3)


class Workers(unittest.TestCase):
    """End-to-end driver runs against scripted workers."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.parity = pathlib.Path(self.dir.name) / "parity.json"
        self.started: list[rb.Worker] = []
        real = rb.Worker
        started = self.started

        class Tracked(real):
            def __init__(self, cmd):
                super().__init__(cmd)
                started.append(self)

        patcher = mock.patch.object(rb, "Worker", Tracked)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.dir.cleanup()

    def run_mode(self, mode, ref=None):
        return rb.run_inference([sys.executable, FAKE, mode, str(self.parity), "4", "1"], self.parity, ref,
                                f"fake/{mode}", 4, 1)

    def assert_settled(self):
        for w in self.started:
            self.assertIsNotNone(w.proc.poll(), "worker still running")
            self.assertTrue(w.proc.stdout.closed and w.proc.stderr.closed and w.proc.stdin.closed)
            for t in w.threads:
                self.assertFalse(t.is_alive(), "reader thread still running")

    def test_a_valid_worker_is_recorded(self):
        rec, ref = self.run_mode("ok")
        self.assertEqual(ref, [0.5] * 4)
        self.assertEqual(rec["exit_status"], 0)
        self.assertEqual(rec["parity_max_abs_error"], 0.0)
        self.assert_settled()

    def test_every_failure_is_named_and_settles_the_worker(self):
        modes = ["missing-batch", "empty", "short", "wrong-width", "nan", "malformed-parity",
                 "malformed-results", "exit-before-parity", "nonzero-exit"]
        for mode in modes:
            with self.subTest(mode=mode):
                with self.assertRaises(rb.BenchError) as e:
                    self.run_mode(mode)
                self.assertIn("fake_worker.py", str(e.exception))
        self.assert_settled()

    def test_no_unclosed_resource_warnings(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for mode in ("ok", "malformed-parity", "nonzero-exit"):
                try:
                    self.run_mode(mode)
                except rb.BenchError:
                    pass
            self.started.clear()
            gc.collect()
        leaks = [w for w in caught if issubclass(w.category, ResourceWarning)]
        self.assertEqual(leaks, [], [str(w.message) for w in leaks])

    def test_a_mismatch_against_the_reference_aborts(self):
        with self.assertRaises(rb.BenchError):
            self.run_mode("ok", ref=[0.5, 0.5, 0.5, 0.9])
        self.assert_settled()

    def test_hung_worker_times_out_and_is_killed(self):
        with mock.patch.object(rb, "STEP_TIMEOUT_S", 1):
            w = rb.Worker([sys.executable, "-c", "import sys; sys.stdin.readline()"])
            with self.assertRaises(rb.BenchError):
                w.expect("FIRST")
        self.assert_settled()


class Monitoring(unittest.TestCase):
    def test_worker_that_exits_before_monitoring_starts(self):
        gone = mock.Mock(side_effect=psutil.NoSuchProcess(pid=1))
        with mock.patch.object(rb.psutil, "Process", gone), mock.patch.object(rb, "WINDOWS", True):
            w = rb.Worker([sys.executable, "-c", "print('FIRST', flush=True)"])
            self.assertIsNone(w.ps)
            self.assertEqual(w.expect("FIRST"), "FIRST")
            code, peak = rb.wait_with_peak(w.proc, 30, w.ps)
            w.close()
        self.assertEqual((code, peak), (0, 0))

    def test_process_vanishing_between_polls_is_not_an_error(self):
        ps = mock.Mock()
        ps.memory_info.side_effect = psutil.NoSuchProcess(pid=1)
        with mock.patch.object(rb, "WINDOWS", True):
            proc = rb.subprocess.Popen([sys.executable, "-c", "pass"])
            code, _ = rb.wait_with_peak(proc, 30, ps)
        self.assertEqual(code, 0)

    def test_access_denied_is_reported_not_swallowed(self):
        ps = mock.Mock()
        ps.memory_info.side_effect = psutil.AccessDenied(pid=1)
        with mock.patch.object(rb, "WINDOWS", True):
            proc = rb.subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.readline()"],
                                       stdin=rb.subprocess.PIPE)
            try:
                with self.assertRaises(rb.BenchError):
                    rb.wait_with_peak(proc, 30, ps)
            finally:
                rb.settle(proc, [])


class MeasuredRuns(unittest.TestCase):
    def test_nonzero_exit_of_a_measured_run_aborts(self):
        with self.assertRaises(rb.BenchError):
            rb.run_measured([sys.executable, "-c", "import sys; sys.exit(3)"])

    def test_measured_run_reports_a_real_peak(self):
        # Allocate ~50 MB, then exit: the OS high-water mark must see it.
        rec = rb.run_measured([sys.executable, "-c", "b = bytearray(50_000_000); b[::4096] = b'x' * len(b[::4096])"])
        self.assertEqual(rec["exit_status"], 0)
        self.assertGreater(rec["peak_memory_bytes"], 50_000_000)

    def test_timeout_kills_the_child(self):
        with tempfile.TemporaryDirectory() as d:
            pidfile = pathlib.Path(d) / "pid"
            code = f"import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(600)"
            with mock.patch.object(rb, "TRAIN_TIMEOUT_S", 2):
                with self.assertRaises(rb.BenchError):
                    rb.run_measured([sys.executable, "-c", code])
            pid = int(pidfile.read_text())
        self.assertFalse(psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE)


if __name__ == "__main__":
    unittest.main()
