"""Checks that the benchmark driver fails closed: `python -m unittest bench/test_run_bench.py`."""

import pathlib
import sys
import unittest

sys.argv = sys.argv[:1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import run_bench as rb  # noqa: E402

REF = [0.5, 1.0, 2.0]


class ParityGate(unittest.TestCase):
    def test_mismatch_nonfinite_and_length_abort(self):
        for bad in ([0.5, 1.0, 2.01], [0.5, float("nan"), 2.0], [0.5, float("inf"), 2.0], [0.5, 1.0]):
            with self.assertRaises(rb.BenchError, msg=bad):
                rb.parity_error({"1": bad}, REF, "t")

    def test_every_batch_size_is_checked(self):
        with self.assertRaises(rb.BenchError):
            rb.parity_error({"1": REF, "7": [0.5, 1.0, 2.5]}, REF, "t")

    def test_within_tolerance_reports_the_largest_error(self):
        err = rb.parity_error({"1": [0.5, 1.0, 2.00001]}, REF, "t")
        self.assertAlmostEqual(err, 1e-5, places=9)


class WorkerFailures(unittest.TestCase):
    def test_nonzero_exit_of_a_measured_run_aborts(self):
        with self.assertRaises(rb.BenchError):
            rb.run_measured([sys.executable, "-c", "import sys; sys.exit(3)"])

    def test_measured_run_reports_a_real_peak(self):
        # Allocate ~50 MB, then exit: the OS high-water mark must see it.
        rec = rb.run_measured([sys.executable, "-c", "b = bytearray(50_000_000); b[::4096] = b'x' * len(b[::4096])"])
        self.assertEqual(rec["exit_status"], 0)
        self.assertGreater(rec["peak_memory_bytes"], 50_000_000)

    def test_worker_exiting_early_aborts(self):
        w = rb.Worker([sys.executable, "-c", "print('FIRST', flush=True); import sys; sys.exit(2)"])
        self.assertEqual(w.expect("FIRST"), "FIRST")
        with self.assertRaises(rb.BenchError):
            w.expect("PARITY")

    def test_hung_worker_times_out_and_is_killed(self):
        old = rb.STEP_TIMEOUT_S
        rb.STEP_TIMEOUT_S = 1
        try:
            w = rb.Worker([sys.executable, "-c", "import time; time.sleep(60)"])
            with self.assertRaises(rb.BenchError):
                w.expect("FIRST")
            self.assertIsNotNone(w.proc.poll())
        finally:
            rb.STEP_TIMEOUT_S = old


if __name__ == "__main__":
    unittest.main()
