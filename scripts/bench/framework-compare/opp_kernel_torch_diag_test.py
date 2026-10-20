"""opp_kernel_torch_diag.py の回帰テスト（イシュー #3179）。stdlib の unittest のみ。

torch／numpy 不要の純粋関数（形状表・算術・JSONL 検証・集計の fail-closed）だけを検査する。
summarize_test.py と同じくファイルパス指定の importlib で読み込む。
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_P = Path(__file__).with_name("opp_kernel_torch_diag.py")
_spec = importlib.util.spec_from_file_location("opp_kernel_torch_diag", _P)
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


def row(run=1, phase="window", sid="sq256", arm="base", dev="cpu", med=1e-3, ck=10.0, m=256, n=256, k=256):
    return dict(issue=3179, arm=arm, device=dev, shape_id=sid, m=m, n=n, k=k, trans="NN", phase=phase,
                median_s=med, q1_s=med, q3_s=med, warmup=20, iters=20, run=run, torch_threads=10,
                veclib_max_threads="", torch_version="t", numpy_version="n", bench_py_sha256="s", checksum=ck)


def write_runs(root, rows):
    for r in rows:
        d = Path(root) / f"run{r['run']}"
        d.mkdir(exist_ok=True)
        with open(d / "x.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")


def full(**kw):
    return [row(run=run, phase=ph, **kw) for run in range(1, 6)
            for ph in ("mm_pure", "readout", "checksum", "window")]


REF = {("cpu", 256): {"median_s": 1e-3, "checksum": 10.0}}


def match(a, b):
    return abs(a - b) < 1e-3


class Tests(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual([s[1] for s in D.shapes_for("square")], [256, 512, 1024, 2048, 4096])
        self.assertEqual(len(D.MLP_SHAPES), 6)
        self.assertEqual(D.MLP_SHAPES[1][1:], (784, 256, 64, "TN"))

    def test_arithmetic(self):
        self.assertAlmostEqual(D.ns_per_element(65536 * 3.5e-9, 256, 256), 3.5)
        self.assertAlmostEqual(D.gflops(2, 2, 2, 16e-9), 1.0)

    def test_validate_rejects(self):
        D.validate_row(row())
        r = row()
        del r["checksum"]
        with self.assertRaises(ValueError):
            D.validate_row(r)
        with self.assertRaises(ValueError):
            D.validate_row(row(med=float("inf")))
        with self.assertRaises(ValueError):
            D.validate_row(row(phase="bogus"))

    def test_aggregate_ok(self):
        with tempfile.TemporaryDirectory() as t:
            write_runs(t, full())
            out = D.aggregate_rows(D.load_runs(t), REF, match)
            win = [e for e in out if e["phase"] == "window"][0]
            self.assertAlmostEqual(win["ratio_to_series_b"], 1.0)
            self.assertAlmostEqual(win["sum_over_window"], 3.0)

    def test_missing_run(self):
        with tempfile.TemporaryDirectory() as t:
            write_runs(t, [r for r in full() if not (r["run"] == 3 and r["phase"] == "checksum")])
            with self.assertRaises(ValueError):
                D.load_runs(t)

    def test_duplicate_run(self):
        with tempfile.TemporaryDirectory() as t:
            write_runs(t, full() + [row(run=2)])
            with self.assertRaises(ValueError):
                D.load_runs(t)

    def test_checksum_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            write_runs(t, full(ck=99.0))
            with self.assertRaises(ValueError):
                D.aggregate_rows(D.load_runs(t), REF, match)

    def test_series_b_absent_cpu_4096_recorded_not_raised(self):
        with tempfile.TemporaryDirectory() as t:
            write_runs(t, full(sid="sq4096", m=4096, n=4096, k=4096))
            out = D.aggregate_rows(D.load_runs(t), REF, match)
            win = [e for e in out if e["phase"] == "window"][0]
            self.assertTrue(win["series_b_absent"])
            self.assertNotIn("ratio_to_series_b", win)
            self.assertIn("系列B不在", D.render_md(out))

    def test_series_b_absent_other_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            write_runs(t, full(sid="sq512", m=512, n=512, k=512))
            with self.assertRaises(ValueError):
                D.aggregate_rows(D.load_runs(t), REF, match)

    def test_checksum_checked_for_non_base_arm(self):
        with tempfile.TemporaryDirectory() as t:
            write_runs(t, full() + full(arm="threads1", ck=99.0))
            with self.assertRaises(ValueError):
                D.aggregate_rows(D.load_runs(t), REF, match)

    def test_checksum_checked_for_non_window_phase(self):
        with tempfile.TemporaryDirectory() as t:
            rows = full()
            for r in rows:
                if r["phase"] == "readout":
                    r["checksum"] = 99.0
            write_runs(t, rows)
            with self.assertRaises(ValueError):
                D.aggregate_rows(D.load_runs(t), REF, match)

    def _complete_rows(self, drop=lambda g: False):
        rows = []
        for (arm, dev, sid, ph) in sorted(D.expected_groups()):
            if drop((arm, dev, sid, ph)):
                continue
            veclib = ""
            a = arm
            if arm.startswith("base+vl"):
                a, veclib = "base", arm[len("base+vl"):]
            n = int(sid[2:]) if sid.startswith("sq") else 64
            for run in range(1, 6):
                r = row(run=run, phase=ph, sid=sid, arm=a, dev=dev, m=n, n=n, k=n)
                r["veclib_max_threads"] = veclib
                rows.append(r)
        return rows

    def test_check_complete_ok(self):
        D.check_complete(self._complete_rows())

    def test_check_complete_missing_phase_everywhere(self):
        rows = self._complete_rows(lambda g: g[3] == "checksum" and g[0] == "base" and g[1] == "cpu")
        with self.assertRaises(ValueError):
            D.check_complete(rows)

    def test_check_complete_missing_arm_file(self):
        with self.assertRaises(ValueError):
            D.check_complete(self._complete_rows(lambda g: g[0] == "threads4"))

    def test_numpy_unavailable_only_when_declared(self):
        rows = self._complete_rows(lambda g: g[0] == "numpy")
        with self.assertRaises(ValueError):
            D.check_complete(rows)
        D.check_complete(rows, numpy_unavailable=True)

    def test_load_numpy_unavailable(self):
        with tempfile.TemporaryDirectory() as t:
            for k in range(1, 6):
                d = Path(t) / f"run{k}"
                d.mkdir()
                (d / "unavailable.txt").write_text("unavailable numpy-cpu-vlunset\n")
            self.assertTrue(D.load_numpy_unavailable(t))
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(D.load_numpy_unavailable(t))
            d = Path(t) / "run1"
            d.mkdir()
            (d / "unavailable.txt").write_text("unavailable numpy-cpu-vlunset\n")
            with self.assertRaises(ValueError):
                D.load_numpy_unavailable(t)

    def test_bench_py_sha_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "b.py"
            p.write_text("x = 1\n")
            with self.assertRaises(SystemExit):
                D.load_bench_py(p)


if __name__ == "__main__":
    unittest.main()
