"""torch_mps_train_diag.py の回帰テスト（イシュー #4154）。stdlib の unittest のみ。

torch／numpy 不要の部分（行・env の厳格検証、統計、腕と item の定義、集計側との整合）だけを検査する。
実機（MPS）経路は Linux で実行できないため対象外（Mac mini の初回実行で確認する）。
手動実行: python3 scripts/bench/framework-compare/torch_mps_train_diag_test.py
"""
import importlib.util
import unittest
from pathlib import Path

_P = Path(__file__).with_name("torch_mps_train_diag.py")
_spec = importlib.util.spec_from_file_location("torch_mps_train_diag", _P)
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

CTX = {"torch_version": "2.14.0", "bench_sha": "s", "self_sha": "d"}


def ok(arm="split", item="issue_wall", run=1):
    return D.make_row(CTX, arm, item, (1e-3, 9e-4, 1.1e-3), "実測", 20, 80, run, checksum=0.5)


class RowTest(unittest.TestCase):
    def test_ok_row(self):
        self.assertEqual(ok()["status"], "ok")

    def test_clock_per_item(self):
        self.assertEqual(ok("event", "gpu_span")["clock"], "mps_event")
        self.assertEqual(ok("split", "sync_wall")["clock"], "wall")

    def test_unavailable_has_null_values(self):
        r = D.make_row(CTX, "event", "gpu_span", None, "実測", 20, 80, 1, reason="x")
        self.assertIsNone(r["median_s"])
        self.assertEqual(r["tag"], "未測定")
        with self.assertRaises(ValueError):
            D.validate_row(dict(r, median_s=1.0))
        with self.assertRaises(ValueError):
            D.validate_row(dict(r, reason=""))

    def test_strict_rejections(self):
        r = ok()
        for bad in (dict(r, median_s=float("inf")), dict(r, median_s=-1.0), dict(r, item="nope"), dict(r, arm="nope"),
                    dict(r, run=0), dict(r, tag="x"), dict(r, status="x"), dict(r, issue=1), dict(r, clock="mps_event"),
                    dict(r, checksum=float("nan"))):
            with self.assertRaises(ValueError):
                D.validate_row(bad)
        with self.assertRaises(ValueError):
            D.validate_row({k: v for k, v in r.items() if k != "tag"})


class OtherTest(unittest.TestCase):
    def test_stats(self):
        self.assertEqual(D.stats_of([1.0, 2.0, 3.0, 4.0, 5.0]), (3.0, 2.0, 4.0))

    def test_arm_items_match_aggregate(self):
        p = Path(__file__).resolve().parents[3] / "docs/perf/logs/diag-0140-m-mtl-trn-4154/aggregate_4154.py"
        sp = importlib.util.spec_from_file_location("aggregate_4154", p)
        A = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(A)
        self.assertEqual(A.ARM_ITEMS, D.ARM_ITEMS)

    def test_env_requires_torch_2_14_and_mps(self):
        d = {k: None for k in D.ENV_KEYS}
        d.update(issue=4154, torch_version="2.13.0", mps_available=True)
        with self.assertRaises(ValueError):
            D.validate_env(d)
        d["torch_version"] = "2.14.0"
        D.validate_env(d)
        d["mps_available"] = False
        with self.assertRaises(ValueError):
            D.validate_env(d)

    def test_bench_py_sha_mismatch_stops(self):
        with self.assertRaises(SystemExit):
            D.load_bench_py(_P)


if __name__ == "__main__":
    unittest.main()
