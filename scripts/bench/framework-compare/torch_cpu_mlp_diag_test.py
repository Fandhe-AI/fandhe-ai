"""torch_cpu_mlp_diag.py の回帰テスト（イシュー #4148）。stdlib の unittest のみ。

torch／numpy 不要の部分（行・env・profile の厳格検証、統計、item 集合）だけを検査する。
opp_kernel_torch_diag_test.py と同じくファイルパス指定の importlib で読み込む。
"""
import importlib.util
import unittest
from pathlib import Path

_P = Path(__file__).with_name("torch_cpu_mlp_diag.py")
_spec = importlib.util.spec_from_file_location("torch_cpu_mlp_diag", _P)
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

CTX = {"threads": 4, "torch_version": "2.14.0", "numpy_version": "2", "bench_sha": "s", "self_sha": "d"}


def ok(item="window", section="gemm256", run=1):
    return D.make_row(CTX, section, item, (1e-3, 9e-4, 1.1e-3), "実測", 20, 20, run)


class RowTest(unittest.TestCase):
    def test_ok_row(self):
        self.assertEqual(ok()["status"], "ok")

    def test_unavailable_has_null_values(self):
        r = D.make_row(CTX, "gemm256", "accelerate_batch10", None, "実測", 20, 20, 1, reason="x")
        self.assertIsNone(r["median_s"])
        self.assertEqual(r["tag"], "未測定")

    def test_unavailable_with_value_rejected(self):
        r = D.make_row(CTX, "gemm256", "accelerate_batch10", None, "実測", 20, 20, 1, reason="x")
        with self.assertRaises(ValueError):
            D.validate_row(dict(r, median_s=1.0))
        with self.assertRaises(ValueError):
            D.validate_row(dict(r, reason=""))

    def test_strict_rejections(self):
        r = ok()
        for bad in (dict(r, median_s=float("inf")), dict(r, median_s=-1.0), dict(r, item="nope"),
                    dict(r, section="nope"), dict(r, run=0), dict(r, tag="x"), dict(r, status="x"),
                    dict(r, issue=1)):
            with self.assertRaises(ValueError):
                D.validate_row(bad)
        with self.assertRaises(ValueError):
            D.validate_row({k: v for k, v in r.items() if k != "tag"})

    def test_ok_with_unmeasured_tag_rejected(self):
        with self.assertRaises(ValueError):
            D.validate_row(dict(ok(), tag="未測定"))


class OtherTest(unittest.TestCase):
    def test_stats(self):
        self.assertEqual(D.stats_of([1.0, 2.0, 3.0, 4.0, 5.0]), (3.0, 2.0, 4.0))

    def test_items(self):
        self.assertEqual(len(D.SECTION_ITEMS["shapes"]), 18)
        self.assertIn("window", D.SECTION_ITEMS["train"])

    def test_items_match_aggregate(self):
        p = Path(__file__).resolve().parents[3] / "docs/perf/logs/diag-0140-torch-cpu-mlp-macmini-4148/aggregate_4148.py"
        sp = importlib.util.spec_from_file_location("aggregate_4148", p)
        A = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(A)
        self.assertEqual(A.SECTION_ITEMS, D.SECTION_ITEMS)
        self.assertEqual((A.TRAIN_FWD, A.TRAIN_BWD, A.INFER_G), (D.TRAIN_GEMM_FWD, D.TRAIN_GEMM_BWD, D.INFER_GEMM))

    def test_env_requires_torch_2_14(self):
        d = {k: None for k in D.ENV_KEYS}
        d.update(issue=4148, torch_version="2.13.0")
        with self.assertRaises(ValueError):
            D.validate_env(d)
        d["torch_version"] = "2.14.0"
        D.validate_env(d)

    def test_profile_validation(self):
        d = {k: None for k in D.PROFILE_KEYS}
        d.update(issue=4148, section="train", steps=10, n_ops_aten_total=5, n_ops_aten_toplevel=9)
        with self.assertRaises(ValueError):
            D.validate_profile(d)
        d["n_ops_aten_toplevel"] = 4
        D.validate_profile(d)


if __name__ == "__main__":
    unittest.main()
