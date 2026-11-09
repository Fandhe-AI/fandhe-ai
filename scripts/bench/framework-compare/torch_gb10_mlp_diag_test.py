"""torch_gb10_mlp_diag.py と aggregate_4149.py の回帰テスト（イシュー #4149）。stdlib の unittest のみ。

torch／numpy 不要の部分（行・env・profile の厳格検証、item 集合の突き合わせ、集計の自己テスト、起動キットの構文、
公開面の文字列検査、shim の C 構文）だけを検査する。torch_cpu_mlp_diag_test.py と同じくファイルパス指定の importlib で読み込む。
CI（ci.yml）には未接続で手動実行する（#4148 の前例どおり）: python3 -m unittest scripts/bench/framework-compare/torch_gb10_mlp_diag_test.py
"""
import importlib.util
import re
import shutil
import subprocess
import unittest
from pathlib import Path

_P = Path(__file__).with_name("torch_gb10_mlp_diag.py")
_spec = importlib.util.spec_from_file_location("torch_gb10_mlp_diag", _P)
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT / "docs/perf/logs/diag-0140-torch-gb10-mlp-4149"
_sp = importlib.util.spec_from_file_location("aggregate_4149", KIT / "aggregate_4149.py")
A = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(A)

CTX = {"threads": 4, "torch_version": "2.14.0", "numpy_version": "2", "bench_sha": "s", "self_sha": "d"}


def ok(item="sync_idle", section="cuda_fixed", run=1):
    return D.make_row(CTX, section, item, (1e-5, 9e-6, 1.1e-5), "実測", 20, 20, run)


class RowTest(unittest.TestCase):
    def test_ok_row(self):
        self.assertEqual(ok()["status"], "ok")

    def test_unavailable_has_null_values(self):
        r = D.make_row(CTX, "cuda_infer", "graph_replay_ref", None, "実測", 20, 20, 1, reason="x")
        self.assertIsNone(r["median_s"])
        self.assertEqual(r["tag"], "未測定")
        with self.assertRaises(ValueError):
            D.validate_row(dict(r, median_s=1.0))
        with self.assertRaises(ValueError):
            D.validate_row(dict(r, reason=""))

    def test_strict_rejections(self):
        r = ok()
        for bad in (dict(r, median_s=float("inf")), dict(r, median_s=-1.0), dict(r, item="nope"),
                    dict(r, section="nope"), dict(r, run=0), dict(r, tag="x"), dict(r, status="x"),
                    dict(r, issue=4148), dict(r, tag="未測定")):
            with self.assertRaises(ValueError):
                D.validate_row(bad)
        with self.assertRaises(ValueError):
            D.validate_row({k: v for k, v in r.items() if k != "tag"})

    def test_item_belongs_to_section(self):
        with self.assertRaises(ValueError):
            D.make_row(CTX, "cuda_fixed", "issue", (1e-5, 9e-6, 1.1e-5), "実測", 1, 1, 1)


class EnvProfileTest(unittest.TestCase):
    def test_env_requires_torch_2_14(self):
        d = {k: None for k in D.ENV_KEYS}
        d.update(issue=4149, torch_version="2.13.0")
        with self.assertRaises(ValueError):
            D.validate_env(d)
        d["torch_version"] = "2.14.0"
        D.validate_env(d)

    def test_profile_validation(self):
        d = {k: None for k in D.PROFILE_KEYS}
        d.update(issue=4149, section="cpu_train", steps=10, n_ops_aten_total=5, n_ops_aten_toplevel=9)
        with self.assertRaises(ValueError):
            D.validate_profile(d)
        d["n_ops_aten_toplevel"] = 4
        D.validate_profile(d)

    def test_cuda_profile_needs_activity_or_reason(self):
        d = {k: None for k in D.PROFILE_KEYS}
        d.update(issue=4149, section="cuda_infer", steps=10, n_ops_aten_total=5, n_ops_aten_toplevel=4)
        with self.assertRaises(ValueError):
            D.validate_profile(d)
        d["cuda_unavailable_reason"] = "CUPTI 不通"
        D.validate_profile(d)
        d["cuda_unavailable_reason"] = None
        d["cuda"] = {"kernels": {}, "api": {}, "graph_launch": False}
        D.validate_profile(d)


class ConsistencyTest(unittest.TestCase):
    def test_items_match_aggregate(self):
        self.assertEqual(A.SECTION_ITEMS, D.SECTION_ITEMS)
        self.assertEqual(A.PROFILE_SECTIONS, D.PROFILE_SECTIONS)
        self.assertEqual((A.TRAIN_FWD, A.TRAIN_BWD, A.INFER_G), (D.TRAIN_GEMM_FWD, D.TRAIN_GEMM_BWD, D.INFER_GEMM))
        self.assertEqual(A.SHAPE_IDS, D.SHAPE_IDS)
        self.assertEqual(A.ROW_REQUIRED_KEYS, D.REQUIRED_KEYS)

    def test_item_counts(self):
        self.assertEqual(len(D.SECTION_ITEMS["cpu_shapes"]), 18)
        self.assertEqual(len(D.SECTION_ITEMS["cuda_shapes"]), 6)

    def test_aggregate_self_test(self):
        A.self_test()

    def test_template_check(self):
        A.check_template(str(KIT / "RULE.template.txt"))

    def test_template_params_match_aggregate(self):
        text = (KIT / "RULE.template.txt").read_text(encoding="utf-8")
        _, body = A.split_params(text)
        self.assertEqual(set(A.parse_kv(body)), set(A.PARAM_KEYS))

    def test_bench_py_sha_matches_helper(self):
        h = D.cpu_diag()
        self.assertIn(h.BENCH_PY_SHA256, (KIT / "run_round.sh").read_text(encoding="utf-8"))


class KitTest(unittest.TestCase):
    def test_bash_syntax(self):
        if not shutil.which("bash"):
            self.skipTest("bash なし")
        for p in (KIT / "run_round.sh", KIT / "probe/run_probe.sh"):
            r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True, check=False)
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_shim_compiles(self):
        if not shutil.which("gcc"):
            self.skipTest("gcc なし")
        r = subprocess.run(["gcc", "-fsyntax-only", "-Wall", str(KIT / "probe/probe_shim_4149.c")],
                           capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_public_surface_has_no_home_or_user_paths(self):
        pat = re.compile(r"/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+")
        files = [p for p in KIT.rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".c", ".txt", ".md")] + [_P]
        for p in files:
            text = p.read_text(encoding="utf-8")
            self.assertIsNone(pat.search(text), f"絶対ホームパスの混入: {p}")

    def test_timing_arm_refuses_ld_preload(self):
        self.assertIn('[ -z "${LD_PRELOAD:-}" ]', (KIT / "run_round.sh").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
