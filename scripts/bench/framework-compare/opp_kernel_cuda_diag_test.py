"""opp_kernel_cuda_diag.py の回帰テスト（イシュー #3185）。stdlib の unittest のみ。

torch／numpy 不要の純粋関数（検証・パーサー・集計の fail-closed・maps 抽出）だけを検査する。
opp_kernel_torch_diag_test.py（#3179）と同じくファイルパス指定の importlib で読み込む。
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_P = Path(__file__).with_name("opp_kernel_cuda_diag.py")
_spec = importlib.util.spec_from_file_location("opp_kernel_cuda_diag", _P)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

DEVICE = {"fp32": "cuda", "tf32": "cuda", "cpu": "cpu"}


def row(arm="fp32", n=256, phase="mm_pure", run=1, med=1e-3, ck=10.0):
    return dict(issue=3185, arm=arm, device=DEVICE[arm], shape_id=f"sq{n}", n=n, phase=phase,
                median_s=med, q1_s=med, q3_s=med, warmup=20, iters=20, run=run,
                allow_tf32=(arm == "tf32"), fp32_matmul_precision="highest", fp32_precision="",
                torch_version="t", cuda_version="13.0", numpy_version="n", bench_py_sha256="s",
                checksum=ck, gpu_name="g")


def full_py(skip=None, runs=range(1, 6), tf32_ck=11.0):
    out = []
    for (arm, n, ph) in sorted(G.expected_py_groups()):
        if skip and skip(arm, n, ph):
            continue
        for r in runs:
            out.append(row(arm, n, ph, r, ck=tf32_ck if arm == "tf32" else 10.0))
    return out


def frow(n, layer, phase, run, med=1e-3):
    return dict(issue=3185, source="fandhe", n=n, layer=layer, phase=phase, clock="wall",
                median_s=med, q1_s=med, q3_s=med, run=run)


def full_fandhe(runs=range(1, 6), skip=None):
    return [frow(n, ly, ph, r) for (n, ly, ph) in sorted(G.expected_fandhe_groups())
            for r in runs if not (skip and skip(n, ly, ph))]


def match(a, b):
    return abs(a - b) < 1e-3


REF = {("cuda", n): 10.0 for n in G.SIZES}
REF.update({("cpu", n): 10.0 for n in G.SIZES})

LAYERB_OK = """running 1 test
  N=1024 kernel=Select (median over 20 trials, 20 warmup):
    h2d_a: median=0.1000 ms  q1=0.1000 ms  q3=0.1000 ms
    launch_issue: median=0.0500 ms  q1=0.0400 ms  q3=0.0600 ms
    kernel_wait: median=0.2000 ms  q1=0.1900 ms  q3=0.2100 ms
  N=1024 kernel=Classic (median over 20 trials, 20 warmup):
    launch_issue: median=9.0000 ms  q1=9.0000 ms  q3=9.0000 ms
    kernel_wait: median=9.0000 ms  q1=9.0000 ms  q3=9.0000 ms
  N=2048 kernel=Select (median over 20 trials, 20 warmup):
    launch_issue: median=0.0500 ms  q1=0.0400 ms  q3=0.0600 ms
    kernel_wait: median=1.1650 ms  q1=1.1600 ms  q3=1.1700 ms
  N=4096 kernel=Select (median over 20 trials, 20 warmup):
    launch_issue: median=0.0500 ms  q1=0.0400 ms  q3=0.0600 ms
    kernel_wait: median=9.0000 ms  q1=8.9000 ms  q3=9.1000 ms
test result: ok. 1 passed; 0 failed; 1 ignored
"""


def diag_text(skip=None, extra=""):
    lines = []
    for n in G.DIAG_SIZES:
        for (ly, ph) in G.DIAG_WANTED:
            if skip and skip == (n, ly, ph):
                continue
            lines.append(f"DIAG2109 n={n} layer={ly} phase={ph} median_us=12.500 q1_us=12.000 q3_us=13.000")
    lines.append("DIAG2109 n=256 layer=L2 phase=h2d_a median_us=1.0 q1_us=1.0 q3_us=1.0")
    return "\n".join(lines) + extra + "\ntest result: ok. 1 passed; 0 failed\n"


class Tests(unittest.TestCase):
    def test_base_module_reused_and_untouched(self):
        self.assertEqual(G.D.ISSUE, 3179)
        self.assertEqual(G.D.SQUARE_SIZES, (256, 512, 1024, 2048, 4096))
        self.assertEqual(G.SIZES, G.D.SQUARE_SIZES)
        self.assertEqual(len(G.D.BENCH_PY_SHA256), 64)

    def test_arithmetic(self):
        self.assertAlmostEqual(G.D.ns_per_element(65536 * 2.7e-9, 256, 256), 2.7)
        self.assertAlmostEqual(G.D.gflops(2, 2, 2, 16e-9) / 1e3, 0.001)

    def test_validate_rejects(self):
        G.validate_row(row())
        r = row()
        del r["gpu_name"]
        with self.assertRaises(ValueError):
            G.validate_row(r)
        for bad in ({"median_s": float("inf")}, {"phase": "x"}, {"arm": "z"}, {"issue": 3179},
                    {"allow_tf32": True}, {"device": "cpu"}, {"n": 300}, {"run": 0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                G.validate_row({**row(), **bad})
        with self.assertRaises(ValueError):
            G.validate_row(row(arm="cpu", phase="mm_event"))
        G.validate_row(row(arm="tf32"))

    def test_tf32_readback(self):
        G.check_tf32_readback("fp32", False)
        G.check_tf32_readback("tf32", True)
        G.check_tf32_readback("cpu", False)
        with self.assertRaises(RuntimeError):
            G.check_tf32_readback("tf32", False)
        with self.assertRaises(RuntimeError):
            G.check_tf32_readback("fp32", True)

    def test_load_py_runs_missing_and_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            for r in full_py():
                p = Path(d) / f"run{r['run']}"
                p.mkdir(exist_ok=True)
                with open(p / f"{r['arm']}.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
            self.assertEqual(len(G.load_py_runs(d)), len(full_py()))
            with open(Path(d) / "run1" / "fp32.jsonl", "a") as f:
                f.write(json.dumps(row()) + "\n")
            with self.assertRaises(ValueError):
                G.load_py_runs(d)
        with tempfile.TemporaryDirectory() as d:
            for r in full_py(runs=range(1, 5)):
                p = Path(d) / f"run{r['run']}"
                p.mkdir(exist_ok=True)
                with open(p / "x.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
            with self.assertRaises(ValueError):
                G.load_py_runs(d)

    def test_check_complete_arm_file_missing(self):
        G.check_complete(full_py(), full_fandhe())
        with self.assertRaises(ValueError):
            G.check_complete(full_py(skip=lambda a, n, p: a == "tf32"), full_fandhe())
        with self.assertRaises(ValueError):
            G.check_complete(full_py(), full_fandhe(skip=lambda n, ly, ph: ly == "D"))

    def test_aggregate_fp32_checksum_mismatch_is_fail_closed(self):
        rows = full_py()
        G.aggregate_rows(rows, full_fandhe(), REF, match)
        bad = [dict(r, checksum=99.0) if r["arm"] == "fp32" and r["n"] == 512 else r for r in rows]
        with self.assertRaises(ValueError):
            G.aggregate_rows(bad, full_fandhe(), REF, match)
        badcpu = [dict(r, checksum=99.0) if r["arm"] == "cpu" else r for r in rows]
        with self.assertRaises(ValueError):
            G.aggregate_rows(badcpu, full_fandhe(), REF, match)

    def test_aggregate_tf32_mismatch_does_not_raise(self):
        agg = G.aggregate_rows(full_py(tf32_ck=11.0), full_fandhe(), REF, match)
        self.assertFalse(agg["ratios"][0]["tf32_checksum_bit_equal_fp32"])
        self.assertAlmostEqual(agg["ratios"][0]["tf32_checksum_minus_fp32"], 1.0)
        same = G.aggregate_rows(full_py(tf32_ck=10.0), full_fandhe(), REF, match)
        self.assertTrue(same["ratios"][0]["tf32_checksum_bit_equal_fp32"])

    def test_aggregate_ratio_tables_clock_split(self):
        agg = G.aggregate_rows(full_py(), full_fandhe(), REF, match)
        by = {r["n"]: r for r in agg["ratios"]}
        self.assertIsNone(by[2048]["event_ratio_seg_over_cublas_fp32"])  # N>=1024 に event 時計なし
        self.assertIsNotNone(by[256]["event_ratio_seg_over_cublas_fp32"])
        self.assertAlmostEqual(by[2048]["fandhe_wall_s"], 2e-3)  # launch_issue + kernel_wait
        self.assertIsNone(by[256]["fandhe_wall_s"])  # L2S は投入を含まないため未計測
        self.assertIn("該当計測なし", G.render_md(agg))

    def test_layerb_parser(self):
        rows = G.parse_layerb(LAYERB_OK, 3)
        self.assertEqual(len(rows), 6)
        got = {(r["n"], r["phase"]): r["median_s"] for r in rows}
        self.assertAlmostEqual(got[(2048, "kernel_wait")], 1.165e-3)  # Classic の 9ms を拾わない
        self.assertTrue(all(r["run"] == 3 and r["clock"] == "wall" for r in rows))

    def test_layerb_parser_rejects(self):
        with self.assertRaises(ValueError):  # test result 不在
            G.parse_layerb(LAYERB_OK.replace("test result: ok", "test result: FAILED"), 1)
        with self.assertRaises(ValueError):  # 欠落（N=4096 を削除）
            G.parse_layerb(LAYERB_OK.replace("  N=4096 kernel=Select", "  N=4096 kernel=Classic"), 1)
        dup = LAYERB_OK.replace(
            "  N=2048 kernel=Select (median over 20 trials, 20 warmup):",
            "  N=1024 kernel=Select (median over 20 trials, 20 warmup):")
        with self.assertRaises(ValueError):  # 重複
            G.parse_layerb(dup, 1)
        commented = LAYERB_OK.replace("    launch_issue: median=0.0500 ms  q1=0.0400 ms  q3=0.0600 ms\n    kernel_wait: median=1.1650",
                                      "    // launch_issue: median=0.0500 ms  q1=0.0400 ms  q3=0.0600 ms\n    kernel_wait: median=1.1650")
        with self.assertRaises(ValueError):  # コメント行の部分一致で通らない
            G.parse_layerb(commented, 1)

    def test_diag_parser(self):
        rows = G.parse_diag2109(diag_text(), 2)
        self.assertEqual(len(rows), 9)
        self.assertAlmostEqual(rows[0]["median_s"], 12.5e-6)
        clocks = {(r["layer"], r["phase"]): r["clock"] for r in rows}
        self.assertEqual(clocks[("D", "dev_kernel_seg")], "event")
        self.assertEqual(clocks[("L2S", "kernel_wait")], "wall")

    def test_diag_parser_rejects(self):
        with self.assertRaises(ValueError):
            G.parse_diag2109(diag_text(skip=(512, "D", "dev_kernel_b2b")), 1)
        with self.assertRaises(ValueError):
            G.parse_diag2109(diag_text(extra="\nDIAG2109 n=256 layer=D phase=dev_kernel_seg median_us=1 q1_us=1 q3_us=1"), 1)
        with self.assertRaises(ValueError):
            G.parse_diag2109(diag_text().replace("test result: ok", "test result: FAILED"), 1)
        # 行頭アンカー: コメント行・インデント付きの部分一致は読まない
        with self.assertRaises(ValueError):
            G.parse_diag2109(diag_text().replace("DIAG2109 n=128 layer=D phase=dev_kernel_b2b",
                                                 "// DIAG2109 n=128 layer=D phase=dev_kernel_b2b"), 1)

    def test_maps_extraction_drops_directory(self):
        maps = ("ffff0000-ffff1000 r-xp 00000000 fd:01 1 /home/someone/venv/lib/libopenblas.so.0\n"
                "ffff2000-ffff3000 r-xp 00000000 fd:01 2 /usr/lib/libc.so.6\n"
                "ffff4000-ffff5000 r-xp 00000000 fd:01 3 /x/torch/lib/libdnnl.so.3\n"
                "ffff6000-ffff7000 r-xp 00000000 fd:01 4 /home/someone/notes/mkl.txt\n")
        libs = G.blas_libs_from_maps(maps)
        self.assertEqual(libs, ["libdnnl.so.3", "libopenblas.so.0"])
        self.assertFalse(any("/" in x or "someone" in x for x in libs))

    def test_judge_blas(self):
        self.assertEqual(G.judge_blas("", [])["verdict"], "確認できなかった")
        self.assertEqual(G.judge_blas("note: onednn_verbose is not here\n", [])["verdict"], "確認できなかった")
        j = G.judge_blas("onednn_verbose,v1,primitive,exec,cpu,matmul\n", [])
        self.assertEqual(j["verdict"], "確認できた")
        self.assertEqual(len(j["evidence_lines"]), 1)
        # ロードのみ・初期化情報のみは確認済みにしない（参考情報として残る）
        j = G.judge_blas("onednn_verbose,v1,info,oneDNN v3\n", ["libopenblas.so.0"])
        self.assertEqual(j["verdict"], "確認できなかった")
        self.assertEqual(j["libs"], ["libopenblas.so.0"])
        self.assertEqual(len(j["info_lines"]), 1)

    def test_load_bench_py_rejects_wrong_sha(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bench_py.py"
            p.write_text("x = 1\n")
            with self.assertRaises(SystemExit):
                G.D.load_bench_py(p)

    def test_reference_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            a = {"framework": "pytorch", "task": "gemm", "device": "cuda", "size": 256, "checksum": 1.0}
            p.write_text(json.dumps(a) + "\n" + json.dumps({**a, "checksum": 2.0}) + "\n")
            with self.assertRaises(ValueError):
                G.load_reference(p)
            p2 = Path(d) / "r2.jsonl"
            p2.write_text(json.dumps(a) + "\n" + json.dumps(a) + "\n")
            self.assertEqual(G.load_reference(p2), {("cuda", 256): 1.0})


if __name__ == "__main__":
    unittest.main()
