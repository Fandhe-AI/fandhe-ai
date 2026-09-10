#!/usr/bin/env python3
"""`compare_gemm_ab.py` の単体テスト（イシュー #1306）。

`compare_managed_ab_test.py`・`compare_ab_test.py` と同じ方式（ファイルパス
指定 import・tempfile への合成 JSONL 書き出し）。CI（`ci.yml` の
`deps-forbidden` ジョブ）は
`python3 -m unittest scripts/bench/framework-compare/compare_gemm_ab_test.py`
で本ファイルを実行する。

検証観点:
- 正常系: 8 セル（size×mode）とも非後退（ratio<=threshold・checksum 完全
  一致）。
- 後退セル（ratio>threshold）は「後退」判定。
- 件数不足（5 件未満／超過）は判定不能。
- checksum 複合判定 pass だが完全一致でない場合は「複合判定 ok」表示・
  非後退セルでも `checksum_exact_match` は False として区別される。
- checksum が複合判定を外れる場合は判定不能。
- 不正行（`framework` 不一致・`tf32:true`・`managed:true`・`task`/
  `device`/`size`/`mode` 不正）は理由付きで警告しスキップし、main() は
  終了コード 2（入力不能）を返す。
- 終了コード: 全セル非後退なら 0、後退または判定不能ありなら 3、入力自体
  が不能なら 2。
"""

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))

_SPEC = importlib.util.spec_from_file_location(
    "compare_gemm_ab", os.path.join(HERE, "compare_gemm_ab.py")
)
compare_gemm_ab = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_gemm_ab)


def _rec(median_s, checksum=1.23456, size=1024, mode="reuse", version="0.7.0",
         warmup=20, iters=20, task="gemm", device="metal", framework="fandhe-ai",
         tf32=None, managed=None, parity_fail_count=0):
    r = {
        "framework": framework,
        "version": version,
        "task": task,
        "device": device,
        "size": size,
        "median_s": median_s,
        "q1_s": median_s,
        "q3_s": median_s,
        "checksum": checksum,
        "warmup": warmup,
        "iters": iters,
        "mode": mode,
        "parity_fail_count": parity_fail_count,
    }
    if tf32 is not None:
        r["tf32"] = tf32
    if managed is not None:
        r["managed"] = managed
    return r


def _write_jsonl(rows):
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


_SIZES = (512, 1024, 2048, 4096)
_MODES = ("fresh", "reuse")


def _all_cells_rows(before_median, after_median, checksum=1.23456, sizes=_SIZES, device="metal"):
    """指定 device のセル（`sizes` × `_MODES`）分の before/after 各 5 件を
    生成する（既定は metal の 8 セル。イシュー #1364 で `device`／`sizes`
    を引数化し cpu の 6 セル生成にも流用する）。
    """
    before = []
    after = []
    for size in sizes:
        for mode in _MODES:
            for _ in range(5):
                before.append(
                    _rec(before_median, checksum=checksum, size=size, mode=mode, device=device)
                )
                after.append(
                    _rec(after_median, checksum=checksum, size=size, mode=mode, device=device)
                )
    return before, after


class LoadRowsTest(unittest.TestCase):
    def test_valid_rows_are_loaded(self):
        rows = [_rec(0.001), _rec(0.0009)]
        path = _write_jsonl(rows)
        try:
            loaded, warnings = compare_gemm_ab.load_rows(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(warnings, [])
        finally:
            os.unlink(path)

    def test_invalid_json_line_is_skipped_with_warning(self):
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ).name
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not valid json\n")
            f.write(json.dumps(_rec(0.001)) + "\n")
        try:
            loaded, warnings = compare_gemm_ab.load_rows(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(len(warnings), 1)
            self.assertIn("invalid JSON", warnings[0])
        finally:
            os.unlink(path)

    def test_non_fandhe_ai_framework_is_skipped(self):
        rows = [_rec(0.001, framework="candle")]
        path = _write_jsonl(rows)
        try:
            loaded, warnings = compare_gemm_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("framework", warnings[0])
        finally:
            os.unlink(path)

    def test_tf32_true_is_skipped(self):
        rows = [_rec(0.001, tf32=True)]
        path = _write_jsonl(rows)
        try:
            loaded, warnings = compare_gemm_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertIn("tf32:true", warnings[0])
        finally:
            os.unlink(path)

    def test_managed_true_is_skipped(self):
        rows = [_rec(0.001, managed=True)]
        path = _write_jsonl(rows)
        try:
            loaded, warnings = compare_gemm_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertIn("managed:true", warnings[0])
        finally:
            os.unlink(path)

    def test_wrong_task_or_device_or_size_is_skipped(self):
        rows = [
            _rec(0.001, task="train"),
            _rec(0.001, device="cuda"),
            _rec(0.001, size=999),
            _rec(0.001, mode="bogus"),
        ]
        path = _write_jsonl(rows)
        try:
            loaded, warnings = compare_gemm_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertEqual(len(warnings), 4)
        finally:
            os.unlink(path)


class EvaluateCellTest(unittest.TestCase):
    def test_non_regression_exact_checksum_match(self):
        before = [_rec(0.002) for _ in range(5)]
        after = [_rec(0.0019) for _ in range(5)]
        result = compare_gemm_ab.evaluate_cell(before, after, 1.05)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["verdict"], "非後退")
        self.assertTrue(result["checksum_exact_match"])
        self.assertAlmostEqual(result["ratio"], 0.95, places=6)

    def test_regression_ratio_over_threshold(self):
        before = [_rec(0.002) for _ in range(5)]
        after = [_rec(0.0025) for _ in range(5)]
        result = compare_gemm_ab.evaluate_cell(before, after, 1.05)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["verdict"], "後退")
        self.assertAlmostEqual(result["ratio"], 1.25, places=6)

    def test_count_mismatch_is_undeterminable(self):
        before = [_rec(0.002) for _ in range(4)]
        after = [_rec(0.002) for _ in range(5)]
        result = compare_gemm_ab.evaluate_cell(before, after, 1.05)
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("5 件を要求", result["reason"])

    def test_checksum_composite_ok_but_not_exact(self):
        before = [_rec(0.002, checksum=1.0) for _ in range(5)]
        # 相対誤差 1e-3 未満だが完全一致ではない値。
        after = [_rec(0.002, checksum=1.0000005) for _ in range(5)]
        result = compare_gemm_ab.evaluate_cell(before, after, 1.05)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["checksum_composite_match"])
        self.assertFalse(result["checksum_exact_match"])

    def test_checksum_outside_composite_is_undeterminable(self):
        before = [_rec(0.002, checksum=1.0) for _ in range(5)]
        after = [_rec(0.002, checksum=2.0) for _ in range(5)]
        result = compare_gemm_ab.evaluate_cell(before, after, 1.05)
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("checksum", result["reason"])

    def test_version_mismatch_is_undeterminable(self):
        before = [_rec(0.002, version="0.7.0") for _ in range(5)]
        after = [_rec(0.002, version="0.6.0") for _ in range(5)]
        result = compare_gemm_ab.evaluate_cell(before, after, 1.05)
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("version", result["reason"])

    def test_parity_fail_count_positive_is_undeterminable(self):
        before = [_rec(0.002, parity_fail_count=1) for _ in range(5)]
        after = [_rec(0.002) for _ in range(5)]
        result = compare_gemm_ab.evaluate_cell(before, after, 1.05)
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("parity_fail_count", result["reason"])


class MainTest(unittest.TestCase):
    def test_all_cells_non_regression_exit_zero(self):
        before, after = _all_cells_rows(0.002, 0.0019)
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(["prog", before_path, after_path])
            self.assertEqual(code, 0)
            self.assertEqual(out.getvalue().count("非後退"), 8)
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_one_cell_regression_exit_three(self):
        before, after = _all_cells_rows(0.002, 0.0019)
        # 1 セル（4096/reuse）だけ後退させる。
        for r in after:
            if r["size"] == 4096 and r["mode"] == "reuse":
                r["median_s"] = 0.01
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(["prog", before_path, after_path])
            self.assertEqual(code, 3)
            self.assertIn("後退", out.getvalue())
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_invalid_rows_exit_two(self):
        before_path = _write_jsonl([_rec(0.002, framework="candle")])
        after_path = _write_jsonl([_rec(0.002)])
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(["prog", before_path, after_path])
            self.assertEqual(code, 2)
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_missing_cell_is_undeterminable_and_exit_three(self):
        # 8 セル中 1 セル（4096/reuse）を before/after 双方から欠落させる
        # （codex-review P2・Cursor Bugbot Medium 指摘。イシュー #1306）。
        before, after = _all_cells_rows(0.002, 0.0019)
        before = [r for r in before if not (r["size"] == 4096 and r["mode"] == "reuse")]
        after = [r for r in after if not (r["size"] == 4096 and r["mode"] == "reuse")]
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(["prog", before_path, after_path])
            self.assertEqual(code, 3)
            self.assertIn("欠測セル", out.getvalue())
            self.assertEqual(out.getvalue().count("非後退"), 7)
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_empty_input_exit_two(self):
        before_path = _write_jsonl([])
        after_path = _write_jsonl([])
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(["prog", before_path, after_path])
            self.assertEqual(code, 2)
        finally:
            os.unlink(before_path)
            os.unlink(after_path)


class DeviceCpuTest(unittest.TestCase):
    """`--device cpu`（イシュー #1364。既定スレッド数限定 on/off 比較にも
    流用する N=512/1024/2048 限定セル集合）の挙動を検証する。
    """

    _CPU_SIZES = (512, 1024, 2048)

    def test_device_defaults_to_metal(self):
        # `--device` 省略時は既存 8 セル（metal）のまま（後方互換）。
        before, after = _all_cells_rows(0.002, 0.0019)
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(["prog", before_path, after_path])
            self.assertEqual(code, 0)
            self.assertEqual(out.getvalue().count("非後退"), 8)
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_device_cpu_uses_three_size_six_cells(self):
        before, after = _all_cells_rows(
            0.002, 0.0019, sizes=self._CPU_SIZES, device="cpu"
        )
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(
                    ["prog", "--device", "cpu", before_path, after_path]
                )
            self.assertEqual(code, 0)
            self.assertEqual(out.getvalue().count("非後退"), 6)
            # 4096 は cpu のセル集合に含まれない。
            self.assertNotIn("4096", out.getvalue())
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_device_cpu_regression_cell_exit_three(self):
        before, after = _all_cells_rows(
            0.002, 0.0019, sizes=self._CPU_SIZES, device="cpu"
        )
        for r in after:
            if r["size"] == 2048 and r["mode"] == "reuse":
                r["median_s"] = 0.01
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(
                    ["prog", "--device", "cpu", before_path, after_path]
                )
            self.assertEqual(code, 3)
            self.assertIn("後退", out.getvalue())
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_metal_rows_rejected_when_device_cpu_requested(self):
        # device=metal 指定の行は --device cpu 実行では判定不能行として
        # 除外される（`_valid_cell_identity` の row_device != device 分岐）。
        before, after = _all_cells_rows(0.002, 0.0019)  # device="metal"
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(
                    ["prog", "--device", "cpu", before_path, after_path]
                )
            self.assertEqual(code, 2)
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_invalid_device_value_rejected(self):
        # イシュー #1337 で `cuda` を有効な `--device` へ追加したため、
        # ここでは依然として無効な文字列（`rocm`）を使う。
        before_path = _write_jsonl([_rec(0.002)])
        after_path = _write_jsonl([_rec(0.002)])
        try:
            err = io.StringIO()
            with self.assertRaises(SystemExit):
                with redirect_stderr(err):
                    compare_gemm_ab.main(
                        ["prog", "--device", "rocm", before_path, after_path]
                    )
        finally:
            os.unlink(before_path)
            os.unlink(after_path)


class GateCudaAndModesTest(unittest.TestCase):
    """イシュー #1337: `--device cuda`・`--sizes gate`・`--modes` の挙動を
    検証する（`run_gemm_gate.sh` 由来の cuda/metal reuse 専用入力を想定）。
    """

    _CUDA_SIZES = (1024, 2048, 4096)

    def test_device_cuda_uses_three_size_six_cells(self):
        # `--sizes` 省略時（既定 full）でも cuda は元々 3 サイズ ×
        # 2 モード＝6 セル（`_VALID_SIZES_BY_DEVICE["cuda"]` が既に
        # gate と同一集合のため）。
        before, after = _all_cells_rows(
            0.002, 0.0019, sizes=self._CUDA_SIZES, device="cuda"
        )
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(
                    ["prog", "--device", "cuda", before_path, after_path]
                )
            self.assertEqual(code, 0)
            self.assertEqual(out.getvalue().count("非後退"), 6)
            self.assertNotIn("512", out.getvalue())
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_sizes_gate_restricts_metal_to_three_sizes(self):
        # `run_gemm_gate.sh` は metal でも N=1024/2048/4096 のみ発行する
        # （512 行はそもそも存在しない。gate 出力の実形状を模す）。
        # `--sizes gate` はこの 6 セル（1024/2048/4096 × fresh/reuse）を
        # 期待セルとして判定する（既定 `--sizes full` は 512 込み 8 セルを
        # 期待するため、512 行が無いと欠測セル扱いで判定不能になる差分）。
        before, after = _all_cells_rows(
            0.002, 0.0019, sizes=(1024, 2048, 4096), device="metal"
        )
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(
                    ["prog", "--sizes", "gate", before_path, after_path]
                )
            self.assertEqual(code, 0)
            self.assertEqual(out.getvalue().count("非後退"), 6)
            self.assertNotIn("512/", out.getvalue())
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_sizes_full_default_flags_missing_512_cells_when_input_lacks_them(self):
        # 対照テスト: `--sizes gate` を使わず同じ 6 セル入力（512 なし）を
        # 既定（`--sizes full`）で判定すると、512 の 2 セルが欠測扱いに
        # なり判定不能（exit 3）になることを固定する（`--sizes gate` の
        # 意義そのものの回帰点）。
        before, after = _all_cells_rows(
            0.002, 0.0019, sizes=(1024, 2048, 4096), device="metal"
        )
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(["prog", before_path, after_path])
            self.assertEqual(code, 3)
            self.assertIn("欠測セル", out.getvalue())
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_modes_reuse_ignores_fresh_rows_without_warning(self):
        # cuda/metal のゲート出力は reuse のみ（`run_gemm_gate.sh` は
        # cuda/metal に対し bench-fandhe の fresh モードを起動しない）。
        # `--modes reuse` を指定すると fresh 行が交じっていても警告や
        # 判定不能を出さず、reuse セルのみで判定する。
        before = []
        after = []
        for size in self._CUDA_SIZES:
            for _ in range(5):
                before.append(_rec(0.002, size=size, mode="reuse", device="cuda"))
                after.append(_rec(0.0019, size=size, mode="reuse", device="cuda"))
            # fresh 行（参考記録。cuda では通常発生しないが、混入しても
            # 無害であることを固定する）。
            before.append(_rec(0.001, size=size, mode="fresh", device="cuda"))
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(
                    [
                        "prog",
                        "--device",
                        "cuda",
                        "--modes",
                        "reuse",
                        before_path,
                        after_path,
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(err.getvalue(), "")
            self.assertEqual(out.getvalue().count("非後退"), 3)
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_invalid_modes_value_rejected(self):
        before_path = _write_jsonl([_rec(0.002)])
        after_path = _write_jsonl([_rec(0.002)])
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(
                    ["prog", "--modes", "bogus", before_path, after_path]
                )
            self.assertEqual(code, 2)
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_default_gemm_output_is_byte_unchanged(self):
        """イシュー #1517: `--task` 追加後も既定（`--task` 省略・gemm）の
        出力が旧実装とバイト単位で不変であることを固定する（実装計画
        §6「既定 gemm 出力バイト不変」）。
        """
        before, after = _all_cells_rows(0.002, 0.0019)
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(["prog", before_path, after_path])
            self.assertEqual(code, 0)
            self.assertEqual(err.getvalue(), "")
            stdout = out.getvalue()
            self.assertIn("| size/mode | before median | after median |", stdout)
            self.assertEqual(stdout.count("非後退"), 8)
            # `--task`/`--phases` 追加により既定出力へフェーズ表等の余計な
            # 出力が混入していないこと（末尾が判定表のみで終わること）を
            # 固定する。
            self.assertTrue(stdout.rstrip("\n").endswith("| 非後退 |"))
        finally:
            os.unlink(before_path)
            os.unlink(after_path)


def _rec_train(median_s, checksum=-1.5, mode="reuse", version="0.8.0", warmup=5, iters=15):
    """`bench-fandhe --task train` 行の複製。`train` タスクは `parity`
    フィールドを `None` として emit しない（`Record.parity: Option<
    ParityStats>`）ため `parity_fail_count` キー自体を持たない
    （`_rec` の gemm 用フィクスチャとは異なる。実装計画 §3.3 手順 6 で
    確認済み）。`size` は `bench-fandhe` の `BATCH` 定数（64）固定。
    """
    return {
        "framework": "fandhe-ai",
        "version": version,
        "task": "train",
        "device": "metal",
        "size": 64,
        "median_s": median_s,
        "q1_s": median_s,
        "q3_s": median_s,
        "checksum": checksum,
        "warmup": warmup,
        "iters": iters,
        "mode": mode,
    }


def _all_train_cells_rows(before_median, after_median, checksum=-1.5):
    before = []
    after = []
    for mode in _MODES:
        for _ in range(5):
            before.append(_rec_train(before_median, checksum=checksum, mode=mode))
            after.append(_rec_train(after_median, checksum=checksum, mode=mode))
    return before, after


class TaskTrainTest(unittest.TestCase):
    """イシュー #1517: `--task train`（split-K 結線前後 A/B の train 2 セル）
    の判定ロジックを検証する。"""

    def test_all_cells_non_regression(self):
        before, after = _all_train_cells_rows(0.010, 0.0099)
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(["prog", "--task", "train", before_path, after_path])
            self.assertEqual(code, 0)
            self.assertEqual(err.getvalue(), "")
            self.assertEqual(out.getvalue().count("非後退"), 2)
            self.assertIn("64/fresh", out.getvalue())
            self.assertIn("64/reuse", out.getvalue())
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_regression_cell_detected(self):
        before, after = _all_train_cells_rows(0.010, 0.020)
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(["prog", "--task", "train", before_path, after_path])
            self.assertEqual(code, 3)
            self.assertEqual(out.getvalue().count("後退"), 2)
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_gemm_rows_excluded_when_task_train(self):
        """`task:"gemm"` の行が紛れ込むと `--task train` 実行時は不正行
        として警告付きでスキップされ、fail-closed の「判定不能」（終了
        コード 2）になることを確認する（`load_rows` の既存 fail-closed
        方針〈不正行が 1 件でもあれば判定不能〉を `--task train` でも
        維持する。混在を黙って無視して「非後退」と誤判定しない）。"""
        before, after = _all_train_cells_rows(0.010, 0.0099)
        before.append(_rec(0.002))
        after.append(_rec(0.002))
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(["prog", "--task", "train", before_path, after_path])
            self.assertEqual(code, 2)
            self.assertIn("WARNING", err.getvalue())
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_phases_requires_train_task(self):
        before_path = _write_jsonl(_all_train_cells_rows(0.010, 0.0099)[0])
        after_path = _write_jsonl(_all_train_cells_rows(0.010, 0.0099)[1])
        phases_path = _write_jsonl([])
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(
                    [
                        "prog",
                        "--phases",
                        phases_path,
                        phases_path,
                        before_path,
                        after_path,
                    ]
                )
            self.assertEqual(code, 2)
        finally:
            os.unlink(before_path)
            os.unlink(after_path)
            os.unlink(phases_path)

    def test_phases_diagnostic_table_rendered(self):
        before, after = _all_train_cells_rows(0.010, 0.0099)
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        before_phases_path = _write_jsonl(
            [
                {
                    "framework": "fandhe-ai",
                    "version": "0.8.0",
                    "task": "train_phases",
                    "device": "metal",
                    "size": 64,
                    "median_s": 0.001,
                    "q1_s": 0.001,
                    "q3_s": 0.001,
                    "checksum": -1.5,
                    "warmup": 5,
                    "iters": 15,
                    "mode": "reuse",
                    "phase": "backward",
                    "phase_index": 2,
                }
            ]
        )
        after_phases_path = _write_jsonl(
            [
                {
                    "framework": "fandhe-ai",
                    "version": "0.8.0",
                    "task": "train_phases",
                    "device": "metal",
                    "size": 64,
                    "median_s": 0.0009,
                    "q1_s": 0.0009,
                    "q3_s": 0.0009,
                    "checksum": -1.5,
                    "warmup": 5,
                    "iters": 15,
                    "mode": "reuse",
                    "phase": "backward",
                    "phase_index": 2,
                }
            ]
        )
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(
                    [
                        "prog",
                        "--task",
                        "train",
                        "--phases",
                        before_phases_path,
                        after_phases_path,
                        before_path,
                        after_path,
                    ]
                )
            self.assertEqual(code, 0)
            stdout = out.getvalue()
            self.assertIn("フェーズ分解（診断用・reuse・単発計測・非判定）", stdout)
            self.assertIn("| backward |", stdout)
        finally:
            os.unlink(before_path)
            os.unlink(after_path)
            os.unlink(before_phases_path)
            os.unlink(after_phases_path)


class PerRunTest(unittest.TestCase):
    """イシュー #1517 実装計画 §4 rule (b)（run 内比 5/5 run 符号一貫の
    機械判定）向け `--per-run` 診断列を検証する。"""

    def test_default_output_unaffected_by_per_run_flag_availability(self):
        """`--per-run` を渡さない既定実行は列追加前と出力が完全一致する
        （実装計画 §6「既定出力バイト不変」の `--per-run` 追加後の再確認）。
        """
        before, after = _all_cells_rows(0.002, 0.0019)
        before_path = _write_jsonl(before)
        after_path = _write_jsonl(after)
        try:
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                compare_gemm_ab.main(["prog", before_path, after_path])
            baseline = out.getvalue()
        finally:
            os.unlink(before_path)
            os.unlink(after_path)
        self.assertNotIn("run 内比", baseline)
        self.assertNotIn("符号一貫", baseline)

    def test_per_run_sign_consistent_regression_flagged(self):
        """全 5 run で after が before を上回る（後退方向で一貫）場合、
        `sign_consistent` 相当の列が「はい」になることを確認する
        （実装計画 §4 rule (b) の機械判定対象ケース）。fresh セルは
        非後退の 5 件を与え、reuse セル 1 つだけの符号一貫性を
        `--per-run` 列で確認できることを検証する（`_all_expected_cells`
        が両セルを要求するため）。
        """
        before_rows = list(_all_train_cells_rows(0.010, 0.0099)[0])
        after_rows = list(_all_train_cells_rows(0.010, 0.0099)[1])
        # reuse セルだけ 5 run 一貫して後退する行へ差し替える。
        before_rows = [r for r in before_rows if r["mode"] != "reuse"]
        after_rows = [r for r in after_rows if r["mode"] != "reuse"]
        for _ in range(5):
            before_rows.append(_rec_train(0.010, mode="reuse"))
            after_rows.append(_rec_train(0.0103, mode="reuse"))
        before_path = _write_jsonl(before_rows)
        after_path = _write_jsonl(after_rows)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_gemm_ab.main(
                    [
                        "prog",
                        "--task",
                        "train",
                        "--threshold",
                        "1.05",
                        "--per-run",
                        before_path,
                        after_path,
                    ]
                )
            stdout = out.getvalue()
            self.assertIn("run 内比（5 run）", stdout)
            self.assertIn("符号一貫（全 run >1.00）", stdout)
            # 64/reuse セルは 5 run とも ratio=1.03>1.0 のため「はい」。
            self.assertIn("| はい |", stdout)
            # 判定自体（ratio<=threshold=1.05）は 2 セルとも非後退のまま
            # （--per-run は終了コード・verdict に影響しない診断列）。
            self.assertEqual(code, 0)
        finally:
            os.unlink(before_path)
            os.unlink(after_path)

    def test_per_run_helper_returns_none_on_count_mismatch(self):
        before = [_rec(0.01) for _ in range(4)]
        after = [_rec(0.01) for _ in range(5)]
        self.assertIsNone(compare_gemm_ab.per_run_ratios(before, after))

    def test_per_run_helper_ratios(self):
        before = [_rec(0.010) for _ in range(5)]
        after = [_rec(0.011) for _ in range(5)]
        ratios = compare_gemm_ab.per_run_ratios(before, after)
        self.assertEqual(len(ratios), 5)
        for r in ratios:
            self.assertAlmostEqual(r, 1.1, places=6)

    def test_per_run_table_column_count_matches_header_for_all_row_kinds(self):
        """codex-review P2・Cursor Bugbot Low Severity 指摘（イシュー #1517
        PR #1531）の再発防止: `--per-run` 出力のヘッダー・区切り行・
        すべてのデータ行（欠測セル／`ratios is None` セル／通常セル）が
        同一列数（パイプ `|` の出現数が同一）であることを機械的に検証
        する。以前は文字列スライス（`header[:-2]`／行末直接連結）に
        よって区切り行が 1 列少なく・データ行が重複パイプで 1 列多く
        描画され GitHub Markdown として壊れていた。
        """
        # 欠測セル（rows が空）・通常セル（ratios あり）の双方を含む
        # 入力: fresh のみ用意し reuse セルを欠測させる。
        before_rows = [_rec(0.010, size=512, mode="fresh") for _ in range(5)]
        after_rows = [_rec(0.0105, size=512, mode="fresh") for _ in range(5)]
        before_path = _write_jsonl(before_rows)
        after_path = _write_jsonl(after_rows)
        try:
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                code = compare_gemm_ab.main(
                    ["prog", "--threshold", "1.05", "--per-run", before_path, after_path]
                )
            stdout = out.getvalue()
        finally:
            os.unlink(before_path)
            os.unlink(after_path)
        lines = [ln for ln in stdout.splitlines() if ln.startswith("|")]
        self.assertGreaterEqual(len(lines), 3, "header・sep・データ行が出力されていない")
        pipe_counts = {ln.count("|") for ln in lines}
        self.assertEqual(
            len(pipe_counts),
            1,
            f"--per-run 出力の列数（'|' 出現数）が行ごとに異なる: {pipe_counts}\n{stdout}",
        )
        # 8 列（size/mode・before・after・after/before・checksum・判定・
        # run 内比・符号一貫）= パイプ 9 個。
        self.assertEqual(pipe_counts.pop(), 9)
        # reuse セルを意図的に欠測させているため判定不能セルが残り、
        # 終了コードは 3（any_bad）になる。列数の検証が本テストの主眼で
        # あり終了コード自体は本題ではないため存在確認のみ行う。
        self.assertEqual(code, 3)


if __name__ == "__main__":
    unittest.main()
