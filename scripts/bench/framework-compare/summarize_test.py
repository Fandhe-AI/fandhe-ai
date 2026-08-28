#!/usr/bin/env python3
"""`summarize.py` の単体テスト（イシュー #970）。

`unittest`（stdlib のみ、追加依存なし）で、合成 JSONL を tempfile に書き
`summarize.py` の関数を直接呼ぶ。CI（`ci.yml` の `deps-forbidden` ジョブ、
`フレームワーク横並びベンチ bench-common のテスト実行` ステップの直後）は
`python3 -m unittest scripts/bench/framework-compare/summarize_test.py` で
本ファイルを実行する。

検証観点:
- キー欠損（旧形式 JSONL）は「未検証」として区別され、「無効」と混同されない。
- `parity_fail_count > 0` の行は無効表示・GFLOP/s `-`・`--strict` で exit 2。
- `parity_max_abs_err`/`parity_max_rel_err` が `null`（Python では `None`）の
  行も同様に無効。
- checksum 突合（イシュー #965）との併記・`--strict` の両立。
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

# 本体の import パスに追加せず、ファイルパス指定で summarize.py を直接
# ロードする（`scripts/bench/framework-compare/` はパッケージ化されておらず、
# sys.path 汚染を避けるため）。
_SPEC = importlib.util.spec_from_file_location(
    "summarize", os.path.join(HERE, "summarize.py")
)
summarize = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summarize)


def _base_row(framework="fandhe-ai", device="cpu", size=256, mode="fresh", checksum=1.0):
    return {
        "framework": framework,
        "version": "0.3.0",
        "task": "gemm",
        "device": device,
        "size": size,
        "median_s": 0.001,
        "q1_s": 0.0009,
        "q3_s": 0.0011,
        "gflops": 100.0,
        "checksum": checksum,
        "warmup": 20,
        "iters": 20,
        "mode": mode,
    }


def _with_parity(row, total=65536, fail_count=0, max_abs_err=0.0, max_rel_err=0.0):
    row = dict(row)
    row["parity_total"] = total
    row["parity_fail_count"] = fail_count
    row["parity_max_abs_err"] = max_abs_err
    row["parity_max_rel_err"] = max_rel_err
    return row


def _write_jsonl(rows):
    """rows を tempfile（JSONL）へ書き、そのパスを返す（呼び出し側が unlink）。"""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


class ParityStatusTests(unittest.TestCase):
    def test_missing_keys_is_unverified(self):
        row = _base_row()
        self.assertEqual(summarize.parity_status(row), "unverified")

    def test_zero_fail_count_is_ok(self):
        row = _with_parity(_base_row())
        self.assertEqual(summarize.parity_status(row), "ok")

    def test_positive_fail_count_is_fail(self):
        row = _with_parity(_base_row(), fail_count=3)
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_null_max_abs_err_is_fail(self):
        row = _with_parity(_base_row())
        row["parity_max_abs_err"] = None
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_null_max_rel_err_is_fail(self):
        row = _with_parity(_base_row())
        row["parity_max_rel_err"] = None
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_non_numeric_fail_count_is_fail(self):
        row = _with_parity(_base_row())
        row["parity_fail_count"] = "0"
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_bool_fail_count_is_fail(self):
        # bool は int のサブクラスなので明示的に除外していることを確認する。
        row = _with_parity(_base_row())
        row["parity_fail_count"] = False
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_missing_total_only_is_fail(self):
        # parity_fail_count キーはあるが parity_total が欠けている（部分的な
        # 破損データ）場合も無効として扱う。
        row = _with_parity(_base_row())
        del row["parity_total"]
        self.assertEqual(summarize.parity_status(row), "fail")


class GemmParityHelpersTests(unittest.TestCase):
    def test_gemm_parity_failures_filters_task_and_status(self):
        rows = [
            _with_parity(_base_row(framework="fandhe-ai"), fail_count=0),
            _with_parity(_base_row(framework="candle"), fail_count=2),
            dict(_base_row(framework="burn"), task="train"),  # gemm 以外は対象外
        ]
        failures = summarize.gemm_parity_failures(rows)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["framework"], "candle")

    def test_gemm_parity_unverified_filters_task_and_status(self):
        rows = [
            _base_row(framework="fandhe-ai"),  # parity キーなし = 旧形式
            _with_parity(_base_row(framework="candle")),
        ]
        unverified = summarize.gemm_parity_unverified(rows)
        self.assertEqual(len(unverified), 1)
        self.assertEqual(unverified[0]["framework"], "fandhe-ai")


class SectionRenderingTests(unittest.TestCase):
    def test_fail_row_marked_invalid_with_dash_gflops(self):
        rows = [_with_parity(_base_row(), fail_count=5, max_abs_err=1.2e-3, max_rel_err=4.5e-2)]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, has_checksum_mismatch, has_parity_failure = summarize.section(
                "dummy.jsonl", rows
            )
        text = "\n".join(lines)
        self.assertTrue(has_parity_failure)
        self.assertFalse(has_checksum_mismatch)
        self.assertIn("無効: 要素誤差超過", text)
        self.assertIn("fail=5/65536", text)
        # 無効行の GFLOP/s 列は "-"（性能値として見せない）。
        self.assertIn("| - |", text)

    def test_ok_row_not_marked_invalid(self):
        rows = [_with_parity(_base_row())]
        lines, has_checksum_mismatch, has_parity_failure = summarize.section(
            "dummy.jsonl", rows
        )
        text = "\n".join(lines)
        self.assertFalse(has_parity_failure)
        self.assertFalse(has_checksum_mismatch)
        self.assertNotIn("無効", text)

    def test_old_format_row_reported_as_unverified_not_invalid(self):
        rows = [_base_row()]
        lines, has_checksum_mismatch, has_parity_failure = summarize.section(
            "dummy.jsonl", rows
        )
        text = "\n".join(lines)
        self.assertFalse(has_parity_failure)
        self.assertIn("未検証（旧形式）", text)
        self.assertNotIn("無効", text)

    def test_checksum_and_parity_reasons_are_both_listed(self):
        # 同一 size に相互一致するクラスタが無いと checksum 突合不能になる
        # ため、複数フレームワークを用意して不一致を作る。
        rows = [
            _with_parity(_base_row(framework="fandhe-ai", checksum=100.0)),
            _with_parity(
                _base_row(framework="candle", checksum=999.0),  # checksum 不一致
                fail_count=1,  # かつ要素誤差超過
            ),
        ]
        lines, has_checksum_mismatch, has_parity_failure = summarize.section(
            "dummy.jsonl", rows
        )
        text = "\n".join(lines)
        self.assertTrue(has_checksum_mismatch)
        self.assertTrue(has_parity_failure)
        self.assertIn("checksum 不一致", text)
        self.assertIn("要素誤差超過", text)
        # 両理由が 1 つのセルに併記されていること（セミコロン区切り）。
        self.assertIn("checksum 不一致; 要素誤差超過", text)

    def test_reuse_table_fail_row_marked_invalid_with_dash_gflops(self):
        # (a') reuse 表は (a) fresh 表と別の描画コード（`fw_col`/`gflops_col`
        # の組み立てが独立した分岐）のため、fresh 側のテストとは独立に
        # 検証する（advisor 指摘: reuse 表の無効行描画は未検証だった）。
        row = _with_parity(_base_row(mode="reuse"), fail_count=2)
        row["init_s"] = 0.01
        rows = [row]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, _, has_parity_failure = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertTrue(has_parity_failure)
        self.assertIn("(a')", text)
        self.assertIn("無効: 要素誤差超過", text)
        self.assertIn("fail=2/65536", text)
        self.assertIn("| - |", text)

    def test_reuse_table_ok_row_shows_gflops(self):
        row = _with_parity(_base_row(mode="reuse"))
        row["init_s"] = 0.01
        rows = [row]
        lines, _, has_parity_failure = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertFalse(has_parity_failure)
        self.assertIn("(a')", text)
        self.assertNotIn("無効", text)
        self.assertIn("100.0", text)  # gflops=100.0（_base_row の既定値）


class MainStrictExitCodeTests(unittest.TestCase):
    def _run_main(self, path, strict=False):
        argv = [path]
        if strict:
            argv.append("--strict")
        old_argv = sys.argv
        sys.argv = ["summarize.py", *argv]
        buf_out, buf_err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                code = summarize.main()
        finally:
            sys.argv = old_argv
        return code, buf_out.getvalue(), buf_err.getvalue()

    def test_parity_failure_triggers_strict_exit_2(self):
        path = _write_jsonl([_with_parity(_base_row(), fail_count=1)])
        try:
            code, _, err = self._run_main(path, strict=True)
            self.assertEqual(code, 2)
            self.assertIn("要素単位検証の閾値超過", err)
        finally:
            os.unlink(path)

    def test_parity_failure_without_strict_exits_0(self):
        path = _write_jsonl([_with_parity(_base_row(), fail_count=1)])
        try:
            code, _, _ = self._run_main(path, strict=False)
            self.assertEqual(code, 0)
        finally:
            os.unlink(path)

    def test_all_ok_exits_0_with_strict(self):
        path = _write_jsonl([_with_parity(_base_row())])
        try:
            code, _, _ = self._run_main(path, strict=True)
            self.assertEqual(code, 0)
        finally:
            os.unlink(path)

    def test_old_format_exits_0_with_strict(self):
        path = _write_jsonl([_base_row()])
        try:
            code, _, _ = self._run_main(path, strict=True)
            self.assertEqual(code, 0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
