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
import re
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


def _train_row(
    framework="fandhe-ai", device="cpu", mode="fresh", median_s=0.01, checksum=0.08, init_s=None
):
    """train タスク（(b)/(b') 節）用の合成行（イシュー #957/#958/#959）。

    `_base_row` は task="gemm" 固定・`gflops` 必須のため train 行には流用
    しない（gemm と train は集計対象列が異なる。summarize.py `Record` 参照）。
    """
    return {
        "framework": framework,
        "version": "0.4.0",
        "task": "train",
        "device": device,
        "size": 64,
        "median_s": median_s,
        "q1_s": median_s * 0.9,
        "q3_s": median_s * 1.1,
        "gflops": None,
        "checksum": checksum,
        "warmup": 20,
        "iters": 80,
        "mode": mode,
        "init_s": init_s,
    }


class ParityStatusTests(unittest.TestCase):
    def test_missing_keys_is_unverified(self):
        row = _base_row()
        self.assertEqual(summarize.parity_status(row), "unverified")

    def test_zero_fail_count_is_ok(self):
        row = _with_parity(_base_row())
        self.assertEqual(summarize.parity_status(row), "ok")

    def test_partial_missing_keys_is_fail_not_unverified(self):
        # イシュー #970 PR #978 codex-review P0 指摘3: parity_fail_count
        # のみが欠落し他 3 キー（parity_total 等）が存在する外部 JSONL は
        # 旧形式（4 キー全欠損）ではなく部分的な破損・改変とみなし
        # "unverified" ではなく "fail" にする（fail-open 防止）。
        row = _with_parity(_base_row())
        del row["parity_fail_count"]
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_partial_missing_total_is_fail_not_unverified(self):
        row = _with_parity(_base_row())
        del row["parity_total"]
        self.assertEqual(summarize.parity_status(row), "fail")

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

    # --- イシュー #970 PR #978 codex-review P0 指摘の回帰テスト ---
    # 型が数値であっても値域・有限性を検証しない実装は、外部 JSONL の
    # 不正な parity 値を "ok" 判定へ通してしまっていた（fail-open）。
    # 以下は境界値ごとに fail-closed になることを確認する。

    def test_negative_fail_count_is_fail(self):
        row = _with_parity(_base_row(), fail_count=-1)
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_zero_total_is_fail(self):
        # parity_total=0 は「比較対象要素数ゼロ」であり検証したことになら
        # ない。fail_count=0 と組み合わせても "ok" にしてはならない。
        row = _with_parity(_base_row(), total=0, fail_count=0)
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_negative_total_is_fail(self):
        row = _with_parity(_base_row(), total=-1, fail_count=0)
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_fail_count_exceeds_total_is_fail(self):
        row = _with_parity(_base_row(), total=10, fail_count=11)
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_fail_count_equals_total_is_fail(self):
        # fail_count == total > 0 は境界値だが、fail_count > 0 の既存分岐で
        # 引き続き "fail" になることを確認する（値域検証追加後の非後退）。
        row = _with_parity(_base_row(), total=10, fail_count=10)
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_negative_max_abs_err_is_fail(self):
        row = _with_parity(_base_row(), max_abs_err=-1e-6)
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_negative_max_rel_err_is_fail(self):
        row = _with_parity(_base_row(), max_rel_err=-1e-6)
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_nan_max_abs_err_is_fail(self):
        # JSON の json.loads は既定で NaN/Infinity を受理する（RFC 8259
        # 非準拠の拡張）ため、型検査だけでは弾けない非有限値を明示検証する。
        row = _with_parity(_base_row())
        row["parity_max_abs_err"] = float("nan")
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_infinite_max_rel_err_is_fail(self):
        row = _with_parity(_base_row())
        row["parity_max_rel_err"] = float("inf")
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_nan_fail_count_is_fail(self):
        row = _with_parity(_base_row())
        row["parity_fail_count"] = float("nan")
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_boundary_fail_count_equals_zero_total_positive_is_ok(self):
        # 値域検証を追加しても、正当な "ok" ケース（fail_count=0,
        # total>0 かつ size*size と一致）が誤って fail 化されない非後退確認。
        # size=1 の正方行列は要素数 1 のため total=1 が正しい期待値になる
        # （size のデフォルト値 256 のままだと total=1 は期待要素数
        # 65536 と不一致になり、期待要素数検証〈イシュー #970 PR #978
        # codex-review P0 指摘2〉により意図的に "fail" 化される）。
        row = _with_parity(_base_row(size=1), total=1, fail_count=0)
        self.assertEqual(summarize.parity_status(row), "ok")

    def test_total_mismatched_with_size_squared_is_fail(self):
        # イシュー #970 PR #978 codex-review P0 指摘2: parity_total が
        # GEMM の期待要素数（size*size）と一致しない場合、たとえ
        # parity_total>0・parity_fail_count<=total であっても「ok」に
        # してはならない（破損・改変 JSONL が結果の一部しか検証していない
        # のを見逃さない）。
        row = _with_parity(_base_row(size=256), total=1, fail_count=0)
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_total_matches_size_squared_is_ok(self):
        # 上記の対比: size*size と完全一致する total は "ok" のまま。
        row = _with_parity(_base_row(size=256), total=65536, fail_count=0)
        self.assertEqual(summarize.parity_status(row), "ok")

    def test_non_integer_total_is_fail(self):
        # parity_total が整数値でない（例: 65536.5）場合は不正入力として
        # fail-closed で "fail" にする（イシュー #970 PR #978 codex-review
        # P0 指摘2: fail_count/total の整数性検証）。
        row = _with_parity(_base_row(size=256), total=65536.5, fail_count=0)
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_non_integer_fail_count_is_fail(self):
        row = _with_parity(_base_row(size=256), total=65536, fail_count=0.5)
        self.assertEqual(summarize.parity_status(row), "fail")

    # --- イシュー #970 PR #978 codex-review P0 指摘（巨大整数）の回帰テスト ---
    # `_is_plain_number` が `int` にも `math.isfinite()` を適用していた
    # 実装は、桁数の大きい `int`（Python は任意精度）で `float` 変換に
    # 失敗し `OverflowError` を送出して集計全体が例外終了していた
    # （fail-closed 契約違反）。以下は例外を送出せず "fail" を返すことを
    # 確認する非後退テスト。

    def test_huge_parity_total_is_fail_without_exception(self):
        row = _with_parity(_base_row(size=256), total=10**1000, fail_count=0)
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_huge_parity_fail_count_is_fail_without_exception(self):
        row = _with_parity(_base_row(size=256), total=65536, fail_count=10**1000)
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_huge_size_is_fail_without_exception(self):
        row = _with_parity(_base_row(size=10**1000), total=65536, fail_count=0)
        self.assertEqual(summarize.parity_status(row), "fail")

    def test_huge_max_abs_err_does_not_raise(self):
        # max_abs_err 自体は fail_count>0 と組み合わせないと "fail" になら
        # ないため、`_parity_reason` の指数表記フォーマット
        # （`_format_maybe_huge`）が巨大整数で例外送出しないことを、
        # "fail" 行（fail_count>0）で直接確認する。
        row = _with_parity(_base_row(size=256), total=65536, fail_count=1)
        row["parity_max_abs_err"] = 10**1000
        self.assertEqual(summarize.parity_status(row), "fail")
        reason = summarize._parity_reason(row)
        self.assertIn(str(10**1000), reason)


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
            lines, has_checksum_mismatch, has_parity_failure, _ = summarize.section(
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
        lines, has_checksum_mismatch, has_parity_failure, _ = summarize.section(
            "dummy.jsonl", rows
        )
        text = "\n".join(lines)
        self.assertFalse(has_parity_failure)
        self.assertFalse(has_checksum_mismatch)
        self.assertNotIn("無効", text)

    def test_old_format_row_reported_as_unverified_not_invalid(self):
        rows = [_base_row()]
        lines, has_checksum_mismatch, has_parity_failure, has_unverified = (
            summarize.section("dummy.jsonl", rows)
        )
        text = "\n".join(lines)
        self.assertFalse(has_parity_failure)
        # 「無効（要素誤差超過）」とは区別されるが、要素単位検証を一度も
        # 受けていない点では検証済みと同列に扱えないため、旧形式行の
        # 存在は has_unverified（--strict 対象フラグ）で個別に伝える。
        self.assertTrue(has_unverified)
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
        lines, has_checksum_mismatch, has_parity_failure, _ = summarize.section(
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
            lines, _, has_parity_failure, _ = summarize.section("dummy.jsonl", rows)
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
        lines, _, has_parity_failure, _ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertFalse(has_parity_failure)
        self.assertIn("(a')", text)
        self.assertNotIn("無効", text)
        self.assertIn("100.0", text)  # gflops=100.0（_base_row の既定値）


class TrainReuseSectionTests(unittest.TestCase):
    """(b') train reuse 節の集計（イシュー #957/#958/#959）。"""

    def test_no_train_reuse_rows_omits_section(self):
        # 旧 JSONL（train fresh のみ、reuse 行なし）では (b') を出力しない
        # （互換維持。モジュール docstring 参照）。
        rows = [_train_row(mode="fresh")]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertNotIn("(b')", text)

    def test_reuse_row_renders_init_and_fresh_reference(self):
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
            _train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=0.005),
        ]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("(b')", text)
        self.assertIn("5.000 ms", text)  # init_s=0.005 の fmt_ms 表示
        self.assertIn("2.00 倍", text)  # fresh 0.02 / reuse 0.01
        # (b') 表の行自体（gemm 側「データ有効性」節の無関係な「突合不能」
        # 記述と混同しないよう、(b') 行のみを取り出して検証する）。
        # (b) 表・(b') 表とも "| cpu |" で始まるため列数（"|" の個数）で区別
        # する: (b) は 5 列（6 個の "|"）、(b') は 9 列（10 個の "|"）。
        b_prime_row = next(
            line for line in lines if line.startswith("| cpu |") and line.count("|") == 10
        )
        self.assertIn("一致", b_prime_row)
        self.assertNotIn("突合不能", b_prime_row)

    def test_reuse_row_without_fresh_shows_unmeasured(self):
        rows = [_train_row(mode="reuse", init_s=0.005)]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("(b')", text)
        self.assertIn("未計測", text)
        self.assertIn("突合不能", text)
        self.assertIn("| - |", text)  # fresh/reuse 比の "-" 列

    def test_reuse_loss_mismatch_marked_invalid(self):
        # 契約外の乖離（不一致）。
        rows = [
            _train_row(mode="fresh", checksum=0.08),
            _train_row(mode="reuse", checksum=0.20),
        ]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("無効: fresh と最終 loss 不一致", text)
        self.assertIn("不一致", buf.getvalue())

        # 契約内（複合判定: 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）は
        # 「一致」表示のまま。
        rows_ok = [
            _train_row(mode="fresh", checksum=0.080541),
            _train_row(mode="reuse", checksum=0.080542),
        ]
        lines_ok, *_ = summarize.section("dummy.jsonl", rows_ok)
        text_ok = "\n".join(lines_ok)
        self.assertNotIn("無効: fresh と最終 loss 不一致", text_ok)
        self.assertIn("一致", text_ok)

    def test_train_reuse_rows_do_not_affect_gemm_checksum_reference(self):
        # gemm の checksum 突合（`gemm_checksum_mismatches`）は task でフィルタ
        # 済みのはずで、train reuse 行を混ぜても結果が変わらないことの回帰確認。
        gemm_only = [
            _with_parity(_base_row(framework="fandhe-ai", checksum=100.0)),
            _with_parity(_base_row(framework="candle", checksum=100.0)),
        ]
        mismatches_before = summarize.gemm_checksum_mismatches(gemm_only)
        mixed = gemm_only + [
            _train_row(mode="fresh", checksum=0.08),
            _train_row(mode="reuse", checksum=999.0),  # gemm 突合には無関係
        ]
        mismatches_after = summarize.gemm_checksum_mismatches(mixed)
        self.assertEqual(len(mismatches_before), len(mismatches_after))
        self.assertEqual(mismatches_before, mismatches_after)

    def test_main_strict_exit_code_unaffected_by_train_reuse_rows(self):
        # train reuse の最終 loss 不一致は --strict の対象（4-tuple・exit code）
        # を変えない設計（計画 §4.2「`section()` の戻り値 4-tuple・`--strict`
        # の判定は変更しない」）。gemm 側が全て正常なら train 側に不一致が
        # あっても exit 0 のまま。
        path = _write_jsonl(
            [
                _with_parity(_base_row()),
                _train_row(mode="fresh", checksum=0.08),
                _train_row(mode="reuse", checksum=999.0),
            ]
        )
        old_argv = sys.argv
        sys.argv = ["summarize.py", path, "--strict"]
        buf_out, buf_err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                code = summarize.main()
        finally:
            sys.argv = old_argv
            os.unlink(path)
        self.assertEqual(code, 0)


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

    def test_old_format_exits_nonzero_with_strict(self):
        # 旧形式（parity キー欠損）行は要素単位検証を一度も受けていない
        # ため、--strict では checksum 不一致・要素誤差超過と同様に
        # 拒否対象とする（codex-review PR #978 P1 指摘。イシュー #970）。
        path = _write_jsonl([_base_row()])
        try:
            code, _, err = self._run_main(path, strict=True)
            self.assertEqual(code, 2)
            self.assertIn("要素単位検証を受けていない", err)
        finally:
            os.unlink(path)

    def test_old_format_exits_0_without_strict(self):
        path = _write_jsonl([_base_row()])
        try:
            code, _, _ = self._run_main(path, strict=False)
            self.assertEqual(code, 0)
        finally:
            os.unlink(path)


class ToleranceDriftTests(unittest.TestCase):
    """`CHECKSUM_*`/`PARITY_*`（summarize.py）が本体 `backend-cpu::parity`
    の契約値から乖離していないことを機械照合する。

    このハーネスは本体 workspace 外の独立 workspace（`.claude/rules/
    deps-policy.md` 第 9 区分）のため本体クレートを import できず、
    閾値は summarize.py・bench-common（Rust 側）双方に再定義している。
    本体側だけが変更されると静かに乖離しうるため、代わりに本体ソースを
    直接読んで数値を照合し、乖離・読み取り失敗のいずれも fail-closed
    （test failure）で検知する（イシュー #970 codex-review 指摘・
    PR #978 P1）。Rust 側の同趣旨テストは
    `bench-common::parity::tests::parity_tolerances_match_backend_cpu_contract`。
    """

    def test_summarize_tolerances_match_backend_cpu_contract(self):
        # HERE (scripts/bench/framework-compare) からリポジトリルートまでは
        # 3 階層上（framework-compare → bench → scripts → root）。
        backend_cpu_parity_path = os.path.join(
            HERE, "..", "..", "..", "crates", "backend-cpu", "src", "parity.rs"
        )
        try:
            with open(backend_cpu_parity_path, encoding="utf-8") as f:
                source = f.read()
        except OSError as err:
            self.fail(
                f"本体 parity.rs を読めない（{backend_cpu_parity_path}）: {err}。"
                "パスがずれていないか確認すること"
            )

        rel_tol = _extract_f64_const(source, "RELATIVE_TOLERANCE")
        abs_tol = _extract_f64_const(source, "ABSOLUTE_RESCUE_THRESHOLD")

        self.assertEqual(
            rel_tol,
            summarize.CHECKSUM_REL_TOL,
            "CHECKSUM_REL_TOL/PARITY_REL_TOL が backend-cpu::parity::"
            "RELATIVE_TOLERANCE から乖離している。閾値の変更はユーザー承認"
            "必須（.claude/rules/coding-rust.md）",
        )
        self.assertEqual(
            abs_tol,
            summarize.CHECKSUM_ABS_TOL,
            "CHECKSUM_ABS_TOL/PARITY_ABS_TOL が backend-cpu::parity::"
            "ABSOLUTE_RESCUE_THRESHOLD から乖離している。閾値の変更は"
            "ユーザー承認必須（.claude/rules/coding-rust.md）",
        )


def _extract_f64_const(source, name):
    """`pub const <name>: f64 = <value>;` 形式の宣言から数値を取り出す。

    本体 `crates/backend-cpu/src/parity.rs` の宣言スタイル固定を前提に
    した簡易パーサー（正規表現クレート追加を避けるため stdlib `re` の
    みで十分）。宣言が見つからない・数値化できない場合は fail-closed に
    例外を送出する（呼び出し側の `TestCase` が失敗として報告する）。
    """
    match = re.search(
        rf"pub const {re.escape(name)}: f64 = ([^;]+);", source
    )
    if match is None:
        raise AssertionError(
            f"本体 parity.rs に `pub const {name}: f64 = ...;` の宣言が"
            "見つからない（宣言スタイルが変わった可能性）"
        )
    return float(match.group(1).strip())


if __name__ == "__main__":
    unittest.main()
