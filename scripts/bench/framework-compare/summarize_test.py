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


def _train_phases_row(
    device="cpu",
    mode="fresh",
    phase="tape_build",
    phase_index=0,
    median_s=0.001,
    init_s=None,
):
    """train_phases タスク（(b'') 節。イシュー #1009）用の合成行。

    `_train_row` と異なりフィールド集合が異なる（`phase`/`phase_index`
    を持ち `gflops` を持たない。`bench_common::PhaseRecord::to_json_line`
    参照）。
    """
    row = {
        "framework": "fandhe-ai",
        "version": "0.4.0",
        "task": "train_phases",
        "device": device,
        "size": 64,
        "median_s": median_s,
        "q1_s": median_s * 0.9,
        "q3_s": median_s * 1.1,
        "checksum": 0.08054,
        "warmup": 20,
        "iters": 80,
        "mode": mode,
        "phase": phase,
        "phase_index": phase_index,
    }
    if init_s is not None:
        row["init_s"] = init_s
    return row


def _train_phases_group(device="cpu", mode="fresh", init_s=None):
    """1 step 分の典型的な train_phases 行グループを構成する。

    `summarize._TRAIN_PHASES_REQUIRED_PHASES[mode]` の必須 phase 名を
    `phase_index` 0 始まりの連番で全件含める（`_train_phases_validate` の
    必須 phase 集合・順序・件数チェック — codex-review 指摘・PR #1055 —
    に対して「有効な group」の基準となるため、producer 側 `PHASE_*` 定数
    と同じ集合を実プロダクションの一部として合成する）。`step_total` 以外
    の `median_s` は小さめの固定値とし、その和が `step_total` の
    `median_s`（0.01）以下になるよう構成する
    （`train_phases_each_step_phase_sum_does_not_exceed_total` と同じ
    不変条件）。
    """
    phases = summarize._TRAIN_PHASES_REQUIRED_PHASES[mode]
    rows = []
    for phase_index, phase in enumerate(phases):
        median_s = 0.01 if phase == "step_total" else 0.0005
        rows.append(_train_phases_row(device, mode, phase, phase_index, median_s, init_s))
    return rows


def _infer_row(framework="fandhe-ai", device="cpu", median_s=0.0005, checksum=13.9):
    """infer タスク（(c) 節）用の合成行（イシュー #1051 のゲート判定用）。

    `_train_row` と異なり `throughput_per_s` を持ち `gflops`/`init_s` を
    持たない（実データ形状。results/raw/*.jsonl の infer 行を参照）。
    infer には reuse モード自体が無い（モジュール docstring 参照）ため
    `mode` は常に "fresh" 固定とする。
    """
    return {
        "framework": framework,
        "version": "0.4.0",
        "task": "infer",
        "device": device,
        "size": 64,
        "median_s": median_s,
        "q1_s": median_s * 0.9,
        "q3_s": median_s * 1.1,
        "throughput_per_s": 1.0 / median_s,
        "checksum": checksum,
        "warmup": 20,
        "iters": 20,
        "mode": "fresh",
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


class SafeNumberOverflowTests(unittest.TestCase):
    """イシュー #959 codex-review 2 巡目 P0 指摘の回帰テスト。

    `_is_plain_number` は任意精度の `int`（`bool`・`NaN`・`Infinity` 以外は
    有限として許容）を通すため、`10**1000` のような外部 JSONL 由来の巨大
    整数を `float()` へ渡すと `OverflowError: int too large to convert to
    float` になる。`_safe_time_s`・`_safe_finite_number` はこれを捕捉して
    `None`（無効値）へ倒すべきで、集計スクリプト全体を落としてはならない。
    """

    def test_safe_time_s_huge_int_does_not_raise(self):
        self.assertIsNone(summarize._safe_time_s(10**1000))

    def test_safe_finite_number_huge_int_does_not_raise(self):
        self.assertIsNone(summarize._safe_finite_number(10**1000))

    def test_safe_time_s_huge_negative_int_does_not_raise(self):
        self.assertIsNone(summarize._safe_time_s(-(10**1000)))

    def test_reuse_row_with_huge_int_median_does_not_raise(self):
        # `section()` 経由でも巨大整数混入時に例外終了しないことを end-to-end
        # で固定する（呼び出し元は `_safe_time_s` の戻り値のみを扱うため
        # 直接は落ちないはずだが、呼び出し経路自体を回帰対象として残す）。
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
            {
                **_train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=0.005),
                "median_s": 10**1000,
            },
        ]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("無効な値", text)

    def test_reuse_row_with_huge_int_checksum_does_not_raise(self):
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
            {
                **_train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=0.005),
                "checksum": 10**1000,
            },
        ]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("突合不能（無効値）", text)


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
            lines, has_checksum_mismatch, has_parity_failure, _, _, _ = summarize.section(
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
        lines, has_checksum_mismatch, has_parity_failure, _, _, _ = summarize.section(
            "dummy.jsonl", rows
        )
        text = "\n".join(lines)
        self.assertFalse(has_parity_failure)
        self.assertFalse(has_checksum_mismatch)
        self.assertNotIn("無効", text)

    def test_old_format_row_reported_as_unverified_not_invalid(self):
        rows = [_base_row()]
        lines, has_checksum_mismatch, has_parity_failure, has_unverified, _, _ = (
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
        lines, has_checksum_mismatch, has_parity_failure, _, _, _ = summarize.section(
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
            lines, _, has_parity_failure, _, _, _ = summarize.section("dummy.jsonl", rows)
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
        lines, _, has_parity_failure, _, _, _ = summarize.section("dummy.jsonl", rows)
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

    def test_reuse_row_with_zero_median_does_not_raise(self):
        # イシュー #959 codex-review P0 指摘: 外部 JSONL の reuse median_s が
        # 0 の行で fresh/reuse 比を除算すると ZeroDivisionError で集計全体
        # が停止していた。0 は無効データとして "計測不正" 表示に倒し、
        # 例外を送出しないことを確認する。
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
            _train_row(mode="reuse", median_s=0.0, checksum=0.08054, init_s=0.005),
        ]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("(b')", text)
        # reuse median_s=0 は無効な時間値として「無効な値」表示に倒す。
        # fresh 自体は有効なので fresh 側は「計測不正」にならない
        # （cursor bugbot 指摘の回帰確認。上の
        # `test_reuse_invalid_median_with_valid_fresh_still_shows_fresh_median`
        # と同じ設計）。
        self.assertIn("無効な値", text)

    def test_reuse_row_with_negative_median_does_not_raise(self):
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
            _train_row(mode="reuse", median_s=-0.01, checksum=0.08054, init_s=0.005),
        ]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("無効な値", text)

    def test_reuse_row_with_non_finite_fresh_median_does_not_raise(self):
        rows = [
            _train_row(mode="fresh", median_s=float("nan"), checksum=0.08054),
            _train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=0.005),
        ]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("計測不正", text)

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

    def test_reuse_row_with_non_numeric_init_s_shows_dash_not_typeerror(self):
        # イシュー #959 codex-review P0 指摘: init_s が文字列（"bad" 等）
        # の場合、旧実装は `is not None` のみ検査していたため `fmt_ms`
        # の比較演算（`s >= 1.0`）で TypeError になり集計全体が停止して
        # いた。不正値は "-"（未計測相当）に倒し、例外を送出しないことを
        # 確認する。
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
            {
                **_train_row(mode="reuse", median_s=0.01, checksum=0.08054),
                "init_s": "bad",
            },
        ]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("(b')", text)
        b_prime_row = next(
            line for line in lines if line.startswith("| cpu |") and line.count("|") == 10
        )
        self.assertIn("| - |", b_prime_row)

    def test_reuse_row_with_negative_init_s_shows_dash(self):
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
            {
                **_train_row(mode="reuse", median_s=0.01, checksum=0.08054),
                "init_s": -1.0,
            },
        ]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        b_prime_row = next(
            line for line in lines if line.startswith("| cpu |") and line.count("|") == 10
        )
        self.assertIn("| - |", b_prime_row)

    def test_reuse_row_with_invalid_q1_q3_shows_invalid_value_not_raise(self):
        # q1_s/q3_s も外部 JSONL 由来であり、bool・非有限値・文字列が
        # 混入しても `fmt_ms` へ渡す前に検証して落ちないことを確認する
        # （イシュー #959 codex-review P0 指摘）。
        row = {
            **_train_row(mode="reuse", median_s=0.01, checksum=0.08054),
            "q1_s": float("nan"),
            "q3_s": True,
        }
        lines, *_ = summarize.section("dummy.jsonl", [row])
        text = "\n".join(lines)
        self.assertIn("(b')", text)
        self.assertIn("無効な値", text)

    def test_reuse_row_with_bool_checksum_marked_unverifiable_not_raise(self):
        # checksum（最終 loss）に bool が混入すると `int` のサブクラスの
        # ため型検査だけでは通過し、`f"{x:.6f}"` 自体は成立するが数値として
        # 無意味な突合になる。`_safe_finite_number` で弾き「突合不能
        # （無効値）」表示に倒すことを確認する（イシュー #959 codex-review
        # P0 指摘）。
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
            {
                **_train_row(mode="reuse", median_s=0.01, checksum=True),
            },
        ]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("突合不能（無効値）", text)
        self.assertIn("checksum が不正な値", text)

    def test_reuse_row_with_infinite_checksum_marked_unverifiable_not_raise(self):
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=float("inf")),
            _train_row(mode="reuse", median_s=0.01, checksum=0.08054),
        ]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("突合不能（無効値）", text)

    # 以下、イシュー #959 codex-review P1 指摘の回帰テスト: `section()` の
    # 戻り値（5-tuple 目・has_train_reuse_invalid）が表示上の「無効」判定
    # と一致することを確認する。

    def test_section_flags_train_reuse_checksum_mismatch_as_invalid(self):
        rows = [
            _train_row(mode="fresh", checksum=0.08),
            _train_row(mode="reuse", checksum=999.0),
        ]
        *_, has_train_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_reuse_invalid)

    def test_section_flags_train_reuse_invalid_checksum_as_invalid(self):
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=float("inf")),
            _train_row(mode="reuse", median_s=0.01, checksum=0.08054),
        ]
        *_, has_train_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_reuse_invalid)

    def test_section_flags_train_reuse_invalid_median_as_invalid(self):
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
            _train_row(mode="reuse", median_s=-0.01, checksum=0.08054),
        ]
        *_, has_train_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_reuse_invalid)

    def test_section_does_not_flag_ok_train_reuse_row(self):
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
            _train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=0.005),
        ]
        *_, has_train_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertFalse(has_train_reuse_invalid)

    def test_section_does_not_flag_train_reuse_row_without_fresh(self):
        # fresh 欠落のみ（比較対象なしで突合不能）は値そのものの正当性を
        # 否定しないため無効扱いにしない（gemm の「突合不能（検証対象外）」
        # と同じ位置づけ）。init_s は本節が計測する必須フィールドのため
        # 有効値を明示し、「fresh 欠落」のみを分離検証する（init_s 欠損の
        # 検証は下の `test_section_flags_train_reuse_missing_init_s_as_invalid`
        # に分離。イシュー #959 codex-review 2 巡目 P0 指摘）。
        rows = [_train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=0.005)]
        *_, has_train_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertFalse(has_train_reuse_invalid)

    def test_section_flags_train_reuse_missing_init_s_as_invalid(self):
        # イシュー #959 codex-review 2 巡目 P0 指摘: reuse 行の init_s は
        # 本節（(b')）が計測する初期化コストの主対象であり必須フィールド
        # だが、旧実装は表示列（"-"）にのみ反映し `has_train_reuse_invalid`
        # へ反映していなかったため `--strict` が fail-open だった。
        rows = [_train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=None)]
        *_, has_train_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_reuse_invalid)

    def test_section_flags_train_reuse_invalid_init_s_as_invalid(self):
        rows = [_train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=-1.0)]
        *_, has_train_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_reuse_invalid)

    def test_section_flags_train_reuse_invalid_fresh_median_as_invalid(self):
        # イシュー #959 codex-review 2 巡目 P0 指摘: fresh 行自体は存在する
        # のに fresh 側 median_s（fresh/reuse 比の算出に使う値）が不正
        # （NaN 等）な場合、表示は「計測不正」になるだけで
        # `has_train_reuse_invalid` に反映されておらず `--strict` が
        # fail-open だった。fresh 行が存在しない「突合不能」（上の
        # `test_section_does_not_flag_train_reuse_row_without_fresh`）とは
        # 区別する。
        rows = [
            _train_row(mode="fresh", median_s=float("nan"), checksum=0.08054),
            _train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=0.005),
        ]
        *_, has_train_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_reuse_invalid)

    def test_main_strict_exit_code_reflects_train_reuse_missing_init_s(self):
        # `has_train_reuse_invalid` の反映が `--strict` の終了コードまで
        # 一貫していることを固定する回帰テスト（イシュー #959 codex-review
        # 2 巡目 P0 指摘）。
        path = _write_jsonl(
            [
                _with_parity(_base_row()),
                _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
                _train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=None),
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
        self.assertEqual(code, 2)
        self.assertIn("train reuse", buf_err.getvalue())

    def test_reuse_invalid_median_with_valid_fresh_still_shows_fresh_median(self):
        # cursor bugbot 指摘: reuse の median_s が無効でも、対応する fresh
        # の計測が有効なら「fresh 中央値（参考）」列に "計測不正" で
        # 上書きせず、有効な fresh 計測値をそのまま表示する。
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
            _train_row(mode="reuse", median_s=-0.01, checksum=0.08054),
        ]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        b_prime_row = next(
            line for line in lines if line.startswith("| cpu |") and line.count("|") == 10
        )
        self.assertIn("20.000 ms", b_prime_row)  # fresh median_s=0.02 の表示
        self.assertNotIn("計測不正", b_prime_row)

    def test_main_strict_exit_code_reflects_train_reuse_mismatch(self):
        # イシュー #959 codex-review P1 指摘: train reuse (b') の最終 loss
        # 不一致は表示上「無効」判定されるにもかかわらず `section()` の
        # 戻り値（4-tuple）に反映されておらず、`--strict` が fail-open
        # （終了コード 0 のまま）だった。fail-closed（終了コード 2）に
        # 修正し、本テストもその挙動を固定する（旧テスト名
        # `test_main_strict_exit_code_unaffected_by_train_reuse_rows` は
        # fail-open を固定していたため置き換え）。
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
        self.assertEqual(code, 2)
        self.assertIn("train reuse", buf_err.getvalue())

    def test_main_strict_exit_code_unaffected_by_train_reuse_fresh_missing(self):
        # fresh 行が存在しない（比較対象なしで突合不能）だけの reuse 行は
        # 値そのものの正当性を否定しないため「無効」扱いにせず、gemm 側が
        # 全て正常なら --strict でも exit 0 のまま（gemm の「突合不能
        # （検証対象外）」と同じ位置づけ）。init_s は本節の必須フィールド
        # のため有効値を明示し、「fresh 欠落」のみを分離検証する（init_s
        # 欠損側の --strict 回帰は
        # `test_main_strict_exit_code_reflects_train_reuse_missing_init_s`
        # に分離。イシュー #959 codex-review 2 巡目 P0 指摘）。
        path = _write_jsonl(
            [
                _with_parity(_base_row()),
                _train_row(mode="reuse", checksum=0.08, init_s=0.005),
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


class TrainPhasesSectionTests(unittest.TestCase):
    """(b'') train_phases 節の集計（イシュー #1009）。"""

    def test_no_train_phases_rows_omits_section(self):
        # 旧 JSONL（train_phases 行なし）では (b'') を出力しない
        # （(a')/(b') と同じ互換維持方針）。
        rows = [_train_row(mode="fresh")]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertNotIn("(b'')", text)

    def test_valid_group_renders_table_in_phase_index_order(self):
        rows = _train_phases_group(device="cpu", mode="fresh")
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("(b'')", text)
        self.assertIn("CPU / fresh", text)
        # phase_index 昇順（tape_build → forward → backward → step_total）。
        # 各 phase のテーブル行（"| <phase> |" で始まる）の出現位置で判定する
        # （列見出し「step_total 比」に "step_total" が部分文字列として
        # 含まれるため、素の `text.index(phase)` はヘッダ行を拾ってしまう）。
        order = [text.index(f"| {p} |") for p in ["tape_build", "forward", "backward", "step_total"]]
        self.assertEqual(order, sorted(order))
        self.assertIn("100.0%", text)  # step_total 行自身の比
        self.assertNotIn("無効", text)

    def test_reuse_group_shows_init_s(self):
        rows = _train_phases_group(device="cpu", mode="reuse", init_s=0.002)
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("CPU / reuse", text)
        self.assertIn("初期化(init_s): 2.000 ms", text)

    def test_missing_step_total_is_invalid_and_strict_fails(self):
        rows = [r for r in _train_phases_group() if r["phase"] != "step_total"]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            *_, has_train_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("step_total", buf.getvalue())

    def test_missing_required_non_step_total_phase_is_invalid(self):
        # codex-review 指摘（PR #1055）: `backward` 等の必須 phase 行が
        # 欠落していても `step_total` 行さえ残っていれば修正前は有効判定
        # されていた（`_train_phases_validate` が `step_total` の存在のみ
        # を検証していたため）。`phase_index` を詰め直さず欠番のまま残す
        # ことで、mode ごとの必須 phase 集合・順序チェックが単独で欠落を
        # 検出できることを固定する。
        rows = [r for r in _train_phases_group(mode="fresh") if r["phase"] != "backward"]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_, has_train_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("必須 phase 集合と不一致", buf.getvalue())
        self.assertIn("backward", buf.getvalue())

    def test_missing_required_phase_fails_with_strict_even_with_step_total(self):
        rows = [r for r in _train_phases_group(mode="reuse") if r["phase"] != "device_update"]
        path = _write_jsonl(rows)
        old_argv = sys.argv
        sys.argv = ["summarize.py", path, "--strict"]
        buf_out, buf_err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                code = summarize.main()
        finally:
            sys.argv = old_argv
            os.unlink(path)
        self.assertEqual(code, 2)
        self.assertIn("train_phases", buf_err.getvalue())

    def test_extra_unknown_phase_alongside_full_required_set_is_invalid(self):
        # 必須集合を全て満たしたうえで余剰 phase（producer 契約に無い名前）
        # が混入した場合も、件数不一致として無効化する。
        rows = _train_phases_group(mode="fresh")
        extra = dict(rows[0])
        extra["phase"] = "unexpected_extra_phase"
        extra["phase_index"] = max(r["phase_index"] for r in rows) + 1
        rows = rows + [extra]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            *_, has_train_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("必須 phase 集合と不一致", buf.getvalue())

    def test_duplicate_phase_index_is_invalid(self):
        rows = _train_phases_group()
        rows[1] = dict(rows[1])
        rows[1]["phase_index"] = rows[0]["phase_index"]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_, has_train_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("重複", "\n".join(lines))

    def test_non_string_phase_is_invalid(self):
        rows = _train_phases_group()
        rows[0] = dict(rows[0])
        rows[0]["phase"] = 123
        *_, has_train_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)

    def test_phase_with_markdown_injection_chars_is_invalid(self):
        # producer 契約は `[a-z0-9_]+`（`bench_common::validate_phase_name`）。
        # 非空文字列チェックのみでは改行・`|` を含む値が検証を素通りし表へ
        # 無加工出力される（codex-review 指摘・PR #1055）。
        rows = _train_phases_group()
        rows[0] = dict(rows[0])
        rows[0]["phase"] = "tape_build|injected\n# hijacked heading"
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_, has_train_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        text = "\n".join(lines)
        self.assertNotIn("injected", text)
        self.assertNotIn("hijacked", text)

    def test_unhashable_device_does_not_raise(self):
        # `device`/`mode` は外部 JSONL 由来のためグループ化キーへ使う前に
        # 型検証する。配列等の unhashable な値をそのまま辞書キーにすると
        # `TypeError` で集計全体が例外終了する（codex-review 指摘・PR #1055）。
        rows = _train_phases_group()
        rows[0] = dict(rows[0])
        rows[0]["device"] = ["cpu"]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_, has_train_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("(b'')", "\n".join(lines))

    def test_unallowlisted_mode_does_not_raise(self):
        rows = _train_phases_group()
        rows[0] = dict(rows[0])
        rows[0]["mode"] = "evil"
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_, has_train_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("(b'')", "\n".join(lines))

    def test_duplicate_phase_name_with_distinct_index_is_invalid(self):
        # phase_index が別々でも同じ phase 名（"step_total"）を混入させると、
        # 修正前は最初の行だけが分母として無検証に採用されていた
        # （codex-review 指摘・PR #1055）。
        rows = _train_phases_group()
        extra = dict(rows[-1])  # 2 つ目の "step_total"（phase_index だけ変える）
        extra["phase_index"] = max(r["phase_index"] for r in rows) + 1
        extra["median_s"] = 0.05
        rows = rows + [extra]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_, has_train_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        text = "\n".join(lines)
        self.assertIn("重複", text)
        # 分母が一意に決まらないため比の算出不能警告が出る（修正前は
        # 最初の "step_total" 行が無検証に分母として使われていた）。
        self.assertIn("step_total 行が欠落または不正", buf.getvalue())

    def test_negative_phase_index_is_invalid(self):
        rows = _train_phases_group()
        rows[0] = dict(rows[0])
        rows[0]["phase_index"] = -1
        *_, has_train_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)

    def test_non_finite_median_is_invalid(self):
        rows = _train_phases_group()
        rows[1] = dict(rows[1])
        rows[1]["median_s"] = float("nan")
        lines, *_, has_train_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("無効な値", "\n".join(lines))

    def test_reuse_missing_init_s_is_invalid(self):
        rows = _train_phases_group(mode="reuse", init_s=0.001)
        rows[1] = dict(rows[1])
        del rows[1]["init_s"]
        *_, has_train_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)

    def test_phase_median_exceeding_step_total_is_invalid(self):
        # 計時区間の合計が全体（step_total）を超えるのは不整合（コメント
        # 「各 phase の中央値が `step_total` の中央値を上回る」参照）。
        rows = _train_phases_group()
        rows[1] = dict(rows[1])
        rows[1]["median_s"] = rows[-1]["median_s"] * 2  # step_total の 2 倍
        rows[1]["q1_s"] = rows[1]["median_s"] * 0.9
        rows[1]["q3_s"] = rows[1]["median_s"] * 1.1
        lines, *_, has_train_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("100% を超過", "\n".join(lines))

    def test_train_phases_rows_do_not_affect_train_section(self):
        # (b)/(b') は task == "train" のみを読むため、train_phases 行を
        # 混ぜても (b)/(b') の集計結果に影響しないことを固定する。
        rows_without_phases = [_train_row(mode="fresh", median_s=0.02, checksum=0.08054)]
        rows_with_phases = rows_without_phases + _train_phases_group()
        lines_a, *_ = summarize.section("dummy.jsonl", rows_without_phases)
        lines_b, *_ = summarize.section("dummy.jsonl", rows_with_phases)
        # (b) の行（"| cpu | fandhe-ai |" で始まる 5 列の行）は不変。
        b_row_a = next(line for line in lines_a if line.startswith("| cpu | fandhe-ai |"))
        b_row_b = next(line for line in lines_b if line.startswith("| cpu | fandhe-ai |"))
        self.assertEqual(b_row_a, b_row_b)

    def test_train_phases_rows_do_not_affect_gemm_or_devices_in(self):
        gemm_rows = [_with_parity(_base_row())]
        rows = gemm_rows + _train_phases_group()
        self.assertEqual(
            summarize.gemm_checksum_mismatches(rows), summarize.gemm_checksum_mismatches(gemm_rows)
        )
        self.assertEqual(summarize.devices_in(rows, "gemm"), summarize.devices_in(gemm_rows, "gemm"))

    def test_main_strict_exit_code_reflects_train_phases_invalid(self):
        rows = [r for r in _train_phases_group() if r["phase"] != "step_total"]
        path = _write_jsonl(rows)
        old_argv = sys.argv
        sys.argv = ["summarize.py", path, "--strict"]
        buf_out, buf_err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                code = summarize.main()
        finally:
            sys.argv = old_argv
            os.unlink(path)
        self.assertEqual(code, 2)
        self.assertIn("train_phases", buf_err.getvalue())

    def test_main_strict_exit_code_unaffected_by_valid_train_phases_rows(self):
        rows = _train_phases_group()
        path = _write_jsonl(rows)
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


class TargetGateTests(unittest.TestCase):
    """`target_gate`/`target_gate_section`（イシュー #1051）の判定規則。"""

    def test_gemm_reuse_preferred_over_fresh_and_achieved(self):
        # fandhe-ai に reuse 行がある場合は reuse 側の中央値を使う
        # （実装計画 §3「fandhe-ai 側の行」）。fandhe reuse 0.5ms <
        # candle fresh 1.0ms → achieved・ratio 2.0。
        rows = [
            _with_parity(_base_row(framework="fandhe-ai", mode="fresh")),
            dict(
                _with_parity(_base_row(framework="fandhe-ai", mode="reuse")),
                median_s=0.0005,
            ),
            _with_parity(_base_row(framework="candle", mode="fresh"), total=65536),
        ]
        rows[2]["median_s"] = 0.001
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "gemm" and r["size"] == 256)
        self.assertEqual(rec["status"], "achieved")
        self.assertEqual(rec["fandhe_mode"], "reuse")
        self.assertAlmostEqual(rec["ratio"], 2.0)

    def test_fandhe_fresh_only_uses_fresh(self):
        rows = [
            _with_parity(_base_row(framework="fandhe-ai", mode="fresh")),
            _with_parity(_base_row(framework="candle", mode="fresh")),
        ]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "gemm")
        self.assertEqual(rec["fandhe_mode"], "fresh")

    def test_infer_unmet_when_fandhe_slower(self):
        rows = [
            _infer_row(framework="fandhe-ai", median_s=0.002),
            _infer_row(framework="candle", median_s=0.0005),
        ]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "infer")
        self.assertEqual(rec["status"], "unmet")
        self.assertIsNone(rec["size"])

    def test_train_undeterminable_when_target_unmeasured(self):
        rows = [_train_row(framework="fandhe-ai", device="cuda")]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "train" and r["device"] == "cuda")
        self.assertEqual(rec["status"], "undeterminable")
        self.assertIn("candle 未計測", rec["reason"])

    def test_undeterminable_when_fandhe_row_has_parity_failure(self):
        rows = [
            _with_parity(_base_row(framework="fandhe-ai"), fail_count=3),
            _with_parity(_base_row(framework="candle")),
        ]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "gemm")
        self.assertEqual(rec["status"], "undeterminable")
        self.assertIn("無効データ", rec["reason"])
        self.assertIn("fandhe-ai", rec["reason"])

    def test_undeterminable_when_train_reuse_checksum_mismatches_fresh(self):
        rows = [
            _train_row(framework="fandhe-ai", mode="fresh", checksum=0.08),
            dict(
                _train_row(framework="fandhe-ai", mode="reuse", init_s=0.001),
                checksum=999.0,
            ),
            _train_row(framework="candle", mode="fresh", median_s=0.005),
        ]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "train")
        self.assertEqual(rec["status"], "undeterminable")
        self.assertIn("無効データ", rec["reason"])

    def test_undeterminable_when_median_is_non_finite_number(self):
        # NaN/文字列/巨大 int が混入しても例外で落ちず判定不能に倒す
        # （`_safe_time_s` の fail-closed 契約と同じ。イシュー #970 系の
        # 教訓を踏襲）。
        for bad_median in (float("nan"), "not-a-number", 10**1000, True):
            with self.subTest(bad_median=bad_median):
                rows = [
                    dict(_with_parity(_base_row(framework="fandhe-ai")), median_s=bad_median),
                    _with_parity(_base_row(framework="candle")),
                ]
                records = summarize.target_gate(rows, "candle")
                rec = next(r for r in records if r["task"] == "gemm")
                self.assertEqual(rec["status"], "undeterminable")

    def test_old_format_gemm_row_is_judged_with_note(self):
        # 旧形式（parity キー欠損）行は判定は行うが備考に注記する
        # （実装計画 §3「旧形式（parity 未検証）行」）。
        rows = [
            _base_row(framework="fandhe-ai", checksum=100.0),
            _base_row(framework="candle", checksum=100.0),
        ]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "gemm")
        self.assertIn(rec["status"], ("achieved", "unmet"))
        self.assertEqual(rec["note"], "未検証（旧形式）")

    def test_target_gate_section_renders_unmet_marker_and_list(self):
        rows = [
            _infer_row(framework="fandhe-ai", median_s=0.002),
            _infer_row(framework="candle", median_s=0.0005),
        ]
        records = summarize.target_gate(rows, "candle")
        lines = summarize.target_gate_section("dummy.jsonl", records, "candle")
        text = "\n".join(lines)
        self.assertIn("**未達**", text)
        self.assertIn("infer/CPU", text)

    def test_files_are_not_cross_matched(self):
        # ファイルをまたいだ突合はしない: 1 回の target_gate 呼び出しは
        # 1 ファイル分の rows のみを対象とする。片方に candle のみ・他方に
        # fandhe-ai のみのケースは、それぞれの呼び出し内で双方
        # 判定不能になる（main() がファイルごとに target_gate を呼ぶ
        # ことで保証される契約。ここでは関数単体で確認する）。
        rows_fandhe_only = [_infer_row(framework="fandhe-ai")]
        rows_candle_only = [_infer_row(framework="candle")]
        rec_a = next(
            r for r in summarize.target_gate(rows_fandhe_only, "candle") if r["task"] == "infer"
        )
        rec_b = next(
            r for r in summarize.target_gate(rows_candle_only, "candle") if r["task"] == "infer"
        )
        self.assertEqual(rec_a["status"], "undeterminable")
        self.assertIn("candle 未計測", rec_a["reason"])
        self.assertEqual(rec_b["status"], "undeterminable")
        self.assertIn("fandhe-ai 未計測", rec_b["reason"])

    def test_missing_task_for_measured_device_is_undeterminable(self):
        # P0 回帰（イシュー #1051 codex-review 指摘）: cpu で infer は
        # fandhe-ai/candle 双方計測済みだが、gemm/train が丸ごと未計測
        # （実行時失敗等で 0 件）の場合、以前は devices_in がタスク単位
        # だったため cpu×gemm/train の組そのものが列挙されず「全達成」に
        # 混入しなかった。デバイス集合をファイル横断に変更した現在は
        # cpu×gemm/cpu×train の組が判定不能として明示的に列挙される。
        rows = [
            _infer_row(framework="fandhe-ai", device="cpu"),
            _infer_row(framework="candle", device="cpu"),
        ]
        records = summarize.target_gate(rows, "candle")
        gemm_rec = next(r for r in records if r["task"] == "gemm" and r["device"] == "cpu")
        train_rec = next(r for r in records if r["task"] == "train" and r["device"] == "cpu")
        self.assertEqual(gemm_rec["status"], "undeterminable")
        self.assertIn("gemm 未計測", gemm_rec["reason"])
        self.assertEqual(train_rec["status"], "undeterminable")
        self.assertIn("fandhe-ai 未計測", train_rec["reason"])

    def test_devices_restricted_to_fandhe_and_target_framework(self):
        # P1 回帰（イシュー #1051 codex-review 指摘）: burn 専用デバイス
        # （metal）は --target candle 指定時のデバイス集合に含めない。
        # 含めてしまうと、candle 側が計測していない metal に対して
        # 「candle 未計測」の判定不能レコードが生成されてしまう
        # （burn/candle いずれも計測していないだけで判定不能とすべき
        # 対象ではない）。
        rows = [
            _infer_row(framework="fandhe-ai", device="cpu"),
            _infer_row(framework="candle", device="cpu"),
            _infer_row(framework="burn", device="metal"),
        ]
        records = summarize.target_gate(rows, "candle")
        self.assertFalse(any(r["device"] == "metal" for r in records))
        self.assertTrue(any(r["device"] == "cpu" for r in records))


class MainTargetExitCodeTests(unittest.TestCase):
    def _run_main(self, path, target=None, strict=False):
        argv = [path]
        if target:
            argv += ["--target", target]
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

    def test_unmet_exits_3_and_reports_stderr(self):
        path = _write_jsonl(
            [
                _infer_row(framework="fandhe-ai", median_s=0.002),
                _infer_row(framework="candle", median_s=0.0005),
            ]
        )
        try:
            code, out, err = self._run_main(path, target="candle")
            self.assertEqual(code, 3)
            self.assertIn("未達", err)
            self.assertIn("目標達成ゲート", out)
        finally:
            os.unlink(path)

    def test_all_achieved_exits_0(self):
        # イシュー #1051 P0 修正（codex-review 指摘）: `target_gate` は
        # デバイス集合をファイル全体（gemm/train/infer 横断）から導出する
        # ため、GATE_TASKS の一部タスクが丸ごと欠落したまま「全達成」を
        # 装うのを避ける目的で、gemm/train/infer の 3 タスク全てを
        # fandhe-ai/candle 双方で満たす完全なフィクスチャにする。
        path = _write_jsonl(
            [
                _with_parity(_base_row(framework="fandhe-ai", checksum=1.0)),
                _with_parity(_base_row(framework="candle", checksum=1.0)),
                _train_row(framework="fandhe-ai", checksum=0.08, median_s=0.0005),
                _train_row(framework="candle", checksum=0.09, median_s=0.01),
                _infer_row(framework="fandhe-ai", median_s=0.0001),
                _infer_row(framework="candle", median_s=0.0005),
            ]
        )
        try:
            code, _, _ = self._run_main(path, target="candle")
            self.assertEqual(code, 0)
        finally:
            os.unlink(path)

    def test_without_target_exits_0_unchanged(self):
        # 既存契約（--target 省略時は挙動不変）の回帰確認。
        path = _write_jsonl(
            [
                _infer_row(framework="fandhe-ai", median_s=0.002),
                _infer_row(framework="candle", median_s=0.0005),
            ]
        )
        try:
            code, _, _ = self._run_main(path)
            self.assertEqual(code, 0)
        finally:
            os.unlink(path)

    def test_gate_records_empty_exits_3_not_0(self):
        # P0 回帰（イシュー #1051 codex-review 指摘）: 入力 JSONL に
        # fandhe-ai/target いずれの行も存在しない（ここでは burn のみ）
        # 場合、以前は gate_records_all が空のまま unmet=0・
        # undeterminable=0 で exit 0（「全達成」の誤判定）になっていた。
        # 計測対象が丸ごと欠落した入力は判定不能として非ゼロ終了する。
        path = _write_jsonl([_infer_row(framework="burn", median_s=0.002)])
        try:
            code, out, err = self._run_main(path, target="candle")
            self.assertEqual(code, 3)
            self.assertIn("判定不能", err)
            self.assertIn("目標達成ゲート", out)
        finally:
            os.unlink(path)

    def test_target_outside_allowlist_is_argparse_error(self):
        path = _write_jsonl([_infer_row(framework="fandhe-ai")])
        try:
            old_argv = sys.argv
            sys.argv = ["summarize.py", path, "--target", "fandhe-ai"]
            buf_err = io.StringIO()
            try:
                with contextlib.redirect_stderr(buf_err), self.assertRaises(SystemExit) as ctx:
                    summarize.main()
                self.assertEqual(ctx.exception.code, 2)
            finally:
                sys.argv = old_argv
        finally:
            os.unlink(path)

    def test_strict_takes_priority_over_gate_exit_code(self):
        # 旧形式 gemm 行（--strict 対象）と unmet な gemm ゲート判定が
        # 両立する場合、データ無効の解消を優先し終了コードは 2
        # （実装計画 §3「--strict と併用し旧形式 gemm 行 → exit 2 が
        # 優先される」）。
        path = _write_jsonl(
            [
                dict(_base_row(framework="fandhe-ai"), median_s=0.002),
                dict(_base_row(framework="candle"), median_s=0.0005),
            ]
        )
        try:
            code, _, err = self._run_main(path, target="candle", strict=True)
            self.assertEqual(code, 2)
            self.assertIn("要素単位検証を受けていない", err)
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
