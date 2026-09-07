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
import shutil
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


def _gemm_phases_row(device, size, phase, phase_index, median_s, init_s=0.05):
    """`gemm_phases` タスク（(a'') 節。イシュー #1182）用の合成行。

    `_train_phases_row` と同型だが `size` を持ち（gemm は N 別に集計する）
    `mode` は常に "reuse"（`bench-fandhe` の producer 側が reuse のみを
    出力する。モジュール docstring 参照）。
    """
    return {
        "framework": "fandhe-ai",
        "version": "0.6.0",
        "task": "gemm_phases",
        "device": device,
        "size": size,
        "median_s": median_s,
        "q1_s": median_s * 0.9,
        "q3_s": median_s * 1.1,
        "checksum": 12345.6,
        "warmup": 19,
        "iters": 20,
        "mode": "reuse",
        "init_s": init_s,
        "phase": phase,
        "phase_index": phase_index,
    }


def _gemm_phases_group(device="cuda", size=1024, init_s=0.05):
    """1 反復分の典型的な gemm_phases 行グループを構成する。

    `summarize._GEMM_PHASES_REQUIRED_PHASES["reuse"]` の必須 phase 名を
    `phase_index` 0 始まりの連番で全件含める（`_train_phases_group` と
    同じ理由）。`iter_total` 以外の `median_s` の和が `iter_total`
    （0.01）以下になるよう構成する。
    """
    phases = summarize._GEMM_PHASES_REQUIRED_PHASES["reuse"]
    rows = []
    for phase_index, phase in enumerate(phases):
        median_s = 0.01 if phase == "iter_total" else 0.002
        rows.append(_gemm_phases_row(device, size, phase, phase_index, median_s, init_s))
    return rows


def _infer_row(
    framework="fandhe-ai",
    device="cpu",
    median_s=0.0005,
    checksum=13.9,
    mode="fresh",
    init_s=None,
):
    """infer タスク（(c)/(c') 節）用の合成行（イシュー #1051 のゲート判定用・
    イシュー #1217 で reuse 行にも対応）。

    `_train_row` と異なり `throughput_per_s` を持ち `gflops` を持たない
    （実データ形状。results/raw/*.jsonl の infer 行を参照）。`mode="reuse"`
    のとき `init_s`（既定 0.0002）を含める（`_train_row` の `init_s` 引数
    と同型。省略時 `mode="fresh"` は `init_s` キー自体を持たない、実際の
    `bench_common::Record::to_json_line` の `init_s: None` 省略と対応）。
    """
    row = {
        "framework": framework,
        "version": "0.6.0",
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
        "mode": mode,
    }
    if mode == "reuse":
        row["init_s"] = 0.0002 if init_s is None else init_s
    return row


def _infer_phases_row(device, mode, phase, phase_index, median_s, init_s=0.0002):
    """`infer_phases` タスク（(c'') 節。イシュー #1217）用の合成行。

    `_gemm_phases_row` と同型だが `mode` が `"fresh"`/`"reuse"` の双方を
    取りうる（`_GEMM_PHASES_REQUIRED_PHASES` は `"reuse"` 固定だが
    `_INFER_PHASES_REQUIRED_PHASES` は `(mode, device_class)` キー）。
    `init_s` は reuse 行にのみ含める。
    """
    row = {
        "framework": "fandhe-ai",
        "version": "0.6.0",
        "task": "infer_phases",
        "device": device,
        "size": 64,
        "median_s": median_s,
        "q1_s": median_s * 0.9,
        "q3_s": median_s * 1.1,
        "checksum": 13.9,
        "warmup": 20,
        "iters": 20,
        "mode": mode,
        "phase": phase,
        "phase_index": phase_index,
    }
    if mode == "reuse":
        row["init_s"] = init_s
    return row


def _infer_phases_group(device="cpu", mode="fresh"):
    """1 反復分の典型的な infer_phases 行グループを構成する
    （`_gemm_phases_group` と同じ理由）。`device_class` は `device` から
    `summarize._infer_phases_device_class` で決まる。
    """
    device_class = summarize._infer_phases_device_class(device)
    phases = summarize._INFER_PHASES_REQUIRED_PHASES[(mode, device_class)]
    rows = []
    for phase_index, phase in enumerate(phases):
        median_s = 0.001 if phase == "iter_total" else 0.0001
        rows.append(_infer_phases_row(device, mode, phase, phase_index, median_s))
    return rows


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


class LoadRowsTf32ValidationTests(unittest.TestCase):
    """`load_rows()` の `tf32` フィールド型検証（イシュー #1042
    codex-review P0 指摘・PR #1091）。`bool(r.get("tf32", False))` は
    文字列 `"false"` 等の非 bool 値も真として誤って受理する fail-open
    だったため、キー欠損（`False` 扱い）または厳密な `bool` 型のみを
    許容し、それ以外は `ValueError` でロード全体を失敗させることを
    検証する。
    """

    def test_missing_tf32_key_loads_as_untouched_row(self):
        path = _write_jsonl([_base_row()])
        try:
            rows = summarize.load_rows(path)
        finally:
            os.unlink(path)
        self.assertNotIn("tf32", rows[0])

    def test_strict_bool_true_and_false_load_unchanged(self):
        rows_in = [
            dict(_base_row(framework="fandhe-ai"), tf32=True),
            dict(_base_row(framework="candle"), tf32=False),
        ]
        path = _write_jsonl(rows_in)
        try:
            rows = summarize.load_rows(path)
        finally:
            os.unlink(path)
        self.assertIs(rows[0]["tf32"], True)
        self.assertIs(rows[1]["tf32"], False)

    def test_string_tf32_value_raises_value_error(self):
        path = _write_jsonl([dict(_base_row(), tf32="false")])
        try:
            with self.assertRaises(ValueError):
                summarize.load_rows(path)
        finally:
            os.unlink(path)

    def test_truthy_non_bool_tf32_value_raises_value_error(self):
        # `bool(1) == True` かつ `bool("x") == True` だが、いずれも
        # 外部 JSONL 由来の非 bool 型であり fail-closed で拒否する。
        for bad_value in (1, "x", [], {}, None):
            with self.subTest(bad_value=bad_value):
                path = _write_jsonl([dict(_base_row(), tf32=bad_value)])
                try:
                    with self.assertRaises(ValueError):
                        summarize.load_rows(path)
                finally:
                    os.unlink(path)


class SectionRenderingTests(unittest.TestCase):
    def test_gemm_row_with_unhashable_size_does_not_raise(self):
        # イシュー #1051 codex-review 指摘の防御的スイープ（PR #1082）:
        # (a)/(a') 節の size 集合化・`sorted()` も `target_gate` と同じ
        # `_valid_gate_size` フィルタを適用する。`main()` は `section()`
        # と `--target` ゲートを同一実行内で呼ぶため、片方だけ防御しても
        # 不正 size の gemm 行で全体が traceback 停止しうる。
        rows = [dict(_with_parity(_base_row()), size=["not", "hashable"])]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, _, _, _, _, _, _, _, _ = summarize.section("dummy.jsonl", rows)  # 例外を送出しないこと
        self.assertIn("### (a) GEMM", "\n".join(lines))

    def test_fail_row_marked_invalid_with_dash_gflops(self):
        rows = [_with_parity(_base_row(), fail_count=5, max_abs_err=1.2e-3, max_rel_err=4.5e-2)]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, has_checksum_mismatch, has_parity_failure, _, _, _, _, _, _ = summarize.section(
                "dummy.jsonl", rows
            )
        text = "\n".join(lines)
        self.assertTrue(has_parity_failure)
        self.assertFalse(has_checksum_mismatch)
        self.assertIn("無効: 要素誤差超過", text)
        self.assertIn("fail=5/65536", text)
        # 無効行の GFLOP/s 列は "-"（性能値として見せない）。
        self.assertIn("| - |", text)

    def test_tf32_section_absent_without_tf32_rows(self):
        # イシュー #1042: `--tf32` 行が 1 件も無いファイルでは
        # (a-tf32) 節自体を出力しない（既存ファイルとの表示差分を
        # 作らない）。
        rows = [_with_parity(_base_row())]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        self.assertNotIn("(a-tf32)", "\n".join(lines))

    def test_tf32_section_rendered_when_tf32_row_present(self):
        rows = [
            _with_parity(_base_row()),
            dict(_with_parity(_base_row(framework="candle")), tf32=True, gflops=200.0),
        ]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("(a-tf32) GEMM TF32", text)
        self.assertIn("| 200.0 |", text)

    def test_tf32_row_parity_failure_marked_invalid_in_tf32_section(self):
        rows = [
            dict(
                _with_parity(_base_row(), fail_count=2, max_abs_err=1e-2, max_rel_err=1e-2),
                tf32=True,
            ),
        ]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("(a-tf32)", text)
        self.assertIn("無効: 要素誤差超過", text)

    def test_ok_row_not_marked_invalid(self):
        rows = [_with_parity(_base_row())]
        lines, has_checksum_mismatch, has_parity_failure, _, _, _, _, _, _ = summarize.section(
            "dummy.jsonl", rows
        )
        text = "\n".join(lines)
        self.assertFalse(has_parity_failure)
        self.assertFalse(has_checksum_mismatch)
        self.assertNotIn("無効", text)

    def test_old_format_row_reported_as_unverified_not_invalid(self):
        rows = [_base_row()]
        lines, has_checksum_mismatch, has_parity_failure, has_unverified, _, _, _, _, _ = (
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
        lines, has_checksum_mismatch, has_parity_failure, _, _, _, _, _, _ = summarize.section(
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
            lines, _, has_parity_failure, _, _, _, _, _, _ = summarize.section("dummy.jsonl", rows)
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
        lines, _, has_parity_failure, _, _, _, _, _, _ = summarize.section("dummy.jsonl", rows)
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
        *_, has_train_reuse_invalid, _, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_reuse_invalid)

    def test_section_flags_train_reuse_invalid_checksum_as_invalid(self):
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=float("inf")),
            _train_row(mode="reuse", median_s=0.01, checksum=0.08054),
        ]
        *_, has_train_reuse_invalid, _, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_reuse_invalid)

    def test_section_flags_train_reuse_invalid_median_as_invalid(self):
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
            _train_row(mode="reuse", median_s=-0.01, checksum=0.08054),
        ]
        *_, has_train_reuse_invalid, _, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_reuse_invalid)

    def test_section_does_not_flag_ok_train_reuse_row(self):
        rows = [
            _train_row(mode="fresh", median_s=0.02, checksum=0.08054),
            _train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=0.005),
        ]
        *_, has_train_reuse_invalid, _, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertFalse(has_train_reuse_invalid)

    def test_section_does_not_flag_train_reuse_row_without_fresh(self):
        # fresh 欠落のみ（比較対象なしで突合不能）は値そのものの正当性を
        # 否定しないため無効扱いにしない（gemm の「突合不能（検証対象外）」
        # と同じ位置づけ）。init_s は本節が計測する必須フィールドのため
        # 有効値を明示し、「fresh 欠落」のみを分離検証する（init_s 欠損の
        # 検証は下の `test_section_flags_train_reuse_missing_init_s_as_invalid`
        # に分離。イシュー #959 codex-review 2 巡目 P0 指摘）。
        rows = [_train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=0.005)]
        *_, has_train_reuse_invalid, _, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertFalse(has_train_reuse_invalid)

    def test_section_flags_train_reuse_missing_init_s_as_invalid(self):
        # イシュー #959 codex-review 2 巡目 P0 指摘: reuse 行の init_s は
        # 本節（(b')）が計測する初期化コストの主対象であり必須フィールド
        # だが、旧実装は表示列（"-"）にのみ反映し `has_train_reuse_invalid`
        # へ反映していなかったため `--strict` が fail-open だった。
        rows = [_train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=None)]
        *_, has_train_reuse_invalid, _, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_reuse_invalid)

    def test_section_flags_train_reuse_invalid_init_s_as_invalid(self):
        rows = [_train_row(mode="reuse", median_s=0.01, checksum=0.08054, init_s=-1.0)]
        *_, has_train_reuse_invalid, _, _, _, _ = summarize.section("dummy.jsonl", rows)
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
        *_, has_train_reuse_invalid, _, _, _, _ = summarize.section("dummy.jsonl", rows)
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
            *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("step_total", buf.getvalue())

    # イシュー #1010: sub-100 ns 区間（`tape_build` 等）は、9 桁固定小数
    # シリアライズ自体は ns 単位を表現できるため丸まらないが、計時クロック
    # の分解能未満の標本では連続する `Instant::now()` が同一時刻を返し
    # median_s/q1_s/q3_s が 0.0 と計測されることがある。
    # `_safe_phase_time_s` は phase 行（`step_total` を除く）に限りこれを
    # 妥当な下限として許容し、`--strict` を誤って落とさない
    # （`_safe_phase_time_s` docstring 参照）。

    def test_phase_zero_median_is_valid_and_strict_passes(self):
        rows = _train_phases_group(device="cpu", mode="fresh")
        rows[0] = dict(rows[0])  # tape_build（phase_index 0・step_total 以外）
        assert rows[0]["phase"] == "tape_build"
        rows[0]["median_s"] = 0.0
        rows[0]["q1_s"] = 0.0
        rows[0]["q3_s"] = 0.0
        buf_err = io.StringIO()
        with contextlib.redirect_stderr(buf_err):
            lines, *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertFalse(has_train_phases_invalid)
        text = "\n".join(lines)
        self.assertIn("0.0 µs", text)
        self.assertNotIn("無効", text)

    def test_step_total_zero_median_is_still_invalid(self):
        # 比の分母（`step_total`）はゼロ除算回避のため引き続き 0 秒を
        # 許容しない（`_safe_time_s`（> 0）のまま。phase 行の緩和対象外）。
        rows = _train_phases_group(device="cpu", mode="fresh")
        rows = [dict(r) for r in rows]
        for r in rows:
            if r["phase"] == "step_total":
                r["median_s"] = 0.0
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)

    def test_init_s_zero_is_still_invalid(self):
        # `init_s`（reuse の初期化コスト）も緩和対象外（`_safe_time_s`
        # のまま。実測として 0 秒はあり得ないため fail-closed を維持）。
        rows = _train_phases_group(device="cpu", mode="reuse", init_s=0.0)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)

    def test_phase_negative_median_is_still_invalid(self):
        rows = _train_phases_group(device="cpu", mode="fresh")
        rows[0] = dict(rows[0])
        rows[0]["median_s"] = -0.0001
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)

    def test_phase_nan_median_is_still_invalid(self):
        rows = _train_phases_group(device="cpu", mode="fresh")
        rows[0] = dict(rows[0])
        rows[0]["median_s"] = float("nan")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)

    def test_safe_phase_time_s_accepts_zero_rejects_invalid(self):
        # `_safe_phase_time_s` 単体の境界値検証（`_safe_time_s` との差分は
        # 0 の扱いのみで、負値・NaN・Infinity・bool・非数は同じく無効）。
        self.assertEqual(summarize._safe_phase_time_s(0.0), 0.0)
        self.assertEqual(summarize._safe_phase_time_s(0), 0.0)
        self.assertIsNone(summarize._safe_phase_time_s(-0.0001))
        self.assertIsNone(summarize._safe_phase_time_s(float("nan")))
        self.assertIsNone(summarize._safe_phase_time_s(float("inf")))
        self.assertIsNone(summarize._safe_phase_time_s(True))
        self.assertIsNone(summarize._safe_phase_time_s("0.0"))
        self.assertIsNone(summarize._safe_phase_time_s(10**1000))

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
            lines, *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
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
            *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("必須 phase 集合と不一致", buf.getvalue())

    def test_duplicate_phase_index_is_invalid(self):
        rows = _train_phases_group()
        rows[1] = dict(rows[1])
        rows[1]["phase_index"] = rows[0]["phase_index"]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("重複", "\n".join(lines))

    def test_non_string_phase_is_invalid(self):
        rows = _train_phases_group()
        rows[0] = dict(rows[0])
        rows[0]["phase"] = 123
        *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
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
            lines, *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
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
            lines, *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("(b'')", "\n".join(lines))

    def test_unallowlisted_mode_does_not_raise(self):
        rows = _train_phases_group()
        rows[0] = dict(rows[0])
        rows[0]["mode"] = "evil"
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
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
            lines, *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
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
        *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)

    def test_non_finite_median_is_invalid(self):
        rows = _train_phases_group()
        rows[1] = dict(rows[1])
        rows[1]["median_s"] = float("nan")
        lines, *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("無効な値", "\n".join(lines))

    def test_reuse_missing_init_s_is_invalid(self):
        rows = _train_phases_group(mode="reuse", init_s=0.001)
        rows[1] = dict(rows[1])
        del rows[1]["init_s"]
        *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)

    def test_phase_median_exceeding_step_total_is_invalid(self):
        # 計時区間の合計が全体（step_total）を超えるのは不整合（コメント
        # 「各 phase の中央値が `step_total` の中央値を上回る」参照）。
        rows = _train_phases_group()
        rows[1] = dict(rows[1])
        rows[1]["median_s"] = rows[-1]["median_s"] * 2  # step_total の 2 倍
        rows[1]["q1_s"] = rows[1]["median_s"] * 0.9
        rows[1]["q3_s"] = rows[1]["median_s"] * 1.1
        lines, *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_train_phases_invalid)
        self.assertIn("100% を超過", "\n".join(lines))

    def test_managed_group_does_not_invalidate_normal_group(self):
        # イシュー #1353（github-actions レビュー指摘・2 巡目）:
        # `_train_phases_groups` が `managed` を区別しないと、`managed:true`
        # の train_phases 行が正常な device-only グループと同一
        # `(device, mode)` へ混在し、`_train_phases_validate` の
        # `phase_index`/`phase` 名重複検査を誤って発火させ、正常な行まで
        # 無効化されてしまう。同一 `(device="cuda", mode="fresh")` の
        # 正常グループと managed グループを混在させても、正常グループが
        # 無効化されないことを確認する。
        normal_group = _train_phases_group(device="cuda", mode="fresh")
        managed_group = []
        for r in _train_phases_group(device="cuda", mode="fresh"):
            r = dict(r)
            r["managed"] = True
            managed_group.append(r)
        rows = normal_group + managed_group
        lines, *_, has_train_phases_invalid, _, _, _ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertFalse(has_train_phases_invalid)
        self.assertNotIn("無効", text)
        self.assertIn("CUDA / fresh", text)

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


class GemmPhasesSectionTests(unittest.TestCase):
    """(a'') gemm_phases 節の集計（イシュー #1182）。`TrainPhasesSectionTests`
    と同型のテスト群。
    """

    def test_no_gemm_phases_rows_omits_section(self):
        # 旧 JSONL（gemm_phases 行なし）では (a'') を出力しない
        # （`_train_phases_group` テストと同じ互換維持方針）。
        rows = [_with_parity(_base_row())]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertNotIn("(a'')", text)

    def test_valid_group_renders_table_in_phase_index_order(self):
        rows = _gemm_phases_group(device="cuda", size=1024)
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("(a'')", text)
        self.assertIn("CUDA / reuse / N=1024", text)
        order = [
            text.index(f"| {p} |")
            for p in ["matmul", "to_tensor", "host_copy", "checksum", "iter_total"]
        ]
        self.assertEqual(order, sorted(order))
        self.assertIn("100.0%", text)  # iter_total 行自身の比
        self.assertNotIn("無効", text)

    def test_group_shows_init_s(self):
        rows = _gemm_phases_group(device="cuda", size=2048, init_s=0.123)
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("CUDA / reuse / N=2048", text)
        self.assertIn("初期化(init_s): 123.000 ms", text)

    def test_multiple_runs_same_device_mode_size_are_not_flagged_as_duplicates(self):
        # Cursor Bugbot 指摘（イシュー #1182・PR #1195）: producer
        # （`bench-fandhe`）は実行識別子を出力しないため、同一
        # (device, mode, size) に対しハーネスを複数回実行した結果を
        # そのまま 1 つの raw JSONL へ追記すると `phase_index` 列
        # （各回とも 0 始まり）が同一グループへ混在する
        # （実例: `results/raw/results-dgx-gemm-phases-0.6.0-extra.jsonl`）。
        # `_gemm_phases_split_runs` による実行単位分割前は、2 回目以降の
        # 行がすべて「phase_index が重複」＝無効データと誤判定され
        # `--strict` が exit 2 になっていた。分割後は各実行が独立に
        # 検証されて有効データのまま複数の run 表として表示されることを
        # 固定する。
        rows = (
            _gemm_phases_group(device="cuda", size=1024)
            + _gemm_phases_group(device="cuda", size=1024)
            + _gemm_phases_group(device="cuda", size=1024)
            + _gemm_phases_group(device="cuda", size=1024)
        )
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_, has_gemm_phases_invalid, _, _ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertFalse(has_gemm_phases_invalid)
        self.assertEqual(buf.getvalue(), "")
        self.assertNotIn("無効", text)
        self.assertNotIn("重複", text)
        for run_label in ["run 1/4", "run 2/4", "run 3/4", "run 4/4"]:
            self.assertIn(f"CUDA / reuse / N=1024 / {run_label}", text)

    def test_malformed_duplicate_index_does_not_reconstruct_as_valid_runs(self):
        # codex-review 指摘（P0。PR #1195）: 旧 `_gemm_phases_split_runs`
        # は `phase_index` の再出現を無条件に run 境界とみなしていたため、
        # 完全な run（0..4）の直後に重複した末尾 index（4）と不完全な
        # 残り（0..3）が続く壊れた入力を、最初の正常 run と
        # 「4,0,1,2,3」（ソート後は完全な 0..4 に見える）という
        # 2 つの「正常な」run に誤って再構成し、両方が
        # `_gemm_phases_validate` を通過してしまっていた（`--strict` の
        # fail-open）。本テストは同じ壊れた入力（[0,1,2,3,4,4,0,1,2,3]）
        # を渡し、修正後は run 分割されず 1 run のまま
        # `phase_index` 重複としてエラーになる（fail-closed）ことを固定
        # する。
        phases = summarize._GEMM_PHASES_REQUIRED_PHASES["reuse"]
        indices = [0, 1, 2, 3, 4, 4, 0, 1, 2, 3]
        rows = [
            _gemm_phases_row(
                "cuda", 1024, phases[idx], idx,
                0.01 if phases[idx] == "iter_total" else 0.002,
            )
            for idx in indices
        ]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_, has_gemm_phases_invalid, _, _ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertTrue(has_gemm_phases_invalid)
        self.assertIn("重複", buf.getvalue())
        # 壊れた入力が「2 つの正常な run」として表示されてはならない
        # （fail-open の再発防止）。
        self.assertNotIn("run 2/2", text)

    def test_single_run_header_has_no_run_suffix(self):
        # run が 1 件のみの場合は従来どおりヘッダーに run 番号を付けない
        # （既存 JSONL・`test_valid_group_renders_table_in_phase_index_order`
        # との表示互換を維持する）。
        rows = _gemm_phases_group(device="cuda", size=1024)
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("CUDA / reuse / N=1024\n", text)
        self.assertNotIn("run 1/1", text)

    def test_missing_iter_total_is_invalid_and_strict_fails(self):
        rows = [r for r in _gemm_phases_group() if r["phase"] != "iter_total"]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            *_, has_gemm_phases_invalid, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_gemm_phases_invalid)
        self.assertIn("iter_total", buf.getvalue())

    def test_different_sizes_are_grouped_separately(self):
        rows = _gemm_phases_group(size=1024) + _gemm_phases_group(size=4096)
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("N=1024", text)
        self.assertIn("N=4096", text)

    def test_phase_sum_exceeding_iter_total_is_invalid(self):
        rows = [dict(r) for r in _gemm_phases_group()]
        for r in rows:
            if r["phase"] == "matmul":
                r["median_s"] = 1.0  # iter_total（0.01）を大幅に超過
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            *_, has_gemm_phases_invalid, _, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_gemm_phases_invalid)

    def test_gemm_phases_does_not_affect_gemm_section(self):
        # イシュー #1182: `gemm_phases` 行の追加が既存 (a) GEMM 節・
        # checksum 突合・parity 集計（`task == "gemm"` のみ読む）に一切
        # 影響しないことを固定する。
        gemm_rows = [_with_parity(_base_row())]
        phases_rows = _gemm_phases_group(device="cpu", size=64)
        lines, has_checksum_mismatch, has_parity_failure, *_ = summarize.section(
            "dummy.jsonl", gemm_rows + phases_rows
        )
        text = "\n".join(lines)
        self.assertFalse(has_checksum_mismatch)
        self.assertFalse(has_parity_failure)
        self.assertIn("### (a) GEMM", text)
        self.assertIn("(a'')", text)


class InferReuseSectionTests(unittest.TestCase):
    """(c') infer reuse 節の集計（イシュー #1217）。`TrainReuseSectionTests`
    と同型の検証を `_reuse_row_invalid_reason` の一般化（train/infer 共通
    ロジック）に対して行う。
    """

    def test_no_infer_reuse_rows_omits_section(self):
        rows = [_infer_row(mode="fresh")]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertNotIn("(c')", text)

    def test_reuse_row_renders_init_throughput_and_fresh_reference(self):
        rows = [
            _infer_row(mode="fresh", median_s=0.001, checksum=13.9),
            _infer_row(mode="reuse", median_s=0.0005, checksum=13.9, init_s=0.02),
        ]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("(c')", text)
        self.assertIn("20.000 ms", text)  # init_s=0.02 の fmt_ms 表示
        self.assertIn("2.00 倍", text)  # fresh 0.001 / reuse 0.0005
        # (c') 表の行（size 列追加後は 11 列＝"|" 12 個）を fresh 表
        # (c. 6 列＝"|" 7 個) と区別する（codex-review P2 指摘・PR #1229:
        # size 列を追加し異なるバッチサイズを区別できるようにした）。
        c_prime_row = next(
            line for line in lines if line.startswith("| cpu |") and line.count("|") == 12
        )
        self.assertIn("一致", c_prime_row)
        self.assertIn("2000", c_prime_row)  # throughput_per_s = 1/0.0005

    def test_reuse_row_without_fresh_shows_unmeasured(self):
        rows = [_infer_row(mode="reuse", init_s=0.02)]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("(c')", text)
        self.assertIn("未計測", text)
        self.assertIn("突合不能", text)

    def test_reuse_checksum_mismatch_marked_invalid(self):
        rows = [
            _infer_row(mode="fresh", checksum=13.9),
            _infer_row(mode="reuse", checksum=99.0),
        ]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("無効: fresh と checksum 不一致", text)
        self.assertIn("不一致", buf.getvalue())

    def test_section_flags_infer_reuse_checksum_mismatch_as_invalid(self):
        rows = [
            _infer_row(mode="fresh", checksum=13.9),
            _infer_row(mode="reuse", checksum=99.0),
        ]
        *_, has_infer_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_infer_reuse_invalid)

    def test_section_does_not_flag_ok_infer_reuse_row(self):
        rows = [
            _infer_row(mode="fresh", median_s=0.001, checksum=13.9),
            _infer_row(mode="reuse", median_s=0.0005, checksum=13.9, init_s=0.02),
        ]
        *_, has_infer_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertFalse(has_infer_reuse_invalid)

    def test_section_does_not_flag_infer_reuse_row_without_fresh(self):
        # fresh 欠落のみ（比較対象なしで突合不能）は値そのものの正当性を
        # 否定しないため無効扱いにしない（(b') と同方針）。
        rows = [_infer_row(mode="reuse", init_s=0.02)]
        *_, has_infer_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertFalse(has_infer_reuse_invalid)

    def test_section_flags_infer_reuse_row_with_invalid_checksum_without_fresh(self):
        # codex-review P0 指摘（PR #1229）: fresh 行が存在しない場合でも
        # reuse 自身の checksum が非数値・NaN・Infinity 等で不正なら、
        # fresh の有無と独立に無効判定に含めなければならない。旧実装は
        # `if fresh:` 分岐の内側でしか checksum を検証しておらず、fresh
        # 欠落時は checksum が不正でも `has_infer_reuse_invalid` が
        # false のまま `--strict` を通過していた。
        rows = [_infer_row(mode="reuse", checksum=float("nan"), init_s=0.02)]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            lines, *_, has_infer_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertTrue(has_infer_reuse_invalid)
        self.assertIn("無効: checksum が不正な値", text)

    def test_section_flags_duplicate_infer_reuse_rows_as_invalid(self):
        # codex-review P0 指摘（PR #1229）: `get()` は同一キー
        # （framework/task/device/mode）に一致する最初の 1 行だけを返す
        # ため、正常な reuse 行の後に checksum 不一致の壊れた reuse 行が
        # 続いても後続行は一切検証されず `--strict` が不正データを通して
        # いた。同一キーの重複自体を無効条件に含める（`_pick_row_for_gate`
        # の重複キー検出と同じ fail-closed 方針）。
        rows = [
            _infer_row(mode="fresh", checksum=13.9),
            _infer_row(mode="reuse", median_s=0.0005, checksum=13.9, init_s=0.02),
            _infer_row(mode="reuse", median_s=0.0005, checksum=999.0, init_s=0.02),
        ]
        *_, has_infer_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_infer_reuse_invalid)

    def test_section_does_not_flag_different_size_infer_reuse_rows_as_duplicate(self):
        # codex-review P2 指摘（PR #1229 4 巡目）: reuse_matches/
        # fresh_matches が size を区別せずキー化していたため、同一
        # framework/device に size=64 と size=128 の正常な計測が各 1 件
        # あるだけで「重複キー」と誤判定され `--strict` が正常データを
        # 不正データとして弾いてしまっていた。size ごとに独立してキー化
        # した後は、両 size とも正常なら無効判定されないことを確認する。
        rows = [
            dict(_infer_row(mode="fresh", checksum=1.0), size=64),
            dict(
                _infer_row(mode="reuse", median_s=0.0005, checksum=1.0, init_s=0.02),
                size=64,
            ),
            dict(_infer_row(mode="fresh", checksum=2.0), size=128),
            dict(
                _infer_row(mode="reuse", median_s=0.0005, checksum=2.0, init_s=0.02),
                size=128,
            ),
        ]
        lines, *_, has_infer_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertFalse(has_infer_reuse_invalid)
        text = "\n".join(lines)
        self.assertNotIn("重複キー", text)

    def test_section_still_flags_same_size_duplicate_infer_reuse_rows_as_invalid(self):
        # 上記の size 分離が本来の重複検出（同一 size 内の重複）を
        # 無効化していないことを確認する回帰テスト。
        rows = [
            dict(_infer_row(mode="fresh", checksum=1.0), size=64),
            dict(
                _infer_row(mode="reuse", median_s=0.0005, checksum=1.0, init_s=0.02),
                size=64,
            ),
            dict(
                _infer_row(mode="reuse", median_s=0.0005, checksum=999.0, init_s=0.02),
                size=64,
            ),
        ]
        *_, has_infer_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_infer_reuse_invalid)

    def test_managed_reuse_row_does_not_trigger_false_duplicate(self):
        # イシュー #1353（github-actions レビュー指摘）: `--managed` A/B
        # 計測（`run_ab_managed_cuda.sh`）が出力する JSONL には同一
        # framework/task/device/size/mode の行が `managed:true`（on）／
        # `managed` キー欠損（off）で交互に混在する。(c') 節の集計が
        # managed 行を除外せず通常行と同一グループへ混ぜると、正常な
        # 2 行（off・on）が「重複キー」と誤判定され `--strict` を
        # 誤って失敗させる。managed:true 行を除外すれば off 単独では
        # 重複にならないことを確認する。
        rows = [
            dict(_infer_row(mode="fresh", checksum=1.0), size=64),
            dict(
                _infer_row(mode="reuse", median_s=0.0005, checksum=1.0, init_s=0.02),
                size=64,
            ),
            dict(
                _infer_row(mode="reuse", median_s=0.0009, checksum=999.0, init_s=0.02),
                size=64,
                managed=True,
            ),
        ]
        lines, *_, has_infer_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertFalse(has_infer_reuse_invalid)
        text = "\n".join(lines)
        self.assertNotIn("重複", text)

    def test_managed_fresh_row_is_excluded_from_reuse_checksum_match(self):
        # 上記と対の観点: fresh 側に managed:true 行が混在していても、
        # reuse 側（off）の突合先として誤って選ばれない（checksum が
        # 異なる managed fresh 行と誤って「不一致」判定されない）ことを
        # 確認する。
        rows = [
            dict(
                _infer_row(mode="fresh", checksum=999.0),
                size=64,
                managed=True,
            ),
            dict(_infer_row(mode="fresh", checksum=1.0), size=64),
            dict(
                _infer_row(mode="reuse", median_s=0.0005, checksum=1.0, init_s=0.02),
                size=64,
            ),
        ]
        lines, *_, has_infer_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertFalse(has_infer_reuse_invalid)
        text = "\n".join(lines)
        self.assertIn("一致", text)

    def test_infer_reuse_rows_do_not_affect_gemm_or_train_sections(self):
        gemm_rows = [_with_parity(_base_row())]
        train_rows = [_train_row(mode="fresh")]
        infer_reuse_rows = [
            _infer_row(mode="fresh", checksum=13.9),
            _infer_row(mode="reuse", checksum=13.9, init_s=0.02),
        ]
        lines, has_checksum_mismatch, has_parity_failure, *_ = summarize.section(
            "dummy.jsonl", gemm_rows + train_rows + infer_reuse_rows
        )
        text = "\n".join(lines)
        self.assertFalse(has_checksum_mismatch)
        self.assertFalse(has_parity_failure)
        self.assertIn("### (a) GEMM", text)
        self.assertIn("(c')", text)

    def test_pick_row_for_gate_prefers_infer_reuse(self):
        # `_pick_row_for_gate` は gemm/train と同様 infer でも reuse を
        # 優先する（イシュー #1217 でモードループ自体は変更していない
        # ため、reuse 行の存在だけで自然に優先される想定を固定する）。
        rows = [
            _infer_row(mode="fresh", median_s=0.001, checksum=13.9),
            _infer_row(mode="reuse", median_s=0.0004, checksum=13.9, init_s=0.02),
        ]
        row, mode, dup_reason, used_tf32 = summarize._pick_row_for_gate(
            rows, "fandhe-ai", "infer", "cpu", 64
        )
        self.assertIsNotNone(row)
        self.assertEqual(mode, "reuse")
        self.assertIsNone(dup_reason)
        self.assertFalse(used_tf32)

    def test_section_flags_duplicate_infer_fresh_rows_as_invalid(self):
        # codex-review P0 指摘（PR #1229 3 巡目）: fresh 側の取得に
        # `get()`（最初に一致した行だけを返す）を使っていたため、同一
        # キー（framework/task/device/mode）の fresh 行が複数存在すると
        # 先頭の 1 行だけが reuse と突合され、残りの不一致な fresh 行が
        # 握りつぶされていた。checksum が一致する fresh 行を先頭、不一致
        # な fresh 行を 2 番目に置いても「一致」側へすり抜けず無効判定に
        # なることを確認する（`_reuse_row_invalid_reason` の同種検証と
        # 揃える）。
        rows = [
            _infer_row(mode="fresh", checksum=13.9, median_s=0.001),
            _infer_row(mode="fresh", checksum=999.0, median_s=0.0011),
            _infer_row(mode="reuse", checksum=13.9, median_s=0.0005, init_s=0.02),
        ]
        *_, has_infer_reuse_invalid, _ = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_infer_reuse_invalid)

    def test_target_gate_infer_reuse_invalid_checksum_without_fresh_is_undeterminable(self):
        # codex-review P0 指摘（PR #1229 3 巡目）: `_reuse_row_invalid_
        # reason`（`target_gate` が `_gate_row_invalid_reason` 経由で
        # infer reuse 行の有効性判定に使う）は、fresh 行が存在しない
        # 場合に checksum 検証前に `None`（有効）を返していたため、
        # checksum が不正な reuse 行と正常な target（candle）行を
        # 組み合わせると「達成」と誤判定していた。fresh 欠落時も
        # reuse 自身の checksum を検証し判定不能へ倒すことを確認する。
        rows = [
            _infer_row(
                framework="fandhe-ai",
                mode="reuse",
                checksum=float("nan"),
                median_s=0.0001,
                init_s=0.02,
            ),
            _infer_row(framework="candle", mode="fresh", median_s=0.03),
        ]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "infer" and r["size"] == 64)
        self.assertEqual(rec["status"], "undeterminable")
        self.assertIn("無効データ", rec["reason"])


class InferPhasesSectionTests(unittest.TestCase):
    """(c'') infer_phases 節の集計（イシュー #1217）。`GemmPhasesSectionTests`
    と同型の検証を `(mode, device_class)` ごとに異なる必須 phase 集合へ
    適用する。
    """

    def test_no_infer_phases_rows_omits_section(self):
        rows = [_infer_row(mode="fresh")]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertNotIn("(c'')", text)

    def test_fresh_cpu_group_renders_without_init_s(self):
        rows = _infer_phases_group(device="cpu", mode="fresh")
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("(c'')", text)
        self.assertIn("CPU / fresh / batch=64", text)
        self.assertIn("predict", text)
        self.assertNotIn("初期化(init_s)", text)
        self.assertNotIn("無効", text)

    def test_fresh_gpu_group_includes_leaf_register_and_forward(self):
        rows = _infer_phases_group(device="metal", mode="fresh")
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("Metal / fresh / batch=64", text)
        self.assertIn("leaf_register", text)
        self.assertIn("forward", text)
        self.assertIn("to_tensor", text)

    def test_reuse_group_renders_with_init_s(self):
        rows = _infer_phases_group(device="cpu", mode="reuse")
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        self.assertIn("CPU / reuse / batch=64", text)
        self.assertIn("predict_resident", text)
        self.assertIn("初期化(init_s)", text)
        self.assertNotIn("無効", text)

    def test_missing_iter_total_is_invalid_and_strict_fails(self):
        rows = [r for r in _infer_phases_group(device="cpu", mode="fresh") if r["phase"] != "iter_total"]
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            *_, has_infer_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_infer_phases_invalid)
        self.assertIn("iter_total", buf.getvalue())

    def test_phase_set_mismatch_between_device_classes_is_invalid(self):
        # cpu fresh の必須 phase 集合（predict/host_copy/checksum/
        # iter_total）を gpu device のラベル付きで出す（欠落: leaf_register/
        # forward/to_tensor）と不一致検出される。
        rows = _infer_phases_group(device="cpu", mode="fresh")
        for r in rows:
            r["device"] = "metal"
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            *_, has_infer_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_infer_phases_invalid)
        self.assertIn("必須 phase 集合と不一致", buf.getvalue())

    def test_phase_sum_exceeding_iter_total_is_invalid(self):
        rows = [dict(r) for r in _infer_phases_group(device="cpu", mode="fresh")]
        for r in rows:
            if r["phase"] == "predict":
                r["median_s"] = 1.0  # iter_total（0.001）を大幅に超過
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            *_, has_infer_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_infer_phases_invalid)

    def test_phase_index_non_contiguous_offset_is_invalid(self):
        # codex-review P2 指摘（PR #1229）: `phase_index` の相対順序（名前の
        # 並び順）が `required` と一致していても、実際の値が 0 始まりの
        # 連番（`range(len(required))`）でなければ不正とみなす。旧実装は
        # `actual_order`（sorted(keyed) で並べた名前列）のみを見ており、
        # phase_index に 10,11,12,... のような一律オフセットが入っていても
        # 名前の相対順序さえ保たれていれば誤って有効判定していた。
        rows = [dict(r) for r in _infer_phases_group(device="cpu", mode="fresh")]
        for r in rows:
            r["phase_index"] += 10
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            *_, has_infer_phases_invalid = summarize.section("dummy.jsonl", rows)
        self.assertTrue(has_infer_phases_invalid)
        self.assertIn("0 始まり連番でない", buf.getvalue())

    def test_infer_phases_does_not_affect_infer_section(self):
        # イシュー #1217: `infer_phases` 行の追加が既存 (c)/(c') 節・
        # ゲート判定（`task == "infer"` のみ読む）に一切影響しないことを
        # 固定する（`_gemm_phases_section` と同型の独立性テスト）。
        infer_rows = [_infer_row(mode="fresh")]
        phases_rows = _infer_phases_group(device="cpu", mode="fresh")
        lines, *_ = summarize.section("dummy.jsonl", infer_rows + phases_rows)
        text = "\n".join(lines)
        self.assertIn("### (c) 推論スループット", text)
        self.assertIn("(c'')", text)


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

    def test_tf32_optin_row_excluded_from_gate(self):
        # イシュー #1042: 同一 size に TF32 opt-in 行（10 倍高速。もし
        # 誤って拾われれば ratio ≈ 10.0 になる）と FP32 行（等速）が同居
        # する場合、`target_gate` は FP32 行のみを使う（fail-open 防止。
        # `_pick_row_for_gate`/`get()` の tf32 除外フィルタ参照）。
        rows = [
            _with_parity(_base_row(framework="fandhe-ai", mode="fresh")),
            dict(
                _with_parity(_base_row(framework="fandhe-ai", mode="fresh")),
                median_s=0.0001,
                tf32=True,
            ),
            _with_parity(_base_row(framework="candle", mode="fresh")),
        ]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "gemm" and r["size"] == 256)
        self.assertAlmostEqual(rec["ratio"], 1.0)

    def test_tf32_only_size_excluded_from_gate_size_set(self):
        # イシュー #1042 codex-review P2 指摘（PR #1091）:
        # `_pick_row_for_gate` は gemm について tf32=False の行しか
        # 選ばないため、FP32 側に存在しない size を持つ TF32 専用行が
        # `candidate_rows`/`sizes` に混入すると、その size は両
        # フレームワークとも「該当行なし」として undeterminable に
        # なってしまう（size=512 は tf32=True 行しか無く、FP32 行は
        # size=256 のみ）。修正後は size=512 が sizes 集合から除外され、
        # gemm のゲート判定は size=256（achieved）の 1 件のみになる。
        rows = [
            _with_parity(_base_row(framework="fandhe-ai", size=256, mode="fresh")),
            _with_parity(_base_row(framework="candle", size=256, mode="fresh")),
            dict(
                _with_parity(_base_row(framework="fandhe-ai", size=512, mode="fresh")),
                tf32=True,
            ),
        ]
        records = summarize.target_gate(rows, "candle")
        gemm_records = [r for r in records if r["task"] == "gemm"]
        self.assertEqual(len(gemm_records), 1)
        self.assertEqual(gemm_records[0]["size"], 256)
        self.assertEqual(gemm_records[0]["status"], "achieved")

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
        # イシュー #1051 codex-review 追加指摘（PR #1082）以降、infer も
        # gemm と同じ経路で実データの size（`_infer_row` 既定 64）を
        # 列挙するため `None` 固定ではなくなった。
        self.assertEqual(rec["size"], 64)

    def test_infer_reuse_gate_rejects_invalid_throughput_even_if_faster(self):
        # codex-review P0 指摘（PR #1229 4 巡目）: `_reuse_row_invalid_
        # reason` は throughput_per_s を検証していなかったため、時間値・
        # checksum が正常でも throughput_per_s が不正な reuse 行を
        # `target_gate` が「達成」と誤判定しうる。`--target` 側より
        # median_s が明確に小さい（=「達成」条件を満たす）infer reuse 行
        # の throughput_per_s だけを不正値にしても、undeterminable として
        # fail-closed に扱われることを確認する。
        rows = [
            dict(
                _infer_row(framework="fandhe-ai", mode="reuse", median_s=0.0001, init_s=0.001),
                throughput_per_s=None,
            ),
            _infer_row(framework="candle", mode="fresh", median_s=0.002),
        ]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "infer")
        self.assertEqual(rec["status"], "undeterminable")

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
        # PR #1082 の修正（size 列挙を task 共通化）以降、train 行が
        # 1 件も無い場合は「0 件」の分岐（size=None）で捕捉される
        # （以前は sizes=[None] 固定で `_pick_row_for_gate` 経由の
        # 「fandhe-ai 未計測」に到達していたが、実データが無いこと自体を
        # より正確に表す理由文言になった）。
        self.assertIn("train 未計測", train_rec["reason"])
        self.assertIsNone(train_rec["size"])

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

    def test_train_multiple_sizes_are_all_evaluated_not_just_first(self):
        # P0 回帰（PR #1082 codex-review 追加指摘）: 修正前は
        # task != "gemm" を `sizes = [None]` 固定にしており、
        # `get(..., size=None)` は size 条件を適用せず最初に一致した
        # 行を返すだけだったため、train/infer に複数 size の行が
        # 混在すると先頭の 1 行しか評価されず、他 size の未達・
        # target 未計測が黙って無視されて全体が「全達成」側へ
        # fail-open していた。ここでは size=64 が達成・size=128 が
        # target 未計測（判定不能）という混在データを与え、両方が
        # 個別レコードとして列挙されることを確認する。
        rows = [
            dict(_train_row(framework="fandhe-ai", median_s=0.0005), size=64),
            dict(_train_row(framework="candle", median_s=0.001), size=64),
            dict(_train_row(framework="fandhe-ai", median_s=0.02), size=128),
            # candle 側 size=128 は計測なし（意図的に欠落させる）。
        ]
        records = summarize.target_gate(rows, "candle")
        train_recs = {r["size"]: r for r in records if r["task"] == "train"}
        self.assertEqual(set(train_recs), {64, 128})
        self.assertEqual(train_recs[64]["status"], "achieved")
        self.assertEqual(train_recs[128]["status"], "undeterminable")
        self.assertIn("candle 未計測", train_recs[128]["reason"])

    def test_infer_multiple_sizes_unmet_size_is_not_hidden_by_achieved_size(self):
        # 上記と対の観点: size=64 が achieved・size=128 が unmet の場合、
        # 先頭行（size=64）の achieved だけを見て unmet を握りつぶさない
        # ことを確認する。
        rows = [
            dict(_infer_row(framework="fandhe-ai", median_s=0.0001), size=64),
            dict(_infer_row(framework="candle", median_s=0.0005), size=64),
            dict(_infer_row(framework="fandhe-ai", median_s=0.01), size=128),
            dict(_infer_row(framework="candle", median_s=0.001), size=128),
        ]
        records = summarize.target_gate(rows, "candle")
        infer_recs = {r["size"]: r for r in records if r["task"] == "infer"}
        self.assertEqual(set(infer_recs), {64, 128})
        self.assertEqual(infer_recs[64]["status"], "achieved")
        self.assertEqual(infer_recs[128]["status"], "unmet")

    def test_duplicate_key_row_is_undeterminable_not_fail_open(self):
        # P0 修正（codex-review 指摘・PR #1082）: 同一
        # (framework, task, device, size, mode) に複数行が存在する場合、
        # 旧実装は `get()` が返す先頭行だけを採用し残りを検証しなかった。
        # 正常行（速い）を先に置き、未達（遅い）な fandhe-ai 行を後置する
        # と未達を隠して「達成」判定を返してしまう fail-open があった。
        # ここでは fandhe-ai/gemm/cpu/size=256/fresh に 2 行を与え、先頭を
        # 速い行（達成条件を満たす）にしても「重複キー」で判定不能へ倒れ、
        # 誤って「達成」を返さないことを確認する。
        rows = [
            dict(_with_parity(_base_row(framework="fandhe-ai", mode="fresh")), median_s=0.0001),
            dict(_with_parity(_base_row(framework="fandhe-ai", mode="fresh")), median_s=0.5),
            _with_parity(_base_row(framework="candle", mode="fresh")),
        ]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "gemm" and r["size"] == 256)
        self.assertEqual(rec["status"], "undeterminable")
        self.assertIn("重複キー", rec["reason"])
        self.assertIsNone(rec["fandhe_median"])

    def test_duplicate_key_on_target_side_is_also_undeterminable(self):
        # 上記と対称のケース: 重複が target（candle）側にある場合も同様に
        # 判定不能へ倒れることを確認する。
        rows = [
            _with_parity(_base_row(framework="fandhe-ai", mode="fresh")),
            dict(_with_parity(_base_row(framework="candle", mode="fresh")), median_s=0.001),
            dict(_with_parity(_base_row(framework="candle", mode="fresh")), median_s=0.002),
        ]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "gemm" and r["size"] == 256)
        self.assertEqual(rec["status"], "undeterminable")
        self.assertIn("重複キー", rec["reason"])

    def test_invalid_size_values_do_not_raise_and_are_undeterminable(self):
        # P0 修正（codex-review 指摘・PR #1082）: 外部 JSONL 由来の `size`
        # を型・値域未検証のまま set 内包・`sorted()` へ渡すと、配列／
        # オブジェクトは `unhashable type`、文字列と整数の混在は比較
        # `TypeError` で集計全体が traceback 停止しうる。producer 契約
        # （正の整数）に反する値は例外にせず判定不能レコードへ倒す
        # （security.md「外部フォーマットのパース時検証（A03）」）。
        bad_sizes = [
            ["not", "hashable"],
            {"nested": "object"},
            "64",  # candle 側の int 64 と型が混在し sorted() 比較で TypeError になりうる
            0,
            -5,
            1.5,
            True,
        ]
        for bad_size in bad_sizes:
            with self.subTest(bad_size=bad_size):
                rows = [
                    dict(_infer_row(framework="fandhe-ai"), size=bad_size),
                    _infer_row(framework="candle"),
                ]
                # 例外を送出せず判定不能レコードを返すことを確認する
                # （fail-closed。旧実装は traceback で停止していた）。
                records = summarize.target_gate(rows, "candle")
                invalid_rec = next(
                    r for r in records if r["task"] == "infer" and r["size"] is None
                )
                self.assertEqual(invalid_rec["status"], "undeterminable")
                self.assertIn("size が不正な値", invalid_rec["reason"])
                # candle 側の有効な size=64 は不正値混入の影響を受けず
                # 独立に判定される（fandhe-ai 側に有効な size が無いため
                # 「fandhe-ai 未計測」となる）。
                valid_rec = next(
                    r for r in records if r["task"] == "infer" and r["size"] == 64
                )
                self.assertEqual(valid_rec["status"], "undeterminable")
                self.assertIn("fandhe-ai 未計測", valid_rec["reason"])

    def test_gemm_row_with_unhashable_size_does_not_raise(self):
        # advisor 指摘（PR #1082 2 巡目）: `target_gate` は
        # `gemm_checksum_mismatches(rows)` を先頭で無条件に呼ぶため、
        # 不正な size を持つ gemm 行は task フィルタで弾かれる infer 行の
        # テストでは再現しない。`gemm_checksum_mismatches`/
        # `gemm_checksum_unverifiable` 内の `reference.get(r["size"], ...)`
        # は `size` が配列等の unhashable 値だと `TypeError` を送出しうる
        # （`gemm_checksum_reference` 側の `_valid_gate_size` フィルタとは
        # 別の dict 参照経路のため個別に防御が必要）。
        rows = [
            dict(_with_parity(_base_row(framework="fandhe-ai")), size=["x", "y"]),
            _with_parity(_base_row(framework="candle")),
        ]
        records = summarize.target_gate(rows, "candle")  # 例外を送出しないこと
        invalid_rec = next(r for r in records if r["task"] == "gemm" and r["size"] is None)
        self.assertEqual(invalid_rec["status"], "undeterminable")
        self.assertIn("size が不正な値", invalid_rec["reason"])

    def test_get_function_missing_size_key_does_not_raise(self):
        # Bugbot Medium 指摘（PR #1082 2 巡目・summarize.py L243-253）:
        # `get()` が `r["size"]` を直接アクセスするため、`size` キー欠損の
        # 行が framework/task/device/mode まで一致すると `KeyError` を
        # 送出しうる。`r.get("size")` 経由での検証に切り替え、欠損行は
        # 一致しないものとして扱う。
        bad_row = _base_row(framework="fandhe-ai")
        del bad_row["size"]
        good_row = _base_row(framework="fandhe-ai", size=256)
        rows = [bad_row, good_row]
        r = summarize.get(rows, "fandhe-ai", "gemm", "cpu", 256)  # 例外を送出しないこと
        self.assertIs(r, good_row)

    def test_get_function_excludes_bool_size(self):
        # `True == 1` により `size: true` の行が `size=1` のクエリで
        # 誤選択されないことを確認する（Bugbot Medium 指摘・PR #1082
        # 2 巡目）。
        bool_row = _base_row(framework="fandhe-ai", size=True)
        rows = [bool_row]
        r = summarize.get(rows, "fandhe-ai", "gemm", "cpu", 1)
        self.assertIsNone(r)

    def test_get_function_defaults_to_excluding_tf32_rows(self):
        # イシュー #1042: `get()` の既定 `tf32=False` は TF32 opt-in 行を
        # 拾わない（fail-open 防止。同一 size に FP32/TF32 双方が存在する
        # 場合、明示指定なしの呼び出しは常に FP32 行を返す）。
        tf32_row = dict(_base_row(framework="fandhe-ai", size=256), tf32=True)
        fp32_row = _base_row(framework="fandhe-ai", size=256)
        rows = [tf32_row, fp32_row]
        self.assertIs(summarize.get(rows, "fandhe-ai", "gemm", "cpu", 256), fp32_row)
        self.assertIs(
            summarize.get(rows, "fandhe-ai", "gemm", "cpu", 256, tf32=True), tf32_row
        )

    def test_pick_row_for_gate_missing_size_key_does_not_raise(self):
        # Bugbot Medium 指摘（PR #1082 2 巡目・summarize.py L702-712）:
        # `_pick_row_for_gate` の候補列挙が `r["size"]` を直接アクセス
        # するため、size キー欠損行が framework/task/device/mode まで
        # 一致すると `KeyError` を送出しうる。
        bad_row = _base_row(framework="fandhe-ai", mode="fresh")
        del bad_row["size"]
        good_row = _base_row(framework="fandhe-ai", mode="fresh", size=256)
        rows = [bad_row, good_row]
        row, mode, dup_reason, used_tf32 = summarize._pick_row_for_gate(
            rows, "fandhe-ai", "gemm", "cpu", 256
        )  # 例外を送出しないこと
        self.assertIs(row, good_row)
        self.assertEqual(mode, "fresh")
        self.assertIsNone(dup_reason)
        self.assertFalse(used_tf32)

    def test_pick_row_for_gate_excludes_bool_size(self):
        bool_row = _base_row(framework="fandhe-ai", mode="fresh", size=True)
        row, mode, dup_reason, used_tf32 = summarize._pick_row_for_gate(
            [bool_row], "fandhe-ai", "gemm", "cpu", 1
        )
        self.assertIsNone(row)
        self.assertIsNone(mode)
        self.assertIsNone(dup_reason)
        self.assertFalse(used_tf32)

    def test_train_reuse_missing_size_key_row_does_not_raise_via_target_gate(self):
        # end-to-end 確認: size キー欠損の train 行が rows に混在しても
        # `target_gate` 全体が例外終了せず、他の正常な size の判定へ
        # 影響しないことを確認する。
        bad_row = _train_row(framework="fandhe-ai", mode="fresh", checksum=99.0, median_s=0.02)
        del bad_row["size"]
        rows = [
            dict(
                _train_row(framework="fandhe-ai", mode="fresh", checksum=0.08, median_s=0.01),
                size=64,
            ),
            dict(
                _train_row(
                    framework="fandhe-ai", mode="reuse", checksum=0.08, init_s=0.001, median_s=0.005
                ),
                size=64,
            ),
            bad_row,
            dict(_train_row(framework="candle", mode="fresh", median_s=0.03), size=64),
        ]
        records = summarize.target_gate(rows, "candle")  # 例外を送出しないこと
        rec = next(r for r in records if r["task"] == "train" and r["size"] == 64)
        self.assertEqual(rec["status"], "achieved")

    def test_train_reuse_bool_size_row_is_not_treated_as_size_one_match(self):
        # Bugbot Medium 指摘（PR #1082 2 巡目・summarize.py L758-767）:
        # `_train_reuse_row_invalid_reason` の fresh 候補列挙が
        # `True == 1` により `size: true` の行を `size=1` の reuse 行と
        # 誤って突合しうる。ここでは `size=True` の fresh 行（checksum
        # 999.0・reuse とは不一致）を混入させても、正当な同一 size(1) の
        # fresh 行が存在しない場合と同じ扱い（突合不能ではなく fresh 側
        # 「未計測」＝有効）になり、reuse 行の実測（candle より高速）が
        # そのまま「達成」判定に使われることを確認する（bool 行に化けた
        # 誤った不一致検出ですり抜けさせない）。
        rows = [
            dict(
                _train_row(
                    framework="fandhe-ai", mode="reuse", checksum=5.0, init_s=0.001, median_s=0.005
                ),
                size=1,
            ),
            dict(
                _train_row(framework="fandhe-ai", mode="fresh", checksum=999.0, median_s=0.02),
                size=True,
            ),
            dict(_train_row(framework="candle", mode="fresh", median_s=0.03), size=1),
        ]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "train" and r["size"] == 1)
        self.assertEqual(rec["status"], "achieved")

    def test_train_reuse_size_matches_same_size_fresh_row_not_other_size(self):
        # Bugbot Medium 指摘（PR #1082）: `_train_reuse_row_invalid_reason`
        # が fresh 行の検索に `size` を渡していなかったため、複数 size の
        # train データが存在する場合に reuse 行が別 size の fresh 行と
        # 誤って突合されていた。ここでは size=64/128 それぞれで
        # reuse checksum と同一 size の fresh checksum が一致する
        # （正当な）データを与え、両方とも達成として正しく判定される
        # ことを確認する（size を渡さない旧実装では size=128 の reuse
        # 行が size=64 の fresh checksum〈0.08〉と誤って突合され
        # 不一致「無効データ」判定に落ちてしまっていた）。
        rows = [
            dict(
                _train_row(framework="fandhe-ai", mode="fresh", checksum=0.08, median_s=0.01),
                size=64,
            ),
            dict(
                _train_row(
                    framework="fandhe-ai", mode="reuse", checksum=0.08, init_s=0.001, median_s=0.005
                ),
                size=64,
            ),
            dict(
                _train_row(framework="fandhe-ai", mode="fresh", checksum=5.0, median_s=0.02),
                size=128,
            ),
            dict(
                _train_row(
                    framework="fandhe-ai", mode="reuse", checksum=5.0, init_s=0.002, median_s=0.015
                ),
                size=128,
            ),
            dict(_train_row(framework="candle", mode="fresh", median_s=0.03), size=64),
            dict(_train_row(framework="candle", mode="fresh", median_s=0.03), size=128),
        ]
        records = summarize.target_gate(rows, "candle")
        train_recs = {r["size"]: r for r in records if r["task"] == "train"}
        self.assertEqual(set(train_recs), {64, 128})
        self.assertEqual(train_recs[64]["status"], "achieved")
        self.assertEqual(train_recs[128]["status"], "achieved")

    def test_train_reuse_size_mismatch_detected_against_same_size_fresh(self):
        # 上記と対の観点: reuse checksum が「同一 size」の fresh と
        # 不一致な場合は正しく無効判定される（別 size の fresh とたまたま
        # 一致してすり抜けない）ことを確認する。size=64 は reuse checksum
        # が同一 size fresh（0.08）と不一致（999.0）。size=128 の fresh
        # checksum（0.08 と同じ値）とは偶然一致してしまう配置にしてあり、
        # size を渡さない旧実装ならこの一致をすり抜けて「達成」と
        # 誤判定しうる。
        rows = [
            dict(
                _train_row(framework="fandhe-ai", mode="fresh", checksum=0.08, median_s=0.01),
                size=64,
            ),
            dict(
                _train_row(
                    framework="fandhe-ai", mode="reuse", checksum=999.0, init_s=0.001, median_s=0.005
                ),
                size=64,
            ),
            dict(
                _train_row(framework="fandhe-ai", mode="fresh", checksum=0.08, median_s=0.02),
                size=128,
            ),
            dict(_train_row(framework="candle", mode="fresh", median_s=0.03), size=64),
        ]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "train" and r["size"] == 64)
        self.assertEqual(rec["status"], "undeterminable")
        self.assertIn("無効データ", rec["reason"])

    def test_train_reuse_duplicate_fresh_row_is_undeterminable_not_fail_open(self):
        # advisor 指摘（PR #1082 3 巡目）: `_train_reuse_row_invalid_reason`
        # の fresh 側検索が `get()`（最初に一致した行だけを返す）のままだと、
        # 同一 size に複数 fresh 行がある場合、先頭が checksum 一致する
        # fresh 行であれば他の不一致な fresh 行を握りつぶして「有効」判定
        # してしまう。ここでは reuse checksum（999.0）と一致する fresh
        # 行を先頭に、不一致な fresh 行（0.08）を 2 番目に置いても
        # 「達成」側へすり抜けず判定不能になることを確認する。
        rows = [
            dict(
                _train_row(framework="fandhe-ai", mode="fresh", checksum=999.0, median_s=0.01),
                size=64,
            ),
            dict(
                _train_row(framework="fandhe-ai", mode="fresh", checksum=0.08, median_s=0.011),
                size=64,
            ),
            dict(
                _train_row(
                    framework="fandhe-ai", mode="reuse", checksum=999.0, init_s=0.001, median_s=0.005
                ),
                size=64,
            ),
            dict(_train_row(framework="candle", mode="fresh", median_s=0.03), size=64),
        ]
        records = summarize.target_gate(rows, "candle")
        rec = next(r for r in records if r["task"] == "train" and r["size"] == 64)
        self.assertEqual(rec["status"], "undeterminable")
        self.assertIn("無効データ", rec["reason"])

    def test_reuse_row_invalid_reason_ignores_managed_fresh_duplicate(self):
        # イシュー #1353・codex-review 指摘: `_pick_row_for_gate` は
        # `managed:true` 行を候補から除外するが、`_reuse_row_invalid_
        # reason` の fresh 側突合（`fresh_matches`）はその除外を適用して
        # いなかったため、通常 fresh・managed fresh が各 1 件ずつ存在する
        # （managed A/B 計測で自然に生じる）通常のデータでも「同一 size
        # の fresh 行が複数」判定に化け、本来 checksum が一致し達成する
        # はずの reuse 行が判定不能になっていた。managed fresh 行の
        # checksum を意図的に不一致（999.0）にしても、除外により無視され
        # 通常 fresh（checksum 一致）とだけ突合されることを確認する。
        normal_fresh = dict(
            _train_row(framework="fandhe-ai", mode="fresh", checksum=0.08, median_s=0.01),
            size=64,
        )
        managed_fresh = dict(
            _train_row(framework="fandhe-ai", mode="fresh", checksum=999.0, median_s=0.01),
            size=64,
        )
        managed_fresh["managed"] = True
        reuse = dict(
            _train_row(
                framework="fandhe-ai", mode="reuse", checksum=0.08, init_s=0.001, median_s=0.005
            ),
            size=64,
        )
        rows = [normal_fresh, managed_fresh, reuse]
        reason = summarize._reuse_row_invalid_reason(rows, reuse, "train")
        self.assertIsNone(reason)


class GetDevicesInManagedExclusionTests(unittest.TestCase):
    """イシュー #1353・Cursor Bugbot 指摘: `get()`／`devices_in()` は
    `_pick_row_for_gate` と同じ理由で `managed:true` 行を除外すべき
    （除外しないと表とゲート判定が食い違いうる）。
    """

    def test_get_skips_managed_row_and_returns_device_only_row(self):
        managed_row = dict(_base_row(framework="fandhe-ai", device="cuda", size=1024))
        managed_row["checksum"] = 999.0
        managed_row["managed"] = True
        device_only_row = dict(_base_row(framework="fandhe-ai", device="cuda", size=1024))
        device_only_row["checksum"] = 1.0
        # managed 行を先に置き、`get()` が最初の一致行をそのまま返す旧
        # 実装ならこちらを拾ってしまうことを確認する配置。
        rows = [managed_row, device_only_row]
        found = summarize.get(rows, "fandhe-ai", "gemm", "cuda", size=1024, mode="fresh")
        self.assertIsNotNone(found)
        self.assertEqual(found["checksum"], 1.0)

    def test_get_returns_none_when_only_managed_row_present(self):
        managed_row = dict(_base_row(framework="fandhe-ai", device="cuda", size=1024))
        managed_row["managed"] = True
        found = summarize.get([managed_row], "fandhe-ai", "gemm", "cuda", size=1024, mode="fresh")
        self.assertIsNone(found)

    def test_devices_in_excludes_device_with_only_managed_rows(self):
        # cuda は managed 行しか持たない（device-only 計測なし）ため
        # 「計測不可」として扱われるべきで一覧に含まれてはならない。
        managed_row = dict(_base_row(framework="fandhe-ai", device="cuda", size=1024))
        managed_row["managed"] = True
        normal_row = dict(_base_row(framework="fandhe-ai", device="cpu", size=1024))
        rows = [managed_row, normal_row]
        devices = summarize.devices_in(rows, "gemm", mode="fresh")
        self.assertIn("cpu", devices)
        self.assertNotIn("cuda", devices)

    def test_devices_in_train_infer_excludes_device_with_only_managed_rows(self):
        # イシュー #1353（Cursor Bugbot 指摘）: `_devices_in_train_infer`
        # が managed 行を除外しないと、cuda が「計測済み」として一覧に
        # 挙がる一方 `_get_train_infer_row`（`get()` 経由で managed 行を
        # 除外済み）は該当なしで `None` を返し、(b)/(c) 節が
        # 「計測不可」と誤表示する不整合が起こる。
        managed_row = dict(
            _train_row(framework="fandhe-ai", device="cuda", mode="fresh")
        )
        managed_row["managed"] = True
        normal_row = dict(_train_row(framework="fandhe-ai", device="cpu", mode="fresh"))
        rows = [managed_row, normal_row]
        devices = summarize._devices_in_train_infer(rows, "train", mode="fresh")
        self.assertIn("cpu", devices)
        self.assertNotIn("cuda", devices)


class GemmChecksumManagedExclusionTests(unittest.TestCase):
    """イシュー #1353（github-actions レビュー指摘）: `managed:true` 行を
    `gemm_checksum_reference`／`gemm_checksum_mismatches` からも除外する。
    除外しないと `_row_key` が managed を区別しない旧実装のもとで、同一
    `(framework, device, size, mode)` の managed 行の checksum 相違が
    正常な device-only 行の「不一致」判定へ誤伝播しうる（`get()`／
    `devices_in()` に既に適用済みの除外条件と揃える）。
    """

    def test_reference_ignores_managed_row_with_divergent_checksum(self):
        # 参照値は fandhe-ai/cpu 優先（`_REFERENCE_PRIORITY`）。managed
        # 行（checksum 999.0 で孤立した誤値）が候補に混入しても、参照値
        # 算出・多数決クラスタ判定へ影響しないことを確認する。
        cpu_fresh = _base_row(framework="fandhe-ai", device="cpu", size=1024, checksum=1.0)
        candle_fresh = _base_row(framework="candle", device="cpu", size=1024, checksum=1.0)
        managed_fresh = dict(
            _base_row(framework="fandhe-ai", device="cuda", size=1024, checksum=999.0)
        )
        managed_fresh["managed"] = True
        rows = [cpu_fresh, candle_fresh, managed_fresh]
        reference = summarize.gemm_checksum_reference(rows)
        ref, ref_label, candidate_count = reference[1024]
        self.assertEqual(ref, 1.0)
        # managed 行が候補集合から除外されていれば候補は 2 件
        # （cpu_fresh・candle_fresh のみ）。
        self.assertEqual(candidate_count, 2)

    def test_mismatches_do_not_flag_normal_row_due_to_managed_collision(self):
        # 除外前の実装では、managed 行の checksum 相違が同一キーに登録
        # され device-only 行まで不一致扱いされ得た。除外後は
        # device-only 行（checksum 一致）が mismatches に現れないことを
        # 確認する。managed 行自体も突合対象から除外されるため現れない。
        cpu_fresh = _base_row(framework="fandhe-ai", device="cpu", size=1024, checksum=1.0)
        candle_fresh = _base_row(framework="candle", device="cpu", size=1024, checksum=1.0)
        cuda_device_only = _base_row(
            framework="fandhe-ai", device="cuda", size=1024, checksum=1.0
        )
        managed_divergent = dict(
            _base_row(framework="fandhe-ai", device="cuda", size=1024, checksum=999.0)
        )
        managed_divergent["managed"] = True
        rows = [cpu_fresh, candle_fresh, cuda_device_only, managed_divergent]
        mismatches = summarize.gemm_checksum_mismatches(rows)
        mismatched_rows = [r for r, _ref, _label in mismatches]
        self.assertNotIn(cuda_device_only, mismatched_rows)
        self.assertNotIn(managed_divergent, mismatched_rows)

    def test_unverifiable_excludes_managed_row(self):
        # イシュー #1353（github-actions レビュー指摘・2 巡目）:
        # `gemm_checksum_unverifiable` は `gemm_checksum_mismatches` と
        # 同じ理由で managed 行を除外すべき。除外しないと、managed 行が
        # 突合対象から外れている（`gemm_checksum_reference` の候補集合に
        # 含まれない）にもかかわらず、本関数がそのまま managed 行自身を
        # 突合不能候補として返しうる。
        cpu_fresh = _base_row(framework="fandhe-ai", device="cpu", size=1024, checksum=1.0)
        candle_fresh = _base_row(framework="candle", device="cpu", size=1024, checksum=1.0)
        managed_fresh = dict(
            _base_row(framework="fandhe-ai", device="cuda", size=1024, checksum=999.0)
        )
        managed_fresh["managed"] = True
        rows = [cpu_fresh, candle_fresh, managed_fresh]
        unverifiable = summarize.gemm_checksum_unverifiable(rows)
        self.assertNotIn(managed_fresh, unverifiable)

    def test_verified_total_excludes_managed_row_with_divergent_checksum(self):
        # イシュー #1353（github-actions レビュー指摘・2 巡目）:
        # `section()` の `verified_total`（`gemm_checksum_unverifiable` の
        # 除外キー集合 `unverifiable_keys` に含まれない gemm 行の件数）が
        # managed 行を除外しないと、同一 size に一致する通常行が 2 件
        # 以上存在する場合、checksum が異なる managed 行も
        # `unverifiable_keys` に現れず（`gemm_checksum_unverifiable` が
        # managed 行を除外済みのため管理下候補にすらならない）、実際には
        # 一度も checksum 突合していないにもかかわらず「相互突合できた」
        # 件数へ誤って計上されうる（fail-open のおそれ）。
        cpu_fresh = _base_row(framework="fandhe-ai", device="cpu", size=1024, checksum=1.0)
        candle_fresh = _base_row(framework="candle", device="cpu", size=1024, checksum=1.0)
        managed_divergent = dict(
            _base_row(framework="fandhe-ai", device="cuda", size=1024, checksum=999.0)
        )
        managed_divergent["managed"] = True
        rows = [cpu_fresh, candle_fresh, managed_divergent]
        lines, *_ = summarize.section("dummy.jsonl", rows)
        text = "\n".join(lines)
        # 相互突合できたのは cpu_fresh・candle_fresh の 2 行のみ（managed
        # 行を含めた 3 行ではない）。
        self.assertIn("相互突合できた 2 行の checksum が参照値と一致", text)


class MainTargetExitCodeTests(unittest.TestCase):
    def _run_main(self, path, target=None, strict=False):
        # `path` は単一パス（str）または複数パス（list。複数入力ファイル
        # をまたぐ回帰確認用）のいずれも受け付ける。
        argv = list(path) if isinstance(path, (list, tuple)) else [path]
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

    def test_invalid_size_exits_3_via_main(self):
        # main() の実配線経路（`--target` あり・`--strict` なし）でも
        # size 不正検出が exit 3 に到達することを確認する（records の
        # 中身だけでなく main() の集計・終了コード判定まで含めた回帰）。
        path = _write_jsonl(
            [
                dict(_infer_row(framework="fandhe-ai"), size=["bad"]),
                _infer_row(framework="candle"),
            ]
        )
        try:
            code, out, err = self._run_main(path, target="candle")
            self.assertEqual(code, 3)
            self.assertIn("判定不能", err)
        finally:
            os.unlink(path)

    def test_duplicate_key_exits_3_via_main(self):
        path = _write_jsonl(
            [
                dict(_with_parity(_base_row(framework="fandhe-ai")), median_s=0.0001),
                dict(_with_parity(_base_row(framework="fandhe-ai")), median_s=0.5),
                _with_parity(_base_row(framework="candle")),
            ]
        )
        try:
            code, out, err = self._run_main(path, target="candle")
            self.assertEqual(code, 3)
            self.assertIn("判定不能", err)
        finally:
            os.unlink(path)

    def test_empty_input_file_does_not_cause_fail_open_when_other_file_achieves(self):
        # codex P0（PR #1082 2 巡目指摘）: `rows` が空の入力ファイルを
        # 無条件に `continue` で読み飛ばすと、`--target` 指定時に複数
        # 入力のうち 1 ファイルが空でも、他ファイルが全達成なら
        # `gate_records_all` が非空となり判定不能に数えられず exit 0 に
        # なる fail-open があった（計測が丸ごと欠落したファイルを
        # 「対象外」と黙って扱ってしまう）。ここでは空ファイル 1 件 +
        # 全達成ファイル 1 件を与え、exit 3（判定不能扱い）になることを
        # 確認する。
        empty_path = _write_jsonl([])
        achieved_path = _write_jsonl(
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
            code, out, err = self._run_main([empty_path, achieved_path], target="candle")
            self.assertEqual(code, 3)
            self.assertIn("判定不能", err)
            self.assertIn("有効な行が無い", out)
            # 空ファイル分の判定不能 1 件が集計に確実に載っていることを
            # 明示的に確認する（`achieved_path` 単体は
            # `test_all_achieved_exits_0` で「達成のみ」を確認済みのため、
            # ここでの判定不能はもっぱら空ファイルに由来する）。
            self.assertIn("判定不能 1", out)
            self.assertIn("達成", out)
            self.assertNotIn("達成 0", out)
        finally:
            os.unlink(empty_path)
            os.unlink(achieved_path)

    def test_skip_log_entirely_failed_device_forces_undeterminable_exit_3(self):
        # codex P0（PR #1082 3 巡目指摘）: `_gate_devices`/`target_gate` は
        # 成功して JSONL に残った fandhe-ai/target 行だけからデバイス集合を
        # 導出するため、あるデバイスの全実行が失敗し skipped*.log にしか
        # 記録が残らなかった場合、そのデバイスは判定対象に一切現れず
        # 「全達成」に混入する fail-open があった。CPU は全達成の正常
        # データ、CUDA は skipped.log のみに記録がある（JSONL には一切
        # 現れない）ケースで exit 3 になることを確認する。
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "results.jsonl")
            rows = [
                _with_parity(_base_row(framework="fandhe-ai", device="cpu", checksum=1.0)),
                _with_parity(_base_row(framework="candle", device="cpu", checksum=1.0)),
                _train_row(framework="fandhe-ai", device="cpu", checksum=0.08, median_s=0.0005),
                _train_row(framework="candle", device="cpu", checksum=0.09, median_s=0.01),
                _infer_row(framework="fandhe-ai", device="cpu", median_s=0.0001),
                _infer_row(framework="candle", device="cpu", median_s=0.0005),
            ]
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(os.path.join(tmpdir, "skipped-cuda.log"), "w") as f:
                f.write(
                    "bench-fandhe task=gemm device=cuda size=256 mode=fresh "
                    "extra=none : CUDA driver not found\n"
                )
            code, out, err = self._run_main(path, target="candle")
            self.assertEqual(code, 3)
            self.assertIn("判定不能", err)
            # `DEVICE_LABEL` により表示は大文字化される
            # （`target_gate_section`・`DEVICE_LABEL` 定義参照）。
            self.assertIn("CUDA", out)
        finally:
            shutil.rmtree(tmpdir)

    def test_skip_log_single_size_failure_on_known_device_forces_undeterminable_exit_3(self):
        # 上記と対の観点（codex P0・PR #1082 3 巡目指摘の 2 点目）: CPU
        # デバイス自体は他 size で計測済み（`_gate_devices` には現れる）
        # だが、gemm cpu size=1024 の実行だけが失敗し skipped.log にしか
        # 記録がないケース。`sizes` は実データからのみ導出されるため、
        # 実データに一切現れない size=1024 は黙って判定対象から漏れて
        # いた。全達成データ（size=256）+ skipped.log の size=1024 失敗
        # 記録を与え、exit 3 になることを確認する。
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "results.jsonl")
            rows = [
                _with_parity(_base_row(framework="fandhe-ai", size=256, checksum=1.0)),
                _with_parity(_base_row(framework="candle", size=256, checksum=1.0)),
                _train_row(framework="fandhe-ai", checksum=0.08, median_s=0.0005),
                _train_row(framework="candle", checksum=0.09, median_s=0.01),
                _infer_row(framework="fandhe-ai", median_s=0.0001),
                _infer_row(framework="candle", median_s=0.0005),
            ]
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(os.path.join(tmpdir, "skipped.log"), "w") as f:
                f.write(
                    "bench-fandhe task=gemm device=cpu size=1024 mode=fresh "
                    "extra=none : OOM\n"
                )
            code, out, err = self._run_main(path, target="candle")
            self.assertEqual(code, 3)
            self.assertIn("判定不能", err)
            self.assertIn("N=1024", out)
        finally:
            shutil.rmtree(tmpdir)

    def test_skip_log_failure_already_covered_by_real_data_is_not_double_counted(self):
        # 回帰確認: 既に実データ側で判定済み（例えば candle 側が計測
        # できておらず「candle 未計測」として既に判定不能）の組は、
        # 同じ組を指す skipped.log の記録があっても二重にレコードを
        # 追加しない（`_inject_skip_failures_into_gate` の
        # `existing_keys` 抑制）。skipped.log の有無で判定不能件数が
        # 変わらないことを確認する（水増しされない）。
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "results.jsonl")
            rows = [_infer_row(framework="fandhe-ai")]
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            code_without, out_without, _ = self._run_main(path, target="candle")
            with open(os.path.join(tmpdir, "skipped.log"), "w") as f:
                f.write(
                    "bench-candle task=infer device=cpu size=64 mode=fresh "
                    "extra=none : timeout\n"
                )
            code_with, out_with, _ = self._run_main(path, target="candle")
            self.assertEqual(code_without, 3)
            self.assertEqual(code_with, 3)
            count_without = int(re.search(r"判定不能 (\d+)", out_without).group(1))
            count_with = int(re.search(r"判定不能 (\d+)", out_with).group(1))
            self.assertEqual(count_with, count_without)
        finally:
            shutil.rmtree(tmpdir)


    def test_skip_log_unparseable_build_failure_forces_undeterminable_exit_3(self):
        # advisor 指摘（PR #1082 4 巡目）: `run_all_cuda.sh` の `build()` が
        # 書く `"$crate BUILD FAILED: ..."` は `_SKIP_LINE_RE` に一致せず
        # framework も特定できないため、framework 判定を先に行う実装だと
        # `bench-fandhe BUILD FAILED`（＝当該スイープで fandhe-ai データが
        # 丸ごと生成されなかった、最も深刻なケース）が framework 不明を
        # 理由に無条件で握りつぶされ、CPU が全達成なら exit 0 になって
        # しまっていた。ここでは CPU 全達成データ + ビルド失敗のみの
        # skipped-cuda.log を与え、exit 3（判定不能扱い）になることを
        # 確認する。
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "results.jsonl")
            rows = [
                _with_parity(_base_row(framework="fandhe-ai", device="cpu", checksum=1.0)),
                _with_parity(_base_row(framework="candle", device="cpu", checksum=1.0)),
                _train_row(framework="fandhe-ai", device="cpu", checksum=0.08, median_s=0.0005),
                _train_row(framework="candle", device="cpu", checksum=0.09, median_s=0.01),
                _infer_row(framework="fandhe-ai", device="cpu", median_s=0.0001),
                _infer_row(framework="candle", device="cpu", median_s=0.0005),
            ]
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(os.path.join(tmpdir, "skipped-cuda.log"), "w") as f:
                f.write(
                    "bench-fandhe BUILD FAILED: error[E0432]: unresolved import\n"
                )
            code, out, err = self._run_main(path, target="candle")
            self.assertEqual(code, 3)
            self.assertIn("判定不能", err)
            self.assertIn("未解析の失敗記録あり", out)
        finally:
            shutil.rmtree(tmpdir)

    def test_skip_log_cross_environment_is_not_suppressed_by_other_file_achieving(self):
        # codex P0・Bugbot High 指摘（PR #1082 4 巡目・同一原因）:
        # `_inject_skip_failures_into_gate` の `existing_keys` を全入力
        # ファイル横断の `gate_records_all` から作っていたため、環境 A の
        # JSONL に `gemm/cpu/N=256` の達成行があると、環境 B（別ディレクトリ
        # ・自身の JSONL には当該組が一切現れず、`skipped.log` にしか
        # 記録が残っていない）の同じ組の失敗が `existing_keys` に紛れて
        # 注入されなくなる（`target_gate` 自身の「ファイルをまたいだ突合は
        # 環境混同になるため行わない」契約違反）。
        #
        # 環境 A: gemm/cpu/N=256・train/infer とも全達成（この
        # `gemm/cpu/N=256` の達成行が旧実装での「毒」になる）。
        # 環境 B: gemm/cpu/N=512・train/infer は全達成だが gemm/cpu/N=256
        # の行自体は無く、`skipped.log` にのみ同じ組の失敗が記録されている
        # （env A/B とも他の組が全て達成のため、この 1 件の判定不能以外に
        # exit 3 を招く要因が無い状態にして原因を一意にする）。
        env_a_dir = tempfile.mkdtemp()
        env_b_dir = tempfile.mkdtemp()
        try:
            env_a_path = os.path.join(env_a_dir, "results.jsonl")
            env_a_rows = [
                _with_parity(_base_row(framework="fandhe-ai", size=256, checksum=1.0)),
                _with_parity(_base_row(framework="candle", size=256, checksum=1.0)),
                _train_row(framework="fandhe-ai", checksum=0.08, median_s=0.0005),
                _train_row(framework="candle", checksum=0.09, median_s=0.01),
                _infer_row(framework="fandhe-ai", median_s=0.0001),
                _infer_row(framework="candle", median_s=0.0005),
            ]
            with open(env_a_path, "w") as f:
                for r in env_a_rows:
                    f.write(json.dumps(r) + "\n")

            env_b_path = os.path.join(env_b_dir, "results.jsonl")
            env_b_rows = [
                _with_parity(_base_row(framework="fandhe-ai", size=512, checksum=2.0), total=512 * 512),
                _with_parity(_base_row(framework="candle", size=512, checksum=2.0), total=512 * 512),
                _train_row(framework="fandhe-ai", checksum=0.08, median_s=0.0005),
                _train_row(framework="candle", checksum=0.09, median_s=0.01),
                _infer_row(framework="fandhe-ai", median_s=0.0001),
                _infer_row(framework="candle", median_s=0.0005),
            ]
            with open(env_b_path, "w") as f:
                for r in env_b_rows:
                    f.write(json.dumps(r) + "\n")
            with open(os.path.join(env_b_dir, "skipped.log"), "w") as f:
                f.write(
                    "bench-fandhe task=gemm device=cpu size=256 mode=fresh "
                    "extra=none : segfault\n"
                )

            code, out, err = self._run_main([env_a_path, env_b_path], target="candle")
            # 環境 A・B とも実データ側の (task, device, size) は全て達成
            # であることを前提にしたテストのため、それ自体を先に検証する
            # （このアサーションが崩れると原因が別要因〈parity 等〉に
            # すり替わり、本テストの識別力が失われるため）。
            self.assertIn("| gemm | CPU | 256 | 1.000 ms（fresh） | 1.000 ms（fresh） | 1.00 倍 | 達成", out)
            self.assertIn("| gemm | CPU | 512 | 1.000 ms（fresh） | 1.000 ms（fresh） | 1.00 倍 | 達成", out)
            self.assertEqual(code, 3)
            self.assertIn("判定不能", err)
            # 環境 B（skipped.log 由来）の判定不能が実際に記録されている
            # ことを確認する（環境 A の達成データに紛れて握りつぶされて
            # いないこと）。
            self.assertIn("skipped*.log に実行時失敗", out)
        finally:
            shutil.rmtree(env_a_dir)
            shutil.rmtree(env_b_dir)

    def test_skip_log_mode_aware_downgrades_achieved_when_used_mode_never_succeeded(self):
        # codex-review P0 指摘その1（PR #1082 5 巡目）: `existing_keys` が
        # `(task, device, size)` のみで skip 行の `framework`/`mode` を
        # 無視していたため、fandhe-ai の fresh 実行が失敗して
        # skipped.log に記録されていても、同じ組の fandhe-ai reuse 実行が
        # 成功し（`_pick_row_for_gate` は reuse を優先）target（candle）の
        # fresh 実行も成功していれば「達成」のまま握りつぶされ exit 0 に
        # なっていた（train reuse は比較対象の fresh 行が存在しない場合を
        # 有効値として扱う仕様のため、fresh 実行が実際に試みられて失敗
        # したという証拠があっても checksum 突合不能な状態を見逃す）。
        # ここでは fandhe-ai の fresh 行を意図的に欠落させ（＝失敗）、
        # reuse 行のみを与え、candle の fresh 行は正常に成功させたうえで
        # skipped.log に fandhe-ai の fresh 失敗を記録する。exit 3
        # （判定不能）になることを確認する。
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "results.jsonl")
            rows = [
                _with_parity(_base_row(framework="fandhe-ai", checksum=1.0)),
                _with_parity(_base_row(framework="candle", checksum=1.0)),
                _train_row(
                    framework="fandhe-ai", mode="reuse", checksum=0.08, init_s=0.001, median_s=0.005
                ),
                _train_row(framework="candle", mode="fresh", median_s=0.03),
                _infer_row(framework="fandhe-ai", median_s=0.0001),
                _infer_row(framework="candle", median_s=0.0005),
            ]
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(os.path.join(tmpdir, "skipped.log"), "w") as f:
                f.write(
                    "bench-fandhe task=train device=cpu size=64 mode=fresh "
                    "extra=none : segfault\n"
                )
            code, out, err = self._run_main(path, target="candle")
            # gemm/infer は全達成のため、判定不能の原因は train の
            # skip 失敗のみに一意化される（`_gate_devices` が cpu を
            # gemm/infer からも拾うため、train データを与えないと
            # 「gemm/infer 未計測」という無関係な判定不能が混入し
            # 本テストの識別力が下がる）。
            self.assertIn("| gemm | CPU | 256 | 1.000 ms（fresh） | 1.000 ms（fresh） | 1.00 倍 | 達成", out)
            self.assertIn("| infer | CPU | 64", out)
            self.assertEqual(code, 3)
            self.assertIn("判定不能", err)
            self.assertIn("skipped*.log に実行時失敗", out)
        finally:
            shutil.rmtree(tmpdir)

    def test_skip_log_mode_aware_suppresses_when_exact_mode_actually_succeeded(self):
        # 上記と対の観点（重複抑止側）: skip 失敗と全く同じ
        # (framework, task, device, size, mode) の実行が実際には成功して
        # JSONL に残っている場合（再実行後の stale な skipped.log
        # エントリ）は、達成判定を判定不能へ格下げしない。
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "results.jsonl")
            rows = [
                _with_parity(_base_row(framework="fandhe-ai", checksum=1.0)),
                _with_parity(_base_row(framework="candle", checksum=1.0)),
                _train_row(framework="fandhe-ai", mode="fresh", checksum=0.08, median_s=0.0005),
                _train_row(framework="candle", mode="fresh", median_s=0.03),
                _infer_row(framework="fandhe-ai", median_s=0.0001),
                _infer_row(framework="candle", median_s=0.0005),
            ]
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(os.path.join(tmpdir, "skipped.log"), "w") as f:
                f.write(
                    "bench-fandhe task=train device=cpu size=64 mode=fresh "
                    "extra=none : stale-failure-before-rerun\n"
                )
            code, out, _ = self._run_main(path, target="candle")
            self.assertEqual(code, 0)
            self.assertIn("| train | CPU | 64", out)
            self.assertNotIn("skipped*.log に実行時失敗", out)
        finally:
            shutil.rmtree(tmpdir)

    def test_skip_log_unknown_binary_with_valid_task_device_is_not_silently_dropped(self):
        # codex-review P0 指摘その2（PR #1082 5 巡目）: 正規表現一致で
        # task/device が妥当でも binary 名が `_SKIP_BIN_TO_FRAMEWORK`
        # allowlist 外だと `framework` が `None` になり、旧実装では
        # `sf["framework"] not in (...)` 判定（`None not in (...)` は
        # 常に真）により無条件で無視されていた（fail-open）。CPU が
        # gemm/train/infer 全て達成のデータ + 未知 binary 名の skip 行
        # 1 件を与え、exit 3（判定不能）になることを確認する。
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "results.jsonl")
            rows = [
                _with_parity(_base_row(framework="fandhe-ai", device="cpu", checksum=1.0)),
                _with_parity(_base_row(framework="candle", device="cpu", checksum=1.0)),
                _train_row(framework="fandhe-ai", device="cpu", checksum=0.08, median_s=0.0005),
                _train_row(framework="candle", device="cpu", checksum=0.09, median_s=0.01),
                _infer_row(framework="fandhe-ai", device="cpu", median_s=0.0001),
                _infer_row(framework="candle", device="cpu", median_s=0.0005),
            ]
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(os.path.join(tmpdir, "skipped.log"), "w") as f:
                f.write(
                    "bench-mystery task=gemm device=cpu size=256 mode=fresh "
                    "extra=none : unknown binary\n"
                )
            code, out, err = self._run_main(path, target="candle")
            self.assertEqual(code, 3)
            self.assertIn("判定不能", err)
            self.assertIn("未解析の失敗記録あり", out)
        finally:
            shutil.rmtree(tmpdir)

    def test_skip_log_raw_content_with_pipe_and_script_tag_does_not_break_table(self):
        # codex P0 指摘（PR #1082 6 巡目・security.md A03）: 外部プロセス
        # stderr を含む未信頼文字列 `sf["raw"]` を無加工で `reason` に
        # 格納すると、`target_gate_section()` が `|` 区切りの Markdown
        # 表セル・箇条書きへそのまま埋め込むため、`|` で表構造を、
        # `<script>` 等で出力ページの HTML/Markdown 構文を改変できる。
        # `|` と `<script>` を含む skip 行を与えても、出力に生の `|`
        # 区切り・`<script>` が現れず（エスケープ済みの `\|`・`&lt;`
        # として現れる）、exit 3 になることを確認する。
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "results.jsonl")
            rows = [
                _with_parity(_base_row(framework="fandhe-ai", device="cpu", checksum=1.0)),
                _with_parity(_base_row(framework="candle", device="cpu", checksum=1.0)),
                _train_row(framework="fandhe-ai", device="cpu", checksum=0.08, median_s=0.0005),
                _train_row(framework="candle", device="cpu", checksum=0.09, median_s=0.01),
                _infer_row(framework="fandhe-ai", device="cpu", median_s=0.0001),
                _infer_row(framework="candle", device="cpu", median_s=0.0005),
            ]
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(os.path.join(tmpdir, "skipped.log"), "w") as f:
                f.write(
                    "bench-mystery task=gemm device=cpu size=256 mode=fresh "
                    "extra=none : boom | <script>alert(1)</script>\n"
                )
            code, out, err = self._run_main(path, target="candle")
            self.assertEqual(code, 3)
            self.assertIn("判定不能", err)
            # `## 実行時失敗（skipped*.log）` 節（生ログの informational な
            # 一覧表示・箇条書き）も `_sanitize_skip_raw_for_display` で
            # サニタイズ済みで出力される（イシュー #1085）。ゲート節
            # （`## 目標達成ゲート` の `reason`）と合わせて出力全体（`out`）
            # に対し、生の `|`・`<script>` が現れずエスケープ済み表現の
            # みが現れることを確認する。
            self.assertNotIn(" | <script>", out)
            self.assertNotIn("boom | <", out)
            self.assertNotIn("<script>alert(1)</script>", out)
            self.assertIn("&lt;script&gt;", out)
            self.assertIn("boom \\| ", out)
            self.assertIn("未解析の失敗記録あり", out)
        finally:
            shutil.rmtree(tmpdir)

    def test_skip_failures_section_escapes_raw_log_content(self):
        # イシュー #1085: `## 実行時失敗（skipped*.log）` 節は skip 行を
        # 無加工で箇条書きへ埋め込んでおり、PR #1082 のゲート節向け
        # サニタイズ（`_sanitize_skip_raw_for_display`）の対象外だった
        # （security.md A03 と同種の注入経路）。`--target` 無指定
        # （ゲート節注入と独立の経路）で本節単体のサニタイズを確認する:
        # `<script>` の実体参照化・`|` の Markdown エスケープ・120 文字
        # 超の切り詰め（末尾 `…`）・ログファイル名の bold 表示維持。
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "results.jsonl")
            rows = [_base_row(framework="fandhe-ai", device="cpu", checksum=1.0)]
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            long_token = "x" * 200
            with open(os.path.join(tmpdir, "skipped.log"), "w") as f:
                f.write(
                    "bench-mystery task=gemm device=cpu size=256 mode=fresh "
                    f"extra=none : boom | <script>alert(1)</script> {long_token}\n"
                )
            code, out, err = self._run_main(path)
            section = out[out.index("## 実行時失敗（skipped*.log）"):]
            self.assertNotIn("<script>alert(1)</script>", section)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", section)
            self.assertNotIn("boom | <", section)
            self.assertIn("boom \\| ", section)
            self.assertNotIn(long_token, section)
            self.assertIn("…", section)
            self.assertIn("**skipped.log**", section)
        finally:
            shutil.rmtree(tmpdir)

    def test_skip_log_unresolvable_size_does_not_suppress_via_unrelated_success(self):
        # Bugbot Medium 指摘（PR #1082 6 巡目）: stale 成功判定が `get()` に
        # `sf["size"]` を渡すが、`get()` は `size=None` を「size で絞ら
        # ない」と解釈するため、`_parse_skip_failure` が size を解析
        # できず `None` にした skip 失敗が、同じ framework/task/device/
        # mode の**任意の** size の成功行によって stale 扱いされ握り
        # つぶされていた（fail-open）。ここでは size が不正
        # （`_valid_gate_size` が弾く負数）な skip 行と、同じ
        # framework/task/device/mode の別 size の成功行を与え、
        # 判定不能として exit 3 になることを確認する。
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "results.jsonl")
            rows = [
                _train_row(framework="fandhe-ai", mode="fresh", checksum=0.08, median_s=0.0005),
                _train_row(framework="candle", mode="fresh", median_s=0.03),
                _with_parity(_base_row(framework="fandhe-ai", checksum=1.0)),
                _with_parity(_base_row(framework="candle", checksum=1.0)),
                _infer_row(framework="fandhe-ai", median_s=0.0001),
                _infer_row(framework="candle", median_s=0.0005),
            ]
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(os.path.join(tmpdir, "skipped.log"), "w") as f:
                f.write(
                    "bench-fandhe task=train device=cpu size=-1 mode=fresh "
                    "extra=none : bogus-size-failure\n"
                )
            code, out, err = self._run_main(path, target="candle")
            self.assertEqual(code, 3)
            self.assertIn("判定不能", err)
            self.assertIn("skipped*.log に実行時失敗", out)
        finally:
            shutil.rmtree(tmpdir)

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
