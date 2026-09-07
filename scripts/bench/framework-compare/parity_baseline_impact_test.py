#!/usr/bin/env python3
"""`parity_baseline_impact.py` の stdlib unittest（イシュー #1238）。

`parity_tolerance_candidates_test.py`（#1237）と同じ方針（ファイルパス指定
import・GPU 不要・合成 fixture のみで完結・stdlib のみ）。CI（`ci.yml` の
`deps-forbidden` ジョブ）は
`python3 -m unittest scripts/bench/framework-compare/parity_baseline_impact_test.py`
で本ファイルを実行する。実ファイル
（`crates/backend-cuda/tests/common/parity_baseline.rs`）に対する正式計算は
CI では行わない（`--scale-mode exact` は約 1.2 億要素の生成を伴うため。
README 参照）。

検証観点:
- `Xorshift64Star` の Python 移植が Rust 実装と同一の数列を生成する
  （固定値照合。出典コマンドは `XorshiftPortTest` docstring 参照）
- `bits_extreme_scan` の凸性に基づく O(1) 追跡が、全要素を保持する素朴な
  実装と一致する（境界値・ランダム系列の両方で照合）
- `parse_baselines` が合成 `ParityBaseline { ... }` テキストを正しくパース
  し、行数不一致・欠損フィールド・`total != m*n`・`fail_count > total` を
  fail-closed（`BaselineParseError`）で検出する
- `classify` の 3 クラス分類が境界値（`bound == 1e-5`・`ceiling == bound`・
  `ceiling is None`）で仕様どおりに動作する
- 実ファイルへのスモークパース（`crates/` が存在する開発環境限定。存在し
  ない場合は明示的に `skipTest`——skip は「テストなし」ではなく「実行環境
  制約」であることをメッセージで明示する）
"""

from __future__ import annotations

import importlib.util
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

_SPEC = importlib.util.spec_from_file_location(
    "parity_baseline_impact", os.path.join(HERE, "parity_baseline_impact.py")
)
pbi = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pbi
_SPEC.loader.exec_module(pbi)


class XorshiftPortTest(unittest.TestCase):
    """`bits_extreme_scan`/`bits_to_f32` が `Xorshift64Star`
    （`crates/bench-harness/src/rng.rs`）と同一の数列を生成することの固定値
    照合。

    出典: 下記アルゴリズムを逐語移植したスタンドアロン Rust プログラムを
    `rustc -O` でビルドし、`seed=2000`/`seed=8888` それぞれで `next_u64`/
    `next_f32` の先頭 8 値を出力して得た（`cargo test -p bench-harness`
    〈`[package] name = "bench-harness"`〉の既存 rng テストは数列そのもの
    を出力しないため、本テストのための単発検証として実行した。イシュー
    #1238 実装時）。
    """

    def _next_u64_seq(self, seed: int, count: int) -> list[int]:
        state = pbi.xorshift_seed_state(seed)
        out = []
        s = state
        for _ in range(count):
            x = s
            x ^= (x << 13) & pbi._MASK64
            x ^= x >> 7
            x ^= (x << 17) & pbi._MASK64
            s = x & pbi._MASK64
            v = (s * pbi._XORSHIFT_MUL) & pbi._MASK64
            out.append(v)
        return out

    def test_next_u64_seed_2000(self):
        expected = [
            17857353959250338627,
            5458114834663065368,
            8911073129847362667,
            15635094144986555137,
            6380130334883383129,
            6374596878020430325,
            1700974162000834584,
            1123246461276591293,
        ]
        self.assertEqual(self._next_u64_seq(2000, 8), expected)

    def test_next_u64_seed_8888(self):
        expected = [
            15612732945208664489,
            16062557256306237076,
            6523506771622876257,
            14952567949763288963,
            11234056563648285585,
            9891194839121461213,
            3631576927759455514,
            9219980252184536634,
        ]
        self.assertEqual(self._next_u64_seq(8888, 8), expected)

    def _next_f32_seq(self, seed: int, count: int) -> list[float]:
        state = pbi.xorshift_seed_state(seed)
        out = []
        s = state
        for _ in range(count):
            x = s
            x ^= (x << 13) & pbi._MASK64
            x ^= x >> 7
            x ^= (x << 17) & pbi._MASK64
            s = x & pbi._MASK64
            v = (s * pbi._XORSHIFT_MUL) & pbi._MASK64
            bits = (v >> 40) & 0xFFFFFF
            out.append(pbi.bits_to_f32(bits))
        return out

    def test_next_f32_seed_2000(self):
        expected = [
            0.9360981,
            -0.40823007,
            -0.03385961,
            0.69516027,
            -0.30826497,
            -0.30886483,
            -0.8155801,
            -0.87821746,
        ]
        actual = self._next_f32_seq(2000, 8)
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, places=6)

    def test_next_f32_seed_8888(self):
        expected = [
            0.6927358,
            0.74150586,
            -0.29272008,
            0.6211606,
            0.21799874,
            0.07240546,
            -0.60626376,
            -0.00036776066,
        ]
        actual = self._next_f32_seq(8888, 8)
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, places=6)

    def test_zero_seed_is_corrected(self):
        # `Xorshift64Star::new(0)` が不動点（常に 0）に陥らないことの再現
        # （`crates/bench-harness/src/rng.rs::tests::zero_seed_is_corrected`
        # と同じ契約）。
        self.assertNotEqual(pbi.xorshift_seed_state(0), 0)
        self.assertEqual(pbi.xorshift_seed_state(0), 0x9E3779B97F4A7C15)
        self.assertEqual(pbi.xorshift_seed_state(42), 42)


class BitsExtremeScanTest(unittest.TestCase):
    """`bits_extreme_scan`（凸性に基づく O(1) 追跡）が、全要素を保持して
    素朴に min/max を取る実装と一致することを確認する。
    """

    def _naive_extreme(self, seed: int, count: int) -> tuple[int, int, int]:
        if count == 0:
            return pbi.xorshift_seed_state(seed), -1, -1
        state = pbi.xorshift_seed_state(seed)
        bits_list = []
        s = state
        for _ in range(count):
            x = s
            x ^= (x << 13) & pbi._MASK64
            x ^= x >> 7
            x ^= (x << 17) & pbi._MASK64
            s = x & pbi._MASK64
            v = (s * pbi._XORSHIFT_MUL) & pbi._MASK64
            bits_list.append((v >> 40) & 0xFFFFFF)
        return s, min(bits_list), max(bits_list)

    def test_matches_naive_various_seeds_and_counts(self):
        for seed in (1, 42, 2000, 8888, 0, 0xBEEF):
            for count in (1, 2, 7, 100, 1000):
                with self.subTest(seed=seed, count=count):
                    state = pbi.xorshift_seed_state(seed)
                    got = pbi.bits_extreme_scan(state, count)
                    want = self._naive_extreme(seed, count)
                    self.assertEqual(got, want)

    def test_empty_count(self):
        state = pbi.xorshift_seed_state(1)
        self.assertEqual(pbi.bits_extreme_scan(state, 0), (state, -1, -1))

    def test_convexity_extreme_abs_matches_naive_max(self):
        # 「abs 最大は bits の min/max のいずれかで達成される」という凸性の
        # 主張自体を、素朴な全要素 abs 最大探索と突合して検証する
        # （`bits_extreme_scan` docstring の証明要旨のテスト版）。
        for seed in (7, 12345, 99999):
            count = 500
            state = pbi.xorshift_seed_state(seed)
            _next_state, bmin, bmax = pbi.bits_extreme_scan(state, count)
            extreme_abs = max(abs(pbi.bits_to_f32(bmin)), abs(pbi.bits_to_f32(bmax)))

            s = state
            naive_max_abs = 0.0
            for _ in range(count):
                x = s
                x ^= (x << 13) & pbi._MASK64
                x ^= x >> 7
                x ^= (x << 17) & pbi._MASK64
                s = x & pbi._MASK64
                v = (s * pbi._XORSHIFT_MUL) & pbi._MASK64
                bits = (v >> 40) & 0xFFFFFF
                naive_max_abs = max(naive_max_abs, abs(pbi.bits_to_f32(bits)))

            self.assertAlmostEqual(extreme_abs, naive_max_abs, places=9)


class RoundF32ToF16AbsTest(unittest.TestCase):
    def test_matches_struct_half_roundtrip(self):
        for x in (0.0, 0.5, -0.5, 0.999, -0.999, 1.0 - 2**-30, 0.0001):
            expected = abs(struct.unpack("<e", struct.pack("<e", x))[0])
            self.assertEqual(pbi.round_f32_to_f16_abs(x), expected)

    def test_near_one_rounds_to_exactly_one(self):
        # half の 1.0 近傍の分解能は 2^-10 のため、1.0 に極めて近い値は
        # 1.0 ちょうどへ丸められることがある（実データ観測: 本スクリプトの
        # f16 経路行で S_A/S_B が 1.000000 と表示される理由）。
        self.assertEqual(pbi.round_f32_to_f16_abs(0.99995), 1.0)


class ComputeScaleTest(unittest.TestCase):
    def test_upper_bound_mode_is_trivial_one(self):
        self.assertEqual(
            pbi.compute_scale(123, 4, 4, 4, False, "upper-bound"), (1.0, 1.0)
        )

    def test_exact_mode_within_input_range(self):
        s_a, s_b = pbi.compute_scale(2000, 32, 32, 32, False, "exact")
        self.assertGreater(s_a, 0.0)
        self.assertLessEqual(s_a, 1.0)
        self.assertGreater(s_b, 0.0)
        self.assertLessEqual(s_b, 1.0)

    def test_memoization_returns_identical_result(self):
        pbi._scale_cache.clear()
        first = pbi.compute_scale(999, 8, 8, 8, False, "exact")
        # 2 回目はキャッシュヒットのはず（同じタプルオブジェクトである
        # 必要はないが、値は完全一致する）。
        second = pbi.compute_scale(999, 8, 8, 8, False, "exact")
        self.assertEqual(first, second)
        self.assertIn((999, 8, 8, 8, False), pbi._scale_cache)


class ClassifyTest(unittest.TestCase):
    def test_ceiling_none_is_unclassifiable(self):
        self.assertEqual(pbi.classify(1e-6, None), pbi.UNCLASSIFIABLE_CEILING)

    def test_bound_below_1e5_strict_is_noop(self):
        self.assertEqual(pbi.classify(9.999e-6, 1e-3), pbi.NO_OP)

    def test_bound_equal_1e5_is_not_noop(self):
        # 境界（bound == 1e-5）は no-op に含めない（`d == 1e-5` の要素が
        # 候補側でのみ救済されうるため。既存救済は `d < 1e-5` の厳密不等号）。
        self.assertNotEqual(pbi.classify(1e-5, 1e-3), pbi.NO_OP)

    def test_ceiling_le_bound_is_full_rescue(self):
        self.assertEqual(pbi.classify(1e-3, 1e-3), pbi.FULL_RESCUE)
        self.assertEqual(pbi.classify(2e-3, 1e-3), pbi.FULL_RESCUE)

    def test_ceiling_gt_bound_is_partial(self):
        self.assertEqual(pbi.classify(1e-4, 1e-3), pbi.PARTIAL)


_SYNTHETIC_BASELINES_HEADER = "pub static BASELINES: &[ParityBaseline] = &[\n"
_SYNTHETIC_BASELINES_FOOTER = "\n];\n"


def _synthetic_entry(
    path="WmmaTf32",
    context="synthetic 4x4x4 seed=1",
    m=4,
    n=4,
    k=4,
    seed="1",
    total_m=4,
    total_n=4,
    fail=1,
    mean_ceil="1.0e-4",
    prov="false",
    max_abs="Some(1.0e-3)",
    max_rel="Some(1.0e-1)",
) -> str:
    return (
        "    ParityBaseline {\n"
        f"        path: ParityPath::{path},\n"
        f'        context: "{context}",\n'
        f"        m: {m},\n"
        f"        n: {n},\n"
        f"        k: {k},\n"
        f"        seed: {seed},\n"
        f"        total: {total_m} * {total_n},\n"
        f"        baseline_fail_count: {fail},\n"
        f"        baseline_mean_abs_diff_ceiling: {mean_ceil},\n"
        "        // 合成コメント行（実ファイルの doc コメントを模す）。\n"
        f"        baseline_provenance_unconfirmed: {prov},\n"
        f"        baseline_max_abs_diff_ceiling: {max_abs},\n"
        f"        baseline_max_rel_err_ceiling: {max_rel},\n"
        "    },"
    )


class ParseBaselinesTest(unittest.TestCase):
    def _write(self, tmp_path: str, body: str) -> str:
        text = _SYNTHETIC_BASELINES_HEADER + body + _SYNTHETIC_BASELINES_FOOTER
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
        return tmp_path

    def test_parses_two_valid_entries(self):
        import tempfile

        body = _synthetic_entry(context="a") + "\n" + _synthetic_entry(context="b", seed="0x1A")
        with tempfile.TemporaryDirectory() as d:
            path = self._write(os.path.join(d, "fixture.rs"), body)
            rows = pbi.parse_baselines(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].context, "a")
        self.assertEqual(rows[1].seed, 0x1A)
        self.assertEqual(rows[1].max_abs_diff_ceiling, 1.0e-3)

    def test_none_ceiling_parses_to_none(self):
        import tempfile

        body = _synthetic_entry(max_abs="None", max_rel="None")
        with tempfile.TemporaryDirectory() as d:
            path = self._write(os.path.join(d, "fixture.rs"), body)
            rows = pbi.parse_baselines(path)
        self.assertIsNone(rows[0].max_abs_diff_ceiling)
        self.assertIsNone(rows[0].max_rel_err_ceiling)

    def test_f16_path_flag(self):
        import tempfile

        body = _synthetic_entry(path="MmaF16")
        with tempfile.TemporaryDirectory() as d:
            path = self._write(os.path.join(d, "fixture.rs"), body)
            rows = pbi.parse_baselines(path)
        self.assertTrue(rows[0].is_f16)

        body2 = _synthetic_entry(path="WmmaTf32")
        with tempfile.TemporaryDirectory() as d:
            path2 = self._write(os.path.join(d, "fixture2.rs"), body2)
            rows2 = pbi.parse_baselines(path2)
        self.assertFalse(rows2[0].is_f16)

    def test_missing_file_is_fail_closed(self):
        with self.assertRaises(pbi.BaselineParseError):
            pbi.parse_baselines("/nonexistent/path/parity_baseline.rs")

    def test_missing_array_marker_is_fail_closed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "broken.rs")
            with open(path, "w", encoding="utf-8") as f:
                f.write("// no BASELINES array here\n")
            with self.assertRaises(pbi.BaselineParseError):
                pbi.parse_baselines(path)

    def test_total_mismatch_with_m_n_is_fail_closed(self):
        import tempfile

        # `total: 4 * 4` のまま m を書き換えて m*n との不整合を作る
        # （フィールド抽出自体は成功するが、行内整合性検査で拒否される）。
        body = _synthetic_entry(m=5)
        with tempfile.TemporaryDirectory() as d:
            path = self._write(os.path.join(d, "fixture.rs"), body)
            with self.assertRaises(pbi.BaselineParseError):
                pbi.parse_baselines(path)

    def test_fail_count_exceeds_total_is_fail_closed(self):
        import tempfile

        body = _synthetic_entry(fail=100)  # total=16 < fail=100
        with tempfile.TemporaryDirectory() as d:
            path = self._write(os.path.join(d, "fixture.rs"), body)
            with self.assertRaises(pbi.BaselineParseError):
                pbi.parse_baselines(path)

    def test_missing_field_is_fail_closed(self):
        import tempfile

        # `baseline_fail_count` を欠落させた不正エントリ（フィールド正規表現
        # が一致せず、ブロック開始数とフィールド抽出数が食い違う）。
        broken_entry = (
            "    ParityBaseline {\n"
            "        path: ParityPath::WmmaTf32,\n"
            '        context: "broken",\n'
            "        m: 4,\n"
            "        n: 4,\n"
            "        k: 4,\n"
            "        seed: 1,\n"
            "        total: 4 * 4,\n"
            "        baseline_mean_abs_diff_ceiling: 1.0e-4,\n"
            "        baseline_provenance_unconfirmed: false,\n"
            "        baseline_max_abs_diff_ceiling: None,\n"
            "        baseline_max_rel_err_ceiling: None,\n"
            "    },"
        )
        with tempfile.TemporaryDirectory() as d:
            path = self._write(os.path.join(d, "fixture.rs"), broken_entry)
            with self.assertRaises(pbi.BaselineParseError):
                pbi.parse_baselines(path)

    def test_empty_array_is_fail_closed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = self._write(os.path.join(d, "fixture.rs"), "")
            with self.assertRaises(pbi.BaselineParseError):
                pbi.parse_baselines(path)


class RealFileSmokeTest(unittest.TestCase):
    """実ファイル（`crates/backend-cuda/tests/common/parity_baseline.rs`）
    に対するパース・自己検査のスモークテスト（`--scale-mode exact` の計算
    自体は実行せず、パース段のみを確認する。実 GEMM 計算は CI では行わない
    ——本ファイル冒頭 docstring 参照）。`crates/` が存在しない開発環境
    （framework-compare 単独チェックアウト等）では明示的に skip する。
    """

    def test_real_file_parses_and_totals_45_rows(self):
        default_path = os.path.normpath(
            os.path.join(HERE, "..", "..", "..", "crates", "backend-cuda", "tests", "common", "parity_baseline.rs")
        )
        if not os.path.isfile(default_path):
            self.skipTest(
                f"実ファイルが見つからないため skip（環境制約。期待パス: {default_path}）"
            )
        rows = pbi.parse_baselines(default_path)
        # 実測される行数は 45（`ParityPath` 別内訳の合計。本スクリプト
        # docstring・`docs/perf/candle-parity-tolerance-baseline-impact.md`
        # §6 参照。単純な `grep -c "ParityBaseline {"` は struct 定義自体を
        # 含んで 46 と数えるため使わない）。
        self.assertEqual(len(rows), 45)
        from collections import Counter

        counts = Counter(r.path for r in rows)
        self.assertEqual(
            counts,
            Counter(
                {
                    "MmaTf32": 11,
                    "WmmaTf32": 9,
                    "WmmaTf32Opt": 9,
                    "WmmaTf32Staged": 8,
                    "SpecializedMmaF16": 3,
                    "WmmaF16": 2,
                    "MmaTf32VsWmmaStaged": 2,
                    "MmaF16": 1,
                }
            ),
        )
        none_ceiling_rows = [r for r in rows if r.max_abs_diff_ceiling is None]
        self.assertEqual(len(none_ceiling_rows), 1)
        self.assertEqual(none_ceiling_rows[0].path, "WmmaTf32Opt")


class MainCliTest(unittest.TestCase):
    def test_main_upper_bound_mode_exits_zero_and_emits_markdown(self):
        import contextlib
        import io as _io

        default_path = os.path.normpath(
            os.path.join(HERE, "..", "..", "..", "crates", "backend-cuda", "tests", "common", "parity_baseline.rs")
        )
        if not os.path.isfile(default_path):
            self.skipTest("実ファイルが見つからないため skip（環境制約）")
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = pbi.main(["--baselines", default_path, "--scale-mode", "upper-bound"])
        self.assertEqual(code, 0)
        self.assertIn("# candle-parity-tolerance-baseline-impact", buf.getvalue())

    def test_main_missing_file_exits_nonzero(self):
        import contextlib
        import io as _io

        buf = _io.StringIO()
        with contextlib.redirect_stderr(buf):
            code = pbi.main(["--baselines", "/nonexistent/parity_baseline.rs"])
        self.assertNotEqual(code, 0)
        self.assertIn("ERROR", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
