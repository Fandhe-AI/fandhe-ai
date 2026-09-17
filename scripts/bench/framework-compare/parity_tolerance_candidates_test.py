#!/usr/bin/env python3
"""`parity_tolerance_candidates.py` の stdlib unittest（イシュー #1237）。

`compare_gemm_gate_test.py`・`compare_gemm_ab_test.py` と同じ方針（ファイル
パス指定 import・GPU 不要・合成 fixture のみで完結）。CI（`ci.yml` の
`deps-forbidden` ジョブ）は
`python3 -m unittest scripts/bench/framework-compare/parity_tolerance_candidates_test.py`
で本ファイルを実行する。実ダンプ（`docs/perf/logs/cuda-gemm-candle-parity-1184/`）
に対する正式計算は CI では行わない（実機ダンプが入力のため。README 参照）。

検証観点:
- 候補 A/B の純粋関数（`evaluate_a`/`evaluate_a4`/`evaluate_b`）が境界値
  （bound ちょうど・bound 未満・bound 超過）で期待どおり救済／非救済を返す
- `compute_metrics` が xorshift 再現の小 n（n=8）合成ダンプに対して
  `fma_bit_match=True`（RNG・FMA 逐次再現が正しく機能している証拠）を返す
- CLI の fail-closed 挙動: 存在しないパス・不正ラベル・`--n` 範囲外・
  bit パターン改変ダンプがいずれも非 0 終了する
"""

import argparse
import importlib.util
import io
import math
import os
import re
import struct
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))

_TRUTH_SPEC = importlib.util.spec_from_file_location(
    "parity_dump_truth", os.path.join(HERE, "parity_dump_truth.py")
)
parity_dump_truth = importlib.util.module_from_spec(_TRUTH_SPEC)
import sys as _sys

_sys.modules[_TRUTH_SPEC.name] = parity_dump_truth
_TRUTH_SPEC.loader.exec_module(parity_dump_truth)

_CAND_SPEC = importlib.util.spec_from_file_location(
    "parity_tolerance_candidates", os.path.join(HERE, "parity_tolerance_candidates.py")
)
ptc = importlib.util.module_from_spec(_CAND_SPEC)
_sys.modules[_CAND_SPEC.name] = ptc
_CAND_SPEC.loader.exec_module(ptc)


def _metric(d=0.0, d_ex=0.0, max_partial=1.0, max_ab=0.25, sum_abs_ab=1.0, exact=1.0):
    return ptc.ElementMetrics(
        idx=0,
        row=0,
        col=0,
        d=d,
        d_ex=d_ex,
        d_ref_ex=0.0,
        max_partial=max_partial,
        max_ab=max_ab,
        sum_abs_ab=sum_abs_ab,
        exact=exact,
        fma_bit_match=True,
    )


class CandidateAEvaluationTest(unittest.TestCase):
    def test_bound_formula_k_fixed_m(self):
        cand = ptc.CandidateA(
            name="test", eps_label="2^-23", eps_value=2.0**-23, c=1.0, k_mode="K", m_mode="fixed0.25"
        )
        m = _metric()
        expected = 1.0 * (2.0**-23) * 2048 * 0.25
        self.assertAlmostEqual(cand.bound(m, 2048), expected)

    def test_bound_formula_sqrtk(self):
        cand = ptc.CandidateA(
            name="test", eps_label="2^-23", eps_value=2.0**-23, c=1.0, k_mode="sqrtK", m_mode="fixed0.25"
        )
        m = _metric()
        expected = 1.0 * (2.0**-23) * math.sqrt(2048) * 0.25
        self.assertAlmostEqual(cand.bound(m, 2048), expected)

    def test_bound_formula_actual_m(self):
        cand = ptc.CandidateA(
            name="test", eps_label="2^-23", eps_value=2.0**-23, c=1.0, k_mode="K", m_mode="actual"
        )
        m = _metric(max_ab=0.5)
        expected = 1.0 * (2.0**-23) * 2048 * 0.5
        self.assertAlmostEqual(cand.bound(m, 2048), expected)

    def test_rescue_at_boundary(self):
        cand = ptc.CandidateA(
            name="test", eps_label="2^-23", eps_value=2.0**-23, c=1.0, k_mode="K", m_mode="fixed0.25"
        )
        bound = cand.bound(_metric(), 2048)
        rescued = _metric(d=bound)
        not_rescued = _metric(d=bound * 1.0001)
        self.assertEqual(ptc.evaluate_a([rescued], cand, 2048), 1)
        self.assertEqual(ptc.evaluate_a([not_rescued], cand, 2048), 0)

    def test_a4_rescue(self):
        cand = ptc.CandidateA4()
        m = _metric(sum_abs_ab=100.0)
        bound = cand.bound(m, 2048)
        self.assertEqual(ptc.evaluate_a4([_metric(d=bound, sum_abs_ab=100.0)], cand, 2048), 1)
        self.assertEqual(
            ptc.evaluate_a4([_metric(d=bound * 2, sum_abs_ab=100.0)], cand, 2048), 0
        )


class CandidateBEvaluationTest(unittest.TestCase):
    def test_bound_uses_ulp_of_base(self):
        cand = ptc.CandidateB(name="test", base_mode="max_partial", t=8.0)
        m = _metric(max_partial=4.0)
        expected = 8.0 * parity_dump_truth.ulp_f32(4.0)
        self.assertAlmostEqual(cand.bound(m), expected)

    def test_base_mode_sum_abs_ab(self):
        cand = ptc.CandidateB(name="test", base_mode="sum_abs_ab", t=1.0)
        m = _metric(sum_abs_ab=128.0)
        self.assertAlmostEqual(cand.base_value(m), 128.0)

    def test_base_mode_exact(self):
        cand = ptc.CandidateB(name="test", base_mode="exact", t=1.0)
        m = _metric(exact=0.5)
        self.assertAlmostEqual(cand.base_value(m), 0.5)

    def test_evaluate_b_d_vs_d_ex_independent(self):
        cand = ptc.CandidateB(name="test", base_mode="sum_abs_ab", t=1.0)
        m = _metric(d=0.0, d_ex=1e9, sum_abs_ab=1.0)
        self.assertEqual(ptc.evaluate_b([m], cand, "d"), 1)
        self.assertEqual(ptc.evaluate_b([m], cand, "d_ex"), 0)

    def test_unknown_base_mode_raises(self):
        cand = ptc.CandidateB(name="test", base_mode="bogus", t=1.0)
        with self.assertRaises(ValueError):
            cand.base_value(_metric())


class ComputeMetricsSmokeTest(unittest.TestCase):
    """n=8 の合成ダンプで `compute_metrics` の end-to-end 経路（xorshift 再現・
    FMA 逐次再現・厳密真値抽出）を通す。実データと同じ乱数源
    （`Xorshift64StarExact`／`SEED_A`／`SEED_B`）を使うことで `fma_bit_match`
    が本物のダンプと同様 True になることを確認する。
    """

    def test_fma_bit_match_true_for_reconstructed_row(self):
        n = 8
        row, col = 2, 5
        a_rows = parity_dump_truth.extract_rows_exact(parity_dump_truth.SEED_A, n, {row})
        b_cols = parity_dump_truth.extract_cols_exact(parity_dump_truth.SEED_B, n, {col})
        fma_f32, _partials = parity_dump_truth.fma_sequential_f32_exact(a_rows[row], b_cols[col])
        ref_bits = struct.unpack("<I", struct.pack("<f", fma_f32))[0]

        # actual を ref からわずかに摂動させ、abs/rel をダンプ書式と同一式
        # （`element_error`）で再計算した合成 PARITY_DUMP 行を作る。
        actual_f32 = fma_f32 + 1e-6
        actual_bits = struct.unpack("<I", struct.pack("<f", actual_f32))[0]
        actual_f32 = struct.unpack("<f", struct.pack("<I", actual_bits))[0]
        abs_err = abs(fma_f32 - actual_f32)
        scale = max(abs(fma_f32), abs(actual_f32), 1e-12)
        rel_err = abs_err / scale
        idx = row * n + col
        line = (
            f"PARITY_DUMP call=1 n={n} idx={idx} row={row} col={col} "
            f"ref={fma_f32!r} ref_bits=0x{ref_bits:08x} "
            f"actual={actual_f32!r} actual_bits=0x{actual_bits:08x} "
            f"abs={abs_err!r} rel={rel_err!r}"
        )
        error_count = [0]
        rows = list(parity_dump_truth.parse_dump_lines([line], n, error_count=error_count))
        self.assertEqual(error_count[0], 0)
        self.assertEqual(len(rows), 1)

        metrics = ptc.compute_metrics(rows, n)
        self.assertEqual(len(metrics), 1)
        self.assertTrue(metrics[0].fma_bit_match)
        self.assertAlmostEqual(metrics[0].d, abs_err, places=6)


class EscapeMdCellTest(unittest.TestCase):
    """`_escape_md_cell` が Markdown テーブルの列区切り `|` を破壊しないことを
    確認する（イシュー #1237 codex-review 指摘・P2。`CandidateA4.name` の
    `"A-4 (K*u*sum|ab|)"` が実例）。
    """

    def test_pipe_escaped(self):
        self.assertEqual(ptc._escape_md_cell("A-4 (K*u*sum|ab|)"), "A-4 (K*u*sum\\|ab\\|)")

    def test_no_pipe_unchanged(self):
        self.assertEqual(ptc._escape_md_cell("A-1 c=1.0 eps=2^-23 K*0.25"), "A-1 c=1.0 eps=2^-23 K*0.25")

    def test_render_markdown_a4_row_has_expected_column_count(self):
        # `render_markdown` が生成する候補 A 表で、A-4 行の `|` 区切り数が
        # 他候補行と一致すること（エスケープ漏れがあれば列がずれて検出
        # できる）を end-to-end で確認する。
        n = 8
        row, col = 2, 5
        a_rows = parity_dump_truth.extract_rows_exact(parity_dump_truth.SEED_A, n, {row})
        b_cols = parity_dump_truth.extract_cols_exact(parity_dump_truth.SEED_B, n, {col})
        fma_f32, _partials = parity_dump_truth.fma_sequential_f32_exact(a_rows[row], b_cols[col])
        ref_bits = struct.unpack("<I", struct.pack("<f", fma_f32))[0]
        idx = row * n + col
        line = (
            f"PARITY_DUMP call=1 n={n} idx={idx} row={row} col={col} "
            f"ref={fma_f32!r} ref_bits=0x{ref_bits:08x} "
            f"actual={fma_f32!r} actual_bits=0x{ref_bits:08x} "
            "abs=0.0 rel=0.0"
        )
        error_count = [0]
        rows = list(parity_dump_truth.parse_dump_lines([line], n, error_count=error_count))
        metrics = ptc.compute_metrics(rows, n)
        doc = ptc.render_markdown({"cuda": metrics}, n)
        table_lines = [ln for ln in doc.splitlines() if ln.startswith("| A-")]
        self.assertTrue(table_lines, "候補 A 表の行が見つからない")
        # エスケープされた `\|` は列区切りではないため数えない
        # （`_escape_md_cell` が正しく機能していれば A-4 行の `sum|ab|` 由来
        # の `|` はすべて `\|` になり、生の列区切り数が他候補行と一致する）。
        unescaped_pipe_re = re.compile(r"(?<!\\)\|")
        pipe_counts = {len(unescaped_pipe_re.findall(ln)) for ln in table_lines}
        self.assertEqual(
            len(pipe_counts), 1, f"候補 A 表の列区切り数が行ごとに異なる: {table_lines}"
        )
        a4_lines = [ln for ln in table_lines if "A-4" in ln]
        self.assertEqual(len(a4_lines), 1)
        self.assertIn("sum\\|ab\\|", a4_lines[0])
        self.assertNotIn("sum|ab|", a4_lines[0])


class ExtraEpsCParameterizationTest(unittest.TestCase):
    """`--extra-eps`/`--extra-c`（イシュー #1984／#1985）の単位丸め `u`
    パラメータ化を検証する。既定出力の不変性・`build_extra_candidates_a`
    の組合せ生成・`bound()` の計算式・CLI パーサの fail-closed 検証を対象
    とする。
    """

    def test_build_extra_candidates_a_empty_when_either_missing(self):
        # eps のみ／c のみでは空リスト（既定出力を byte 単位で不変に保つ契約）。
        self.assertEqual(ptc.build_extra_candidates_a([("u", 1e-3)], []), [])
        self.assertEqual(ptc.build_extra_candidates_a([], [0.5]), [])
        self.assertEqual(ptc.build_extra_candidates_a([], []), [])

    def test_build_extra_candidates_a_full_cross_product(self):
        # 1 eps × 2 c × 2 k_mode(K/sqrtK) = 4 件。
        cands = ptc.build_extra_candidates_a([("tf32_u=2^-11", 2.0**-11)], [0.5, 1.0])
        self.assertEqual(len(cands), 4)
        names = {c.name for c in cands}
        self.assertIn("EXTRA c=0.5 eps=tf32_u=2^-11 K*0.25", names)
        self.assertIn("EXTRA c=0.5 eps=tf32_u=2^-11 sqrtK*0.25", names)
        self.assertIn("EXTRA c=1.0 eps=tf32_u=2^-11 K*0.25", names)
        self.assertIn("EXTRA c=1.0 eps=tf32_u=2^-11 sqrtK*0.25", names)
        for c in cands:
            self.assertEqual(c.eps_value, 2.0**-11)
            self.assertEqual(c.m_mode, "fixed0.25")

    def test_extra_candidate_bound_matches_formula(self):
        # イシュー #1984 相当（u=2^-11・c=0.5・K=4096）の bound 式突合。
        cands = ptc.build_extra_candidates_a([("u", 2.0**-11)], [0.5])
        k_cand = next(c for c in cands if c.k_mode == "K")
        sqrtk_cand = next(c for c in cands if c.k_mode == "sqrtK")
        m = _metric()
        self.assertAlmostEqual(k_cand.bound(m, 4096), 0.5 * (2.0**-11) * 4096 * 0.25)
        self.assertAlmostEqual(sqrtk_cand.bound(m, 4096), 0.5 * (2.0**-11) * math.sqrt(4096) * 0.25)

    def test_render_markdown_default_output_unchanged_with_empty_extras(self):
        # extra_candidates_a=None と extra_candidates_a=[] が既定（何も
        # 指定しない）出力と byte 単位で一致すること（イシュー #1984／#1985
        # の要求「既定出力は byte 単位で不変」の直接検証）。
        n = 8
        row, col = 2, 5
        a_rows = parity_dump_truth.extract_rows_exact(parity_dump_truth.SEED_A, n, {row})
        b_cols = parity_dump_truth.extract_cols_exact(parity_dump_truth.SEED_B, n, {col})
        fma_f32, _partials = parity_dump_truth.fma_sequential_f32_exact(a_rows[row], b_cols[col])
        ref_bits = struct.unpack("<I", struct.pack("<f", fma_f32))[0]
        idx = row * n + col
        line = (
            f"PARITY_DUMP call=1 n={n} idx={idx} row={row} col={col} "
            f"ref={fma_f32!r} ref_bits=0x{ref_bits:08x} "
            f"actual={fma_f32!r} actual_bits=0x{ref_bits:08x} "
            "abs=0.0 rel=0.0"
        )
        error_count = [0]
        rows = list(parity_dump_truth.parse_dump_lines([line], n, error_count=error_count))
        metrics = ptc.compute_metrics(rows, n)
        doc_none = ptc.render_markdown({"cuda": metrics}, n)
        doc_default = ptc.render_markdown({"cuda": metrics}, n, extra_candidates_a=None)
        doc_empty = ptc.render_markdown({"cuda": metrics}, n, extra_candidates_a=[])
        self.assertEqual(doc_none, doc_default)
        self.assertEqual(doc_none, doc_empty)
        self.assertNotIn("EXTRA", doc_none)

    def test_render_markdown_with_extras_appends_extra_rows(self):
        n = 8
        row, col = 2, 5
        a_rows = parity_dump_truth.extract_rows_exact(parity_dump_truth.SEED_A, n, {row})
        b_cols = parity_dump_truth.extract_cols_exact(parity_dump_truth.SEED_B, n, {col})
        fma_f32, _partials = parity_dump_truth.fma_sequential_f32_exact(a_rows[row], b_cols[col])
        ref_bits = struct.unpack("<I", struct.pack("<f", fma_f32))[0]
        idx = row * n + col
        line = (
            f"PARITY_DUMP call=1 n={n} idx={idx} row={row} col={col} "
            f"ref={fma_f32!r} ref_bits=0x{ref_bits:08x} "
            f"actual={fma_f32!r} actual_bits=0x{ref_bits:08x} "
            "abs=0.0 rel=0.0"
        )
        error_count = [0]
        rows = list(parity_dump_truth.parse_dump_lines([line], n, error_count=error_count))
        metrics = ptc.compute_metrics(rows, n)
        extras = ptc.build_extra_candidates_a([("tf32_u=2^-11", 2.0**-11)], [0.5])
        doc = ptc.render_markdown({"cuda": metrics}, n, extra_candidates_a=extras)
        self.assertIn("EXTRA c=0.5 eps=tf32_u=2^-11 K*0.25", doc)
        self.assertIn("EXTRA c=0.5 eps=tf32_u=2^-11 sqrtK*0.25", doc)

    def test_positive_finite_float_rejects_non_positive_and_non_finite(self):
        for bad in ("0", "-1.0", "nan", "inf", "-inf", "not-a-number"):
            with self.assertRaises(argparse.ArgumentTypeError):
                ptc._positive_finite_float(bad, "--extra-c")

    def test_parse_extra_eps_arg_valid(self):
        label, value = ptc._parse_extra_eps_arg("tf32u2^-11=0.00048828125")
        self.assertEqual(label, "tf32u2^-11")
        self.assertAlmostEqual(value, 0.00048828125)

    def test_parse_extra_eps_arg_rejects_missing_equals(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            ptc._parse_extra_eps_arg("no-equals-sign")

    def test_parse_extra_eps_arg_rejects_invalid_label(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            ptc._parse_extra_eps_arg("bad|label=0.5")

    def test_cli_extra_eps_without_extra_c_leaves_output_unchanged(self):
        # イシュー #1984／#1985 の契約「片方のみ指定では追加候補が生成
        # されない」を CLI 経由でも確認する。
        n = 8
        row, col = 2, 5
        a_rows = parity_dump_truth.extract_rows_exact(parity_dump_truth.SEED_A, n, {row})
        b_cols = parity_dump_truth.extract_cols_exact(parity_dump_truth.SEED_B, n, {col})
        fma_f32, _partials = parity_dump_truth.fma_sequential_f32_exact(a_rows[row], b_cols[col])
        ref_bits = struct.unpack("<I", struct.pack("<f", fma_f32))[0]
        idx = row * n + col
        line = (
            f"PARITY_DUMP call=1 n={n} idx={idx} row={row} col={col} "
            f"ref={fma_f32!r} ref_bits=0x{ref_bits:08x} "
            f"actual={fma_f32!r} actual_bits=0x{ref_bits:08x} "
            "abs=0.0 rel=0.0"
        )
        import tempfile

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        f.write(line + "\n")
        f.close()
        try:
            out_base, err_base = io.StringIO(), io.StringIO()
            with redirect_stdout(out_base), redirect_stderr(err_base):
                code_base = ptc.main(["--n", str(n), "--dump", f"cuda={f.name}"])
            out_extra, err_extra = io.StringIO(), io.StringIO()
            with redirect_stdout(out_extra), redirect_stderr(err_extra):
                code_extra = ptc.main(
                    ["--n", str(n), "--dump", f"cuda={f.name}", "--extra-eps", "u=0.5"]
                )
            self.assertEqual(code_base, 0)
            self.assertEqual(code_extra, 0)
            self.assertEqual(out_base.getvalue(), out_extra.getvalue())
            self.assertNotIn("EXTRA", out_extra.getvalue())
        finally:
            os.unlink(f.name)


class CliFailClosedTest(unittest.TestCase):
    def _run(self, args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = ptc.main(args)
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_nonexistent_path_rejected(self):
        code, _out, err = self._run(["--n", "8", "--dump", "cuda=/nonexistent/path.txt"])
        self.assertNotEqual(code, 0)

    def test_invalid_label_rejected(self):
        real_path = os.path.join(HERE, "parity_dump_truth.py")  # 存在確認だけ通す任意の実ファイル
        code, _out, err = self._run(["--n", "8", "--dump", f"BAD_LABEL={real_path}"])
        self.assertNotEqual(code, 0)

    def test_n_out_of_range_rejected(self):
        real_path = os.path.join(HERE, "parity_dump_truth.py")
        code, _out, err = self._run(["--n", "0", "--dump", f"cuda={real_path}"])
        self.assertEqual(code, 2)

    def test_duplicate_idx_mismatch_rejected(self):
        # 同一 idx が複数 call にわたって再登場し、かつ値（ref_bits）が食い違う
        # 破損・非決定的ダンプ。`parse_dump_lines` 自体は個々の行として妥当
        # （書式・自己整合性は満たす）ため malformed としては検出されない。
        # `main` 側の重複 idx 突合（`parity_dump_truth.py::main` と同じ防御）
        # が先勝ち破棄せず fail-closed で検出することを確認する（イシュー
        # #1237 codex-review 指摘・P0）。
        n = 8
        row, col = 1, 1
        idx = row * n + col
        line1 = (
            f"PARITY_DUMP call=1 n={n} idx={idx} row={row} col={col} "
            "ref=1.0 ref_bits=0x3f800000 "
            "actual=1.0 actual_bits=0x3f800000 "
            "abs=0.0 rel=0.0"
        )
        line2 = (
            f"PARITY_DUMP call=2 n={n} idx={idx} row={row} col={col} "
            "ref=2.0 ref_bits=0x40000000 "
            "actual=2.0 actual_bits=0x40000000 "
            "abs=0.0 rel=0.0"
        )
        import tempfile

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        f.write(line1 + "\n" + line2 + "\n")
        f.close()
        try:
            code, _out, err = self._run(["--n", str(n), "--dump", f"cuda={f.name}"])
            self.assertNotEqual(code, 0)
            self.assertIn("重複レコードが不一致", err)
        finally:
            os.unlink(f.name)

    def test_duplicate_idx_consistent_accepted(self):
        # 同一 idx の再登場でも値が完全一致（決定的な再計測）なら先頭の
        # 出現を代表値として黙って採用し、エラーにしない（従来挙動の維持）。
        # `ComputeMetricsSmokeTest` と同じ手順で `fma_bit_match=True` となる
        # ref/actual を実データ同様の乱数源から構成する。
        n = 8
        row, col = 2, 5
        a_rows = parity_dump_truth.extract_rows_exact(parity_dump_truth.SEED_A, n, {row})
        b_cols = parity_dump_truth.extract_cols_exact(parity_dump_truth.SEED_B, n, {col})
        fma_f32, _partials = parity_dump_truth.fma_sequential_f32_exact(a_rows[row], b_cols[col])
        ref_bits = struct.unpack("<I", struct.pack("<f", fma_f32))[0]
        idx = row * n + col
        line_tpl = (
            "PARITY_DUMP call={call} n=" + str(n) + f" idx={idx} row={row} col={col} "
            f"ref={fma_f32!r} ref_bits=0x{ref_bits:08x} "
            f"actual={fma_f32!r} actual_bits=0x{ref_bits:08x} "
            "abs=0.0 rel=0.0"
        )
        import tempfile

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        f.write(line_tpl.format(call=1) + "\n" + line_tpl.format(call=2) + "\n")
        f.close()
        try:
            code, _out, err = self._run(["--n", str(n), "--dump", f"cuda={f.name}"])
            self.assertEqual(code, 0, err)
        finally:
            os.unlink(f.name)

    def test_duplicate_label_rejected(self):
        # 同じ LABEL で --dump を複数指定すると、後段の
        # `metrics_by_label[label] = ...` 代入で先の入力が無言で上書き
        # される（CUDA・CPU に誤って同じラベルを付けると一方の要素と
        # fail 数が報告から静かに消える）。引数解析後・実データ処理前に
        # 重複を検出して fail-closed で拒否することを確認する（イシュー
        # #1237 codex-review 指摘・P2）。
        real_path = os.path.join(HERE, "parity_dump_truth.py")
        code, _out, err = self._run(
            [
                "--n",
                "8",
                "--dump",
                f"cuda={real_path}",
                "--dump",
                f"cuda={real_path}",
            ]
        )
        self.assertNotEqual(code, 0)
        self.assertIn("複数回指定", err)

    def test_corrupted_dump_rejected(self):
        n = 8
        row, col = 1, 1
        idx = row * n + col
        # bit パターンとテキスト表現が矛盾する破損行（`ref_text_f32 != ref_bits_f32`
        # を誘発する）。
        line = (
            f"PARITY_DUMP call=1 n={n} idx={idx} row={row} col={col} "
            "ref=1.0 ref_bits=0xdeadbeef "
            "actual=1.0 actual_bits=0xdeadbeef "
            "abs=0.0 rel=0.0"
        )
        import tempfile

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        f.write(line + "\n")
        f.close()
        try:
            code, _out, err = self._run(["--n", str(n), "--dump", f"cuda={f.name}"])
            self.assertNotEqual(code, 0)
            self.assertIn("不一致", err)
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
