#!/usr/bin/env python3
"""`compare_readout_ab.py` の単体テスト（イシュー #1477）。

`compare_managed_ab_test.py` と同じ方式（ファイルパス指定 import・
tempfile への合成 JSONL 書き出し）。CI（`ci.yml` の `deps-forbidden`
ジョブ）は
`python3 -m unittest scripts/bench/framework-compare/compare_readout_ab_test.py`
で本ファイルを実行する。

検証観点:
- 正常系: legacy/borrowed 各 5 件から中央値・比率・checksum 一致列を
  算出し、全 6 セル非後退なら ADOPT を返す。
- 1 セルでも後退（ratio > threshold）すれば総合判定は REJECT。
- 件数過不足・checksum 不一致セルが 1 つでもあれば総合判定は undetermined。
- `readout` キーを持たない行（通常計測行）は除外する。
- 不正な JSON 行・不正な `readout` フィールド値は理由付きで警告しスキップ
  する（例外を送出しない）。
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
    "compare_readout_ab", os.path.join(HERE, "compare_readout_ab.py")
)
compare_readout_ab = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_readout_ab)


def _rec(
    readout,
    median_s,
    checksum=1.23456,
    task="gemm",
    device="metal",
    size=1024,
    mode="reuse",
    parity_fail_count=0,
):
    r = {
        "framework": "fandhe-ai",
        "version": "0.1.0",
        "task": task,
        "device": device,
        "size": size,
        "median_s": median_s,
        "q1_s": median_s,
        "q3_s": median_s,
        "checksum": checksum,
        "warmup": 20,
        "iters": 20,
        "mode": mode,
        # イシュー #1477 P1（PR #1493 codex-review 指摘）: `evaluate_cell`
        # が parity を見ない問題の回帰テスト用に既定 0 fail を付与する
        # （`bench-fandhe` が emit する `parity_fail_count` フラットキーと
        # 同型。`parity_fail_count=None` で意図的にキー省略できる）。
        "parity_fail_count": parity_fail_count,
    }
    if parity_fail_count is None:
        del r["parity_fail_count"]
    if readout is not None:
        r["readout"] = readout
    return r


def _write_jsonl(rows):
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


def _five_cell(median_legacy, median_borrowed, checksum=1.23456, **kwargs):
    """1 セル分の legacy 5 件 + borrowed 5 件を生成する。"""
    rows = []
    for _ in range(5):
        rows.append(_rec("legacy", median_legacy, checksum=checksum, **kwargs))
        rows.append(_rec("borrowed", median_borrowed, checksum=checksum, **kwargs))
    return rows


def _gate_rows(ratio=0.9, checksum=1.23456):
    """6 セル（size×mode）すべてで非後退な合成データを生成する。"""
    rows = []
    for size in (1024, 2048, 4096):
        for mode in ("fresh", "reuse"):
            legacy_median = 0.01
            borrowed_median = legacy_median * ratio
            rows += _five_cell(
                legacy_median, borrowed_median, checksum=checksum, size=size, mode=mode
            )
    return rows


class LoadRowsTest(unittest.TestCase):
    def test_valid_rows_are_loaded(self):
        rows = [_rec("legacy", 0.001), _rec("borrowed", 0.0009)]
        path = _write_jsonl(rows)
        try:
            loaded, warnings = compare_readout_ab.load_rows(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(warnings, [])
        finally:
            os.unlink(path)

    def test_rows_without_readout_key_are_excluded(self):
        rows = [_rec(None, 0.001), _rec("borrowed", 0.0009)]
        path = _write_jsonl(rows)
        try:
            loaded, warnings = compare_readout_ab.load_rows(path)
            self.assertEqual(len(loaded), 1)
            self.assertTrue(any("readout" in w for w in warnings))
        finally:
            os.unlink(path)

    def test_invalid_readout_value_is_skipped_with_warning(self):
        r = _rec("legacy", 0.001)
        r["readout"] = "legacyish"
        path = _write_jsonl([r])
        try:
            loaded, warnings = compare_readout_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertTrue(any("readout" in w for w in warnings))
        finally:
            os.unlink(path)

    def test_invalid_json_line_is_skipped_with_warning(self):
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ).name
        with open(path, "w", encoding="utf-8") as f:
            f.write("not json\n")
        try:
            loaded, warnings = compare_readout_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertTrue(any("invalid JSON" in w for w in warnings))
        finally:
            os.unlink(path)

    def test_non_metal_device_row_is_excluded_by_valid_cell_identity(self):
        rows = [
            _rec("legacy", 0.001, device="cuda"),
            _rec("borrowed", 0.0009, device="cuda"),
        ]
        path = _write_jsonl(rows)
        try:
            loaded, warnings = compare_readout_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertTrue(warnings)
        finally:
            os.unlink(path)

    def test_managed_true_row_is_excluded(self):
        r = _rec("legacy", 0.001)
        r["managed"] = True
        path = _write_jsonl([r])
        try:
            loaded, warnings = compare_readout_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertTrue(any("managed" in w for w in warnings))
        finally:
            os.unlink(path)

    def test_graph_key_row_is_excluded(self):
        r = _rec("legacy", 0.001)
        r["graph"] = "on"
        path = _write_jsonl([r])
        try:
            loaded, warnings = compare_readout_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertTrue(any("graph" in w for w in warnings))
        finally:
            os.unlink(path)

    def test_size_set_filters_out_of_gate_sizes(self):
        rows = [_rec("legacy", 0.001, size=512), _rec("borrowed", 0.0009, size=512)]
        path = _write_jsonl(rows)
        try:
            loaded, warnings = compare_readout_ab.load_rows(
                path, size_set=compare_readout_ab.GATE_SIZES
            )
            self.assertEqual(loaded, [])
            self.assertTrue(warnings)
        finally:
            os.unlink(path)


class EvaluateCellTest(unittest.TestCase):
    def test_full_cell_computes_ratio_and_exact_match(self):
        rows = _five_cell(0.010, 0.009)
        legacy = [r for r in rows if r["readout"] == "legacy"]
        borrowed = [r for r in rows if r["readout"] == "borrowed"]
        result = compare_readout_ab.evaluate_cell(legacy, borrowed)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["ratio"], 0.9, places=6)
        self.assertTrue(result["checksum_exact_match"])

    def test_missing_rows_is_undeterminable(self):
        rows = _five_cell(0.010, 0.009)
        legacy = [r for r in rows if r["readout"] == "legacy"][:4]
        borrowed = [r for r in rows if r["readout"] == "borrowed"]
        result = compare_readout_ab.evaluate_cell(legacy, borrowed)
        self.assertEqual(result["status"], "undeterminable")

    def test_checksum_mismatch_is_undeterminable(self):
        legacy = [_rec("legacy", 0.010, checksum=1.0) for _ in range(5)]
        borrowed = [_rec("borrowed", 0.009, checksum=999.0) for _ in range(5)]
        result = compare_readout_ab.evaluate_cell(legacy, borrowed)
        self.assertEqual(result["status"], "undeterminable")


class OverallVerdictTest(unittest.TestCase):
    def test_all_cells_nonregressed_yields_adopt(self):
        rows = _gate_rows(ratio=0.8)
        cells = compare_readout_ab.split_legacy_borrowed(rows)
        verdict = compare_readout_ab.overall_verdict(
            cells, threshold=1.00, device="metal", size_set=compare_readout_ab.GATE_SIZES
        )
        self.assertTrue(verdict.startswith("ADOPT"))

    def test_one_cell_regressed_yields_reject(self):
        rows = _gate_rows(ratio=0.8)
        # 1 セル（size=4096, mode=reuse）だけ borrowed を後退させる。
        rows = [
            r
            for r in rows
            if not (r["size"] == 4096 and r["mode"] == "reuse")
        ]
        rows += _five_cell(0.010, 0.012, size=4096, mode="reuse")
        cells = compare_readout_ab.split_legacy_borrowed(rows)
        verdict = compare_readout_ab.overall_verdict(
            cells, threshold=1.00, device="metal", size_set=compare_readout_ab.GATE_SIZES
        )
        self.assertTrue(verdict.startswith("REJECT"))

    def test_missing_cell_yields_undetermined(self):
        rows = _gate_rows(ratio=0.8)
        # 1 セル分の borrowed を欠落させ「判定不能」を誘発する。
        rows = [
            r
            for r in rows
            if not (
                r["size"] == 2048 and r["mode"] == "fresh" and r["readout"] == "borrowed"
            )
        ]
        cells = compare_readout_ab.split_legacy_borrowed(rows)
        verdict = compare_readout_ab.overall_verdict(
            cells, threshold=1.00, device="metal", size_set=compare_readout_ab.GATE_SIZES
        )
        self.assertTrue(verdict.startswith("undetermined"))

    def test_entirely_absent_required_cell_yields_undetermined_not_adopt(self):
        """codex-review 指摘（PR #1493 P1）: 必須 6 セルのうち 1 セルが
        丸ごと欠測（対応する行が 1 件も無い）場合でも、存在する 5 セルが
        全て非後退なら ADOPT になってしまっていた問題の回帰テスト。
        """
        rows = _gate_rows(ratio=0.8)
        # size=4096, mode=reuse セルの行を丸ごと除去する（`_five_cell`
        # で legacy/borrowed 双方生成される行を全削除。`test_missing_
        # cell_yields_undetermined` は borrowed 側のみを間引くのに対し、
        # 本テストはセルキー自体を `cells` 辞書から消す）。
        rows = [
            r for r in rows if not (r["size"] == 4096 and r["mode"] == "reuse")
        ]
        cells = compare_readout_ab.split_legacy_borrowed(rows)
        self.assertNotIn(("gemm", "metal", 4096, "reuse", None), cells)
        verdict = compare_readout_ab.overall_verdict(
            cells, threshold=1.00, device="metal", size_set=compare_readout_ab.GATE_SIZES
        )
        self.assertTrue(
            verdict.startswith("undetermined"),
            f"欠測セルがあるのに ADOPT/REJECT が確定した: {verdict!r}",
        )

    def test_parity_fail_count_positive_yields_reject_even_if_faster(self):
        """codex-review 指摘（PR #1493 P1／Cursor Bugbot Medium）:
        `evaluate_cell` は checksum・所要時間のみを見て parity を検証
        しないため、`parity_fail_count` が正の行があっても ADOPT に
        なりうる問題の回帰テスト。
        """
        rows = _gate_rows(ratio=0.8)
        rows = [
            r for r in rows if not (r["size"] == 1024 and r["mode"] == "fresh")
        ]
        rows += _five_cell(
            0.010,
            0.008,
            size=1024,
            mode="fresh",
            parity_fail_count=1,
        )
        cells = compare_readout_ab.split_legacy_borrowed(rows)
        verdict = compare_readout_ab.overall_verdict(
            cells, threshold=1.00, device="metal", size_set=compare_readout_ab.GATE_SIZES
        )
        self.assertTrue(
            verdict.startswith("REJECT"),
            f"parity_fail_count>0 の行があるのに ADOPT になった: {verdict!r}",
        )

    def test_parity_fail_count_negative_yields_reject(self):
        """codex-review 指摘（PR #1493 スレッド 2）: `parity_fail_count` が
        負整数（例 `-1`）の行は `> 0` 判定のみでは素通りし「全行で parity
        0 fail」契約に反して ADOPT になりうる。`!= 0` 判定で負整数も
        fail 扱いになることの回帰テスト。
        """
        rows = _gate_rows(ratio=0.8)
        rows = [
            r for r in rows if not (r["size"] == 4096 and r["mode"] == "fresh")
        ]
        rows += _five_cell(
            0.010,
            0.008,
            size=4096,
            mode="fresh",
            parity_fail_count=-1,
        )
        cells = compare_readout_ab.split_legacy_borrowed(rows)
        verdict = compare_readout_ab.overall_verdict(
            cells, threshold=1.00, device="metal", size_set=compare_readout_ab.GATE_SIZES
        )
        self.assertTrue(
            verdict.startswith("REJECT"),
            f"parity_fail_count が負整数の行があるのに ADOPT になった: {verdict!r}",
        )

    def test_parity_fail_count_missing_yields_reject(self):
        """`parity_fail_count` キー自体が欠損する行（fail-closed 方針）は
        「未検証」として ADOPT を許さない。
        """
        rows = _gate_rows(ratio=0.8)
        rows = [
            r for r in rows if not (r["size"] == 2048 and r["mode"] == "reuse")
        ]
        rows += _five_cell(
            0.010,
            0.008,
            size=2048,
            mode="reuse",
            parity_fail_count=None,
        )
        cells = compare_readout_ab.split_legacy_borrowed(rows)
        verdict = compare_readout_ab.overall_verdict(
            cells, threshold=1.00, device="metal", size_set=compare_readout_ab.GATE_SIZES
        )
        self.assertTrue(
            verdict.startswith("REJECT"),
            f"parity_fail_count 欠損の行があるのに ADOPT になった: {verdict!r}",
        )


class MainTest(unittest.TestCase):
    def test_main_prints_adopt_table_and_returns_zero(self):
        rows = _gate_rows(ratio=0.8)
        path = _write_jsonl(rows)
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                code = compare_readout_ab.main(
                    ["compare_readout_ab.py", path, "--sizes", "gate"]
                )
            self.assertEqual(code, 0)
            self.assertIn("ADOPT", out.getvalue())
        finally:
            os.unlink(path)

    def test_main_returns_nonzero_on_undeterminable_cell(self):
        rows = _gate_rows(ratio=0.8)
        rows = [
            r
            for r in rows
            if not (
                r["size"] == 1024 and r["mode"] == "reuse" and r["readout"] == "legacy"
            )
        ]
        path = _write_jsonl(rows)
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                code = compare_readout_ab.main(["compare_readout_ab.py", path])
            self.assertEqual(code, 3)
        finally:
            os.unlink(path)

    def test_main_returns_nonzero_and_undetermined_on_missing_required_cell(self):
        """codex-review 指摘（PR #1493 P1）: 6 セル中 1 セル（例: 1024/
        reuse のみ）が丸ごと欠測していても `main()` が ADOPT・終了コード
        0 を返してはならない回帰テスト（`--sizes gate` 既定・入力に
        size=4096, mode=reuse セルの行を含めない）。
        """
        rows = _gate_rows(ratio=0.8)
        rows = [
            r for r in rows if not (r["size"] == 4096 and r["mode"] == "reuse")
        ]
        path = _write_jsonl(rows)
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                code = compare_readout_ab.main(
                    ["compare_readout_ab.py", path, "--sizes", "gate"]
                )
            self.assertEqual(code, 3)
            self.assertIn("undetermined", out.getvalue())
            self.assertNotIn("ADOPT（", out.getvalue())
        finally:
            os.unlink(path)

    def test_main_returns_nonzero_on_malformed_input(self):
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ).name
        with open(path, "w", encoding="utf-8") as f:
            f.write("not json\n")
        try:
            err = io.StringIO()
            with redirect_stderr(err):
                code = compare_readout_ab.main(["compare_readout_ab.py", path])
            self.assertEqual(code, 3)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
