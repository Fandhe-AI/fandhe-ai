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
    }
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
        verdict = compare_readout_ab.overall_verdict(cells, threshold=1.00)
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
        verdict = compare_readout_ab.overall_verdict(cells, threshold=1.00)
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
        verdict = compare_readout_ab.overall_verdict(cells, threshold=1.00)
        self.assertTrue(verdict.startswith("undetermined"))


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
